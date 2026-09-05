"""Mirror a Google Drive folder into a local cache folder.

LabRAG never indexes Drive directly: it downloads the papers into a local
cache directory (mirroring the Drive folder structure) and then indexes that
folder exactly like a NAS folder. That keeps one code path for everything and
means the cache doubles as an offline copy of the lab's papers.

Two ways to authenticate, both read-only:

* **Service account** (best for a lab server): create a service account in
  Google Cloud, download its JSON key, and share the Drive folder with the
  service account's e-mail address. No browser, no expiring tokens.
* **OAuth** (best for a personal laptop): download an OAuth "Desktop app"
  client secret JSON; the first sync opens a browser to sign in, and the
  token is cached next to the index.

Only the Drive REST API v3 is used, through google-auth's AuthorizedSession,
so there is nothing to configure beyond the credentials file.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .parse import SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

# Google-native documents have no bytes to download; we export them as text.
GOOGLE_EXPORTS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}

FILE_FIELDS = "id,name,mimeType,md5Checksum,modifiedTime,size,shortcutDetails"
MANIFEST_NAME = ".labrag-drive-manifest.json"

_FOLDER_URL_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_ID_PARAM_RE = re.compile(r"[?&]id=([A-Za-z0-9_-]+)")


def parse_folder_id(value: str) -> str:
    """Accept a bare folder ID or any Drive folder URL and return the ID."""
    value = value.strip()
    m = _FOLDER_URL_RE.search(value) or _ID_PARAM_RE.search(value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise ValueError(
        f"{value!r} does not look like a Google Drive folder ID or folder link (expected https://drive.google.com/drive/folders/<id>)"
    )


class _Response(Protocol):
    status_code: int

    def json(self): ...
    def iter_content(self, chunk_size: int): ...
    def raise_for_status(self): ...


class Session(Protocol):
    """The subset of requests.Session / AuthorizedSession we use."""

    def get(self, url: str, params=None, stream: bool = False, timeout=None) -> _Response: ...


def load_credentials(
    service_account_file: Path | None = None,
    client_secret_file: Path | None = None,
    token_file: Path | None = None,
):
    """Build read-only Drive credentials from whichever file the user has."""
    if service_account_file:
        from google.oauth2.service_account import Credentials

        return Credentials.from_service_account_file(str(service_account_file), scopes=SCOPES)

    if client_secret_file:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = None
        if token_file and Path(token_file).exists():
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
            creds = flow.run_local_server(port=0)
            if token_file:
                Path(token_file).parent.mkdir(parents=True, exist_ok=True)
                Path(token_file).write_text(creds.to_json())
        return creds

    raise ValueError(
        "Google Drive is configured but no credentials were given. Set "
        "LABRAG_GOOGLE_SERVICE_ACCOUNT (path to a service-account JSON key) or "
        "LABRAG_GOOGLE_CLIENT_SECRET (path to an OAuth client secret JSON)."
    )


def authorized_session(credentials) -> Session:
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(credentials)


@dataclass
class DriveFile:
    id: str
    name: str
    mime: str
    rel_path: str  # path inside the cache folder, including export extension
    md5: str | None = None
    modified: str | None = None
    size: int | None = None
    export_mime: str | None = None

    @property
    def version(self) -> str:
        """Something that changes whenever the content changes."""
        return self.md5 or self.modified or ""


@dataclass
class SyncResult:
    added: list[Path] = field(default_factory=list)
    updated: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    unchanged: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[Path]:
        return self.added + self.updated

    def summary(self) -> str:
        return (
            f"{len(self.added)} new, {len(self.updated)} updated, {len(self.removed)} removed, "
            f"{self.unchanged} unchanged, {len(self.skipped)} skipped" + (f", {len(self.errors)} errors" if self.errors else "")
        )


class DriveClient:
    def __init__(self, session: Session, retries: int = 3, timeout: float = 60.0):
        self.session = session
        self.retries = retries
        self.timeout = timeout

    # -- low level ----------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, stream: bool = False):
        delay = 1.0
        for attempt in range(self.retries + 1):
            resp = self.session.get(url, params=params, stream=stream, timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                log.warning("Drive API %s, retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def list_children(self, folder_id: str) -> list[dict]:
        items: list[dict] = []
        page_token = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": f"nextPageToken,files({FILE_FIELDS})",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._get(f"{API}/files", params=params).json()
            items.extend(data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                return items

    def get_file(self, file_id: str) -> dict:
        return self._get(
            f"{API}/files/{file_id}",
            params={"fields": FILE_FIELDS, "supportsAllDrives": "true"},
        ).json()

    # -- tree walk ----------------------------------------------------------
    def walk(self, folder_id: str) -> Iterator[DriveFile]:
        """Yield every indexable file under a folder, recursively."""
        seen_folders: set[str] = set()
        seen_files: set[str] = set()  # a file reachable directly and via a shortcut is one file
        stack: list[tuple[str, str]] = [(folder_id, "")]
        while stack:
            fid, prefix = stack.pop()
            if fid in seen_folders:
                continue
            seen_folders.add(fid)
            children = self.list_children(fid)
            used_names: set[str] = set()
            # (name, id) gives a total order, so duplicate-name suffixes are stable between runs
            for item in sorted(children, key=lambda i: (i.get("name", ""), i.get("id", ""))):
                if item.get("mimeType") == SHORTCUT_MIME:
                    target = item.get("shortcutDetails", {})
                    target_id = target.get("targetId")
                    if not target_id:
                        continue
                    if target.get("targetMimeType") == FOLDER_MIME:
                        stack.append((target_id, f"{prefix}{_safe_name(item['name'])}/"))
                        continue
                    try:
                        item = self.get_file(target_id)
                    except Exception as exc:  # a dangling shortcut must not stop the whole sync
                        log.warning("Skipping shortcut %r: %s", item.get("name"), exc)
                        continue
                if item.get("mimeType") == FOLDER_MIME:
                    stack.append((item["id"], f"{prefix}{_safe_name(item['name'])}/"))
                    continue
                if item["id"] in seen_files:
                    continue
                df = _to_drive_file(item, prefix, used_names)
                if df is not None:
                    seen_files.add(item["id"])
                    yield df

    # -- download -----------------------------------------------------------
    def download(self, file: DriveFile, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if file.export_mime:
            url = f"{API}/files/{file.id}/export"
            params = {"mimeType": file.export_mime}
        else:
            url = f"{API}/files/{file.id}"
            params = {"alt": "media", "supportsAllDrives": "true"}
        resp = self._get(url, params=params, stream=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1024 * 256):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)


def _safe_name(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name or "unnamed"


def _to_drive_file(item: dict, prefix: str, used_names: set[str]) -> DriveFile | None:
    mime = item.get("mimeType", "")
    name = _safe_name(item.get("name", ""))
    export_mime = None
    if mime in GOOGLE_EXPORTS:
        export_mime, ext = GOOGLE_EXPORTS[mime]
        if not name.lower().endswith(ext):
            name = f"{name}{ext}"
    elif Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    if name.lower() in used_names:
        stem, ext = Path(name).stem, Path(name).suffix
        name = f"{stem} ({item['id'][:6]}){ext}"
    used_names.add(name.lower())
    size = item.get("size")
    return DriveFile(
        id=item["id"],
        name=name,
        mime=mime,
        rel_path=f"{prefix}{name}",
        md5=item.get("md5Checksum"),
        modified=item.get("modifiedTime"),
        size=int(size) if size is not None else None,
        export_mime=export_mime,
    )


def sync_folder(client: DriveClient, folder_id: str, cache_dir: Path) -> SyncResult:
    """Make cache_dir mirror the indexable files under a Drive folder."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_NAME
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            log.warning("Drive manifest %s is corrupt; re-downloading everything", manifest_path)

    result = SyncResult()
    seen: dict[str, dict] = {}
    for f in client.walk(folder_id):
        dest = cache_dir / f.rel_path
        previous = manifest.get(f.id)
        entry = {"rel_path": f.rel_path, "version": f.version, "name": f.name, "modified": f.modified}
        seen[f.id] = entry
        if previous and previous.get("version") == f.version and previous.get("rel_path") == f.rel_path and dest.exists():
            result.unchanged += 1
            continue
        try:
            client.download(f, dest)
        except Exception as exc:  # keep syncing the rest
            log.error("Could not download %s: %s", f.rel_path, exc)
            result.errors.append(f"{f.rel_path}: {exc}")
            if previous:
                seen[f.id] = previous
            else:
                seen.pop(f.id, None)
            continue
        (result.updated if previous else result.added).append(dest)

    # Remove local files that no longer correspond to anything remote. A path that is still
    # in use (a file deleted and re-uploaded gets a new id but the same name) must survive.
    current_paths = {entry["rel_path"] for entry in seen.values()}
    stale: list[str] = []
    for fid, previous in manifest.items():
        if fid not in seen:
            stale.append(previous["rel_path"])
    for fid, entry in seen.items():
        previous = manifest.get(fid)
        if previous and previous.get("rel_path") != entry["rel_path"]:
            stale.append(previous["rel_path"])
    for rel in stale:
        if rel in current_paths:
            continue
        old = cache_dir / rel
        if old.exists():
            old.unlink()
            result.removed.append(old)

    manifest_path.write_text(json.dumps(seen, indent=1, sort_keys=True))
    _remove_empty_dirs(cache_dir)
    return result


def _remove_empty_dirs(root: Path) -> None:
    for p in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
