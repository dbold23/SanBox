import json

import pytest

from labrag.drive import (
    FOLDER_MIME,
    MANIFEST_NAME,
    DriveClient,
    parse_folder_id,
    sync_folder,
)

DOC_MIME = "application/vnd.google-apps.document"


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self._content = content

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeDrive:
    """In-memory Drive: folders map id -> list of file dicts; contents map id -> bytes."""

    def __init__(self):
        self.folders: dict[str, list[dict]] = {}
        self.contents: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.fail_next: list[int] = []

    def get(self, url, params=None, stream=False, timeout=None):
        self.calls.append(url)
        if self.fail_next:
            return FakeResponse(status=self.fail_next.pop(0))
        if url.endswith("/files"):
            folder_id = params["q"].split("'")[1]
            files = self.folders.get(folder_id, [])
            page = params.get("pageToken")
            # emulate pagination: two files per page
            start = int(page) if page else 0
            chunk = files[start : start + 2]
            nxt = str(start + 2) if start + 2 < len(files) else None
            payload = {"files": chunk}
            if nxt:
                payload["nextPageToken"] = nxt
            return FakeResponse(payload=payload)
        if url.endswith("/export"):
            fid = url.rsplit("/", 2)[-2]
            return FakeResponse(content=self.contents[fid])
        fid = url.rsplit("/", 1)[-1]
        if params and params.get("alt") == "media":
            return FakeResponse(content=self.contents[fid])
        # metadata get (shortcut target)
        for files in self.folders.values():
            for f in files:
                if f["id"] == fid:
                    return FakeResponse(payload=f)
        return FakeResponse(status=404)


def pdf(fid, name, md5, modified="2024-01-01T00:00:00Z"):
    return {"id": fid, "name": name, "mimeType": "application/pdf", "md5Checksum": md5, "modifiedTime": modified, "size": "10"}


@pytest.fixture
def drive():
    d = FakeDrive()
    d.folders["root"] = [
        pdf("p1", "Smith_2019_sharks.pdf", "aaa"),
        {"id": "sub", "name": "Telemetry", "mimeType": FOLDER_MIME},
        {"id": "g1", "name": "Lab protocol", "mimeType": DOC_MIME, "modifiedTime": "2024-02-02T00:00:00Z"},
        {"id": "img", "name": "photo.jpg", "mimeType": "image/jpeg", "md5Checksum": "zzz"},
        pdf("dup1", "same.pdf", "d1"),
        pdf("dup2", "same.pdf", "d2"),
    ]
    d.folders["sub"] = [pdf("p2", "Jones_2020_tags.pdf", "bbb")]
    d.contents.update({"p1": b"%PDF-1", "p2": b"%PDF-2", "g1": b"protocol text", "dup1": b"x", "dup2": b"y"})
    return d


def test_parse_folder_id_variants():
    assert parse_folder_id("https://drive.google.com/drive/folders/1AbC_def-GHI?usp=sharing") == "1AbC_def-GHI"
    assert parse_folder_id("https://drive.google.com/drive/u/0/folders/1AbC_def-GHI") == "1AbC_def-GHI"
    assert parse_folder_id("https://drive.google.com/open?id=1AbC_def-GHI") == "1AbC_def-GHI"
    assert parse_folder_id("  1AbC_def-GHIjkl  ") == "1AbC_def-GHIjkl"
    with pytest.raises(ValueError):
        parse_folder_id("not a folder")


def test_walk_recurses_paginates_exports_and_skips_unsupported(drive):
    client = DriveClient(drive)
    files = {f.rel_path: f for f in client.walk("root")}
    assert "Smith_2019_sharks.pdf" in files
    assert "Telemetry/Jones_2020_tags.pdf" in files
    assert "Lab protocol.txt" in files and files["Lab protocol.txt"].export_mime == "text/plain"
    assert not any(p.endswith(".jpg") for p in files)
    # duplicate names in one folder are disambiguated
    assert "same.pdf" in files and "same (dup2).pdf" in files
    # pagination happened (6 items in root -> 3 pages)
    assert sum(1 for c in drive.calls if c.endswith("/files")) >= 4


def test_sync_downloads_then_is_idempotent_then_tracks_changes(drive, tmp_path):
    client = DriveClient(drive)
    cache = tmp_path / "cache"

    r1 = sync_folder(client, "root", cache)
    assert len(r1.added) == 5 and not r1.updated and not r1.removed
    assert (cache / "Telemetry" / "Jones_2020_tags.pdf").read_bytes() == b"%PDF-2"
    assert (cache / "Lab protocol.txt").read_bytes() == b"protocol text"
    manifest = json.loads((cache / MANIFEST_NAME).read_text())
    assert set(manifest) == {"p1", "p2", "g1", "dup1", "dup2"}

    downloads_before = len(drive.calls)
    r2 = sync_folder(client, "root", cache)
    assert not r2.changed and not r2.removed and r2.unchanged == 5
    # listing calls only, no downloads
    assert all(c.endswith("/files") for c in drive.calls[downloads_before:])

    # change one file, delete another, rename a third
    drive.folders["root"][0] = pdf("p1", "Smith_2019_sharks.pdf", "aaa2")
    drive.folders["sub"] = []
    drive.folders["root"][2] = {"id": "g1", "name": "Lab protocol v2", "mimeType": DOC_MIME, "modifiedTime": "2024-03-03T00:00:00Z"}
    drive.contents["p1"] = b"%PDF-1b"
    r3 = sync_folder(client, "root", cache)
    assert {p.name for p in r3.updated} == {"Smith_2019_sharks.pdf", "Lab protocol v2.txt"}
    assert {p.name for p in r3.removed} == {"Jones_2020_tags.pdf", "Lab protocol.txt"}
    assert (cache / "Smith_2019_sharks.pdf").read_bytes() == b"%PDF-1b"
    assert not (cache / "Telemetry").exists()  # empty dir cleaned
    assert not (cache / "Lab protocol.txt").exists()


def test_sync_survives_download_error_and_retries_5xx(drive, tmp_path, monkeypatch):
    client = DriveClient(drive, retries=1)
    cache = tmp_path / "cache"
    drive.fail_next = [503]  # first request 503 then OK -> retry path
    monkeypatch.setattr("labrag.drive.time.sleep", lambda s: None)
    r = sync_folder(client, "root", cache)
    assert len(r.added) == 5 and not r.errors

    # a hard failure on one download is recorded and the rest still sync
    drive.folders["root"].append(pdf("bad", "bad.pdf", "bad"))
    del drive.contents["p1"]  # fake will KeyError on p1 -> treated as download error
    drive.folders["root"][0] = pdf("p1", "Smith_2019_sharks.pdf", "changed")
    drive.contents["bad"] = b"ok"
    r2 = sync_folder(client, "root", cache)
    assert any("Smith_2019_sharks.pdf" in e for e in r2.errors)
    assert [p.name for p in r2.added] == ["bad.pdf"]
    # the failed file keeps its previous manifest entry so it is retried next time
    manifest = json.loads((cache / MANIFEST_NAME).read_text())
    assert manifest["p1"]["version"] == "aaa"
