"""Settings. Everything comes from environment variables, which can live in a
plain KEY=VALUE file so nobody has to export anything by hand.

`labrag init` writes that file. Lookup order for the file:
  1. $LABRAG_ENV                 (explicit)
  2. ./labrag.env                (a project folder)
  3. ~/.labrag/labrag.env        (per user; what `labrag init` writes)
Real environment variables always win over the file.

Paths only, never secrets, are safe to keep on the NAS. API keys stay in the
per-user file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_USER_DIR = Path.home() / ".labrag"
DEFAULT_ENV_NAME = "labrag.env"

DOC = {
    "LABRAG_FOLDERS": "Folders with papers to index, separated by ';'. A NAS share, a Google Drive for Desktop folder, a laptop folder. Optional 'name=path' gives a source a short name.",
    "LABRAG_DATA": "Where the index (labrag.db) and caches live. Default: this machine's disk (~/.labrag/data). Only the machine that runs `labrag serve` needs it.",
    "LABRAG_DRIVE_FOLDER": "Google Drive folder link or ID to sync and index (optional).",
    "LABRAG_GOOGLE_SERVICE_ACCOUNT": "Path to a Google service-account JSON key; share the Drive folder with its e-mail (optional).",
    "LABRAG_GOOGLE_CLIENT_SECRET": "Path to an OAuth 'Desktop app' client secret JSON; first run opens a browser (optional).",
    "ANTHROPIC_API_KEY": "Claude API key. When set, answers are written by Claude.",
    "LABRAG_LLM": "anthropic | openai | ollama | none. Default: auto (anthropic if ANTHROPIC_API_KEY is set, else openai if OPENAI_API_KEY is set, else ollama if it is running, else none).",
    "LABRAG_LLM_MODEL": "Model name for the chosen LLM. Defaults: claude-opus-5 / gpt-4o-mini / llama3.1.",
    "LABRAG_LLM_EFFORT": "Claude reasoning effort: low | medium | high. Default medium.",
    "LABRAG_MAX_TOKENS": "Token budget per answer, shared between the model's reasoning and the text it writes. Default 16000.",
    "OPENAI_API_KEY": "OpenAI (or compatible) API key.",
    "LABRAG_OPENAI_BASE_URL": "Base URL for an OpenAI-compatible server. Default https://api.openai.com/v1.",
    "LABRAG_OLLAMA_URL": "Ollama server URL. Default http://localhost:11434.",
    "LABRAG_EMBED": "fastembed | ollama | openai | hash. Default fastembed (local, no account).",
    "LABRAG_EMBED_MODEL": "Embedding model. Defaults: BAAI/bge-small-en-v1.5 / nomic-embed-text / text-embedding-3-small.",
    "LABRAG_TOP_K": "How many passages to give the model per question. Default 8.",
    "LABRAG_PASSWORD": "Optional password for the web UI (any user name). Leave empty on a trusted lab network.",
    "LABRAG_LOOKUP": "on | off. Look up title/authors/year on Crossref from each paper's DOI while indexing. Default on.",
    "LABRAG_HOST": "Web UI bind address. Default 0.0.0.0 (reachable by the lab).",
    "LABRAG_PORT": "Web UI port. Default 8008.",
}


@dataclass
class Source:
    name: str
    root: Path


@dataclass
class Settings:
    folders: list[Source] = field(default_factory=list)
    data_dir: Path = DEFAULT_USER_DIR / "data"
    drive_folder: str | None = None
    google_service_account: Path | None = None
    google_client_secret: Path | None = None
    llm: str = "auto"
    llm_model: str | None = None
    llm_effort: str = "medium"
    max_tokens: int = 16000
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_url: str = "http://localhost:11434"
    embed: str = "fastembed"
    embed_model: str | None = None
    top_k: int = 8
    host: str = "0.0.0.0"
    port: int = 8008
    password: str | None = None
    lookup: bool = True
    env_file: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "labrag.db"

    @property
    def drive_cache_dir(self) -> Path:
        return self.data_dir / "drive"

    @property
    def google_token_file(self) -> Path:
        return DEFAULT_USER_DIR / "google_token.json"

    @property
    def models_cache_dir(self) -> Path:
        return DEFAULT_USER_DIR / "models"

    def all_sources(self) -> list[Source]:
        sources = list(self.folders)
        if self.drive_folder:
            sources.append(Source("drive", self.drive_cache_dir))
        return sources

    def problems(self) -> list[str]:
        out = []
        if not self.folders and not self.drive_folder:
            out.append("No papers configured. Set LABRAG_FOLDERS and/or LABRAG_DRIVE_FOLDER (or run `labrag init`).")
        for s in self.folders:
            if not s.root.exists():
                out.append(f"Folder for source '{s.name}' does not exist: {s.root} (is the NAS mounted?)")
        if self.drive_folder and not (self.google_service_account or self.google_client_secret):
            out.append("LABRAG_DRIVE_FOLDER is set but neither LABRAG_GOOGLE_SERVICE_ACCOUNT nor LABRAG_GOOGLE_CLIENT_SECRET is.")
        for label, p in (
            ("LABRAG_GOOGLE_SERVICE_ACCOUNT", self.google_service_account),
            ("LABRAG_GOOGLE_CLIENT_SECRET", self.google_client_secret),
        ):
            if p and not p.exists():
                out.append(f"{label} points to a file that does not exist: {p}")
        return out


def find_env_file(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path.cwd() / DEFAULT_ENV_NAME, DEFAULT_USER_DIR / DEFAULT_ENV_NAME]
    for c in candidates:
        if c.is_file():
            return c
    return None


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.sub(r"(^|\s+)#.*$", "", value).strip()  # trailing comment, or a value that is only a comment
        if key:
            values[key] = value
    return values


def write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LabRAG settings. Edit freely; blank values are ignored.", ""]
    for key, value in values.items():
        if key in DOC:
            lines.append(f"# {DOC[key]}")
        lines.append(f"{key}={_quote(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        path.chmod(0o600)  # may hold an API key
    except OSError:
        pass


def _quote(value: str) -> str:
    if value == "" or re.search(r"[\s#\"']", value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def parse_folders(spec: str) -> list[Source]:
    sources: list[Source] = []
    used: set[str] = set()
    for part in re.split(r"[;\n]", spec):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue
        if "=" in part and not Path(part).exists():
            name, _, raw = part.partition("=")
            name = name.strip()
        else:
            name, raw = "", part
        root = Path(raw.strip()).expanduser()
        if not name:
            name = root.name or "papers"
        base, i = name, 2
        while name in used:
            name, i = f"{base}{i}", i + 1
        used.add(name)
        sources.append(Source(name, root))
    return sources


def load_settings(env: dict[str, str] | None = None, env_file: str | None = None) -> Settings:
    """Merge file values (lowest priority) with real environment variables."""
    values: dict[str, str] = {}
    path = find_env_file(env_file or (env or os.environ).get("LABRAG_ENV"))
    if path:
        values.update(read_env_file(path))
    values.update({k: v for k, v in (env if env is not None else os.environ).items() if v is not None})

    def get(key: str, default: str | None = None) -> str | None:
        v = values.get(key)
        return v if v not in (None, "") else default

    def path_or_none(key: str) -> Path | None:
        v = get(key)
        return Path(v).expanduser() if v else None

    s = Settings(env_file=path)
    s.folders = parse_folders(get("LABRAG_FOLDERS", "") or "")
    s.data_dir = Path(get("LABRAG_DATA", str(DEFAULT_USER_DIR / "data"))).expanduser()
    s.drive_folder = get("LABRAG_DRIVE_FOLDER")
    s.google_service_account = path_or_none("LABRAG_GOOGLE_SERVICE_ACCOUNT")
    s.google_client_secret = path_or_none("LABRAG_GOOGLE_CLIENT_SECRET")
    s.llm = (get("LABRAG_LLM", "auto") or "auto").lower()
    s.llm_model = get("LABRAG_LLM_MODEL")
    s.llm_effort = (get("LABRAG_LLM_EFFORT", "medium") or "medium").lower()
    s.max_tokens = int(get("LABRAG_MAX_TOKENS", "16000") or 16000)
    s.anthropic_api_key = get("ANTHROPIC_API_KEY")
    s.openai_api_key = get("OPENAI_API_KEY")
    s.openai_base_url = get("LABRAG_OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    s.ollama_url = get("LABRAG_OLLAMA_URL", "http://localhost:11434") or "http://localhost:11434"
    s.embed = (get("LABRAG_EMBED", "fastembed") or "fastembed").lower()
    s.embed_model = get("LABRAG_EMBED_MODEL")
    s.top_k = int(get("LABRAG_TOP_K", "8") or 8)
    s.host = get("LABRAG_HOST", "0.0.0.0") or "0.0.0.0"
    s.port = int(get("LABRAG_PORT", "8008") or 8008)
    s.password = get("LABRAG_PASSWORD")
    s.lookup = (get("LABRAG_LOOKUP", "on") or "on").lower() not in ("off", "false", "0", "no")
    return s
