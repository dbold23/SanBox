"""Keep the index in step with the folders: add new files, re-index changed
ones, forget deleted ones. Cheap checks first (size + mtime), content hash
second, parsing and embedding only when something really changed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import chunk_pages
from .config import Source
from .embed import Embedder
from .parse import ParsedDoc, is_supported, parse_file
from .store import Store

log = logging.getLogger(__name__)

SKIP_DIR_PREFIXES = (".", "@eaDir", "#recycle", "__MACOSX", "$RECYCLE.BIN", "System Volume Information", "node_modules")
SKIP_FILE_PREFIXES = (".", "~$")
SKIP_FILE_SUFFIXES = (".part", ".crdownload", ".tmp")
MAX_FILE_MB = 200

ProgressFn = Callable[[str], None]


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    needs_ocr: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed

    def summary(self) -> str:
        s = f"{self.added} added, {self.updated} updated, {self.removed} removed, {self.unchanged} unchanged"
        if self.needs_ocr:
            s += f", {len(self.needs_ocr)} scanned PDFs without text"
        if self.errors:
            s += f", {len(self.errors)} failed"
        return s + f" ({self.seconds:.0f}s)"


def scan_folder(root: Path) -> Iterator[Path]:
    """Yield supported files under root, skipping hidden/system folders and temp files."""
    root = Path(root)
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:  # a symlink pointing back up the tree
            dirnames[:] = []
            continue
        visited.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(SKIP_DIR_PREFIXES))
        for name in sorted(filenames):
            if name.startswith(SKIP_FILE_PREFIXES) or name.endswith(SKIP_FILE_SUFFIXES):
                continue
            p = Path(dirpath) / name
            if is_supported(p):
                yield p


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def index_sources(
    store: Store,
    embedder: Embedder,
    sources: list[Source],
    *,
    rebuild: bool = False,
    progress: ProgressFn | None = None,
    parse: Callable[[Path], ParsedDoc] = parse_file,
    lookup=None,
) -> IndexReport:
    """lookup: an object with .enrich(ParsedDoc) -> bool (see lookup.CrossrefLookup), or None."""
    say = progress or (lambda msg: None)
    started = time.monotonic()
    report = IndexReport()
    if rebuild:
        say("Rebuilding the index from scratch")
        store.clear()
    store.check_embedder(embedder.name, embedder.dim)

    for source in sources:
        root = Path(source.root)
        if not root.exists():
            say(f"[{source.name}] folder not found, skipping: {root}")
            continue
        known = store.list_paths(source.name)
        seen: set[str] = set()
        files = list(scan_folder(root))
        say(f"[{source.name}] {len(files)} files in {root}")
        if not files and known:
            # A mount point that lost its share looks like an empty folder. Never wipe an
            # index because the NAS is unplugged.
            say(f"[{source.name}] folder is empty but {len(known)} documents are indexed from it; "
                "not removing anything (is the drive mounted?)")
            report.errors.append((str(root), f"folder empty; kept {len(known)} indexed documents"))
            continue
        for i, path in enumerate(files, start=1):
            key = str(path)
            seen.add(key)
            try:
                st = path.stat()
            except OSError as exc:
                report.errors.append((key, f"cannot stat: {exc}"))
                continue
            if st.st_size > MAX_FILE_MB * 1024 * 1024:
                report.errors.append((key, f"skipped: larger than {MAX_FILE_MB} MB"))
                continue
            existing = known.get(key)
            if existing and existing.size == st.st_size and existing.mtime == st.st_mtime and existing.status != "error":
                report.unchanged += 1
                continue
            digest = sha256_of(path)
            if existing and existing.sha256 == digest and existing.status != "error":
                # touched but identical (e.g. copied to a new NAS) - just refresh the stat cache
                store.upsert_stat(existing.id, st.st_size, st.st_mtime)
                report.unchanged += 1
                continue
            say(f"[{source.name}] ({i}/{len(files)}) {'re-indexing' if existing else 'indexing'} {path.relative_to(root)}")
            rel = str(path.relative_to(root))
            try:
                doc = parse(path)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", path, exc)
                report.errors.append((key, str(exc)))
                store.upsert_document(source=source.name, rel_path=rel, path=key, sha256=digest, size=st.st_size, mtime=st.st_mtime,
                                      title=path.stem, authors=None, year=None, doi=None, n_pages=None, status="error", error=str(exc)[:500])
                continue
            if lookup is not None and doc.doi:
                try:
                    lookup.enrich(doc)
                except Exception as exc:  # never let metadata lookup break indexing
                    log.info("Metadata lookup failed for %s: %s", path.name, exc)
            status = "ok"
            if doc.needs_ocr:
                status = "needs_ocr"
                report.needs_ocr.append(rel)
            chunks = chunk_pages(doc.pages or [doc.text])
            if not chunks and status == "ok":
                status = "empty"
            embeddings = None
            if chunks:
                try:
                    embeddings = embedder.embed([c.text for c in chunks])
                except Exception as exc:
                    log.error("Embedding failed for %s: %s", path, exc)
                    report.errors.append((key, f"embedding failed: {exc}"))
                    status, chunks = "error", []
            store.upsert_document(
                source=source.name, rel_path=rel, path=key, sha256=digest, size=st.st_size, mtime=st.st_mtime,
                title=doc.title, authors=doc.authors, year=doc.year, doi=doc.doi, n_pages=doc.n_pages or None,
                status=status, error=None if status != "error" else "embedding failed", chunks=chunks, embeddings=embeddings,
            )
            if existing:
                report.updated += 1
            else:
                report.added += 1

        for key, row in known.items():
            if key not in seen:
                store.delete_document(row.id)
                report.removed += 1
                say(f"[{source.name}] removed {row.rel_path}")

    store.mark_indexed()
    report.seconds = time.monotonic() - started
    return report
