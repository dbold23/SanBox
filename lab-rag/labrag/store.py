"""One SQLite file holds everything: documents, passages, keyword index, vectors.

Why a single SQLite file: it can sit on the NAS next to the papers, every lab
member's copy of LabRAG can read it, and there is nothing to install or run.
Keyword search uses SQLite's built-in FTS5 (BM25 ranking); vector search is a
plain cosine similarity over a matrix that is loaded into memory on first use
and refreshed whenever the index changes. The two are fused with Reciprocal
Rank Fusion, which needs no score calibration.

Concurrency: many readers, one writer at a time (the indexer). WAL mode is
deliberately not used because it is unsafe on network filesystems.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunk import Chunk

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    source     TEXT NOT NULL,
    rel_path   TEXT NOT NULL,
    path       TEXT NOT NULL UNIQUE,
    sha256     TEXT NOT NULL,
    size       INTEGER,
    mtime      REAL,
    title      TEXT,
    authors    TEXT,
    year       INTEGER,
    doi        TEXT,
    n_pages    INTEGER,
    n_chunks   INTEGER NOT NULL DEFAULT 0,
    status     TEXT NOT NULL DEFAULT 'ok',
    error      TEXT,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    page_start INTEGER,
    page_end   INTEGER,
    text       TEXT NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


@dataclass
class DocumentRow:
    id: int
    source: str
    rel_path: str
    path: str
    sha256: str
    size: int | None
    mtime: float | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    n_pages: int | None
    n_chunks: int
    status: str
    error: str | None
    indexed_at: str

    @property
    def short_citation(self) -> str:
        """e.g. 'Jorgensen 2010' or the title when we know nothing better."""
        who = None
        if self.authors:
            first = re.split(r"[,;&]| and ", self.authors)[0].strip()
            parts = [p.strip(".") for p in re.split(r"\s+", first) if p.strip(".")]
            if parts:
                # "Jorgensen S" / "Jorgensen SJ" -> Jorgensen; "Salvador J. Jorgensen" -> Jorgensen
                who = parts[0] if len(parts[-1]) <= 2 else parts[-1]
        if who and self.year:
            return f"{who} {self.year}"
        if who:
            return who
        title = self.title or ""
        if not title or len(title) > 60:  # a headingless note's first sentence is not a citation
            title = Path(self.rel_path).stem
        return f"{title} ({self.year})" if self.year else title


@dataclass
class Hit:
    chunk_id: int
    doc: DocumentRow
    text: str
    page_start: int | None
    page_end: int | None
    score: float
    keyword_rank: int | None
    vector_rank: int | None

    @property
    def pages(self) -> str:
        if not self.page_start:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}-{self.page_end}"
        return f"p. {self.page_start}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str, timeout: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # one connection may be shared by several web-request threads
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=timeout, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = DELETE")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        if self.get_meta("index_version") is None:
            self.set_meta("index_version", "0")
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None
        self._matrix_version: str | None = None

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------------------------------------------------------------- meta
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _bump_version(self) -> None:
        self.conn.execute("UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'index_version'")

    @property
    def embedding_model(self) -> str | None:
        return self.get_meta("embedding_model")

    def check_embedder(self, name: str, dim: int) -> None:
        """Record the embedder on first use; refuse to mix models afterwards."""
        current = self.get_meta("embedding_model")
        if current is None:
            self.set_meta("embedding_model", name)
            self.set_meta("embedding_dim", str(dim))
        elif current != name:
            raise EmbedderMismatch(
                f"This index was built with embeddings from '{current}' but the configured "
                f"embedder is '{name}'. Run `labrag index --rebuild` to re-embed everything, "
                "or switch the configuration back."
            )

    # ----------------------------------------------------------- documents
    def get_document_by_path(self, path: str) -> DocumentRow | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
            return DocumentRow(**dict(row)) if row else None

    def get_document(self, doc_id: int) -> DocumentRow | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return DocumentRow(**dict(row)) if row else None

    def list_documents(self, source: str | None = None) -> list[DocumentRow]:
        with self._lock:
            if source:
                rows = self.conn.execute("SELECT * FROM documents WHERE source = ? ORDER BY rel_path", (source,))
            else:
                rows = self.conn.execute("SELECT * FROM documents ORDER BY source, rel_path")
            return [DocumentRow(**dict(r)) for r in rows]

    def list_paths(self, source: str) -> dict[str, DocumentRow]:
        return {d.path: d for d in self.list_documents(source)}

    def get_document_by_hash(self, source: str, sha256: str) -> DocumentRow | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE source = ? AND sha256 = ? ORDER BY id LIMIT 1", (source, sha256)
            ).fetchone()
        return DocumentRow(**dict(row)) if row else None

    def update_location(self, doc_id: int, path: str, rel_path: str, size: int, mtime: float) -> None:
        """The same file showed up at a new path (renamed, or the share is mounted elsewhere)."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE documents SET path = ?, rel_path = ?, size = ?, mtime = ? WHERE id = ?",
                (path, rel_path, size, mtime, doc_id),
            )

    def upsert_document(
        self,
        *,
        source: str,
        rel_path: str,
        path: str,
        sha256: str,
        size: int | None,
        mtime: float | None,
        title: str | None,
        authors: str | None,
        year: int | None,
        doi: str | None,
        n_pages: int | None,
        status: str = "ok",
        error: str | None = None,
        chunks: Iterable[Chunk] = (),
        embeddings: np.ndarray | None = None,
    ) -> int:
        """Replace a document and all of its passages in one transaction."""
        with self._lock:
            chunks = list(chunks)
            if embeddings is not None and len(embeddings) != len(chunks):
                raise ValueError("embeddings and chunks must have the same length")
            with self.conn:
                existing = self.conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
                if existing:
                    doc_id = existing["id"]
                    self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                    self.conn.execute(
                        """UPDATE documents SET source=?, rel_path=?, sha256=?, size=?, mtime=?, title=?, authors=?,
                           year=?, doi=?, n_pages=?, n_chunks=?, status=?, error=?, indexed_at=? WHERE id=?""",
                        (
                            source,
                            rel_path,
                            sha256,
                            size,
                            mtime,
                            title,
                            authors,
                            year,
                            doi,
                            n_pages,
                            len(chunks),
                            status,
                            error,
                            _now(),
                            doc_id,
                        ),
                    )
                else:
                    cur = self.conn.execute(
                        """INSERT INTO documents(source, rel_path, path, sha256, size, mtime, title, authors, year, doi,
                           n_pages, n_chunks, status, error, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            source,
                            rel_path,
                            path,
                            sha256,
                            size,
                            mtime,
                            title,
                            authors,
                            year,
                            doi,
                            n_pages,
                            len(chunks),
                            status,
                            error,
                            _now(),
                        ),
                    )
                    doc_id = cur.lastrowid
                rows = []
                for i, ch in enumerate(chunks):
                    blob = np.asarray(embeddings[i], dtype=np.float32).tobytes() if embeddings is not None else None
                    rows.append((doc_id, ch.idx, ch.page_start, ch.page_end, ch.text, blob))
                self.conn.executemany("INSERT INTO chunks(doc_id, idx, page_start, page_end, text, embedding) VALUES (?,?,?,?,?,?)", rows)
                self._bump_version()
            return doc_id

    def upsert_stat(self, doc_id: int, size: int, mtime: float) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE documents SET size = ?, mtime = ? WHERE id = ?", (size, mtime, doc_id))

    def delete_document(self, doc_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._bump_version()

    def clear(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta WHERE key IN ('embedding_model', 'embedding_dim')")
            self._bump_version()

    def stats(self) -> dict:
        with self._lock:
            row = self.conn.execute(
                """SELECT COUNT(*) AS docs,
                          SUM(n_chunks) AS chunks,
                          SUM(status = 'ok') AS ok,
                          SUM(status = 'needs_ocr') AS needs_ocr,
                          SUM(status = 'error') AS errors,
                          SUM(status = 'empty') AS empty
                   FROM documents"""
            ).fetchone()
            sources = {r["source"]: r["n"] for r in self.conn.execute("SELECT source, COUNT(*) AS n FROM documents GROUP BY source")}
            return {
                "documents": row["docs"] or 0,
                "chunks": row["chunks"] or 0,
                "ok": row["ok"] or 0,
                "needs_ocr": row["needs_ocr"] or 0,
                "errors": row["errors"] or 0,
                "empty": row["empty"] or 0,
                "sources": sources,
                "embedding_model": self.get_meta("embedding_model"),
                "last_indexed": self.get_meta("last_indexed"),
                "index_version": self.get_meta("index_version"),
                "db_path": str(self.path),
            }

    def get_chunk_text(self, chunk_id: int) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return row["text"] if row else None

        # ---------------------------------------------------------------- search

    def keyword_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """FTS5 BM25. Tries all-terms first, then any-term. Returns (chunk_id, bm25) best first."""
        with self._lock:
            terms = _fts_terms(query)
            if not terms:
                return []
            for joiner in (" AND ", " OR "):
                fts_query = joiner.join(terms)
                try:
                    rows = self.conn.execute(
                        "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
                    log.warning("FTS query failed (%s): %s", fts_query, exc)
                    return []
                if len(rows) >= min(limit, 10) or joiner == " OR ":
                    return [(r["rowid"], float(r["score"])) for r in rows]
            return []

    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            version = self.get_meta("index_version")
            if self._matrix is not None and self._matrix_version == version:
                return self._matrix_ids, self._matrix
            dim_text = self.get_meta("embedding_dim")
            rows = self.conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id").fetchall()
            if not rows:
                self._matrix_ids = np.zeros(0, dtype=np.int64)
                self._matrix = np.zeros((0, int(dim_text or 1)), dtype=np.float32)
            else:
                self._matrix_ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
                self._matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            self._matrix_version = version
            return self._matrix_ids, self._matrix

    def vector_search(self, query_vec: np.ndarray, limit: int = 50) -> list[tuple[int, float]]:
        ids, matrix = self._load_matrix()
        if len(ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[-1] != matrix.shape[1]:
            raise EmbedderMismatch(f"Query vector has {q.shape[-1]} dimensions but the index has {matrix.shape[1]}; the embedder changed.")
        sims = matrix @ q
        k = min(limit, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def search(
        self,
        query: str,
        query_vec: np.ndarray | None = None,
        k: int = 8,
        per_doc: int = 3,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> list[Hit]:
        """Hybrid search: keyword + vector ranks fused with Reciprocal Rank Fusion."""
        with self._lock:
            kw = self.keyword_search(query, limit=candidates)
            vec = self.vector_search(query_vec, limit=candidates) if query_vec is not None else []
            kw_rank = {cid: r for r, (cid, _) in enumerate(kw, start=1)}
            vec_rank = {cid: r for r, (cid, _) in enumerate(vec, start=1)}
            scores: dict[int, float] = {}
            for cid, r in kw_rank.items():
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r)
            for cid, r in vec_rank.items():
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r)
            if not scores:
                return []
            ordered = sorted(scores, key=lambda c: -scores[c])
            placeholders = ",".join("?" * len(ordered))
            rows = self.conn.execute(
                f"""SELECT c.id AS chunk_id, c.text, c.page_start, c.page_end, d.*
                    FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({placeholders})""",
                ordered,
            ).fetchall()
            by_id = {r["chunk_id"]: r for r in rows}
            hits: list[Hit] = []
            per_doc_count: dict[int, int] = {}
            for cid in ordered:
                r = by_id.get(cid)
                if r is None:
                    continue
                doc_id = r["id"]
                if per_doc_count.get(doc_id, 0) >= per_doc:
                    continue
                per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
                doc = DocumentRow(**{k_: r[k_] for k_ in DocumentRow.__dataclass_fields__})
                hits.append(
                    Hit(
                        chunk_id=cid,
                        doc=doc,
                        text=r["text"],
                        page_start=r["page_start"],
                        page_end=r["page_end"],
                        score=scores[cid],
                        keyword_rank=kw_rank.get(cid),
                        vector_rank=vec_rank.get(cid),
                    )
                )
                if len(hits) >= k:
                    break
            return hits

    def mark_indexed(self) -> None:
        self.set_meta("last_indexed", _now())


class EmbedderMismatch(RuntimeError):
    pass


_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "with",
    "by",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "as",
    "about",
    "any",
    "there",
    "their",
    "they",
    "i",
    "we",
    "you",
    "me",
    "my",
    "our",
    "your",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
}


def _fts_terms(query: str) -> list[str]:
    words = re.findall(r"\w[\w\-']*", query.lower())  # \w is Unicode: México, Müller, Bahía
    terms = [w for w in words if w not in _STOP and len(w) > 1]
    if not terms:
        terms = [w for w in words if len(w) > 1]
    # quote every token so FTS5 syntax characters in user input cannot break the query
    return ['"' + t.replace('"', "") + '"' for t in terms[:32]]
