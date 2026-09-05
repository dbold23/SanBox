"""The web page. One always-on machine runs `labrag serve`; everyone else just
opens the URL. No accounts, optionally one shared password."""

from __future__ import annotations

import base64
import logging
import secrets
import threading
import time
from collections import deque
from importlib import resources
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .engine import Answer, Engine
from .providers import make_embedder, make_llm
from .store import Hit, Store

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=30)
    history: list[list[str]] = Field(default_factory=list)
    passages_only: bool = False


class IndexJob:
    """Runs the indexer in a background thread; the page polls for progress."""

    def __init__(self, run: Callable[[Callable[[str], None]], object]):
        self._run = run
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.log: deque[str] = deque(maxlen=300)
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.last_summary: str | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.log.clear()
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
            self._thread = threading.Thread(target=self._target, name="labrag-index", daemon=True)
            self._thread.start()
            return True

    def _target(self) -> None:
        try:
            report = self._run(self.log.append)
            self.last_summary = getattr(report, "summary", lambda: str(report))()
            self.log.append(f"Done: {self.last_summary}")
        except Exception as exc:
            log.exception("Indexing failed")
            self.error = str(exc)
            self.log.append(f"Failed: {exc}")
        finally:
            self.finished_at = time.time()

    def status(self) -> dict:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_summary": self.last_summary,
            "error": self.error,
            "log": list(self.log)[-60:],
        }


def hit_to_dict(n: int, hit: Hit, cited: bool) -> dict:
    doc = hit.doc
    return {
        "n": n,
        "cited": cited,
        "doc_id": doc.id,
        "citation": doc.short_citation,
        "title": doc.title or Path(doc.rel_path).name,
        "authors": doc.authors,
        "year": doc.year,
        "doi": doc.doi,
        "source": doc.source,
        "rel_path": doc.rel_path,
        "pages": hit.pages,
        "page_start": hit.page_start,
        "snippet": " ".join(hit.text.split())[:400],
        "file_url": f"/file/{doc.id}" + (f"#page={hit.page_start}" if hit.page_start else ""),
    }


def answer_to_dict(ans: Answer) -> dict:
    cited = set(ans.cited)
    return {
        "question": ans.question,
        "answer": ans.text,
        "error": ans.error,
        "model": ans.model,
        "elapsed": round(ans.elapsed, 2),
        "cited": ans.cited,
        "sources": [hit_to_dict(i, h, i in cited) for i, h in enumerate(ans.sources, start=1)],
    }


def build_engine(settings: Settings) -> Engine:
    store = Store(settings.db_path)
    embedder = make_embedder(settings)
    store.check_embedder(embedder.name, embedder.dim)
    llm = make_llm(settings)
    return Engine(store, embedder, llm, k=settings.top_k, max_tokens=settings.max_tokens)


def default_index_runner(settings: Settings, engine: Engine) -> Callable[[Callable[[str], None]], object]:
    from .cli import run_index

    def run(progress):
        return run_index(settings, engine.embedder, progress=progress)

    return run


def create_app(settings: Settings, engine: Engine | None = None, index_runner=None) -> FastAPI:
    engine = engine or build_engine(settings)
    job = IndexJob(index_runner or default_index_runner(settings, engine))
    app = FastAPI(title="LabRAG", version=__version__, docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.job = job

    if settings.password:
        password = settings.password

        @app.middleware("http")
        async def require_password(request: Request, call_next):
            header = request.headers.get("authorization", "")
            if header.startswith("Basic "):
                try:
                    decoded = base64.b64decode(header[6:]).decode("utf-8", "replace")
                except Exception:
                    decoded = ""
                _, _, given = decoded.partition(":")
                if secrets.compare_digest(given.encode(), password.encode()):
                    return await call_next(request)
            return Response("LabRAG: password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="LabRAG"'})

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        return resources.files("labrag").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        st = engine.store.stats()
        return {
            "version": __version__,
            "documents": st["documents"],
            "chunks": st["chunks"],
            "needs_ocr": st["needs_ocr"],
            "errors": st["errors"],
            "sources": st["sources"],
            "last_indexed": st["last_indexed"],
            "model": engine.llm.name if engine.llm else None,
            "embeddings": engine.embedder.name if engine.embedder else None,
            "indexing": job.running,
        }

    @app.post("/api/ask")
    def ask(req: AskRequest):
        history = [(q, a) for q, a in (pair for pair in req.history if len(pair) == 2)]
        if req.passages_only or engine.llm is None:
            hits = engine.search(req.question, k=req.k or max(engine.k, 10))
            return answer_to_dict(Answer(question=req.question, text=None, sources=hits))
        return answer_to_dict(engine.ask(req.question, k=req.k, history=history))

    @app.get("/api/search")
    def search(q: str, k: int = 10):
        if not q.strip():
            raise HTTPException(400, "empty query")
        hits = engine.search(q, k=max(1, min(k, 30)))
        return answer_to_dict(Answer(question=q, text=None, sources=hits))

    @app.get("/api/documents")
    def documents(source: str | None = None, status_filter: str | None = None):
        docs = engine.store.list_documents(source)
        if status_filter:
            docs = [d for d in docs if d.status == status_filter]
        return [
            {"id": d.id, "source": d.source, "rel_path": d.rel_path, "title": d.title, "authors": d.authors, "year": d.year,
             "n_pages": d.n_pages, "n_chunks": d.n_chunks, "status": d.status, "error": d.error, "file_url": f"/file/{d.id}"}
            for d in docs
        ]

    @app.get("/file/{doc_id}")
    def file(doc_id: int):
        doc = engine.store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "unknown document")
        path = Path(doc.path)
        if not path.is_file():
            raise HTTPException(404, f"file is not available on this machine: {doc.rel_path}")
        return FileResponse(str(path), filename=path.name, content_disposition_type="inline")

    @app.post("/api/index")
    def start_index():
        started = job.start()
        return JSONResponse({"started": started, **job.status()}, status_code=202 if started else 200)

    @app.get("/api/index")
    def index_status():
        return job.status()

    return app
