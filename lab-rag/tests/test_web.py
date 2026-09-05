import base64
import time

import pytest
from fastapi.testclient import TestClient

from labrag.chunk import Chunk
from labrag.config import Settings
from labrag.embed import HashEmbedder
from labrag.engine import Engine
from labrag.llm import LLMError
from labrag.store import Store
from labrag.web import IndexJob, create_app


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, max_tokens=8000):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def engine(tmp_path):
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(dim=32)
    store.check_embedder(emb.name, emb.dim)
    pdf = tmp_path / "Smith_2019_sharks.txt"
    pdf.write_text("White sharks eat seals near Ano Nuevo.")
    chunks = [Chunk("White sharks eat seals near Ano Nuevo.", 0, 3, 3)]
    store.upsert_document(source="nas", rel_path="Smith_2019_sharks.txt", path=str(pdf), sha256="s", size=1, mtime=1,
                          title="Shark diet", authors="Smith J", year=2019, doi=None, n_pages=3, chunks=chunks,
                          embeddings=emb.embed([c.text for c in chunks]))
    yield Engine(store, emb, FakeLLM("They eat seals [1]."), k=5)
    store.close()


def make_client(engine, settings=None, runner=None):
    app = create_app(settings or Settings(), engine=engine, index_runner=runner or (lambda progress: "noop"))
    return TestClient(app)


def test_page_status_ask_and_file(engine):
    c = make_client(engine)
    assert "LabRAG" in c.get("/").text
    assert c.get("/favicon.ico").headers["content-type"].startswith("image/svg+xml")
    st = c.get("/api/status").json()
    assert st["documents"] == 1 and st["model"] == "fake" and st["embeddings"] == "hash-32"
    assert st["problem_files"] == []
    engine.store.upsert_document(source="nas", rel_path="scan.pdf", path="/nas/scan.pdf", sha256="z", size=1, mtime=1, title="scan",
                                 authors=None, year=None, doi=None, n_pages=2, status="needs_ocr")
    assert c.get("/api/status").json()["problem_files"] == [{"source": "nas", "rel_path": "scan.pdf", "status": "needs_ocr", "error": None}]

    r = c.post("/api/ask", json={"question": "what do white sharks eat?", "history": [["a", "b"], ["bad"]]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "They eat seals [1]." and body["cited"] == [1]
    src = body["sources"][0]
    assert src["cited"] and src["citation"] == "Smith 2019" and src["pages"] == "p. 3"
    assert src["file_url"] == f"/file/{src['doc_id']}#page=3"

    f = c.get(f"/file/{src['doc_id']}")
    assert f.status_code == 200 and b"seals" in f.content and "inline" in f.headers["content-disposition"]
    assert c.get("/file/999").status_code == 404

    passages = c.post("/api/ask", json={"question": "seals", "passages_only": True}).json()
    assert passages["answer"] is None and passages["sources"]
    assert c.get("/api/search", params={"q": "seals"}).json()["sources"]
    assert c.get("/api/search", params={"q": "  "}).status_code == 400
    docs = c.get("/api/documents").json()
    assert docs[0]["title"] == "Shark diet" and docs[0]["status"] == "ok"
    assert c.post("/api/ask", json={"question": ""}).status_code == 422


def test_llm_error_is_returned_as_json(engine):
    engine.llm = FakeLLM(LLMError("model down"))
    body = make_client(engine).post("/api/ask", json={"question": "sharks"}).json()
    assert body["error"] == "model down" and body["answer"] is None and body["sources"]


def test_password_protection(engine):
    c = make_client(engine, Settings(password="octopus"))
    assert c.get("/api/status").status_code == 401
    assert "Basic" in c.get("/").headers["www-authenticate"]
    bad = base64.b64encode(b"lab:wrong").decode()
    assert c.get("/api/status", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    good = base64.b64encode(b"anyone:octopus").decode()
    assert c.get("/api/status", headers={"Authorization": f"Basic {good}"}).status_code == 200


def test_index_job_runs_in_background(engine):
    import threading

    calls = []
    release = threading.Event()

    def runner(progress):
        progress("step 1")
        calls.append(1)
        release.wait(5)

        class R:
            def summary(self):
                return "1 added"

        return R()

    c = make_client(engine, runner=runner)
    r = c.post("/api/index")
    assert r.status_code == 202 and r.json()["started"] is True
    second = c.post("/api/index")  # already running -> not started again
    assert second.status_code == 200 and second.json()["started"] is False and second.json()["running"] is True
    release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        st = c.get("/api/index").json()
        if not st["running"]:
            break
        time.sleep(0.02)
    assert st["last_summary"] == "1 added" and "step 1" in st["log"] and st["error"] is None
    assert calls == [1]


def test_index_job_reports_failure():
    def boom(progress):
        raise RuntimeError("nas unplugged")

    job = IndexJob(boom)
    assert job.start()
    job._thread.join(2)
    assert job.error == "nas unplugged" and not job.running and job.status()["log"][-1].startswith("Failed")
