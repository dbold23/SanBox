import pytest

from labrag.chunk import Chunk
from labrag.embed import HashEmbedder
from labrag.engine import Engine, build_prompt, extract_citations, retrieval_query
from labrag.llm import LLMError
from labrag.store import Store


class FakeLLM:
    name = "fake"

    def __init__(self, reply="Sharks eat seals [1]. Rays dig pits [2, 9]."):
        self.reply = reply
        self.calls = []

    def complete(self, system, user, max_tokens=8000):
        self.calls.append((system, user, max_tokens))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def populated(tmp_path):
    store = Store(tmp_path / "i.db")
    emb = HashEmbedder(dim=64)
    store.check_embedder(emb.name, emb.dim)

    def add(path, texts, **meta):
        chunks = [Chunk(text=t, idx=i, page_start=i + 1, page_end=i + 1) for i, t in enumerate(texts)]
        store.upsert_document(
            source="nas",
            rel_path=path,
            path="/nas/" + path,
            sha256="s",
            size=1,
            mtime=1,
            title=meta.get("title"),
            authors=meta.get("authors"),
            year=meta.get("year"),
            doi=None,
            n_pages=len(texts),
            chunks=chunks,
            embeddings=emb.embed(texts),
        )

    add("a.pdf", ["White sharks eat seals near Ano Nuevo.", "Methods: we tagged sharks."], title="Diet", authors="Jorgensen", year=2010)
    add("b.pdf", ["Bat rays dig pits in the mudflat."], title="Rays", authors="Smith", year=2020)
    yield store, emb
    store.close()


def test_extract_citations_dedupes_and_bounds():
    assert extract_citations("A [1]. B [2, 1]; C [9] D [3;2]", 3) == [1, 2, 3]
    assert extract_citations("no citations", 3) == []


def test_ask_with_llm_returns_cited_sources(populated):
    store, emb = populated
    llm = FakeLLM()
    eng = Engine(store, emb, llm, k=4)
    ans = eng.ask("What do white sharks eat?")
    assert ans.text.startswith("Sharks eat seals")
    assert ans.model == "fake"
    assert ans.cited == [1, 2]
    assert ans.cited_sources[0][1].doc.title == "Diet"
    system, user, _ = llm.calls[0]
    assert "SOURCES:" in user and '[1] Jorgensen 2010 - "Diet", p. 1' in user
    assert user.rstrip().endswith("QUESTION: What do white sharks eat?")
    assert "cite the source number" in system


def test_ask_without_llm_is_search_only(populated):
    store, emb = populated
    ans = Engine(store, emb, None).ask("mudflat rays")
    assert ans.text is None and ans.model is None
    assert ans.sources and ans.sources[0].doc.title == "Rays"


def test_llm_error_is_reported_not_raised(populated):
    store, emb = populated
    ans = Engine(store, emb, FakeLLM(LLMError("Ollama is down"))).ask("sharks")
    assert ans.error == "Ollama is down" and ans.text is None and ans.sources


def test_no_hits_short_circuits_llm(tmp_path):
    store = Store(tmp_path / "empty.db")  # an empty index is the only way to get zero hits
    llm = FakeLLM()
    ans = Engine(store, HashEmbedder(dim=16), llm).ask("zzqx")
    assert not llm.calls
    assert "Nothing in the indexed papers" in ans.text
    store.close()


def test_history_is_included_and_truncated(populated):
    store, emb = populated
    hits = Engine(store, emb, None).search("sharks")
    prompt = build_prompt("and rays?", hits, [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "x" * 5000)])
    assert "Q: q1" not in prompt and "Q: q2" in prompt
    assert "x" * 1201 not in prompt


def test_retrieval_query_expands_followups_only():
    hist = [("what do white sharks eat near Ano Nuevo?", "Seals [1].")]
    assert retrieval_query("what about leopard sharks?", hist) == "what do white sharks eat near Ano Nuevo? what about leopard sharks?"
    assert retrieval_query("which of those studies used acoustic telemetry", hist).startswith("what do white sharks")
    assert retrieval_query("and rays?", hist).startswith("what do white sharks")
    assert retrieval_query("what about rays?", []) == "what about rays?"
    # ordinary questions that merely contain a pronoun are NOT expanded
    for q in [
        "How deep do salmon sharks dive and where do they go in winter?",
        "Where do leopard sharks aggregate seasonally?",
        "Is there any evidence of philopatry in white sharks?",
        "Which tag types were used on bat rays?",
    ]:
        assert retrieval_query(q, hist) == q


def test_search_falls_back_to_keywords_on_embedder_mismatch(populated):
    from labrag.embed import HashEmbedder as H

    store, _ = populated
    wrong = H(dim=16)  # index was built with 64 dims
    hits = Engine(store, wrong, None).search("mudflat rays")
    assert hits and hits[0].doc.title == "Rays"


def test_answer_model_reports_actual_model(populated):
    store, emb = populated
    llm = FakeLLM("Seals [1].")
    llm.last_model = "claude-opus-4-8"
    assert Engine(store, emb, llm).ask("what do white sharks eat?").model == "claude-opus-4-8"


def test_ask_uses_history_for_retrieval(populated):
    store, emb = populated
    llm = FakeLLM("Rays dig pits [1].")
    eng = Engine(store, emb, llm, k=4)
    ans = eng.ask("and rays?", history=[("where do bat rays forage?", "In mudflats [1].")])
    assert ans.sources and "Q: where do bat rays forage?" in llm.calls[0][1]
