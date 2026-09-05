import numpy as np
import pytest

from labrag.chunk import Chunk
from labrag.embed import HashEmbedder, normalize
from labrag.store import EmbedderMismatch, Store, _fts_terms


def add_doc(store, emb, path, text_chunks, **meta):
    chunks = [Chunk(text=t, idx=i, page_start=i + 1, page_end=i + 1) for i, t in enumerate(text_chunks)]
    vecs = emb.embed([c.text for c in chunks])
    return store.upsert_document(
        source=meta.get("source", "nas"),
        rel_path=meta.get("rel_path", path),
        path=path,
        sha256=meta.get("sha256", "x"),
        size=1,
        mtime=1.0,
        title=meta.get("title"),
        authors=meta.get("authors"),
        year=meta.get("year"),
        doi=None,
        n_pages=len(chunks),
        chunks=chunks,
        embeddings=vecs,
    )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "index.db")
    yield s
    s.close()


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=64)
    a = e.embed(["white shark telemetry", "white shark telemetry", "pizza"])
    assert a.shape == (3, 64)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert a[0] @ a[2] < a[0] @ e.embed_query("shark telemetry data")


def test_fts_terms_are_quoted_and_stopwords_dropped():
    assert _fts_terms('What is the "diet" of white sharks?') == ['"diet"', '"white"', '"sharks"']
    assert _fts_terms("the of") == ['"the"', '"of"']
    assert _fts_terms("") == []


def test_upsert_search_update_delete(store):
    emb = HashEmbedder(dim=64)
    store.check_embedder(emb.name, emb.dim)
    d1 = add_doc(store, emb, "/nas/a.pdf", ["White sharks eat seals near the coast.", "Tagging methods for sharks."],
                 title="Shark diet", authors="Jorgensen S, Smith J", year=2010)
    d2 = add_doc(store, emb, "/nas/b.pdf", ["Bat rays forage in mudflats.", "Estuary ecology of leopard sharks."],
                 title="Rays", year=2020)
    assert store.stats()["documents"] == 2 and store.stats()["chunks"] == 4

    hits = store.search("what do white sharks eat", emb.embed_query("what do white sharks eat"), k=3)
    assert hits and hits[0].doc.id == d1
    assert "seals" in hits[0].text
    assert hits[0].keyword_rank == 1
    assert hits[0].doc.short_citation == "Jorgensen 2010"
    assert hits[0].pages == "p. 1"

    # keyword-only search still works with no vector
    kw = store.search("mudflats", None, k=3)
    assert kw and kw[0].doc.id == d2

    # update replaces chunks and bumps the version so the cached matrix reloads
    v_before = store.get_meta("index_version")
    add_doc(store, emb, "/nas/a.pdf", ["Completely different content about plankton."], title="New", year=2011)
    assert store.get_meta("index_version") != v_before
    assert store.stats()["chunks"] == 3
    hits = store.search("plankton", emb.embed_query("plankton"), k=3)
    assert hits[0].doc.title == "New"
    assert not store.search("seals", emb.embed_query("seals"), k=3) or "seals" not in store.search("seals", None)[0].text if store.search("seals", None) else True

    store.delete_document(d2)
    assert store.stats()["documents"] == 1
    assert store.search("mudflats", None) == []


def test_per_doc_cap_and_k(store):
    emb = HashEmbedder(dim=64)
    add_doc(store, emb, "/nas/a.pdf", [f"shark shark shark chunk {i}" for i in range(6)])
    add_doc(store, emb, "/nas/b.pdf", ["shark once"])
    hits = store.search("shark", emb.embed_query("shark"), k=10, per_doc=2)
    from collections import Counter

    counts = Counter(h.doc.path for h in hits)
    assert counts["/nas/a.pdf"] == 2 and counts["/nas/b.pdf"] == 1


def test_embedder_mismatch_is_detected(store):
    store.check_embedder("hash-64", 64)
    store.check_embedder("hash-64", 64)  # same is fine
    with pytest.raises(EmbedderMismatch):
        store.check_embedder("fastembed:bge", 384)
    emb = HashEmbedder(dim=64)
    add_doc(store, emb, "/nas/a.pdf", ["shark"])
    with pytest.raises(EmbedderMismatch):
        store.vector_search(normalize(np.ones(384))[0])


def test_special_characters_in_query_do_not_crash(store):
    emb = HashEmbedder(dim=32)
    add_doc(store, emb, "/nas/a.pdf", ["C-reactive protein in sharks"])
    for q in ['"unclosed', "AND OR NOT", "shark* NEAR(x)", "(", "c-reactive", "!!!"]:
        store.search(q, emb.embed_query(q))


def test_short_citation_fallbacks():
    from labrag.store import DocumentRow

    base = dict(id=1, source="nas", rel_path="Papers/Some_Long_Title_here.pdf", path="/x", sha256="", size=0, mtime=0,
                doi=None, n_pages=1, n_chunks=1, status="ok", error=None, indexed_at="")
    assert DocumentRow(title=None, authors="Salvador J. Jorgensen; Another", year=2015, **base).short_citation == "Jorgensen 2015"
    assert DocumentRow(title="A title", authors=None, year=2015, **base).short_citation == "A title (2015)"
    assert DocumentRow(title=None, authors="Christopher G. Lowe and Kelly Anderson", year=2000, **base).short_citation == "Lowe 2000"
    assert DocumentRow(title=None, authors="Jorgensen SJ, Reeb CA", year=2010, **base).short_citation == "Jorgensen 2010"
    assert DocumentRow(title=None, authors="Lowe", year=None, **base).short_citation == "Lowe"
    assert DocumentRow(title=None, authors=None, year=None, **base).short_citation == "Some_Long_Title_here"
