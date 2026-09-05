import pytest

from labrag.config import Settings
from labrag.lookup import CrossrefLookup, apply_crossref
from labrag.parse import ParsedDoc
from labrag.providers import DEFAULT_MODELS, ProviderError, make_embedder, make_llm, resolve_llm_kind


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


CROSSREF = {
    "message": {
        "title": ["Philopatry and migration of Pacific white sharks"],
        "author": [{"family": "Jorgensen", "given": "Salvador J."}, {"family": "Reeb", "given": "Carol A."}, {"name": "The Tagging Consortium"}],
        "published-print": {"date-parts": [[2010, 3]]},
        "issued": {"date-parts": [[2009]]},
    }
}


def test_apply_crossref_sets_fields():
    doc = ParsedDoc(text="", title="junk", doi="10.1098/rspb.2009.1155")
    assert apply_crossref(doc, CROSSREF["message"])
    assert doc.title == "Philopatry and migration of Pacific white sharks"
    assert doc.authors == "Jorgensen S, Reeb C, The Tagging Consortium"
    assert doc.year == 2010


def test_lookup_handles_404_errors_and_gives_up_after_failures():
    client = FakeClient([FakeResp(200, CROSSREF), FakeResp(404), ConnectionError("down")] + [ConnectionError("down")] * 10)
    lk = CrossrefLookup(client=client)
    d1 = ParsedDoc(text="", doi="10.1/a")
    assert lk.enrich(d1) and d1.year == 2010
    d2 = ParsedDoc(text="", doi="10.1/b", title="keep")
    assert not lk.enrich(d2) and d2.title == "keep"
    for _ in range(5):
        lk.enrich(ParsedDoc(text="", doi="10.1/c"))
    calls = client.calls
    assert not lk.enrich(ParsedDoc(text="", doi="10.1/d"))
    assert client.calls == calls  # stopped calling after repeated network failures
    assert not lk.enrich(ParsedDoc(text="", doi=None))


def test_resolve_llm_kind_auto(monkeypatch):
    monkeypatch.setattr("labrag.providers.ollama_available", lambda url, timeout=1.5: False)
    assert resolve_llm_kind(Settings()) == "none"
    assert resolve_llm_kind(Settings(anthropic_api_key="k")) == "anthropic"
    assert resolve_llm_kind(Settings(openai_api_key="k")) == "openai"
    monkeypatch.setattr("labrag.providers.ollama_available", lambda url, timeout=1.5: True)
    assert resolve_llm_kind(Settings()) == "ollama"
    assert resolve_llm_kind(Settings(llm="none", anthropic_api_key="k")) == "none"


def test_make_llm_and_embedder(monkeypatch):
    monkeypatch.setattr("labrag.providers.ollama_available", lambda url, timeout=1.5: False)
    assert make_llm(Settings()) is None
    llm = make_llm(Settings(anthropic_api_key="sk-test"))
    assert llm.name == f"anthropic:{DEFAULT_MODELS['anthropic']}" and llm.effort == "medium"
    assert make_llm(Settings(llm="ollama", llm_model="gemma3")).name == "ollama:gemma3"
    assert make_llm(Settings(openai_api_key="k")).name == "openai:gpt-4o-mini"
    with pytest.raises(ProviderError):
        make_llm(Settings(llm="anthropic"))
    with pytest.raises(ProviderError):
        make_llm(Settings(llm="banana"))
    assert make_embedder(Settings(embed="hash")).name == "hash-256"
    with pytest.raises(ProviderError):
        make_embedder(Settings(embed="nope"))
