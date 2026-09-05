"""Retrieve passages, ask the model, return an answer with numbered citations."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .embed import Embedder
from .llm import LLM, LLMError
from .store import Hit, Store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the literature assistant for a marine biology research lab. You answer questions using only the numbered SOURCES provided, which are passages from papers and documents in the lab's own collection.

Rules:
- Base every statement on the sources. After each claim, cite the source number in square brackets, like [2] or [1, 3].
- If the sources do not contain the answer, say so plainly in one sentence, then say what the sources *do* cover that is closest, and suggest a better search phrase. Never invent findings or citations.
- Prefer specific details (species, numbers, locations, methods, years) over generalities. Note disagreements between sources.
- Be concise: a short paragraph or a few bullet points. Plain prose, no headings. Do not repeat the question."""

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_FOLLOWUP_RE = re.compile(
    r"^(and|but|what about|how about|also|then|same|more|any|which of|what of|ok|okay)\b|\b(it|its|they|them|their|those|these|that one|the same|there)\b",
    re.IGNORECASE,
)


@dataclass
class Answer:
    question: str
    text: str | None  # None when no LLM is configured
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1-based source numbers actually cited
    model: str | None = None
    error: str | None = None
    elapsed: float = 0.0

    @property
    def cited_sources(self) -> list[tuple[int, Hit]]:
        return [(n, self.sources[n - 1]) for n in self.cited if 0 < n <= len(self.sources)]


class Engine:
    def __init__(self, store: Store, embedder: Embedder | None, llm: LLM | None = None, k: int = 8, per_doc: int = 3, max_tokens: int = 8000):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.k = k
        self.per_doc = per_doc
        self.max_tokens = max_tokens

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        qvec = None
        if self.embedder is not None:
            try:
                qvec = self.embedder.embed_query(question)
            except Exception as exc:  # embeddings are optional for search
                log.warning("Query embedding failed (%s); falling back to keyword search", exc)
        return self.store.search(question, qvec, k=k or self.k, per_doc=self.per_doc)

    def ask(self, question: str, k: int | None = None, history: list[tuple[str, str]] | None = None) -> Answer:
        started = time.monotonic()
        question = question.strip()
        hits = self.search(retrieval_query(question, history or []), k)
        answer = Answer(question=question, text=None, sources=hits)
        if self.llm is None:
            answer.elapsed = time.monotonic() - started
            return answer
        if not hits:
            answer.text = "Nothing in the indexed papers matches this question. Try different words, or check that the papers you expect have been indexed (`labrag status`)."
            answer.model = self.llm.name
            answer.elapsed = time.monotonic() - started
            return answer
        prompt = build_prompt(question, hits, history or [])
        try:
            text = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=self.max_tokens)
        except LLMError as exc:
            answer.error = str(exc)
            answer.elapsed = time.monotonic() - started
            return answer
        answer.text = text
        answer.model = self.llm.name
        answer.cited = extract_citations(text, len(hits))
        answer.elapsed = time.monotonic() - started
        return answer


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """Follow-ups ("what about rays?", "which of those used telemetry?") retrieve badly on
    their own, so fold the previous question in when the new one looks like a follow-up."""
    if not history:
        return question
    if len(question.split()) <= 6 or _FOLLOWUP_RE.search(question):
        previous = history[-1][0].strip()
        if previous and previous.lower() != question.lower():
            return f"{previous} {question}"
    return question


def format_source(n: int, hit: Hit, max_chars: int = 2500) -> str:
    doc = hit.doc
    label = doc.short_citation
    title = doc.title or doc.rel_path
    where = f", {hit.pages}" if hit.pages else ""
    text = hit.text if len(hit.text) <= max_chars else hit.text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return f"[{n}] {label} - \"{title}\"{where}\n{text}"


def build_prompt(question: str, hits: list[Hit], history: list[tuple[str, str]]) -> str:
    parts = ["SOURCES:"]
    for i, hit in enumerate(hits, start=1):
        parts.append(format_source(i, hit))
    if history:
        parts.append("\nEARLIER IN THIS CONVERSATION (for context only; cite the sources above, not this):")
        for q, a in history[-3:]:
            parts.append(f"Q: {q}\nA: {a[:1200]}")
    parts.append(f"\nQUESTION: {question}")
    return "\n\n".join(parts)


def extract_citations(text: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _CITE_RE.finditer(text):
        for tok in re.split(r"[,;]", m.group(1)):
            try:
                n = int(tok.strip())
            except ValueError:
                continue
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen
