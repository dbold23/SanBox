"""Split a parsed document into overlapping passages that keep their page numbers.

Passages are the unit of retrieval. Around 300 words is a good compromise:
big enough to contain a complete idea, small enough that a handful of them fit
in a prompt. We pack whole paragraphs where we can and split long paragraphs
at sentence boundaries. The reference list at the end of a paper is dropped,
because it matches almost every query and never answers one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_REFERENCES_RE = re.compile(r"^\s*(references|literature cited|bibliography|works cited|references cited)\s*:?\s*$", re.I)


@dataclass
class Chunk:
    text: str
    idx: int
    page_start: int  # 1-based
    page_end: int

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def chunk_pages(
    pages: list[str],
    target_words: int = 300,
    overlap_words: int = 50,
    min_words: int = 40,
) -> list[Chunk]:
    paragraphs = _paragraphs_with_pages(pages)
    paragraphs = _drop_references(paragraphs)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    page_start = page_end = paragraphs[0][1]

    def flush() -> None:
        nonlocal current, current_words, page_start
        if not current:
            return
        text = "\n\n".join(current).strip()
        if text:
            chunks.append(Chunk(text=text, idx=len(chunks), page_start=page_start, page_end=page_end))
        # carry the tail of this chunk into the next one as overlap
        tail = " ".join(text.split()[-overlap_words:]) if overlap_words > 0 else ""
        current = [tail] if tail else []
        current_words = len(tail.split())
        page_start = page_end

    for para, page in paragraphs:
        pieces = [para]
        if len(para.split()) > target_words:
            pieces = _split_long_paragraph(para, target_words)
        for piece in pieces:
            n = len(piece.split())
            if current_words + n > target_words and current_words >= min_words:
                flush()
                page_start = min(page_start, page)
            if not current:
                page_start = page
            current.append(piece)
            current_words += n
            page_end = page
    if current_words >= min_words or not chunks:
        # emit the remainder unless it is only the overlap tail we carried over
        remainder = "\n\n".join(current).strip()
        if remainder and (not chunks or remainder != " ".join(chunks[-1].text.split()[-overlap_words:])):
            chunks.append(Chunk(text=remainder, idx=len(chunks), page_start=page_start, page_end=page_end))
    return chunks


def _paragraphs_with_pages(pages: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for page_no, page in enumerate(pages, start=1):
        for para in re.split(r"\n\s*\n", page or ""):
            para = " ".join(para.split())
            if para:
                out.append((para, page_no))
    return out


def _drop_references(paragraphs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    total = sum(len(p.split()) for p, _ in paragraphs)
    seen = 0
    for i, (para, _) in enumerate(paragraphs):
        if _REFERENCES_RE.match(para) and seen > total * 0.4:
            return paragraphs[:i]
        seen += len(para.split())
    return paragraphs


def _split_long_paragraph(para: str, target_words: int) -> list[str]:
    sentences = _SENTENCE_RE.split(para)
    pieces: list[str] = []
    buf: list[str] = []
    n = 0
    for s in sentences:
        sw = len(s.split())
        if n + sw > target_words and buf:
            pieces.append(" ".join(buf))
            buf, n = [], 0
        if sw > target_words * 1.5:  # a "sentence" with no punctuation: hard-split by words
            words = s.split()
            for i in range(0, len(words), target_words):
                pieces.append(" ".join(words[i : i + target_words]))
            continue
        buf.append(s)
        n += sw
    if buf:
        pieces.append(" ".join(buf))
    return pieces
