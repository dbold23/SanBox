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
_REFERENCES_RE = re.compile(r"^\s*(references|literature cited|bibliography|works cited|references cited)\s*:?\s*$", re.IGNORECASE)


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
    pages = drop_references(pages)
    paragraphs = _paragraphs_with_pages(pages)
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


def drop_references(pages: list[str], min_fraction: float = 0.4) -> list[str]:
    """Cut the document at a 'References' / 'Literature Cited' heading that sits on its own
    line past `min_fraction` of the text. Everything after it (usually the bibliography, on
    this and later pages) is dropped. An early heading (table of contents) is ignored."""
    total = sum(len((p or "").split()) for p in pages)
    if total == 0:
        return pages
    seen = 0
    for page_no, page in enumerate(pages):
        lines = (page or "").split("\n")
        words_before = seen
        for i, line in enumerate(lines):
            if _REFERENCES_RE.match(line) and words_before > total * min_fraction:
                kept = "\n".join(lines[:i]).strip()
                return pages[:page_no] + ([kept] if kept else [])
            words_before += len(line.split())
        seen = words_before
    return pages


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
