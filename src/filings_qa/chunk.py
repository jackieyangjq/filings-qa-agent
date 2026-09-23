"""Split section text into chunks of about 400 words that overlap by 60 words and never cross a section boundary."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .edgar import Filing
    from .parse import Section

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<filing_key>-<item>-<NNN>", e.g. "AAPL-10-K-20241101-1A-003"; item "" is written as "0"
    filing_key: str  # "<TICKER>-<FORM>-<YYYYMMDD of the filing date>"
    item: str
    ordinal: int  # 1-based position within the section (the NNN of the id)
    text: str
    n_words: int


def _count(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Split a paragraph longer than ``max_words`` at sentence ends, and a sentence that is still too long at
    ``max_words``-word intervals."""
    if _count(paragraph) <= max_words:
        return [paragraph]
    pieces: list[str] = []
    current: list[str] = []
    n = 0
    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        if not words:
            continue
        if len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current, n = [], 0
            pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
            continue
        if n + len(words) > max_words:
            pieces.append(" ".join(current))
            current, n = [], 0
        current.append(sentence)
        n += len(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail(pieces: list[str], n_words: int) -> list[str]:
    """The last ``n_words`` words of ``pieces``, keeping line breaks and tabs."""
    out: list[str] = []
    need = n_words
    for piece in reversed(pieces):
        if need <= 0:
            break
        spans = [m.start() for m in _WORD.finditer(piece)]
        if len(spans) <= need:
            out.append(piece)
            need -= len(spans)
        else:
            out.append(piece[spans[-need] :])
            need = 0
    return out[::-1]


def chunk_text(text: str, *, target_words: int = 400, overlap_words: int = 60) -> list[str]:
    """Group the lines (paragraphs, table rows) of ``text`` into chunks of at most ``target_words`` words.

    Each chunk after the first starts with the last ``overlap_words`` words of the previous one. Lines longer than
    ``target_words - overlap_words`` words are split first, so every chunk stays within the target.
    """
    if target_words <= 0 or not 0 <= overlap_words < target_words:
        raise ValueError("need target_words > 0 and 0 <= overlap_words < target_words")
    max_piece = target_words - overlap_words
    pieces = [p for line in text.splitlines() if line.strip() for p in _split_long(line.strip(), max_piece)]
    chunks: list[str] = []
    current: list[str] = []
    n = 0
    fresh = 0  # words in ``current`` that are not overlap from the previous chunk
    for piece in pieces:
        k = _count(piece)
        if fresh and n + k > target_words:
            chunks.append("\n".join(current))
            current = _tail(current, overlap_words)
            n = sum(_count(p) for p in current)
            fresh = 0
        current.append(piece)
        n += k
        fresh += k
    if fresh:
        chunks.append("\n".join(current))
    return chunks


def chunk_filing(
    filing: Filing, sections: Iterable[Section], *, target_words: int = 400, overlap_words: int = 60
) -> list[Chunk]:
    """Chunks of every section of ``filing``, numbered from 1 within each item."""
    chunks: list[Chunk] = []
    counts: dict[str, int] = {}
    for section in sections:
        for text in chunk_text(section.text, target_words=target_words, overlap_words=overlap_words):
            ordinal = counts[section.item] = counts.get(section.item, 0) + 1
            chunk_id = f"{filing.key}-{section.item or '0'}-{ordinal:03d}"
            chunks.append(Chunk(chunk_id, filing.key, section.item, ordinal, text, _count(text)))
    return chunks
