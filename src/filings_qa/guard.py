"""Checks run on a model's answer after it is generated.

``verify_citations`` enforces the rule that every citation is a chunk retrieved for the question: a sentence citing
anything else is dropped. ``advice_check`` flags wording that reads as a recommendation to buy, sell or hold.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .answer import Answer

# Phrases that read as investment advice, matched without regard to case. Variants with extra words in the middle
# ("should you buy", "recommend that you sell") are listed one by one rather than folded into looser patterns, so each
# entry stays easy to read and a false alarm on filing language is easy to trace. "should hold" alone is not listed:
# filings use it about courts and reserves; "you should hold" is.
ADVICE_PATTERNS: tuple[str, ...] = (
    r"\bshould\s+buy\b",
    r"\bshould\s+sell\b",
    r"\byou\s+should\s+hold\b",
    r"\bshould\s+you\s+buy\b",
    r"\bshould\s+you\s+sell\b",
    r"\bshould\s+(?:\w+\s+)?consider\s+buying\b",
    r"\bshould\s+(?:\w+\s+)?consider\s+selling\b",
    r"\brecommend(?:s|ed)?\s+buying\b",
    r"\brecommend(?:s|ed)?\s+selling\b",
    r"\brecommend(?:s|ed)?\s+holding\b",
    r"\brecommend(?:s|ed)?\s+(?:that\s+)?you\s+buy\b",
    r"\brecommend(?:s|ed)?\s+(?:that\s+)?you\s+sell\b",
    r"\brecommend(?:s|ed)?\s+(?:that\s+)?you\s+hold\b",
    r"\bstrong\s+buy\b",
    r"\bstrong\s+sell\b",
    r"\bgood\s+time\s+to\s+buy\b",
    r"\bgood\s+time\s+to\s+sell\b",
)
_ADVICE = [re.compile(p, re.IGNORECASE) for p in ADVICE_PATTERNS]


def advice_check(text: str) -> list[str]:
    """Every phrase of ``text`` that matches an advice pattern, as written in the text, in order of appearance."""
    found = sorted((m.start(), m.group(0)) for pattern in _ADVICE for m in pattern.finditer(text))
    return [phrase for _, phrase in found]


def clean_citation(citation: str) -> str:
    """A cited id without surrounding spaces or square brackets: "[AAPL-10-K-20251031-7-007] " is read as the id."""
    return citation.strip().strip("[]").strip()


def verify_citations(answer: Answer, allowed_ids: Iterable[str]) -> Answer:
    """``answer`` without the sentences that cite an id outside ``allowed_ids`` (the chunks the model was shown).

    ``dropped_sentences`` counts the sentences removed; ``uncited_sentences`` counts the kept sentences that cite
    nothing (kept, as for an abstention's explanation, but counted). Citations are cleaned (``clean_citation``) and
    de-duplicated; empty sentences are removed without being counted.
    """
    allowed = set(allowed_ids)
    kept = []
    dropped = uncited = 0
    for sentence in answer.sentences:
        text = sentence.text.strip()
        citations = list(dict.fromkeys(c for c in map(clean_citation, sentence.citations) if c))
        if not text:
            continue
        if any(c not in allowed for c in citations):
            dropped += 1
            continue
        uncited += not citations
        kept.append(replace(sentence, text=text, citations=citations))
    return replace(
        answer,
        sentences=kept,
        dropped_sentences=answer.dropped_sentences + dropped,
        uncited_sentences=uncited,
    )
