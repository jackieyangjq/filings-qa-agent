import re

import pytest

from filings_qa.chunk import chunk_filing, chunk_text
from filings_qa.parse import Section


def _paragraphs(prefix: str, n_paragraphs: int, words_each: int) -> str:
    """Paragraphs of unique words, so any word identifies its position."""
    return "\n".join(
        " ".join(f"{prefix}{p}_{w}" for w in range(words_each)) for p in range(n_paragraphs)
    )


def test_chunks_reach_about_280_words_and_overlap_by_50():
    chunks = chunk_text(_paragraphs("w", 40, 45))  # 1,800 words
    assert len(chunks) >= 4
    sizes = [len(c.split()) for c in chunks]
    assert all(s <= 280 for s in sizes)
    assert all(s > 230 for s in sizes[:-1])  # paragraphs are packed up to the target
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.split()[-50:] == nxt.split()[:50]  # the end of one chunk opens the next
    # nothing is lost: every word of the text appears in some chunk
    assert set(" ".join(chunks).split()) == set(_paragraphs("w", 40, 45).split())


def test_chunks_keep_paragraph_breaks_and_split_overlong_paragraphs():
    long_paragraph = " ".join(f"s{i}_{w}" for i in range(20) for w in range(49)) + "."  # 980 words, few sentences
    text = "Heading line\n" + long_paragraph + "\nClosing paragraph."
    chunks = chunk_text(text)
    assert chunks[0].startswith("Heading line\n")
    assert all(len(c.split()) <= 280 for c in chunks)
    assert chunks[-1].endswith("\nClosing paragraph.")
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.split()[-50:] == nxt.split()[:50]


def test_short_and_empty_text():
    assert chunk_text("A short section.\nTwo lines.") == ["A short section.\nTwo lines."]
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("x", target_words=50, overlap_words=50)


def test_chunk_filing_ids_and_item_boundaries(sample_filing):
    sections = [
        Section("", "Front matter", _paragraphs("cover", 2, 30)),
        Section("1A", "Risk Factors", "Item 1A. Risk Factors\n" + _paragraphs("risk", 20, 50)),
        Section("7", "MD&A", "Item 7. MD&A\n" + _paragraphs("mda", 12, 50)),
    ]
    chunks = chunk_filing(sample_filing, sections)
    pattern = re.compile(r"^ACME-10-Q-20240802-(0|1A|7)-\d{3}$")
    assert all(pattern.match(c.chunk_id) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.filing_key for c in chunks} == {"ACME-10-Q-20240802"}

    risk = [c for c in chunks if c.item == "1A"]
    assert [c.ordinal for c in risk] == list(range(1, len(risk) + 1))
    assert risk[0].chunk_id == "ACME-10-Q-20240802-1A-001"
    assert chunks[0].chunk_id == "ACME-10-Q-20240802-0-001"  # text outside any item
    for c in chunks:  # no chunk mixes words from two sections
        prefixes = {re.match(r"[a-z]+", w).group() for w in c.text.split() if re.match(r"[a-z]+\d", w)}
        assert prefixes == {"cover": {"cover"}, "1A": {"risk"}, "7": {"mda"}}[c.item or "cover"]
        assert c.n_words == len(c.text.split())
