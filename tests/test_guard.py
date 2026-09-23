import re

import pytest

from filings_qa.answer import Answer, Sentence
from filings_qa.guard import ADVICE_PATTERNS, advice_check, verify_citations


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("You should buy NVDA before the next report.", ["should buy"]),
        ("I recommend selling your Apple shares.", ["recommend selling"]),
        ("Analysts rate the stock a STRONG BUY.", ["STRONG BUY"]),
        ("Now is a good time to buy; later you should hold.", ["good time to buy", "you should hold"]),
        ("We recommend that you sell. Should you buy AMD instead?", ["recommend that you sell", "Should you buy"]),
        ("Investors should seriously consider buying\nthe shares.", ["should seriously consider buying"]),
    ],
)
def test_advice_check_flags_recommendations(text, hits):
    assert advice_check(text) == hits


@pytest.mark.parametrize(
    "text",
    [
        "NVIDIA's data center revenue grew 92% year over year, driven by demand for its computing platforms.",
        "The Board authorized an additional $100 billion for share repurchases.",
        "Tesla may sell regulatory credits to other automakers; strong buyer demand lifted deliveries.",
        "The court should hold that the claims are without merit, and management recommends reading the risk factors.",
        "Members who buy an executive membership renew at a higher rate; the company holds cash in money funds.",
    ],
)
def test_advice_check_leaves_filing_language_alone(text):
    assert advice_check(text) == []


def test_advice_patterns_cover_the_required_phrases():
    assert len(ADVICE_PATTERNS) >= 8
    for phrase in ["should buy", "should sell", "recommend buying", "recommend selling", "strong buy", "strong sell",
                   "good time to buy", "you should hold"]:
        assert any(re.search(p, phrase.upper(), re.IGNORECASE) for p in ADVICE_PATTERNS), phrase


def _answer(sentences, dropped=0):
    return Answer("q", "bm25", sentences, abstained=False, hits=[], dropped_sentences=dropped)


def test_verify_citations_drops_sentences_citing_chunks_not_retrieved():
    raw = _answer(
        [
            Sentence("Revenue grew 10%.", ["A-1", "B-2"]),
            Sentence("Margins fell.", ["A-1", "Z-9"]),  # one citation outside the retrieved chunks drops it
            Sentence("The filing does not give guidance.", []),
            Sentence("Costs rose. ", [" [B-2] ", "B-2"]),  # brackets, spaces and repeats are cleaned
            Sentence("  ", ["A-1"]),  # empty: removed without counting
            Sentence("Made up.", ["NOT-SHOWN"]),
        ]
    )
    checked = verify_citations(raw, ["A-1", "B-2", "C-3"])
    assert [(s.text, s.citations) for s in checked.sentences] == [
        ("Revenue grew 10%.", ["A-1", "B-2"]),
        ("The filing does not give guidance.", []),
        ("Costs rose.", ["B-2"]),
    ]
    assert checked.dropped_sentences == 2 and checked.uncited_sentences == 1
    assert [s.uncited for s in checked.sentences] == [False, True, False]
    assert len(raw.sentences) == 6  # the input is not changed


def test_verify_citations_counts_add_up_over_repeated_checks():
    once = verify_citations(_answer([Sentence("a", ["X"]), Sentence("b", ["Y"])], dropped=1), ["X", "Y"])
    twice = verify_citations(once, ["X"])
    assert (once.dropped_sentences, twice.dropped_sentences) == (1, 2)
    assert [s.text for s in twice.sentences] == ["a"]
