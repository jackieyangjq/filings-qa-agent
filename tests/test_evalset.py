import json
import re
from dataclasses import replace
from datetime import date

import pytest

from filings_qa import cli, evalset
from filings_qa.chunk import Chunk
from filings_qa.evalset import (
    Question,
    build,
    check_reply,
    eligible,
    find_quote,
    load,
    names_company,
    plan,
    save,
    strata,
)
from filings_qa.evaluate import section_of
from filings_qa.llm import FakeLLM
from filings_qa.store import Store

NAMES = {"ACME": ("Acme",), "BETA": ("Beta Corp",), "GAMA": ("Gamma",), "BANK": ("Bank Co",)}
FILLER = " ".join(f"w{i}" for i in range(100))  # makes a chunk long enough to draw a question from


@pytest.fixture
def eval_store(tmp_path, sample_filing):
    """Four companies in tmp_path/index/filings.sqlite. ACME and BETA: a 10-K (items 1A, 7, 8) and a 10-Q (items 1A,
    2, 1), GAMA only the 10-K, each section two chunks; BANK (sampled by filing) a 10-K (item 15) and a 10-Q (no
    items), two chunks each. Every chunk starts with a sentence stating a figure; ACME's 10-Q repeats the risk factors
    of its 10-K word for word. ACME also has a short chunk and a table chunk, which are never drawn."""
    store = Store(tmp_path / "index" / "filings.sqlite")
    number = 0
    texts = {}
    layout = {
        "ACME": {"10-K": ["1A", "7", "8"], "10-Q": ["1A", "2", "1"]},
        "BETA": {"10-K": ["1A", "7", "8"], "10-Q": ["1A", "2", "1"]},
        "GAMA": {"10-K": ["1A", "7", "8"]},
        "BANK": {"10-K": ["15"], "10-Q": [""]},
    }
    for ticker, forms in layout.items():
        for form, items in forms.items():
            filed = date(2024, 2, 16) if form == "10-K" else date(2024, 8, 2)
            filing = replace(sample_filing, ticker=ticker, form=form, filed=filed, accession=f"{ticker}-{form}")
            chunks = []
            for item in items:
                for ordinal in (1, 2):
                    if (ticker, form, item) == ("ACME", "10-Q", "1A"):
                        text = texts["ACME", "10-K", "1A", ordinal]
                    else:
                        number += 1
                        text = f"{NAMES[ticker][0]} reported revenue of ${number} million for the period.\n{FILLER}"
                    texts[ticker, form, item, ordinal] = text
                    chunks.append(_chunk(filing, item, ordinal, text))
            if ticker == "ACME" and form == "10-Q":
                chunks.append(_chunk(filing, "2", 3, "Acme reported revenue of $999 million."))  # too short
                table = "\n".join(f"Revenue\t{i}\t{i + 1}" for i in range(60)) + "\nAcme reported a table."
                chunks.append(_chunk(filing, "2", 4, table))  # mostly table rows
            store.add_filing(filing, len(chunks))
            store.add_chunks(chunks)
    yield store
    store.close()


def _chunk(filing, item, ordinal, text):
    return Chunk(f"{filing.key}-{item or '0'}-{ordinal:03d}", filing.key, item, ordinal, text, len(text.split()))


class PromptLLM(FakeLLM):
    """Replies with ``respond(label, contents)`` to each call, and keeps the labels."""

    def __init__(self, respond):
        super().__init__([])
        self.respond = respond
        self.labels = []

    def generate(self, models, *, label="", **kwargs):
        self.labels.append(label)
        self.script = [self.respond(label, kwargs["contents"])]
        return super().generate(models, label=label, **kwargs)


def good_question(label, contents):
    """A question about the first sentence of the excerpt, quoting it with its line break made a space."""
    company = re.search(r"^Company: (.+) \(", contents, re.M)[1]
    first = contents.split("Excerpt:\n", 1)[1].split("\n", 1)[0]
    figure = re.search(r"\$\d+ million", first)[0]
    return {
        "skip": False,
        "question": f"How much revenue did {company} report for the period?",
        "answer": f"{figure}.",
        "quote": first.replace("revenue of", "revenue  of"),  # a spacing difference is fine
    }


def rewrite(label, contents):
    """Rewrites as the instruction says: swaps the company name, or the period for fiscal 2019."""
    question = re.search(r"^Question: (.+)$", contents, re.M)[1]
    swap = re.search(r"Ask about (.+?) \((\w+)\) instead of (.+?)\.", contents)
    if swap:
        return {"question": question.replace(swap[3], swap[1])}
    return {"question": question.replace("for the period", "for fiscal 2019")}


def respond_well(label, contents):
    return rewrite(label, contents) if label.startswith("rewrite:") else good_question(label, contents)


def build_questions(store, llm, **kwargs):
    kwargs = {"n_answerable": 8, "n_unanswerable": 0, "names": NAMES, "by_filing": ("BANK",), **kwargs}
    return build(store, llm, **kwargs)


def chunk_text(store, chunk_id):
    return store.get_chunks([chunk_id])[0].text


def test_a_quote_not_in_the_chunk_is_dropped_and_another_chunk_of_the_stratum_is_tried(eval_store):
    """The first reply for every stratum quotes words the chunk does not hold: each is rejected, and the stratum's
    question comes from the next chunk tried."""

    def respond(label, contents):
        reply = good_question(label, contents)
        if len(llm.labels) % 2:  # calls 1, 3, 5, ...: the first try of each stratum
            reply["quote"] = "Revenue reached a record high for the period."
        return reply

    llm = PromptLLM(respond)
    log = []
    questions = build_questions(eval_store, llm, n_answerable=13, log=log.append)  # every stratum once
    assert len(questions) == 13 and len(llm.labels) == 26
    tried_first = {label.split(":", 1)[1] for label in llm.labels[0::2]}
    tried_second = [label.split(":", 1)[1] for label in llm.labels[1::2]]
    assert [q.gold_chunk_id for q in questions] == tried_second
    assert not tried_first & {q.gold_chunk_id for q in questions}
    assert sum("rejected (quote not found in the chunk)" in line for line in log) == 13
    for q in questions:
        text = chunk_text(eval_store, q.gold_chunk_id)
        assert q.gold_quote in text and q.gold_quote.endswith("million for the period.")  # as written in the chunk
        assert not q.reviewed and q.answerable
        assert (q.gold_filing_key, q.gold_item) == section_of(q.gold_chunk_id)
        # every chunk of the company holding the quote is a gold chunk: ACME's 10-Q repeats its 10-K's risk factors
        if q.stratum == "ACME/1A":
            ordinal = q.gold_chunk_id[-3:]
            assert q.gold_chunk_ids == [f"ACME-10-K-20240216-1A-{ordinal}", f"ACME-10-Q-20240802-1A-{ordinal}"]
        else:
            assert q.gold_chunk_ids == [q.gold_chunk_id]


def test_strata_cover_every_company_and_spread_over_sections_and_filings(eval_store):
    questions = build_questions(eval_store, PromptLLM(respond_well), n_answerable=8)
    by_company = {}
    for q in questions:
        by_company.setdefault(q.ticker, []).append(q)
    assert {t: len(qs) for t, qs in by_company.items()} == {"ACME": 2, "BETA": 2, "GAMA": 2, "BANK": 2}
    for qs in by_company.values():
        assert len({q.stratum for q in qs}) == 2  # two different sections, or for BANK two filings
    assert {q.stratum for q in by_company["BANK"]} == {"BANK/BANK-10-K-20240216", "BANK/BANK-10-Q-20240802"}
    kinds = [q.stratum.split("/")[1] for q in questions if q.ticker != "BANK"]
    assert len(set(kinds)) >= 3  # the kinds are rotated from one company to the next
    assert [q.qid for q in questions] == [f"q{i:02d}" for i in range(1, 9)]
    # the short chunk and the table chunk are never drawn
    assert not {"ACME-10-Q-20240802-2-003", "ACME-10-Q-20240802-2-004"} & {q.gold_chunk_id for q in questions}


def test_unanswerable_questions_are_rewrites_about_another_company_or_2019(eval_store):
    def respond(label, contents):
        if label.startswith("rewrite:") and label.endswith(":NFLX") and "NFLX" not in rejected:
            rejected.append("NFLX")
            return {"question": "How much revenue did Acme and Netflix report for the period?"}  # still names Acme
        return respond_well(label, contents)

    rejected = []
    log = []
    questions = build_questions(eval_store, PromptLLM(respond), n_answerable=6, n_unanswerable=5, log=log.append)
    answerable, unanswerable = questions[:6], questions[6:]
    assert all(q.answerable for q in answerable) and len(unanswerable) == 5
    assert [q.kind for q in unanswerable] == ["other_company"] * 3 + ["other_period"] * 2
    assert [q.ticker for q in unanswerable[:3]] == ["NFLX", "INTC", "DIS"]
    assert [q.qid for q in unanswerable] == ["q07", "q08", "q09", "q10", "q11"]
    names = {**evalset.COMPANY_NAMES, **NAMES}
    sources = {q.qid: q for q in answerable}
    for q in unanswerable:
        assert not q.answerable and q.gold_chunk_id is None and q.gold_chunk_ids == [] and not q.reviewed
        source = sources[q.source_qid]
        if q.kind == "other_company":
            assert names_company(q.question, q.ticker, names) and not names_company(q.question, source.ticker, names)
        else:
            assert "fiscal 2019" in q.question and names_company(q.question, source.ticker, names)
    assert len({q.source_qid for q in unanswerable}) == 5
    assert rejected == ["NFLX"] and any("rejected (still names" in line for line in log)


def test_a_second_build_reuses_the_cached_replies(eval_store, tmp_path):
    cache = tmp_path / "cache" / "evalset"
    first = build_questions(eval_store, PromptLLM(respond_well), n_unanswerable=2, cache_dir=cache)
    assert len(list(cache.glob("*.json"))) == 10
    again = build_questions(eval_store, FakeLLM([]), n_unanswerable=2, cache_dir=cache)  # any call would fail
    assert again == first


def test_questions_round_trip_through_json_lines(tmp_path):
    questions = [
        Question("q01", "How much revenue did Acme report?", "$1 million.", "Acme reported revenue of $1 million.",
                 "ACME-10-K-20240216-7-001", ["ACME-10-K-20240216-7-001"], "ACME-10-K-20240216", "7", "ACME",
                 "ACME/7"),
        Question("q02", "How much revenue did Netflix report?", "(none)", ticker="NFLX", kind="other_company",
                 source_qid="q01", answerable=False),
    ]
    path = tmp_path / "eval" / "questions.jsonl"
    save(questions, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["reviewed"] is False
    assert load(path) == questions


@pytest.mark.parametrize(
    "quote, found",
    [
        ("Net sales rose 5% to $94.9 billion.", "Net sales rose 5% to\t$94.9 billion."),  # a tab in the chunk
        ("“the Company’s products”", "the Company's\nproducts"),  # curly quotes, a line break
        ('"Net sales rose 5%"', "Net sales rose 5%"),  # quotation marks around the quote
        ("Net sales rose 6% to $94.9 billion.", None),  # another figure
        ("Net sales rose ... $94.9 billion.", None),  # an ellipsis
        ("net sales rose 5%", None),  # case matters
    ],
)
def test_find_quote(quote, found):
    text = "Overview\nNet sales rose 5% to\t$94.9 billion.\nWe sell the Company's\nproducts worldwide."
    assert find_quote(quote, text) == found


def test_check_reply_rejects_unusable_questions():
    chunk = Chunk("COST-10-K-20251008-7-001", "COST-10-K-20251008", "7", 1, "Costco raised its fee to $65.", 6)
    ok = {"skip": False, "question": "What is Costco's annual membership fee?", "answer": "$65.",
          "quote": "Costco raised its fee to $65."}
    assert check_reply(ok, chunk, "COST") == ("Costco raised its fee to $65.", "")
    cases = {
        "the model found no fact to ask about": {"skip": True},
        "yes/no question": {"question": "Did Costco raise its membership fee?"},
        "refers to the excerpt": {"question": "According to the passage, what is Costco's fee?"},
        "does not name the company": {"question": "What is the annual cost of a membership?"},  # "cost" is not COST
        "answer longer than 30 words": {"answer": " ".join(["fee"] * 31)},
        "quote not found in the chunk": {"quote": "Costco raised its annual fee to $65."},
        "quote too short or too long": {"quote": "$65."},
        "incomplete reply": {"answer": ""},
    }
    for why, change in cases.items():
        assert check_reply({**ok, **change}, chunk, "COST") == (None, why)


def test_eligible_chunks_and_the_rotation_of_sections(eval_store):
    by_company = strata(eval_store, by_filing=("BANK",))
    acme = {s.name: s for s in by_company["ACME"]}
    assert set(acme) == {"ACME/1A", "ACME/7", "ACME/2", "ACME/fin"}
    assert acme["ACME/2"].chunk_ids == ("ACME-10-Q-20240802-2-001", "ACME-10-Q-20240802-2-002")  # not 003, 004
    assert len(acme["ACME/fin"].chunk_ids) == 4  # the 10-K's item 8 and the 10-Q's item 1
    assert [s.name for s in by_company["BANK"]] == ["BANK/BANK-10-K-20240216", "BANK/BANK-10-Q-20240802"]
    assert not eligible(Chunk("x", "f", "7", 1, "Too short.", 2))
    assert not eligible(Chunk("x", "f", "7", 1, " ".join(["word"] * 100), 100))  # no number at all
    assert eligible(Chunk("x", "f", "7", 1, FILLER, 100))

    order = plan(by_company, seed=0)
    assert len(order) == 13 and len({s.name for s in order}) == 13
    first_round = order[:4]
    assert len({s.ticker for s in first_round}) == 4  # every company before any company's second stratum
    normal = [s.name.split("/")[1] for s in first_round if s.ticker != "BANK"]
    assert len(set(normal)) == 3  # three companies start with three different sections
    assert [s.name for s in order if s.ticker == "BANK"][0] == "BANK/BANK-10-K-20240216"  # the 10-K first


def test_cli_builds_questions_then_evaluates_them(eval_store, tmp_path, monkeypatch, capsys):
    def respond(label, contents):
        if label.startswith("answer:"):
            first_id = re.search(r"^\[(\S+)\]", contents.split("Excerpts from SEC filings:\n\n", 1)[1], re.M)[1]
            return {"abstained": False, "sentences": [{"text": "Revenue was reported.", "citations": [first_id]}]}
        if label.startswith("judge:"):
            return {"reason": "Matches.", "verdict": "correct"}
        return respond_well(label, contents)

    llm = PromptLLM(respond)
    monkeypatch.setattr(cli, "_make_llm", lambda: llm)
    monkeypatch.setattr(evalset, "COMPANY_NAMES", {**evalset.COMPANY_NAMES, **NAMES})
    monkeypatch.setattr(evalset, "BY_FILING", ("BANK",))
    data, out = str(tmp_path), tmp_path / "questions.jsonl"
    args = ["evalset", "build", "--answerable", "4", "--unanswerable", "2", "--out", str(out), "--data", data]
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    assert f"wrote 6 questions to {out}" in printed and "unanswerable: 2 (other_company 1, other_period 1)" in printed
    assert cli.main(args) == 1 and "--force" in capsys.readouterr().err  # reviewed questions are not overwritten
    assert len(load(out)) == 6

    results = tmp_path / "results.json"
    base = ["eval", "--questions", str(out), "--data", data, "--out", str(results)]
    assert cli.main([*base, "--strategies", "bm25"]) == 0
    printed = capsys.readouterr().out
    assert re.search(r"^\[bm25 1/6\] q01 gold at \d+, correct \(\d+\.\d s\)$", printed, re.M)
    assert "[bm25 6/6] q06 answered, should have abstained" in printed
    assert "| bm25 | 6/6 |" in printed and f"wrote {results}" in printed
    saved = json.loads(results.read_text())
    assert saved["summary"]["bm25"]["done"] == 6 and len(saved["items"]) == 6
    assert (tmp_path / "cache" / "eval" / "bm25" / "q06.json").exists()

    calls = len(llm.labels)
    assert cli.main([*base, "--strategies", "bm25", "--report-only"]) == 0  # from the cache alone
    assert len(llm.labels) == calls and "| bm25 | 6/6 |" in capsys.readouterr().out
    assert cli.main([*base, "--strategies", "dense", "--report-only"]) == 1  # nothing cached for dense
    assert "| dense | 0/6 |" in capsys.readouterr().out
    assert cli.main([*base, "--strategies", "bm25,best"]) == 2
    assert cli.main([*base, "--strategies", "dense"]) == 1  # no vectors were built
    assert "filings-qa index" in capsys.readouterr().err
