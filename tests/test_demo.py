import os
import re
import socket
import tempfile
from pathlib import Path

from filings_qa import cli, demo
from filings_qa.demo import DemoQuestion
from filings_qa.embed import DenseIndex
from filings_qa.store import Store, db_path

KEYS = ("GEMINI_API_KEY", "SEC_USER_AGENT", "FINNHUB_API_KEY", "FASTEMBED_CACHE_PATH")
README = Path(__file__).resolve().parents[1] / "README.md"


class ReadRecorder(dict):
    """A copy of the environment that records every variable read from it."""

    def __init__(self, data):
        super().__init__(data)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def test_every_answer_cites_chunks_that_are_in_the_index_and_were_retrieved(tmp_path):
    lines = []
    answers = demo.run(write=lines.append)
    questions = demo.load_questions()
    assert [a.question for a in answers] == [q.question for q in questions] and len(answers) == 3
    demo.build(tmp_path)  # the same excerpts, indexed again to look up the ids
    with Store(db_path(tmp_path)) as store:
        indexed = set(store.all_chunk_ids())
    assert set(DenseIndex.load(db_path(tmp_path).parent).ids) == indexed
    for q, result in zip(questions, answers, strict=True):
        assert not result.abstained and result.dropped_sentences == 0 and result.uncited_sentences == 0
        assert [s.text for s in result.sentences] == [s["text"] for s in q.reply["sentences"]]  # nothing dropped
        retrieved = {hit.chunk_id for hit in result.hits}
        assert len(retrieved) == demo.K and retrieved <= indexed
        for sentence in result.sentences:
            assert sentence.citations and set(sentence.citations) <= retrieved
            assert f"{sentence.text} [{', '.join(sentence.citations)}]" in lines  # printed as `ask` prints it
    assert demo.problems(answers) == []
    assert sum(line.startswith("Citation check: 3 sentences kept, 0 dropped") for line in lines) == 3


def test_the_command_needs_no_network_key_or_folder(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("the demo must not use the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    environ = ReadRecorder(os.environ)
    monkeypatch.setattr(os, "environ", environ)
    (tmp_path / "work").mkdir()
    (tmp_path / "tmp").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))

    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert out.count("Citation check:") == 3 and "All 3 answers passed the citation check." in out
    assert not environ.read & set(KEYS)  # no key, user agent or model folder is looked up
    assert not any((tmp_path / "work").iterdir())  # nothing written where it runs
    assert not any((tmp_path / "tmp").iterdir())  # and its temporary folder is gone


def test_the_command_fails_when_a_sentence_cites_a_chunk_not_retrieved(monkeypatch, capsys):
    question = demo.load_questions()[1].question  # NVIDIA's supply commitments
    reply = {
        "abstained": False,
        "sentences": [
            {"text": "The commitments rose to $279 billion.", "citations": ["NVDA-10-Q-20260826-2-005"]},
            {"text": "Apple bought back shares.", "citations": ["AAPL-10-K-20251031-5-001"]},  # indexed, not retrieved
        ],
    }
    monkeypatch.setattr(demo, "load_questions", lambda: [DemoQuestion(question, reply)])
    assert cli.main(["demo"]) == 1
    captured = capsys.readouterr()
    assert "filings-qa demo: 1 question answered offline" in captured.out
    assert "Citation check: 1 sentence kept, 1 dropped" in captured.out
    assert "demo check failed: question 1: 1 sentence(s) cited a chunk not retrieved" in captured.err


def test_the_corpus_is_six_short_excerpts_of_three_filings():
    excerpts = demo.load_excerpts()
    assert [e.name for e in excerpts] == [
        "AAPL-10-K-20251031-5",
        "AAPL-10-K-20251031-7",
        "COST-10-K-20251008-1",
        "COST-10-K-20251008-7",
        "NVDA-10-Q-20260826-1A",
        "NVDA-10-Q-20260826-2",
    ]
    for e in excerpts:
        assert len(e.section.text.split()) <= 1500
        assert re.match(rf"Item {e.section.item}\b", e.section.text)  # the start of the section: ids as in ingest
        assert e.filing.key == e.name.rsplit("-", 1)[0] and e.section.item == e.name.rsplit("-", 1)[1]
        assert re.fullmatch(r"https://www\.sec\.gov/Archives/edgar/data/\d+/\d{18}/[a-z]+-\d{8}\.htm", e.filing.url)
        assert e.filing.accession.replace("-", "") in e.filing.url and str(e.filing.cik) in e.filing.url
    assert len({e.filing.url for e in excerpts}) == 3  # two sections of one filing per company


def test_the_readme_quotes_the_demo_output_as_it_is(capsys):
    """The lines of demo output quoted in the README must still be printed by `filings-qa demo`: when the replies, the
    excerpts or the format change, the README must change too."""
    block = README.read_text(encoding="utf-8").split("$ filings-qa demo\n", 1)[1].split("```", 1)[0]
    quoted = [line for line in block.splitlines() if line.strip() and line != "…"]
    assert cli.main(["demo"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(quoted) >= 5 and [line for line in quoted if line not in printed] == []
