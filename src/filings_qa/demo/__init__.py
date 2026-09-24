"""Offline demo: three questions answered from six excerpts of SEC filings, with no network, key or model download.

``corpus/<TICKER>-<FORM>-<YYYYMMDD>-<item>.txt`` is an excerpt of one section of a filing: its first lines, at most
1,500 words, of the text that ``filings-qa ingest`` extracts, after a first line giving the filing's url on sec.gov
(SEC filings are public records). Since every excerpt starts where its section starts, its chunks get the ids that a
full ingest gives them, with the same text (the last one cut short), so the demo's citations can be checked against
the filings. ``replies.json`` lists the questions and the model reply the demo replays for each: the JSON that
``answer.answer`` asks the model for, written from the excerpts, with the chunk ids each sentence relies on.

``build`` indexes the excerpts as ``ingest`` and ``index`` would, except that the vectors come from ``FakeEmbedder``
(hashed words) instead of the embedding model. ``run`` does this in a temporary folder and answers each question with
``answer.answer`` and ``RecordedLLM``: the retrieval, the prompt and the citation check are those of
``filings-qa ask``, and only the model's reply is replayed. Nothing here reads an environment variable.
"""

from __future__ import annotations

import functools
import json
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..answer import Answer, answer
from ..chunk import chunk_filing
from ..edgar import Filing
from ..embed import DenseIndex, FakeEmbedder
from ..llm import RecordedLLM
from ..parse import Section
from ..retrieve import retrieve
from ..store import Store, db_path

ROOT = files(__name__)
STRATEGY = "hybrid"  # the default of `filings-qa ask`
K = 8  # chunks shown to the model, as `filings-qa ask` shows by default
DIM = 256  # coordinates of the hashed word vectors: enough that few words of a question share one by chance
COMPANIES = {"AAPL": "Apple", "NVDA": "NVIDIA", "COST": "Costco"}

_NAME = re.compile(r"(?P<ticker>[A-Z][A-Z.]*)-(?P<form>10-[KQ])-(?P<filed>\d{8})-(?P<item>[0-9A-Z]+(?:-[0-9A-Z]+)?)")
_URL = re.compile(r"https://www\.sec\.gov/Archives/edgar/data/(?P<cik>\d+)/(?P<folder>\d{18})/(?P<doc>[^/\s]+)")


@dataclass(frozen=True)
class Excerpt:
    name: str  # the file name without ".txt", e.g. "AAPL-10-K-20251031-7"
    filing: Filing
    section: Section


@dataclass(frozen=True)
class DemoQuestion:
    question: str
    reply: dict[str, Any]  # the model's JSON reply: {"abstained": ..., "sentences": [{"text", "citations"}]}


def load_excerpts() -> list[Excerpt]:
    """The excerpts in ``corpus/``, in file-name order."""
    excerpts = []
    for entry in sorted(ROOT.joinpath("corpus").iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".txt"):
            continue
        stem = entry.name.removesuffix(".txt")
        url, _, text = entry.read_text(encoding="utf-8").partition("\n")
        name, where = _NAME.fullmatch(stem), _URL.fullmatch(url.strip())
        if not name or not where:
            raise ValueError(
                f"demo excerpt {entry.name}: expected <TICKER>-<FORM>-<YYYYMMDD>-<item>.txt whose first line is the"
                " filing's url on sec.gov"
            )
        filed, folder = name["filed"], where["folder"]
        filing = Filing(
            ticker=name["ticker"],
            cik=int(where["cik"]),
            form=name["form"],
            filed=date(int(filed[:4]), int(filed[4:6]), int(filed[6:])),
            period=None,
            accession=f"{folder[:10]}-{folder[10:12]}-{folder[12:]}",
            primary_doc=where["doc"],
            url=url.strip(),
        )
        excerpts.append(Excerpt(stem, filing, Section(name["item"], "", text.strip())))
    return excerpts


def load_questions() -> list[DemoQuestion]:
    """The questions of ``replies.json``, in order, each with the reply the demo replays."""
    rows = json.loads(ROOT.joinpath("replies.json").read_text(encoding="utf-8"))
    return [DemoQuestion(row["question"], row["reply"]) for row in rows]


def build(data_dir: Path | str, excerpts: Sequence[Excerpt] | None = None) -> DenseIndex:
    """Index ``excerpts`` (default: ``load_excerpts()``) in ``data_dir`` as ``filings-qa ingest`` and ``index`` would:
    chunks in ``<data_dir>/index/filings.sqlite`` with its full-text index, and next to it a vector for each chunk,
    from ``FakeEmbedder(DIM)``. The sections of one filing are chunked together, so its chunk ids are numbered as
    ingest numbers them. Returns the vector index."""
    path = db_path(data_dir)
    by_filing: dict[str, list[Excerpt]] = {}
    for excerpt in excerpts if excerpts is not None else load_excerpts():
        by_filing.setdefault(excerpt.filing.key, []).append(excerpt)
    with Store(path) as store:
        for group in by_filing.values():
            chunks = chunk_filing(group[0].filing, [e.section for e in group])
            store.add_filing(group[0].filing, len(chunks))
            store.add_chunks(chunks)
        return DenseIndex.build(store, FakeEmbedder(DIM), path.parent)


def _listing(items: Sequence[object]) -> str:
    """1 -> "1"; 1, 3 -> "1 and 3"; 1, 3, 6 -> "1, 3 and 6"."""
    words = [str(item) for item in items]
    return " and ".join([", ".join(words[:-1]), words[-1]]) if len(words) > 1 else "".join(words)


def show(result: Answer, number: int, total: int, store: Store, write: Callable[[str], Any] = print) -> None:
    """Print one answer as ``filings-qa ask`` prints it (each sentence followed by its chunk ids), then what the
    citation check found and the url of each filing cited."""
    write(f"[{number}/{total}] {result.question}")
    if result.abstained:
        write("Abstained: the retrieved excerpts do not answer this question.")
    indent = "  " if result.abstained else ""
    for sentence in result.sentences:
        cited = f" [{', '.join(sentence.citations)}]" if sentence.citations else ("" if indent else " [no citation]")
        write(f"{indent}{sentence.text}{cited}")
    kept = len(result.sentences)
    rank = {hit.chunk_id: hit.rank for hit in result.hits}
    ranks = sorted(rank[c] for c in result.cited_ids if c in rank)
    where = f"; they cite the chunks ranked {_listing(ranks)} of the {len(result.hits)} retrieved" if ranks else ""
    write(f"Citation check: {kept} sentence{'' if kept == 1 else 's'} kept, {result.dropped_sentences} dropped{where}.")
    for key in dict.fromkeys(c.filing_key for c in store.get_chunks(result.cited_ids)):
        write(f"Source: {key} {(store.get_filing(key) or {}).get('url', '')}")


def problems(answers: Sequence[Answer]) -> list[str]:
    """What keeps ``answers`` from being a clean demo: an abstention, a dropped sentence or a sentence without a
    citation. Empty when every answer has sentences and each cites only chunks shown to the model."""
    found = []
    for n, result in enumerate(answers, start=1):
        if result.abstained or not result.sentences:
            found.append(f"question {n} was not answered")
        if result.dropped_sentences:
            found.append(f"question {n}: {result.dropped_sentences} sentence(s) cited a chunk not retrieved")
        if result.uncited_sentences:
            found.append(f"question {n}: {result.uncited_sentences} sentence(s) without a citation")
    return found


def run(write: Callable[[str], Any] = print) -> list[Answer]:
    """Build the index in a temporary folder (deleted afterwards), answer every question of ``replies.json`` with its
    replayed reply, print each answer with ``show`` and return the answers."""
    excerpts = load_excerpts()
    questions = load_questions()
    llm = RecordedLLM({f"answer:{q.question}": q.reply for q in questions}, tokens=(0, 0))
    tickers = set(e.filing.ticker for e in excerpts)
    names = [name for t, name in COMPANIES.items() if t in tickers] + sorted(tickers - set(COMPANIES))
    with tempfile.TemporaryDirectory(prefix="filings-qa-demo-") as tmp:
        dense = build(tmp, excerpts)
        plural = "" if len(questions) == 1 else "s"
        write(
            f"filings-qa demo: {len(questions)} question{plural} answered offline from {len(excerpts)} excerpts of SEC"
            f" filings by {_listing(names)} ({len(dense)} chunks, indexed in a temporary folder)."
        )
        write(
            "Stand-ins: hashed word vectors instead of the embedding model, and replies written in advance"
            " (demo/replies.json) instead of Gemini."
        )
        write(
            f"As in `filings-qa ask`: {STRATEGY} retrieval of {K} chunks, the same prompt, and a check that drops any"
            " sentence citing a chunk not shown to the model."
        )
        write("Chunk ids read <ticker>-<form>-<filing date>-<section>-<number>.")
        answers = []
        with Store(db_path(tmp)) as store:
            retriever = functools.partial(retrieve, store=store, dense=dense, embedder=FakeEmbedder(dense.dim))
            for number, q in enumerate(questions, start=1):
                result = answer(q.question, retriever=retriever, llm=llm, store=store, strategy=STRATEGY, k=K)
                answers.append(result)
                write("")
                show(result, number, len(questions), store, write)
    return answers
