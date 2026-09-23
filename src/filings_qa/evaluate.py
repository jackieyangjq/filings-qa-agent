"""Evaluate retrieval and answers: every question with every strategy, graded, with metrics per strategy.

For each strategy and question, ``run`` ranks ``depth`` (10) chunks with the strategy, lets ``answer.answer`` answer
from the first ``k`` (8) of them, and asks a judge model to grade the answer to an answerable question against the
reference answer and quote. Each finished item is saved as ``<cache_dir>/<strategy>/<qid>.json`` and skipped on the
next run, so a run cut short (quota, network) goes on where it stopped; a failed item is not saved and is tried again
(an answer whose grading failed is saved, and only the grading is tried again). ``summarize`` computes the metrics
from the saved items and ``report`` shows them as a Markdown table.

Metrics per strategy, over the answerable questions unless said otherwise:

- recall@5, recall@10: a gold chunk (one holding the reference quote) is among the first 5 / 10 of the ranking.
- section_hit@5 (secondary): a chunk of the same filing and section as a gold chunk is among the first 5. Coarse for
  JPMorgan and Exxon Mobil, whose section labels are unreliable.
- correct, partial, incorrect: the judge's grade. An abstention, or an answer left without sentences after the
  citation check, counts as incorrect without asking the judge.
- citation_ok: of the answer sentences that cite chunks, the share citing a gold chunk or a chunk of the same filing
  and section.
- abstain_ok: the share of the unanswerable questions on which the answer model abstained.
- false_abstain: the share of the answerable questions on which it abstained.
- avg_in_tokens, avg_out_tokens, avg_latency_s, est_cost_usd: of the answer calls, over all questions (the judge is
  counted apart). Latency is the retrieval plus the model's response time, without waits between retries.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .answer import Retriever, answer
from .evalset import Question
from .llm import DailyQuotaError, Gemini, short_error, token_counts, with_quota_wait
from .store import Hit, Store

EVAL_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
JUDGE_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
K = 8  # chunks shown to the answer model, as `ask` shows by default
DEPTH = 10  # chunks ranked per question, for recall@10
MAX_FAILURES_IN_A_ROW = 3  # then the run stops: the model is probably out of quota or unreachable

PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_DATE = "2026-09-23"
# Paid-tier (Standard) prices in USD per million input tokens and per million output tokens (thinking included), as
# PRICING_URL gave them on PRICING_DATE (the page said "Last updated 2026-09-23 UTC"). The evaluation itself ran on
# the free tier, which costs nothing: these prices estimate what the same calls would cost on a paid plan. None when
# the prices are not known; the report then shows the free tier.
PRICING: dict[str, tuple[float, float]] | None = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
}

JUDGE_PROMPT = """\
You grade the answers of a question-answering system over SEC filings. You get the question, the reference answer \
written from the filing, the passage of the filing that holds it, and the system's answer.

Grade the system's answer against the reference answer and the passage only; do not use outside knowledge.
- "correct": it gives the key facts of the reference answer (figures, dates, names, terms) accurately, for the right \
company and period. Other wording, equivalent units or rounding (e.g. $1.2 billion for $1,196 million), and extra \
details that do not contradict the passage are fine.
- "partial": it gives part of the key facts correctly but misses, garbles or hedges another part the question asks \
for.
- "incorrect": its key facts are wrong or belong to another company or period, it contradicts the passage, or it \
does not answer the question (for example it says the information is not available).

First give the reason in one short sentence, then the grade."""


class JudgeSchema(BaseModel):
    reason: str = Field(description="Why, in one short sentence.")
    verdict: Literal["correct", "partial", "incorrect"] = Field(description="The grade.")


@dataclass
class Record:
    """The outcome of one question with one strategy. ``status`` is "done", "failed" (``error`` says why; tried again
    on the next run) or "not run" (the run stopped first). ``ranking`` holds the chunk ids ranked, best first;
    ``rank`` is the 1-based position of the first gold chunk in it and ``section_rank`` that of the first chunk of a
    gold section (None when absent; answerable questions only). The tokens and latency are those of the answer."""

    qid: str
    strategy: str
    question: str
    kind: str
    answerable: bool
    status: str = "not run"
    error: str = ""
    cached: bool = False
    ranking: list[str] = field(default_factory=list)
    rank: int | None = None
    section_rank: int | None = None
    abstained: bool | None = None
    sentences: list[dict[str, Any]] = field(default_factory=list)
    dropped_sentences: int = 0
    verdict: str | None = None
    reason: str = ""
    cited_sentences: int = 0
    citation_ok_sentences: int = 0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    judge_model: str = ""
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0


@dataclass
class Results:
    records: list[Record]  # one per strategy and question, strategy by strategy
    summary: dict[str, dict[str, Any]]  # the metrics of each strategy (see ``summarize``)
    settings: dict[str, Any] = field(default_factory=dict)
    stopped: str = ""  # why the run stopped before the end, if it did

    @property
    def complete(self) -> bool:
        return all(r.status == "done" for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "stopped": self.stopped,
            "summary": self.summary,
            "items": [asdict(r) for r in self.records],
        }


_CHUNK_ID = re.compile(r"^(?P<filing>.+?-\d{8})-(?P<item>.+)-\d{3}$")


def section_of(chunk_id: str) -> tuple[str, str]:
    """(filing key, item) of a chunk id "<ticker>-<form>-<yyyymmdd>-<item, "0" for none>-<nnn>"."""
    m = _CHUNK_ID.match(chunk_id)
    if not m:
        return chunk_id, ""
    return m["filing"], "" if m["item"] == "0" else m["item"]


def gold_ids(q: Question) -> set[str]:
    return set(q.gold_chunk_ids) | ({q.gold_chunk_id} if q.gold_chunk_id else set())


def needs_judge(q: Question, reply: Mapping[str, Any]) -> bool:
    """Whether the answer ``reply`` (an ``Answer`` as a dict) is graded by the judge: an answer with sentences to an
    answerable question. The others are graded without it."""
    return q.answerable and not reply["abstained"] and bool(reply["sentences"])


def is_complete(q: Question, entry: Mapping[str, Any] | None) -> bool:
    return entry is not None and "answer" in entry and (not needs_judge(q, entry["answer"]) or bool(entry.get("judge")))


def cache_file(cache_dir: Path | str, strategy: str, qid: str) -> Path:
    return Path(cache_dir) / strategy / f"{qid}.json"


def load_entry(path: Path, q: Question, k: int = K, depth: int = DEPTH) -> dict[str, Any] | None:
    """The saved item at ``path``, or None when there is none or it was made for another question text, ``k`` or
    ``depth`` (for example after the questions were rebuilt)."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if entry.get("question") != q.question or entry.get("k") != k or entry.get("depth") != depth:
        return None
    return entry


def _save(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def judge(
    q: Question,
    candidate: str,
    *,
    llm: Gemini,
    models: Sequence[str] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """The judge's grade of ``candidate``, the answer to ``q``: verdict, reason, model, usage and latency."""
    config = {
        "system_instruction": JUDGE_PROMPT,
        "response_mime_type": "application/json",
        "response_schema": JudgeSchema,
    }
    contents = (
        f"Question: {q.question}\n\nReference answer: {q.gold_answer}\n\nPassage of the filing: {q.gold_quote}"
        f"\n\nSystem's answer: {candidate}"
    )
    resp, model = with_quota_wait(
        lambda: llm.generate(list(models or JUDGE_MODELS), label=f"judge:{q.qid}", contents=contents, config=config),
        sleep=sleep,
    )
    llm.add_usage(model, resp.usage_metadata)
    parsed = resp.parsed if isinstance(resp.parsed, JudgeSchema) else JudgeSchema.model_validate_json(resp.text or "")
    tokens_in, tokens_out = token_counts(resp.usage_metadata)
    return {
        "verdict": parsed.verdict,
        "reason": " ".join(parsed.reason.split()),
        "model": model,
        "usage": {"in": tokens_in, "out": tokens_out},
        "latency_s": round(llm.last_latency, 3),
    }


def _answer_entry(
    q: Question,
    strategy: str,
    *,
    retriever: Retriever,
    llm: Gemini,
    store: Store,
    k: int,
    depth: int,
    models: Sequence[str],
    sleep: Callable[[float], Any],
) -> dict[str, Any]:
    """Rank ``depth`` chunks for ``q`` and answer from the first ``k``: the item to save, not yet graded."""
    started = time.perf_counter()
    ranking = retriever(q.question, strategy=strategy, k=depth)
    retrieval_s = time.perf_counter() - started
    shown = list(ranking[:k])

    def first_k(*_: Any, **__: Any) -> list[Hit]:  # answer() asks its retriever for k chunks: the ranking's first k
        return shown

    result = with_quota_wait(
        lambda: answer(
            q.question,
            retriever=first_k,
            llm=llm,
            store=store,
            strategy=strategy,
            k=k,
            models=models,
        ),
        sleep=sleep,
    )
    return {
        "qid": q.qid,
        "strategy": strategy,
        "question": q.question,
        "k": k,
        "depth": depth,
        "ranking": [h.chunk_id for h in ranking],
        "retrieval_s": round(retrieval_s, 4),
        "answer": result.to_dict(),
        "judge": None,
    }


def score(q: Question, strategy: str, entry: Mapping[str, Any] | None, *, status: str = "", error: str = "") -> Record:
    """The record of ``q`` with ``strategy`` from its saved item; without a complete item, a record with ``status``
    (default "not run") and ``error``."""
    rec = Record(q.qid, strategy, q.question, q.kind, q.answerable)
    if entry is None or not is_complete(q, entry):
        rec.status, rec.error = status or "not run", error
        return rec
    rec.status = "done"
    gold = gold_ids(q)
    gold_sections = {section_of(c) for c in gold}
    rec.ranking = list(entry["ranking"])
    if q.answerable:
        rec.rank = next((i for i, c in enumerate(rec.ranking, 1) if c in gold), None)
        rec.section_rank = next((i for i, c in enumerate(rec.ranking, 1) if section_of(c) in gold_sections), None)
    reply = entry["answer"]
    rec.abstained = bool(reply["abstained"])
    rec.sentences = [{"text": s["text"], "citations": list(s["citations"])} for s in reply["sentences"]]
    rec.dropped_sentences = reply.get("dropped_sentences", 0)
    rec.model = reply.get("model", "")
    rec.tokens_in, rec.tokens_out = reply["usage"]["in"], reply["usage"]["out"]
    rec.latency_s = round(entry.get("retrieval_s", 0.0) + reply.get("latency_s", 0.0), 3)
    if not q.answerable:
        return rec
    if rec.abstained:
        rec.verdict, rec.reason = "incorrect", "abstained"
        return rec
    if not rec.sentences:
        rec.verdict, rec.reason = "incorrect", "no sentence left after the citation check"
        return rec
    grade = entry["judge"]
    rec.verdict, rec.reason, rec.judge_model = grade["verdict"], grade["reason"], grade["model"]
    rec.judge_tokens_in, rec.judge_tokens_out = grade["usage"]["in"], grade["usage"]["out"]
    for sentence in rec.sentences:
        if sentence["citations"]:
            rec.cited_sentences += 1
            rec.citation_ok_sentences += any(c in gold or section_of(c) in gold_sections for c in sentence["citations"])
    return rec


def run(
    questions: Sequence[Question],
    strategies: Sequence[str],
    *,
    retriever: Retriever,
    llm: Gemini,
    judge_llm: Gemini,
    store: Store,
    cache_dir: Path | str,
    k: int = K,
    depth: int = DEPTH,
    models: Sequence[str] | None = None,
    judge_models: Sequence[str] | None = None,
    on_record: Callable[[Record, int, int], Any] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
    max_failures_in_a_row: int = MAX_FAILURES_IN_A_ROW,
) -> Results:
    """Answer and grade every question with every strategy, strategy by strategy, skipping the items saved in
    ``cache_dir``. ``retriever(question, strategy=..., k=...)`` ranks chunks; ``store`` gives their text. The answers
    come from ``llm`` with ``models`` (default ``EVAL_MODELS``), the grades from ``judge_llm`` with ``judge_models``
    (default ``JUDGE_MODELS``). A per-minute rate limit is waited out; a failed item is recorded and the run goes on,
    but it stops at a used-up daily quota or after ``max_failures_in_a_row`` failures in a row, leaving the rest
    "not run". ``on_record(record, position, number of questions)`` is called after each item."""
    models = list(models or EVAL_MODELS)
    judge_models = list(judge_models or JUDGE_MODELS)
    records: list[Record] = []
    stopped = ""
    failures = 0
    for strategy in strategies:
        for position, q in enumerate(questions, start=1):
            path = cache_file(cache_dir, strategy, q.qid)
            entry = load_entry(path, q, k, depth)
            cached = is_complete(q, entry)
            error = ""
            if not cached and not stopped:
                try:
                    if entry is None:
                        entry = _answer_entry(
                            q,
                            strategy,
                            retriever=retriever,
                            llm=llm,
                            store=store,
                            k=k,
                            depth=depth,
                            models=models,
                            sleep=sleep,
                        )
                        _save(path, entry)
                    if needs_judge(q, entry["answer"]) and not entry.get("judge"):
                        text = " ".join(s["text"] for s in entry["answer"]["sentences"])
                        entry["judge"] = judge(q, text, llm=judge_llm, models=judge_models, sleep=sleep)
                        _save(path, entry)
                    failures = 0
                except DailyQuotaError as e:
                    error = str(e)
                    stopped = f"stopped at {strategy} {q.qid}: {e}"
                except Exception as e:  # quota, network, an unusable reply: recorded, tried again next run
                    error = short_error(e)
                    failures += 1
                    if failures >= max_failures_in_a_row:
                        stopped = f"stopped at {strategy} {q.qid} after {failures} failures in a row; the last: {error}"
            record = score(q, strategy, entry, status="failed" if error else "not run", error=error)
            record.cached = cached
            records.append(record)
            if on_record:
                on_record(record, position, len(questions))
    return Results(records, summarize(records), _settings(records, k, depth), stopped)


def collect(
    questions: Sequence[Question], strategies: Sequence[str], *, cache_dir: Path | str, k: int = K, depth: int = DEPTH
) -> Results:
    """The results saved in ``cache_dir``, without asking any model (``eval --report-only``)."""
    records = []
    for strategy in strategies:
        for q in questions:
            record = score(q, strategy, load_entry(cache_file(cache_dir, strategy, q.qid), q, k, depth))
            record.cached = record.status == "done"
            records.append(record)
    return Results(records, summarize(records), _settings(records, k, depth))


def _settings(records: Sequence[Record], k: int, depth: int) -> dict[str, Any]:
    return {
        "k": k,
        "depth": depth,
        "answer_models": sorted({r.model for r in records if r.model}),
        "judge_models": sorted({r.judge_model for r in records if r.judge_model}),
        "pricing_usd_per_million_tokens": PRICING,
        "pricing_source": f"{PRICING_URL} (read {PRICING_DATE})",
    }


def cost_usd(
    model: str, tokens_in: int, tokens_out: int, pricing: Mapping[str, tuple[float, float]] | None = PRICING
) -> float | None:
    """What the tokens would cost on the paid tier; None when the model's prices are not known."""
    if not tokens_in and not tokens_out:
        return 0.0
    if pricing is None or model not in pricing:
        return None
    price_in, price_out = pricing[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def _rate(records: Sequence[Record], test: Callable[[Record], bool]) -> float | None:
    return round(sum(1 for r in records if test(r)) / len(records), 4) if records else None


def _mean(values: Sequence[float], digits: int) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def _total_cost(costs: Sequence[float | None]) -> float | None:
    return None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 6)


def summarize(
    records: Sequence[Record], pricing: Mapping[str, tuple[float, float]] | None = PRICING
) -> dict[str, dict[str, Any]]:
    """The metrics of each strategy over its finished records (see the module docstring). Rates are fractions from
    0 to 1, None when no record counts toward them."""
    out: dict[str, dict[str, Any]] = {}
    for strategy in dict.fromkeys(r.strategy for r in records):
        mine = [r for r in records if r.strategy == strategy]
        done = [r for r in mine if r.status == "done"]
        ans = [r for r in done if r.answerable]
        una = [r for r in done if not r.answerable]
        cited = sum(r.cited_sentences for r in ans)
        judged = [r for r in ans if r.judge_model]
        out[strategy] = {
            "questions": len(mine),
            "done": len(done),
            "failed": sum(r.status == "failed" for r in mine),
            "not_run": sum(r.status == "not run" for r in mine),
            "answerable": len(ans),
            "unanswerable": len(una),
            "recall_at_5": _rate(ans, lambda r: r.rank is not None and r.rank <= 5),
            "recall_at_10": _rate(ans, lambda r: r.rank is not None and r.rank <= 10),
            "section_hit_at_5": _rate(ans, lambda r: r.section_rank is not None and r.section_rank <= 5),
            "correct": _rate(ans, lambda r: r.verdict == "correct"),
            "partial": _rate(ans, lambda r: r.verdict == "partial"),
            "incorrect": _rate(ans, lambda r: r.verdict == "incorrect"),
            "citation_ok": round(sum(r.citation_ok_sentences for r in ans) / cited, 4) if cited else None,
            "cited_sentences": cited,
            "abstain_ok": _rate(una, lambda r: bool(r.abstained)),
            "false_abstain": _rate(ans, lambda r: bool(r.abstained)),
            "avg_in_tokens": _mean([r.tokens_in for r in done], 1),
            "avg_out_tokens": _mean([r.tokens_out for r in done], 1),
            "avg_latency_s": _mean([r.latency_s for r in done], 2),
            "est_cost_usd": _total_cost([cost_usd(r.model, r.tokens_in, r.tokens_out, pricing) for r in done]),
            "dropped_sentences": sum(r.dropped_sentences for r in done),
            "judge_calls": len(judged),
            "judge_in_tokens": sum(r.judge_tokens_in for r in judged),
            "judge_out_tokens": sum(r.judge_tokens_out for r in judged),
            "judge_est_cost_usd": _total_cost(
                [cost_usd(r.judge_model, r.judge_tokens_in, r.judge_tokens_out, pricing) for r in judged]
            ),
        }
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def report(results: Results) -> str:
    """The metrics as a Markdown table, one row per strategy, with a key to the columns."""
    columns = (
        "Strategy",
        "Done",
        "Recall@5",
        "Recall@10",
        "Section hit@5",
        "Correct",
        "Partial",
        "Incorrect",
        "Citation OK",
        "Abstain OK",
        "False abstain",
        "Avg in tokens",
        "Avg out tokens",
        "Avg latency (s)",
        "Est. cost (USD)",
    )
    lines = ["| " + " | ".join(columns) + " |", "|---|" + "---:|" * (len(columns) - 1)]
    for strategy, s in results.summary.items():
        if PRICING is None:
            cost = "free tier ($0)"
        else:
            cost = "n/a" if s["est_cost_usd"] is None else f"${s['est_cost_usd']:.4f}"
        cells = [
            strategy,
            f"{s['done']}/{s['questions']}",
            *(_pct(s[key]) for key in ("recall_at_5", "recall_at_10", "section_hit_at_5")),
            *(
                _pct(s[key])
                for key in ("correct", "partial", "incorrect", "citation_ok", "abstain_ok", "false_abstain")
            ),
            _num(s["avg_in_tokens"], ",.0f"),
            _num(s["avg_out_tokens"], ",.0f"),
            _num(s["avg_latency_s"], ".2f"),
            cost,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    first = next(iter(results.summary.values()), None)
    n_ans, n_una = (first["answerable"], first["unanswerable"]) if first else (0, 0)
    models = ", ".join(results.settings.get("answer_models") or ["none yet"])
    judges = ", ".join(results.settings.get("judge_models") or ["none yet"])
    price = (
        "; ".join(f"{m} ${i:.2f} in / ${o:.2f} out per million tokens" for m, (i, o) in PRICING.items())
        if PRICING
        else "not known"
    )
    lines += [
        "",
        f"Answer model: {models}. Judge: {judges}. Each strategy ranks {results.settings.get('depth', DEPTH)} chunks"
        f" per question and the answer model sees the first {results.settings.get('k', K)}.",
        f"Recall, section hit, the grades, citation OK and false abstain are over the {n_ans} answerable questions"
        f" done; abstain OK is over the {n_una} unanswerable ones. Tokens, latency and cost are of the answer calls"
        " over all questions done.",
        "- Recall@5 / Recall@10: a chunk holding the reference quote is among the first 5 / 10 chunks retrieved.",
        "- Section hit@5 (secondary): a chunk from the same filing and section as the reference is in the first 5.",
        "- Correct / Partial / Incorrect: the judge's grade against the reference answer and quote; an abstention"
        " counts as incorrect.",
        "- Citation OK: of the answer sentences with citations, the share citing the reference chunk or its section.",
        "- Abstain OK: unanswerable questions the model declined; False abstain: answerable ones it declined.",
        f"- Est. cost: the answer calls at paid-tier prices ({price}; {PRICING_URL}, read {PRICING_DATE}). The runs"
        " used the free tier and cost nothing.",
    ]
    if results.stopped:
        lines += ["", f"Incomplete: {results.stopped}."]
    return "\n".join(lines)
