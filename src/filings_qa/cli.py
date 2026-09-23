"""Command line entry point: ``filings-qa ingest``, ``stats``, ``index``, ``search``, ``ask`` and ``agent``."""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import edgar
from .agent import AGENT_MODELS, DEFAULT_MAX_STEPS, AgentResult, Step, rounds_text, run_agent
from .answer import ANSWER_MODELS, Answer, answer
from .chunk import chunk_filing
from .embed import (
    DEFAULT_MODEL,
    VECTORS_FILE,
    DenseIndex,
    Embedder,
    FakeEmbedder,
    FastEmbedEmbedder,
    embedder_from_spec,
    models_dir,
)
from .llm import Gemini, SetupError, short_error
from .parse import html_to_text, split_items
from .retrieve import STRATEGIES, retrieve
from .store import Store, db_path
from .tools import default_tools, tool_declarations

DEFAULT_FORMS = ("10-K", "10-Q")
DEFAULT_LIMIT = 4
PREVIEW_CHARS = 200
PROGRESS_EVERY_S = 10


def _load_companies(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...], int]:
    """Companies (each with an upper-case ``ticker`` and a list of ``ciks``, empty when the CIK is to be looked up),
    forms and per-company limit from companies.yaml."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    companies = []
    for entry in cfg.get("companies") or []:
        company = dict(entry) if isinstance(entry, dict) else {"ticker": entry}
        company["ticker"] = str(company["ticker"]).strip().upper()
        company["ciks"] = [int(c) for c in company.get("ciks") or []]
        companies.append(company)
    forms = tuple(cfg.get("forms") or DEFAULT_FORMS)
    return companies, forms, int(cfg.get("limit", DEFAULT_LIMIT))


def _ingest_filing(store: Store, filing: edgar.Filing, data: Path) -> int:
    raw = edgar.download(filing, data / "raw")
    text = html_to_text(raw.read_bytes())
    text_path = data / "text" / f"{filing.key}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    sections = split_items(text)
    chunks = chunk_filing(filing, sections)
    store.add_filing(filing, len(chunks))
    store.add_chunks(chunks)
    items = ", ".join(s.item or "-" for s in sections)
    period = filing.period.isoformat() if filing.period else "n/a"
    print(
        f"  {filing.form} filed {filing.filed} (period {period}): {len(text.split()):,} words,"
        f" {len(sections)} sections [{items}], {len(chunks)} chunks"
    )
    return len(chunks)


def cmd_ingest(args: argparse.Namespace) -> int:
    companies, forms, limit = _load_companies(Path(args.companies))
    if args.only:
        companies = [c for c in companies if c["ticker"] == args.only.strip().upper()]
        if not companies:
            print(f"{args.only} is not listed in {args.companies}", file=sys.stderr)
            return 2
    if not os.environ.get(edgar.USER_AGENT_ENV, "").strip():
        print(
            f"warning: {edgar.USER_AGENT_ENV} is not set; SEC usually rejects requests without a contact email"
            " (HTTP 403). See .env.example.",
            file=sys.stderr,
        )
    data = Path(args.data)
    failed: list[str] = []
    total_filings = total_chunks = 0
    with Store(db_path(data)) as store:
        for company in companies:
            ticker = company["ticker"]
            try:
                ciks = company["ciks"] or [edgar.cik_for(ticker, cache_path=data / "cache" / "company_tickers.json")]
                filings = edgar.list_filings(ciks, forms, limit, ticker=ticker)
            except edgar.EdgarError as e:
                print(f"{ticker}: failed: {e}", file=sys.stderr)
                failed.append(ticker)
                if e.status == 403:
                    print("stopping: every other request would be rejected the same way", file=sys.stderr)
                    break
                continue
            print(f"{ticker} (CIK {', '.join(map(str, ciks))}): {len(filings)} filings")
            n_filings = n_chunks = 0
            for filing in filings:
                try:
                    n_chunks += _ingest_filing(store, filing, data)
                    n_filings += 1
                except Exception as e:  # one bad filing should not stop the batch; the exit code reports it
                    print(f"  {filing.key}: failed: {type(e).__name__}: {e}", file=sys.stderr)
                    if ticker not in failed:
                        failed.append(ticker)
            print(f"{ticker}: {n_filings} filings, {n_chunks} chunks")
            total_filings += n_filings
            total_chunks += n_chunks
    summary = f"total: {total_filings} filings, {total_chunks} chunks"
    print(summary + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def _existing_db(data: str) -> Path | None:
    """The SQLite index in ``data``, or None after telling the user to run ingest (opening a missing database would
    create an empty one)."""
    path = db_path(data)
    if path.exists():
        return path
    print(f"no index at {path}; run `filings-qa ingest` first", file=sys.stderr)
    return None


def cmd_stats(args: argparse.Namespace) -> int:
    path = _existing_db(args.data)
    if path is None:
        return 1
    with Store(path) as store:
        s = store.stats()
    forms = sorted(s["by_form"])
    print(f"{'ticker':<8}" + "".join(f"{f:>7}" for f in forms) + f"{'chunks':>9}{'chunk words':>13}")
    for ticker, t in s["by_ticker"].items():
        counts = "".join(f"{t['forms'].get(f, 0):>7}" for f in forms)
        print(f"{ticker:<8}{counts}{t['chunks']:>9,}{t['words']:>13,}")
    print(
        f"total: {s['filings']} filings, {s['chunks']:,} chunks, {s['words']:,} chunk words"
        " (overlapping words count twice)"
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    path = _existing_db(args.data)
    if path is None:
        return 1
    embedder = FakeEmbedder() if args.fake else FastEmbedEmbedder(args.model, cache_dir=models_dir(args.data))
    label = f"hashed bag-of-words vectors ({embedder.dim} dimensions, for tests and demos)" if args.fake else args.model
    started = last = time.perf_counter()

    def progress(done: int, total: int) -> None:
        nonlocal last
        now = time.perf_counter()
        if done < total and now - last >= PROGRESS_EVERY_S:
            print(f"  {done:,} of {total:,} chunks, {now - started:.0f} s", flush=True)
            last = now

    with Store(path) as store:
        n = len(store.all_chunk_ids())
        if not n:
            print(f"no chunks in {path}; run `filings-qa ingest` first", file=sys.stderr)
            return 1
        print(f"embedding {n:,} chunks with {label}", flush=True)
        try:
            index = DenseIndex.build(store, embedder, path.parent, progress=progress)
        except RuntimeError as e:  # fastembed missing
            print(e, file=sys.stderr)
            return 1
    vectors = path.parent / VECTORS_FILE
    print(
        f"wrote {vectors}: {len(index):,} vectors of {index.dim} dimensions, {vectors.stat().st_size / 1e6:.1f} MB,"
        f" in {time.perf_counter() - started:.1f} s"
    )
    return 0


def _warn_if_stale(store: Store, dense: DenseIndex) -> None:
    """Warn when chunks were added or replaced after the dense index was built."""
    in_store, indexed = set(store.all_chunk_ids()), set(dense.ids)
    missing, removed = len(in_store - indexed), len(indexed - in_store)
    if missing or removed:
        print(
            f"warning: the dense index is out of date ({missing:,} chunks have no vector, {removed:,} vectors belong to"
            " chunks no longer stored); run `filings-qa index` again",
            file=sys.stderr,
        )


def _load_dense(store: Store, path: Path, args: argparse.Namespace) -> tuple[DenseIndex | None, Embedder | None]:
    """The dense index next to ``path`` and the embedder for its queries, or (None, None) for ``--strategy bm25``.
    Raises LookupError after printing why they cannot be loaded."""
    if args.strategy == "bm25":
        return None, None
    try:
        dense = DenseIndex.load(path.parent)
        embedder = embedder_from_spec(dense.embedder_spec, cache_dir=models_dir(args.data))
    except (FileNotFoundError, ValueError) as e:  # no index, a damaged one, or an unknown embedder
        print(f"{e} (or search without vectors: --strategy bm25)", file=sys.stderr)
        raise LookupError from e
    _warn_if_stale(store, dense)
    return dense, embedder


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "..."


def cmd_search(args: argparse.Namespace) -> int:
    path = _existing_db(args.data)
    if path is None:
        return 1
    with Store(path) as store:
        try:
            dense, embedder = _load_dense(store, path, args)
        except LookupError:
            return 1
        hits = retrieve(
            args.query,
            store=store,
            dense=dense,
            embedder=embedder,
            strategy=args.strategy,
            k=args.k,
            ticker=args.ticker,
            form=args.form,
        )
        chunks = {c.chunk_id: c for c in store.get_chunks([h.chunk_id for h in hits])}
    filters = [f"{name} {value.upper()}" for name, value in (("ticker", args.ticker), ("form", args.form)) if value]
    scope = f" ({', '.join(filters)})" if filters else ""
    print(f'{args.strategy} search for "{args.query}"{scope}: {len(hits)} results')
    for hit in hits:
        chunk = chunks.get(hit.chunk_id)
        print(f"{hit.rank:>2}. {hit.score:.4f}  {hit.chunk_id}  item {(chunk.item or '-') if chunk else '?'}")
        print(f"    {_preview(chunk.text) if chunk else '(no longer stored)'}")
    return 0


def _make_llm() -> Gemini:
    """The model client of ``ask`` (key from GEMINI_API_KEY); tests replace this function."""
    return Gemini()


def _print_answer(result: Answer) -> None:
    if result.abstained:
        print("Abstained: the retrieved excerpts do not answer this question.")
    elif not result.sentences:
        print(
            f"No answer left: {result.dropped_sentences} sentence(s) cited excerpts that were not retrieved and were"
            " dropped."
            if result.dropped_sentences
            else "No answer: the model returned no sentences."
        )
    indent = "  " if result.abstained else ""
    for sentence in result.sentences:
        cited = f" [{', '.join(sentence.citations)}]" if sentence.citations else ("" if indent else " [no citation]")
        print(f"{indent}{sentence.text}{cited}")
    if result.advice_hits:
        print(
            f'Note: the wording "{"; ".join(result.advice_hits)}" reads like investment advice. This tool only'
            " reports what the filings say; nothing here is a recommendation to buy, sell or hold."
        )
    print(
        f"model={result.model or 'none'} tokens in/out={result.usage['in']}/{result.usage['out']}"
        f" latency={result.latency_s:.1f}s dropped={result.dropped_sentences} uncited={result.uncited_sentences}"
    )


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        llm = _make_llm()
    except SetupError as e:
        print(e, file=sys.stderr)
        return 1
    path = _existing_db(args.data)
    if path is None:
        return 1
    models = _models(args.models)
    with Store(path) as store:
        try:
            dense, embedder = _load_dense(store, path, args)
        except LookupError:
            return 1
        retriever = functools.partial(
            retrieve, store=store, dense=dense, embedder=embedder, ticker=args.ticker, form=args.form
        )
        try:
            result = answer(
                args.question,
                retriever=retriever,
                llm=llm,
                store=store,
                strategy=args.strategy,
                k=args.k,
                models=models,
            )
        except Exception as e:  # quota, network, an unusable reply: a message, not a traceback
            print(f"no answer: {short_error(e)}", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_answer(result)
    return 0


def _models(arg: str | None) -> list[str] | None:
    """The models of a comma-separated ``--models``, or None (the default models) when none is named."""
    models = [m.strip() for m in (arg or "").split(",") if m.strip()]
    return models or None


def _step_lines(step: Step) -> tuple[str, str]:
    """``→ tool(arg=value, ...)`` and the indented one-line summary of its result."""
    args = ", ".join(f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in step.args.items())
    return f"→ {step.tool}({args})", f"  {step.result_summary}"


def _print_agent_result(result: AgentResult) -> None:
    print()
    print(result.final_answer)
    print()
    if result.truncated:
        print(
            f"Note: the model still wanted tools after {rounds_text(result.max_steps)} of tool calls (--max-steps),"
            " so it answered from the results it had."
        )
    if result.unknown_citations:
        print(
            f"Warning: the answer cites {', '.join(result.unknown_citations)}, which no search in this run returned;"
            " treat those statements as unsupported."
        )
    u = result.usage
    trace = f" trace={result.trace_path}" if result.trace_path else ""
    print(
        f"model={','.join(result.models) or 'none'} model_calls={u['calls']} tool_calls={len(result.steps)}"
        f" tokens in/out={u['in']}/{u['out']} latency={result.latency_s:.1f}s wall={result.wall_s:.1f}s{trace}"
    )


def cmd_agent(args: argparse.Namespace) -> int:
    if args.max_steps < 1:
        print("--max-steps must be at least 1", file=sys.stderr)
        return 2
    try:
        llm = _make_llm()
    except SetupError as e:
        print(e, file=sys.stderr)
        return 1
    path = _existing_db(args.data)
    if path is None:
        return 1
    progress = sys.stderr if args.json else sys.stdout  # --json keeps standard output for the JSON alone

    def show(step: Step) -> None:
        for line in _step_lines(step):
            print(line, file=progress, flush=True)

    with Store(path) as store:
        try:
            dense, embedder = _load_dense(store, path, args)
        except LookupError:
            return 1
        retriever = functools.partial(retrieve, store=store, dense=dense, embedder=embedder)
        try:
            result = run_agent(
                args.question,
                llm=llm,
                tools=default_tools(store, retriever, strategy=args.strategy),
                max_steps=args.max_steps,
                trace_dir=Path(args.data) / "traces",
                models=_models(args.models),
                declarations=tool_declarations(store.filings()),
                on_step=show,
            )
        except Exception as e:  # quota, network: a message, not a traceback
            print(f"no answer: {short_error(e)}", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_agent_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="filings-qa", description="Question answering over SEC 10-K/10-Q filings.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download, parse, chunk and index the filings of the companies in companies.yaml")
    p.add_argument("--companies", default="companies.yaml", help="company list (default: companies.yaml)")
    p.add_argument("--data", default="data", help="data folder (default: data)")
    p.add_argument("--only", metavar="TICKER", help="ingest only this company from the list")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("stats", help="count indexed filings and chunks per company")
    p.add_argument("--data", default="data", help="data folder (default: data)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser(
        "index", help="embed every chunk for dense and hybrid search (the model is downloaded to <data>/models once)"
    )
    p.add_argument("--data", default="data", help="data folder (default: data)")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--model", default=DEFAULT_MODEL, help=f"fastembed model (default: {DEFAULT_MODEL})")
    which.add_argument("--fake", action="store_true", help="hashed bag-of-words vectors, no model (tests and demos)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="show the chunks retrieved for a query")
    p.add_argument("query")
    p.add_argument("--strategy", choices=STRATEGIES, default="hybrid", help="(default: hybrid)")
    p.add_argument("--k", type=int, default=8, help="number of results (default: 8)")
    p.add_argument("--ticker", help="only chunks of this company")
    p.add_argument("--form", help="only chunks of this form, e.g. 10-K")
    p.add_argument("--data", default="data", help="data folder (default: data)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question from the filings, citing a chunk id after every sentence")
    p.add_argument("question")
    p.add_argument("--strategy", choices=STRATEGIES, default="hybrid", help="retrieval (default: hybrid)")
    p.add_argument("--k", type=int, default=8, help="number of chunks shown to the model (default: 8)")
    p.add_argument("--ticker", help="only chunks of this company")
    p.add_argument("--form", help="only chunks of this form, e.g. 10-K")
    p.add_argument(
        "--models", help=f"Gemini models to try in order, comma-separated (default: {','.join(ANSWER_MODELS)})"
    )
    p.add_argument("--json", action="store_true", help="print the whole answer as JSON")
    p.add_argument("--data", default="data", help="data folder (default: data)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser(
        "agent", help="answer a compound question with tools: filing search, daily closes (yfinance), Google News"
    )
    p.add_argument("question")
    p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"most rounds of tool calls, i.e. model replies that ask for tools (default: {DEFAULT_MAX_STEPS})",
    )
    p.add_argument("--strategy", choices=STRATEGIES, default="hybrid", help="filing search (default: hybrid)")
    p.add_argument(
        "--models", help=f"Gemini models to try in order, comma-separated (default: {','.join(AGENT_MODELS)})"
    )
    p.add_argument("--json", action="store_true", help="print the whole result as JSON (the steps go to stderr)")
    p.add_argument("--data", default="data", help="data folder; traces are saved in <data>/traces (default: data)")
    p.set_defaults(func=cmd_agent)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
