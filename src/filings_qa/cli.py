"""Command line entry point: ``filings-qa ingest``, ``stats``, ``index`` and ``search``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import edgar
from .chunk import chunk_filing
from .embed import (
    DEFAULT_MODEL,
    VECTORS_FILE,
    DenseIndex,
    FakeEmbedder,
    FastEmbedEmbedder,
    embedder_from_spec,
    models_dir,
)
from .parse import html_to_text, split_items
from .retrieve import STRATEGIES, retrieve
from .store import Store, db_path

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


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "..."


def cmd_search(args: argparse.Namespace) -> int:
    path = _existing_db(args.data)
    if path is None:
        return 1
    with Store(path) as store:
        dense = embedder = None
        if args.strategy != "bm25":
            try:
                dense = DenseIndex.load(path.parent)
                embedder = embedder_from_spec(dense.embedder_spec, cache_dir=models_dir(args.data))
            except (FileNotFoundError, ValueError) as e:  # no index, a damaged one, or an unknown embedder
                print(f"{e} (or search without vectors: --strategy bm25)", file=sys.stderr)
                return 1
            _warn_if_stale(store, dense)
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
