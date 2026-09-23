"""SQLite store of filings and chunks, with an FTS5 full-text index over chunk text for BM25 keyword search."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .chunk import Chunk

if TYPE_CHECKING:
    from .edgar import Filing

DB_FILE = "filings.sqlite"

# Porter stemming lets "revenues" match "revenue". The triggers keep the index in step with the chunks table.
SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    filing_key TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    form TEXT NOT NULL,
    filed TEXT NOT NULL,
    period TEXT,
    accession TEXT NOT NULL,
    url TEXT NOT NULL,
    n_chunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    filing_key TEXT NOT NULL,
    item TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    n_words INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_filing ON chunks (filing_key);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_after_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_after_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;
"""

# Dropped from keyword queries: they match nearly every chunk and carry no signal.
STOPWORDS = frozenset(
    "a about an and are as at be been by can could did do does for from had has have how if in into is it its of on"
    " or than that the their there these this those to was were what when where which who why will with would".split()
)
_TERM = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # higher is better
    rank: int  # 1-based


def db_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "index" / DB_FILE


def fts_query(query: str) -> str:
    """FTS5 query that ORs the words of ``query``, each quoted so punctuation or operators in the question cannot
    break the query syntax. Stopwords and one-letter words are dropped unless nothing else is left."""
    terms = [t.lower() for t in _TERM.findall(query) if len(t) > 1 or t.isdigit()]
    kept = [t for t in terms if t not in STOPWORDS] or terms
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(kept))


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add_filing(self, filing: Filing, n_chunks: int) -> None:
        """Record ``filing``; recording a filing again replaces its row and deletes its old chunks, so a re-ingest
        never leaves stale chunks behind."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO filings (filing_key, ticker, form, filed, period, accession, url, n_chunks)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (filing_key) DO UPDATE SET ticker = excluded.ticker, form = excluded.form,
                     filed = excluded.filed, period = excluded.period, accession = excluded.accession,
                     url = excluded.url, n_chunks = excluded.n_chunks""",
                (
                    filing.key,
                    filing.ticker,
                    filing.form,
                    filing.filed.isoformat(),
                    filing.period.isoformat() if filing.period else None,
                    filing.accession,
                    filing.url,
                    n_chunks,
                ),
            )
            self.conn.execute("DELETE FROM chunks WHERE filing_key = ?", (filing.key,))

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self.conn:
            self.conn.executemany(
                """INSERT INTO chunks (chunk_id, filing_key, item, ordinal, n_words, text) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (chunk_id) DO UPDATE SET filing_key = excluded.filing_key, item = excluded.item,
                     ordinal = excluded.ordinal, n_words = excluded.n_words, text = excluded.text""",
                [(c.chunk_id, c.filing_key, c.item, c.ordinal, c.n_words, c.text) for c in chunks],
            )

    def get_chunks(self, ids: Sequence[str]) -> list[Chunk]:
        """Chunks with the given ids, in the order given; unknown ids are skipped."""
        found: dict[str, Chunk] = {}
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), 500):
            batch = unique[start : start + 500]
            rows = self.conn.execute(
                "SELECT chunk_id, filing_key, item, ordinal, text, n_words FROM chunks"  # Chunk field order
                f" WHERE chunk_id IN ({','.join('?' * len(batch))})",
                batch,
            )
            found.update((r["chunk_id"], Chunk(*r)) for r in rows)
        return [found[i] for i in ids if i in found]

    def get_filing(self, filing_key: str) -> dict[str, Any] | None:
        """The filing's row (ticker, form, filed, period, accession, url, n_chunks) or None."""
        row = self.conn.execute("SELECT * FROM filings WHERE filing_key = ?", (filing_key,)).fetchone()
        return dict(row) if row else None

    def bm25(
        self, query: str, k: int = 8, ticker: str | None = None, form: str | None = None, filed: str | None = None
    ) -> list[Hit]:
        """Top ``k`` chunks for ``query`` by FTS5's BM25, optionally only from one ticker, form and/or filing date
        (``filed`` as YYYY-MM-DD)."""
        match = fts_query(query)
        if not match or k <= 0:
            return []
        sql = """SELECT c.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts
                 JOIN chunks c ON c.rowid = chunks_fts.rowid
                 JOIN filings f ON f.filing_key = c.filing_key
                 WHERE chunks_fts MATCH ?"""
        params: list[Any] = [match]
        if ticker:
            sql += " AND f.ticker = ?"
            params.append(ticker.upper())
        if form:
            sql += " AND f.form = ?"
            params.append(form.upper())
        if filed:
            sql += " AND f.filed = ?"
            params.append(filed)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(k)
        rows = self.conn.execute(sql, params).fetchall()
        return [Hit(r["chunk_id"], -r["s"], rank) for rank, r in enumerate(rows, start=1)]

    def filings(self) -> list[dict[str, Any]]:
        """Every stored filing (filing_key, ticker, form, filed, period), by ticker, newest first."""
        sql = "SELECT filing_key, ticker, form, filed, period FROM filings ORDER BY ticker, filed DESC"
        return [dict(r) for r in self.conn.execute(sql)]

    def all_chunk_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")]

    def chunk_ids(self, ticker: str | None = None, form: str | None = None, filed: str | None = None) -> list[str]:
        """Ids of the chunks of the filings of ``ticker``, of ``form`` and/or filed on ``filed`` (YYYY-MM-DD); every
        chunk when all are None."""
        sql = "SELECT c.chunk_id FROM chunks c JOIN filings f ON f.filing_key = c.filing_key WHERE 1 = 1"
        params: list[Any] = []
        if ticker:
            sql += " AND f.ticker = ?"
            params.append(ticker.upper())
        if form:
            sql += " AND f.form = ?"
            params.append(form.upper())
        if filed:
            sql += " AND f.filed = ?"
            params.append(filed)
        return [r[0] for r in self.conn.execute(sql + " ORDER BY c.rowid", params)]

    def stats(self) -> dict[str, Any]:
        """Totals and per-ticker / per-form counts. ``words`` sums chunk lengths, so overlapping words count twice."""
        rows = self.conn.execute(
            """SELECT f.ticker, f.form, COUNT(DISTINCT f.filing_key) AS filings, COUNT(c.chunk_id) AS chunks,
                      COALESCE(SUM(c.n_words), 0) AS words
               FROM filings f LEFT JOIN chunks c ON c.filing_key = f.filing_key
               GROUP BY f.ticker, f.form ORDER BY f.ticker, f.form"""
        ).fetchall()
        out: dict[str, Any] = {"filings": 0, "chunks": 0, "words": 0, "by_ticker": {}, "by_form": {}}
        for r in rows:
            for key in ("filings", "chunks", "words"):
                out[key] += r[key]
            t = out["by_ticker"].setdefault(r["ticker"], {"filings": 0, "chunks": 0, "words": 0, "forms": {}})
            t["filings"] += r["filings"]
            t["chunks"] += r["chunks"]
            t["words"] += r["words"]
            t["forms"][r["form"]] = r["filings"]
            f = out["by_form"].setdefault(r["form"], {"filings": 0, "chunks": 0})
            f["filings"] += r["filings"]
            f["chunks"] += r["chunks"]
        return out
