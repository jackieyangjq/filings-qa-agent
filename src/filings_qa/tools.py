"""Tools the agent can call: passages from the filings, daily closing prices and recent news headlines.

Each tool is a plain function returning JSON-ready data, and whatever it reaches outside is injectable, so tests run
without network: ``search_filings`` takes the store and a retriever, ``get_price`` a ``closes`` function (yfinance by
default) and ``get_news`` a ``fetch`` function (by default an HTTP GET of Finnhub's company news when the environment
variable FINNHUB_API_KEY is set, else of the Google News RSS search, which needs no key).
``TOOL_DECLARATIONS`` describes the tools to Gemini as function declarations (JSON schema), ``dispatch`` checks a
call against them and runs it, and ``default_tools`` binds the real implementations to a store.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import requests

from .store import STOPWORDS, Hit, Store

NEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}"
FINNHUB_KEY_ENV = "FINNHUB_API_KEY"
USER_AGENT = "filings-qa (https://github.com/jackieyangjq/filings-qa-agent)"
HTTP_TIMEOUT_S = 20
DEFAULT_SEARCH_K = 6
MAX_SEARCH_K = 12
MAX_NEWS = 10  # headlines returned: the first in ranking order (see get_news), then newest first
MAX_NEWS_DAYS = 30
MAX_TRADING_DAYS = 60
MAX_PRICE_DAYS = 400  # calendar days per get_price call: about 275 rows, a year and a month
SUMMARY_CHARS = 300
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)  # New York time
INTRADAY_NOTE = "latest price during today's trading session, not a close"
_TICKER = re.compile(r"\^?[A-Z0-9][A-Z0-9.=-]{0,14}")  # NVDA, BRK-B, BRK.B, ^GSPC
_WORD = re.compile(r"[^\W_]+")

Retriever = Callable[..., list[Hit]]  # called as retriever(query, strategy=, k=, ticker=, form=, filed=)
Closes = Callable[[str, date, date], Sequence[tuple[date, float]]]  # (ticker, first day, last day) -> (day, close)
Fetch = Callable[[str], bytes]  # url -> response body


class ToolError(ValueError):
    """A call that cannot be run: an unknown tool, or arguments its declaration does not allow."""


TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "search_filings",
        "description": (
            "Search SEC 10-K and 10-Q filings for passages about a topic. Returns the most relevant passages first, "
            "each with its chunk_id (cite it), the company's ticker, the form, the date the filing was filed, the "
            "period it covers, its section (item) and its text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Required: what to look for, in words a filing would use, e.g. 'data center "
                    "demand'.",
                },
                "ticker": {"type": "string", "description": "Only passages of this company, by ticker, e.g. NVDA."},
                "form": {
                    "type": "string",
                    "enum": ["10-K", "10-Q"],
                    "description": "Only passages of this form: 10-K (annual report) or 10-Q (quarterly report).",
                },
                "k": {
                    "type": "integer",
                    "description": f"How many passages to return, 1 to {MAX_SEARCH_K} (default {DEFAULT_SEARCH_K}).",
                },
                "filed": {
                    "type": "string",
                    "description": "Only passages of the filing filed on this date, as YYYY-MM-DD: use it with ticker "
                    "to read one filing among several.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_price",
        "description": (
            "Daily closing prices of a stock, oldest first, adjusted for splits and dividends: from start to end, "
            f"both included (at most {MAX_PRICE_DAYS} days apart), or from start for a number of trading_days after "
            "it. Only trading days have a row, and trading_day counts them from the first row (0). While the market "
            "is open, today's row holds the latest price, not a close, and has a note saying so."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock's ticker, e.g. NVDA."},
                "start": {"type": "string", "description": "First date, as YYYY-MM-DD."},
                "end": {"type": "string", "description": "Last date, as YYYY-MM-DD. Give end or trading_days."},
                "trading_days": {
                    "type": "integer",
                    "description": f"Instead of end: the close on start and on each of this many trading days after "
                    f"it (1 to {MAX_TRADING_DAYS}). E.g. start 2026-08-26 and trading_days 5 give the close on "
                    "2026-08-26 and the closes of the five trading days after it: the move over those five days is "
                    "the last close against the first.",
                },
            },
            "required": ["ticker", "start"],
        },
    },
    {
        "name": "get_news",
        "description": (
            f"Recent news headlines about a stock: up to {MAX_NEWS} from the period, newest first, with the date, "
            "source, title and link of each, from Finnhub's company news or a Google News search. Headlines are what "
            "the outlets wrote, not checked facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock's ticker, e.g. NVDA."},
                "days": {
                    "type": "integer",
                    "description": f"How many days back to look, 1 to {MAX_NEWS_DAYS} (default 7).",
                },
                "topic": {
                    "type": "string",
                    "description": "Words the headlines should be about, e.g. 'capital expenditure'. Without a topic "
                    "any recent headline about the stock can come back.",
                },
            },
            "required": ["ticker"],
        },
    },
]
_DECLARED = {d["name"]: d for d in TOOL_DECLARATIONS}


def tool_declarations(filings: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """``TOOL_DECLARATIONS``, with the stored filings (rows with ticker, form and filed, as ``Store.filings`` gives
    them) listed in the description of ``search_filings``, so the model knows which companies it can search and which
    filings are the latest."""
    declarations = copy.deepcopy(TOOL_DECLARATIONS)
    if filings:
        by_ticker: dict[str, list[str]] = {}
        for f in sorted(filings, key=lambda f: str(f["filed"]), reverse=True):
            by_ticker.setdefault(str(f["ticker"]), []).append(f"{f['form']} {f['filed']}")
        listing = "; ".join(f"{ticker}: {', '.join(items)}" for ticker, items in sorted(by_ticker.items()))
        declarations[0]["description"] += (
            " Stored filings, newest first, by form and date filed (a company's last three filings are the first"
            f" three listed for it, whatever their form): {listing}."
        )
    return declarations


def _ticker(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not _TICKER.fullmatch(symbol):
        raise ValueError(f"{value!r} is not a stock ticker such as NVDA")
    return symbol


def _day(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        raise ValueError(f"{name} must be a date written as YYYY-MM-DD, not {value!r}") from None


def search_filings(
    query: str,
    ticker: str | None = None,
    form: str | None = None,
    k: int = DEFAULT_SEARCH_K,
    filed: str | None = None,
    *,
    store: Store,
    retriever: Retriever,
    strategy: str = "hybrid",
) -> list[dict[str, Any]]:
    """The ``k`` (1 to ``MAX_SEARCH_K``) passages that ``retriever`` ranks best for ``query`` with ``strategy``, best
    first, optionally only of one company (``ticker``), one form and/or the filings filed on one date (``filed``,
    YYYY-MM-DD). Each is a dict: chunk_id, ticker, form, filed (the filing date), period (the period the filing
    covers), item (the section, e.g. "7") and text."""
    k = max(1, min(int(k), MAX_SEARCH_K))
    ticker = str(ticker or "").strip().upper() or None
    form = str(form or "").strip().upper() or None
    filed = _day(filed, "filed").isoformat() if filed else None
    hits = retriever(str(query), strategy=strategy, k=k, ticker=ticker, form=form, filed=filed)
    chunks = store.get_chunks([h.chunk_id for h in hits])  # a chunk deleted after indexing is skipped
    filings = {key: store.get_filing(key) or {} for key in dict.fromkeys(c.filing_key for c in chunks)}
    return [
        {
            "chunk_id": c.chunk_id,
            "ticker": filings[c.filing_key].get("ticker"),
            "form": filings[c.filing_key].get("form"),
            "filed": filings[c.filing_key].get("filed"),
            "period": filings[c.filing_key].get("period"),
            "item": c.item,
            "text": c.text,
        }
        for c in chunks
    ]


def yfinance_closes(ticker: str, start: date, end: date) -> list[tuple[date, float]]:
    """Daily closes of ``ticker`` from ``start`` to ``end``, both included, adjusted for splits and dividends, from
    Yahoo Finance through yfinance. An unknown ticker gives no rows."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is not installed: pip install 'filings-qa-agent[tools]'") from e
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # it logs "possibly delisted" for unknown tickers
    frame = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance leaves out the end date
        interval="1d",
        auto_adjust=True,
        actions=False,
    )
    if frame.empty or "Close" not in frame:
        return []
    return [(stamp.date(), float(close)) for stamp, close in frame["Close"].dropna().items()]


def get_price(
    ticker: str,
    start: str,
    end: str | None = None,
    trading_days: int | None = None,
    *,
    closes: Closes | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Daily closes of ``ticker``, oldest first, as [{"date": "2026-08-26", "trading_day": 0, "close": 209.43}, ...],
    rounded to cents; ``trading_day`` counts the rows from 0. They run from ``start`` to ``end`` (YYYY-MM-DD, both
    included) or, when ``trading_days`` is given (``end`` is then ignored), from the first trading day on or after
    ``start`` for that many more trading days. Before 16:00 New York time (``now``, default the current time), a row
    for today is the latest trade, not a close: it gets {"note": INTRADAY_NOTE}. ``closes`` defaults to
    ``yfinance_closes``."""
    symbol = _ticker(ticker)
    first = _day(start, "start")
    if trading_days is not None:
        n_days = int(trading_days)
        if not 1 <= n_days <= MAX_TRADING_DAYS:
            raise ValueError(f"trading_days must be 1 to {MAX_TRADING_DAYS}, not {trading_days!r}")
        last = first + timedelta(days=2 * n_days + 10)  # calendar days enough for n trading days and holidays
    elif end is not None:
        last = _day(end, "end")
        if last < first:
            raise ValueError(f"end ({last}) is before start ({first})")
        if (last - first).days > MAX_PRICE_DAYS:
            raise ValueError(f"ask for at most {MAX_PRICE_DAYS} days at a time, not {(last - first).days}")
    else:
        raise ValueError("give end (a date) or trading_days (how many trading days after start)")
    found = list((closes or yfinance_closes)(symbol, first, last))
    if trading_days is not None:
        found = found[: n_days + 1]
    rows: list[dict[str, Any]] = [
        {"date": day.isoformat(), "trading_day": n, "close": round(float(close), 2)}
        for n, (day, close) in enumerate(found)
    ]
    market_now = (now or datetime.now(UTC)).astimezone(MARKET_TZ)
    if rows and rows[-1]["date"] == market_now.date().isoformat() and market_now.time() < MARKET_CLOSE:
        rows[-1]["note"] = INTRADAY_NOTE
    return rows


def http_get(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


def finnhub_get(url: str, token: str) -> bytes:
    """GET a Finnhub API ``url`` with the key ``token`` in the X-Finnhub-Token header, which Finnhub accepts in place
    of a ``token=`` query parameter: kept out of the url, the key cannot end up in an error message, a trace or a
    reply to the model."""
    headers = {"User-Agent": USER_AGENT, "X-Finnhub-Token": token}
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


def news_url(ticker: str, topic: str | None = None) -> str:
    """The Google News RSS search for "<TICKER> <topic>", or "<TICKER> stock" without a topic."""
    words = " ".join(str(topic or "").split())[:100] or "stock"
    return NEWS_URL.format(query=quote_plus(f"{ticker} {words}"))


def finnhub_news_url(ticker: str, start: date, end: date) -> str:
    """Finnhub's company news of ``ticker`` from ``start`` to ``end``, without the key (see ``finnhub_get``)."""
    return FINNHUB_NEWS_URL.format(symbol=quote_plus(ticker), start=start.isoformat(), end=end.isoformat())


def parse_news_rss(xml: bytes | str) -> list[dict[str, Any]]:
    """Every item of a Google News RSS feed, in feed order, as {"published": datetime in UTC or None, "source",
    "title", "url"}. The " - Source" that Google appends to each title is removed."""
    items = []
    for item in ET.fromstring(xml).iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        source = " ".join((item.findtext("source") or "").split())
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(UTC)
        except (TypeError, ValueError):
            published = None
        items.append({"published": published, "source": source, "title": title, "url": item.findtext("link") or ""})
    return items


def parse_finnhub_news(body: bytes | str) -> list[dict[str, Any]]:
    """Every item of a Finnhub company-news reply (a JSON list), in its order, as {"published": datetime in UTC or
    None, "source", "title", "url", "summary"}. A missing or zero timestamp gives None."""
    items = []
    for row in json.loads(body):
        stamp = row.get("datetime")
        try:
            published = datetime.fromtimestamp(int(stamp), UTC) if stamp else None
        except (TypeError, ValueError, OverflowError, OSError):
            published = None
        items.append(
            {
                "published": published,
                "source": " ".join(str(row.get("source") or "").split()),
                "title": " ".join(str(row.get("headline") or "").split()),
                "url": str(row.get("url") or ""),
                "summary": " ".join(str(row.get("summary") or "").split()),
            }
        )
    return items


def _stem(word: str) -> str:
    """``word`` without a plural ending, so that it matches the start of both the singular and the plural."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def rank_by_topic(items: Sequence[dict[str, Any]], topic: str | None) -> list[dict[str, Any]]:
    """Finnhub has no search, so a topic filters its items: those whose title or summary mentions a word of ``topic``
    (stopwords aside; a word without its plural ending matches the start of a word, so "tariffs" finds "tariff" and
    "tariffs"), the ones mentioning more of its words first, then the newest. Without a topic, every item, newest
    first."""
    words = [w.lower() for w in _WORD.findall(str(topic or ""))]
    terms = list(dict.fromkeys(_stem(w) for w in words if w not in STOPWORDS))

    def newest(item: dict[str, Any]) -> float:
        return item["published"].timestamp() if item["published"] else float("-inf")

    if not terms:
        return sorted(items, key=newest, reverse=True)
    patterns = [re.compile(rf"\b{re.escape(t)}", re.IGNORECASE) for t in terms]
    scored = []
    for item in items:
        text = f"{item['title']} {item.get('summary', '')}"
        found = sum(1 for p in patterns if p.search(text))
        if found:
            scored.append((found, newest(item), item))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [item for _, _, item in scored]


def get_news(
    ticker: str,
    days: int = 7,
    topic: str | None = None,
    *,
    fetch: Fetch | None = None,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Headlines about ``ticker`` published in the last ``days`` days (1 to ``MAX_NEWS_DAYS``), as
    [{"date": "2026-09-22", "source", "title", "url"}]: the first ``MAX_NEWS`` in ranking order, without repeated
    titles, sorted newest first. ``now`` defaults to the current time.

    With the environment variable FINNHUB_API_KEY set, they come from Finnhub's company news, ranked by
    ``rank_by_topic``, and ``fetch`` defaults to ``finnhub_get`` with that key. Without it, they come from the Google
    News RSS search for "<TICKER> stock" (or "<TICKER> <topic>"), ranked as Google ranks the feed, and ``fetch``
    defaults to ``http_get``. Google offers these feeds for personal, non-commercial use; Finnhub has a free key."""
    symbol = _ticker(ticker)
    days = max(1, min(int(days), MAX_NEWS_DAYS))
    end = now or datetime.now(UTC)
    since = end - timedelta(days=days)
    key = os.environ.get(FINNHUB_KEY_ENV, "").strip()
    if key:
        get = fetch or functools.partial(finnhub_get, token=key)
        ranked = rank_by_topic(parse_finnhub_news(get(finnhub_news_url(symbol, since.date(), end.date()))), topic)
    else:
        ranked = parse_news_rss((fetch or http_get)(news_url(symbol, topic)))
    seen: set[str] = set()
    picked = []
    for item in ranked:
        if not item["published"] or item["published"] < since or item["title"].casefold() in seen:
            continue
        seen.add(item["title"].casefold())
        picked.append(item)
        if len(picked) == MAX_NEWS:
            break
    picked.sort(key=lambda i: i["published"], reverse=True)
    return [
        {"date": i["published"].date().isoformat(), "source": i["source"], "title": i["title"], "url": i["url"]}
        for i in picked
    ]


def default_tools(store: Store, retriever: Retriever, *, strategy: str = "hybrid") -> dict[str, Callable[..., Any]]:
    """The three tools by name: ``search_filings`` bound to ``store`` and ``retriever`` (searching with ``strategy``),
    ``get_price`` from yfinance and ``get_news`` from Finnhub (with FINNHUB_API_KEY) or Google News."""
    return {
        "search_filings": functools.partial(search_filings, store=store, retriever=retriever, strategy=strategy),
        "get_price": get_price,
        "get_news": get_news,
    }


def dispatch(name: str, args: Mapping[str, Any] | None, impls: Mapping[str, Callable[..., Any]]) -> dict[str, Any]:
    """Run the tool ``name`` of ``impls`` with ``args`` and return {"output": its result}, the shape Gemini expects of
    a function response. Arguments set to None count as left out. Raises ToolError for a tool not in ``impls`` and for
    arguments the tool's declaration does not list or requires; errors raised by the tool itself pass through."""
    if name not in impls:
        raise ToolError(f"unknown tool {name!r}; the tools are {', '.join(impls)}")
    args = {key: value for key, value in (args or {}).items() if value is not None}
    declared = _DECLARED.get(name)
    if declared:
        params = declared["parameters"]
        unknown = sorted(set(args) - set(params["properties"]))
        missing = [p for p in params.get("required", []) if p not in args]
        if unknown or missing:
            problems = [
                f"unknown argument(s) {', '.join(unknown)}" if unknown else "",
                f"missing {', '.join(missing)}" if missing else "",
            ]
            allowed = ", ".join(params["properties"])
            raise ToolError(f"{name}: {'; '.join(p for p in problems if p)} (it takes {allowed})")
    return {"output": impls[name](**args)}


def clip(text: str, limit: int = SUMMARY_CHARS) -> str:
    """``text`` on one line, cut to at most ``limit`` characters (ending in "...")."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def summarize(name: str, output: Any, limit: int = SUMMARY_CHARS) -> str:
    """One line, at most ``limit`` characters, saying what a tool returned (for traces and the command line)."""
    if isinstance(output, list) and name == "search_filings":
        ids = ", ".join(str(r.get("chunk_id")) for r in output)
        text = f"{_count(len(output), 'passage')}: {ids}" if output else "no passages found"
    elif isinstance(output, list) and name == "get_price":
        text = "no closing prices in this range"
        if output:
            first, last = output[0], output[-1]
            change = (last["close"] / first["close"] - 1) * 100 if first["close"] else 0.0
            text = (
                f"{_count(len(output), 'close')}: {first['date']} {first['close']:.2f} to {last['date']}"
                f" {last['close']:.2f} ({change:+.1f}%)"
                + ("; the last is intraday, not a close" if "note" in last else "")
            )
    elif isinstance(output, list) and name == "get_news":
        text = "no headlines in this period"
        if output:
            dates = sorted(r["date"] for r in output)
            titles = "; ".join(f"{r['title']} ({r['source']})" for r in output)
            text = f"{_count(len(output), 'headline')}, {dates[0]} to {dates[-1]}: {titles}"
    else:
        text = json.dumps(output, ensure_ascii=False, default=str)
    return clip(text, limit)
