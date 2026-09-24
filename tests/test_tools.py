import functools
import inspect
import sys
import types
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from filings_qa import tools
from filings_qa.retrieve import retrieve
from filings_qa.store import Hit
from filings_qa.tools import (
    INTRADAY_NOTE,
    TOOL_DECLARATIONS,
    ToolError,
    default_tools,
    dispatch,
    finnhub_get,  # the real one, as for yfinance_closes
    get_news,
    get_price,
    parse_finnhub_news,
    parse_news_rss,
    search_filings,
    summarize,
    tool_declarations,
    yfinance_closes,  # the real one: conftest replaces tools.yfinance_closes, not this name
)

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 23, 19, 0, tzinfo=UTC)
GROWTH = "ACME-10-Q-20240802-2-001"  # "Data center revenue growth was strong: ..." (conftest's store)
OTHER = "OTHR-10-K-20240216-7-001"


def rss(url):
    return (FIXTURES / "google_news_sample.xml").read_bytes()


def finnhub(url):
    return (FIXTURES / "finnhub_news_sample.json").read_bytes()


class FakeRetriever:
    """Returns ``hits`` and records how it was called."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.hits


def test_rss_items_are_parsed_with_the_source_taken_off_the_title():
    items = parse_news_rss(rss(""))
    assert len(items) == 6
    assert items[0] == {
        "published": datetime(2026, 9, 22, 16, 58, 28, tzinfo=UTC),
        "source": "Example Wire",
        "title": "Acme shares climb after the widget launch",
        "url": "https://news.google.com/rss/articles/CBMiTEST0001?oc=5",
    }
    assert items[1]["title"] == "Acme & Co. report: what the quarter says about margins"  # entity decoded
    assert items[4]["title"] == "Acme vs. rivals - a comparison"  # only the trailing " - Sample Times" is removed
    assert items[4]["published"] == datetime(2026, 9, 23, 6, 6, 20, tzinfo=UTC)  # +0200 read as UTC
    assert items[5]["published"] is None  # no pubDate


def test_get_news_keeps_recent_headlines_newest_first_without_repeats():
    urls = []

    def fetch(url):
        urls.append(url)
        return rss(url)

    news = get_news("acme", fetch=fetch, now=NOW)
    assert urls == ["https://news.google.com/rss/search?q=ACME+stock&hl=en-US&gl=US&ceid=US:en"]
    assert news == [
        {
            "date": "2026-09-23",
            "source": "Sample Times",
            "title": "Acme vs. rivals - a comparison",
            "url": "https://news.google.com/rss/articles/CBMiTEST0005?oc=5",
        },
        {
            "date": "2026-09-22",
            "source": "Example Wire",  # the newer of the two items with this title
            "title": "Acme shares climb after the widget launch",
            "url": "https://news.google.com/rss/articles/CBMiTEST0001?oc=5",
        },
        {
            "date": "2026-09-21",
            "source": "Sample Times",
            "title": "Acme & Co. report: what the quarter says about margins",
            "url": "https://news.google.com/rss/articles/CBMiTEST0002?oc=5",
        },
    ]
    assert [n["date"] for n in get_news("ACME", days=30.0, fetch=fetch, now=NOW)][-1] == "2026-09-02"
    assert [n["date"] for n in get_news("ACME", days=1, fetch=fetch, now=NOW)] == ["2026-09-23"]  # since 09-22 19:00
    with pytest.raises(ValueError, match="not a stock ticker"):
        get_news("ACME stock&q=other", fetch=fetch, now=NOW)
    get_news("ACME", topic=" capital  expenditure&more ", fetch=fetch, now=NOW)
    assert urls[-1] == "https://news.google.com/rss/search?q=ACME+capital+expenditure%26more&hl=en-US&gl=US&ceid=US:en"
    assert len(urls) == 4


def test_get_news_keeps_the_feeds_top_headlines_of_the_period(monkeypatch):
    """Google ranks the feed; the first MAX_NEWS items of the period are kept, whatever their dates."""
    monkeypatch.setattr(tools, "MAX_NEWS", 2)
    news = get_news("ACME", days=30, fetch=rss, now=NOW)
    assert [(n["date"], n["source"]) for n in news] == [("2026-09-22", "Example Wire"), ("2026-09-21", "Sample Times")]


def test_finnhub_items_are_parsed_in_reply_order():
    items = parse_finnhub_news(finnhub(""))
    assert len(items) == 6
    assert items[0] == {
        "published": datetime(2026, 9, 23, 14, 30, tzinfo=UTC),
        "source": "Example Wire",
        "title": "Acme lifts its capital expenditure plan",  # runs of spaces collapsed
        "url": "https://wire.example.com/acme-capex",
        "summary": "The company now expects capital expenditure of $2 billion next year.",
    }
    assert items[5]["published"] is None  # a zero timestamp


def test_get_news_reads_finnhub_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    urls = []

    def fetch(url):
        urls.append(url)
        return finnhub(url)

    news = get_news("acme", fetch=fetch, now=NOW)
    assert urls == ["https://finnhub.io/api/v1/company-news?symbol=ACME&from=2026-09-16&to=2026-09-23"]  # no key
    assert news == [
        {
            "date": "2026-09-23",
            "source": "Example Wire",
            "title": "Acme lifts its capital expenditure plan",
            "url": "https://wire.example.com/acme-capex",
        },
        {
            "date": "2026-09-22",
            "source": "Example Wire",  # the newer of the two items with this title
            "title": "Acme shares climb after the widget launch",
            "url": "https://wire.example.com/acme-widget",
        },
        {
            "date": "2026-09-21",
            "source": "Sample Times",
            "title": "Acme & Co. report: what the quarter says about margins",
            "url": "https://times.example.org/acme-margins",
        },
    ]  # the item of 2026-09-02 is older than 7 days; the undated one is left out
    assert [n["date"] for n in get_news("ACME", days=30, fetch=fetch, now=NOW)][-1] == "2026-09-02"
    assert urls[-1] == "https://finnhub.io/api/v1/company-news?symbol=ACME&from=2026-08-24&to=2026-09-23"
    monkeypatch.setenv("FINNHUB_API_KEY", "  ")  # a blank key counts as none: Google News
    google = []
    assert len(get_news("ACME", fetch=lambda url: google.append(url) or rss(url), now=NOW)) == 3
    assert google == ["https://news.google.com/rss/search?q=ACME+stock&hl=en-US&gl=US&ceid=US:en"]


def test_a_topic_filters_and_ranks_finnhub_headlines(monkeypatch):
    """Finnhub has no search: the headlines whose title or summary mention more words of the topic come first, then
    the newest; MAX_NEWS of them are kept and returned newest first."""
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    def dates(topic):
        return [(n["date"], n["source"]) for n in get_news("ACME", days=30, topic=topic, fetch=finnhub, now=NOW)]

    # capital and expenditure: 2026-09-23 (title) and 2026-09-02 (summary); capital only: 2026-09-21
    assert dates("the Capital Expenditures") == [
        ("2026-09-23", "Example Wire"), ("2026-09-21", "Sample Times"), ("2026-09-02", "Example Wire"),
    ]
    assert dates("margin") == [("2026-09-21", "Sample Times")]  # "margin" finds "margins"
    assert dates("officers") == [("2026-09-02", "Example Wire")]  # and "officers" finds "officer"
    assert dates("tariffs") == []
    assert len(dates("the")) == 4  # only a stopword: no topic
    monkeypatch.setattr(tools, "MAX_NEWS", 2)  # the two that mention both words, though one is the oldest
    assert dates("capital expenditures") == [("2026-09-23", "Example Wire"), ("2026-09-02", "Example Wire")]
    # without a topic, the newest two, whatever the order of the reply
    assert dates(None) == [("2026-09-23", "Example Wire"), ("2026-09-22", "Example Wire")]


def test_finnhub_key_goes_in_a_header_not_the_url(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", " test-key ")
    calls = []

    def fake_finnhub_get(url, token):
        calls.append((url, token))
        return b"[]"

    monkeypatch.setattr(tools, "finnhub_get", fake_finnhub_get)
    assert get_news("ACME", now=NOW) == []
    assert calls == [("https://finnhub.io/api/v1/company-news?symbol=ACME&from=2026-09-16&to=2026-09-23", "test-key")]

    sent = {}

    class Response:
        content = b"[]"

        def raise_for_status(self):
            pass

    def fake_get(url, headers, timeout):
        sent.update(url=url, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    assert finnhub_get(calls[0][0], "test-key") == b"[]"
    assert sent["headers"]["X-Finnhub-Token"] == "test-key" and "test-key" not in sent["url"]


def test_get_price_rounds_the_closes_and_checks_the_dates():
    calls = []

    def closes(ticker, start, end):
        calls.append((ticker, start, end))
        return [(date(2026, 8, 26), 209.425659), (date(2026, 8, 27), 227.725174)]

    rows = get_price("nvda", "2026-08-26", "2026-08-27", closes=closes)
    assert rows == [
        {"date": "2026-08-26", "trading_day": 0, "close": 209.43},
        {"date": "2026-08-27", "trading_day": 1, "close": 227.73},
    ]
    assert calls == [("NVDA", date(2026, 8, 26), date(2026, 8, 27))]
    with pytest.raises(ValueError, match="before start"):
        get_price("NVDA", "2026-08-27", "2026-08-26", closes=closes)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        get_price("NVDA", "last week", "2026-08-26", closes=closes)
    with pytest.raises(ValueError, match="at most 400 days at a time, not 1460"):
        get_price("NVDA", "2022-08-26", "2026-08-25", closes=closes)  # rows the model would have to read
    assert len(calls) == 1


def test_get_price_for_a_number_of_trading_days_after_a_date():
    asked = []
    days = [date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31), date(2026, 9, 1),
            date(2026, 9, 2), date(2026, 9, 3)]

    def closes(ticker, start, end):
        asked.append((start, end))
        return [(day, 200.0 + i) for i, day in enumerate(days)]

    rows = get_price("NVDA", "2026-08-26", trading_days=5.0, closes=closes)
    assert asked == [(date(2026, 8, 26), date(2026, 9, 15))]  # 2 x 5 + 10 calendar days, enough for 5 trading days
    assert [(r["date"], r["trading_day"], r["close"]) for r in rows] == [
        ("2026-08-26", 0, 200.0), ("2026-08-27", 1, 201.0), ("2026-08-28", 2, 202.0),
        ("2026-08-31", 3, 203.0), ("2026-09-01", 4, 204.0), ("2026-09-02", 5, 205.0),
    ]
    assert get_price("NVDA", "2026-08-26", end="2026-08-27", trading_days=3, closes=closes)[-1]["date"] == "2026-08-31"
    with pytest.raises(ValueError, match="give end"):
        get_price("NVDA", "2026-08-26", closes=closes)
    with pytest.raises(ValueError, match="1 to 60"):
        get_price("NVDA", "2026-08-26", trading_days=0, closes=closes)


def test_todays_price_before_the_close_is_marked_intraday():
    def closes(ticker, start, end):
        return [(date(2026, 9, 22), 370.0), (date(2026, 9, 23), 379.98)]

    during = get_price("TSLA", "2026-09-22", "2026-09-23", closes=closes, now=NOW)  # 15:00 in New York
    assert during[0] == {"date": "2026-09-22", "trading_day": 0, "close": 370.0}
    assert during[1] == {"date": "2026-09-23", "trading_day": 1, "close": 379.98, "note": INTRADAY_NOTE}
    assert summarize("get_price", during).endswith("(+2.7%); the last is intraday, not a close")
    after = get_price("TSLA", "2026-09-22", "2026-09-23", closes=closes, now=datetime(2026, 9, 23, 20, 0, tzinfo=UTC))
    assert "note" not in after[1]  # 16:00 in New York: the close


def test_yfinance_closes_asks_for_the_day_after_the_end(monkeypatch):
    """yfinance leaves out its end date, so the last day asked for is included by asking for the day after."""
    pd = pytest.importorskip("pandas")
    asked = {}

    class Ticker:
        def __init__(self, symbol):
            asked["symbol"] = symbol

        def history(self, **kwargs):
            asked.update(kwargs)
            index = pd.DatetimeIndex(["2026-08-26", "2026-08-27", "2026-08-28"]).tz_localize("America/New_York")
            return pd.DataFrame({"Open": [1.0, 2.0, 3.0], "Close": [209.4, float("nan"), 217.3]}, index=index)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    assert yfinance_closes("NVDA", date(2026, 8, 26), date(2026, 8, 28)) == [
        (date(2026, 8, 26), 209.4),
        (date(2026, 8, 28), 217.3),
    ]
    assert (asked["symbol"], asked["start"], asked["end"]) == ("NVDA", "2026-08-26", "2026-08-29")
    assert asked["interval"] == "1d" and asked["auto_adjust"] is True


def test_search_filings_returns_each_passage_with_its_filing(store):
    gone = "ACME-10-Q-20240802-9-001"  # in the index but no longer stored: skipped
    retriever = FakeRetriever([Hit(GROWTH, 0.9, 1), Hit(gone, 0.5, 2), Hit(OTHER, 0.3, 3)])
    rows = search_filings("data center", ticker=" acme", form="10-q", k=3.0, store=store, retriever=retriever)
    kwargs = {"strategy": "hybrid", "k": 3, "ticker": "ACME", "form": "10-Q", "filed": None}
    assert retriever.calls == [("data center", kwargs)]
    assert [r["chunk_id"] for r in rows] == [GROWTH, OTHER]  # best first
    assert rows[0] == {
        "chunk_id": GROWTH,
        "ticker": "ACME",
        "form": "10-Q",
        "filed": "2024-08-02",
        "period": "2024-06-29",
        "item": "2",
        "text": "Data center revenue growth was strong: data center revenue growth.",
    }
    assert rows[1]["ticker"] == "OTHR" and rows[1]["filed"] == "2024-02-16"
    search_filings("revenue", k=50, filed="2024-08-02", store=store, retriever=retriever, strategy="bm25")
    assert retriever.calls[-1][1] == {"strategy": "bm25", "k": 12, "ticker": None, "form": None, "filed": "2024-08-02"}
    with pytest.raises(ValueError, match="filed must be a date"):
        search_filings("revenue", filed="last quarter", store=store, retriever=retriever)


def test_search_filings_can_read_one_filing_with_the_real_retriever(store):
    retriever = functools.partial(retrieve, store=store)
    rows = search_filings("revenue growth", filed="2024-02-16", store=store, retriever=retriever, strategy="bm25")
    assert [r["chunk_id"] for r in rows] == [OTHER]
    assert search_filings("revenue growth", filed="2024-02-17", store=store, retriever=retriever, strategy="bm25") == []


def test_dispatch_runs_a_declared_tool_and_rejects_unknown_tools_and_arguments():
    calls = []

    def fake_price(**kwargs):
        calls.append(kwargs)
        return [{"date": "2026-08-26", "close": 1.0}]

    impls = {"get_price": fake_price}
    args = {"ticker": "NVDA", "start": "2026-08-26", "end": "2026-08-27"}
    assert dispatch("get_price", {**args, "ignored": None}, impls) == {"output": [{"date": "2026-08-26", "close": 1.0}]}
    assert calls == [args]  # an argument set to None counts as left out
    with pytest.raises(ToolError, match="unknown tool 'get_weather'; the tools are get_price"):
        dispatch("get_weather", {}, impls)
    with pytest.raises(ToolError, match="unknown tool 'get_news'"):
        dispatch("get_news", {"ticker": "NVDA"}, impls)  # declared, but not among the tools given
    with pytest.raises(ToolError, match="missing start"):
        dispatch("get_price", {"ticker": "NVDA"}, impls)
    with pytest.raises(ToolError, match=r"unknown argument\(s\) interval .*takes ticker, start, end, trading_days"):
        dispatch("get_price", {**args, "interval": "1h"}, impls)
    assert len(calls) == 1


def test_declarations_match_the_implementations_and_the_sdk(store):
    impls = default_tools(store, FakeRetriever([]))
    assert [d["name"] for d in TOOL_DECLARATIONS] == list(impls) == ["search_filings", "get_price", "get_news"]
    for declaration in TOOL_DECLARATIONS:
        params = declaration["parameters"]
        assert set(params["required"]) <= set(params["properties"])
        assert set(params["properties"]) <= set(inspect.signature(impls[declaration["name"]]).parameters)

    filings = [
        {"ticker": "NVDA", "form": "10-K", "filed": "2026-02-25"},
        {"ticker": "AAPL", "form": "10-Q", "filed": "2026-07-31"},
        {"ticker": "NVDA", "form": "10-Q", "filed": "2026-08-26"},
    ]
    described = tool_declarations(filings)
    assert described[0]["description"].endswith(
        "first three listed for it, whatever their form):"
        " AAPL: 10-Q 2026-07-31; NVDA: 10-Q 2026-08-26, 10-K 2026-02-25."
    )
    assert "Stored filings" not in TOOL_DECLARATIONS[0]["description"]  # a copy
    assert described[1:] == TOOL_DECLARATIONS[1:]

    genai_types = pytest.importorskip("google.genai.types")
    tool = genai_types.Tool.model_validate({"function_declarations": described})
    assert [f.name for f in tool.function_declarations] == ["search_filings", "get_price", "get_news"]
    assert tool.function_declarations[0].parameters.properties["form"].enum == ["10-K", "10-Q"]


def test_summaries_are_one_line_of_at_most_300_characters():
    passages = [{"chunk_id": f"NVDA-10-Q-20260826-2-{i:03d}"} for i in range(1, 13)]
    assert summarize("search_filings", passages[:1]) == "1 passage: NVDA-10-Q-20260826-2-001"
    long = summarize("search_filings", passages)
    assert long.startswith("12 passages: NVDA-10-Q-20260826-2-001, ") and long.endswith("...") and len(long) <= 300
    assert summarize("search_filings", []) == "no passages found"
    prices = [{"date": "2026-08-26", "close": 200.0}, {"date": "2026-08-27", "close": 190.0}]
    assert summarize("get_price", prices) == "2 closes: 2026-08-26 200.00 to 2026-08-27 190.00 (-5.0%)"
    assert summarize("get_price", []) == "no closing prices in this range"
    news = [{"date": "2026-09-22", "source": "Wire", "title": "Line one\nline two " * 30}]
    text = summarize("get_news", news)
    assert text.startswith("1 headline, 2026-09-22 to 2026-09-22: Line one line two") and len(text) <= 300
    assert summarize("other", {"a": 1}) == '{"a": 1}'
