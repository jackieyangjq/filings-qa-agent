import json
from datetime import date

import pytest
import requests

from filings_qa import edgar
from filings_qa.edgar import EdgarError, Response

UNDECLARED_TOOL_PAGE = (
    b"<!DOCTYPE html><html><head><title>SEC.gov | Your Request Originates from an Undeclared Automated Tool</title>"
    b"</head><body>To allow for equitable access to all users, SEC reserves the right to limit requests ...</body>"
)


def _serve(pages):
    """Fake ``edgar.fetch`` answering from a {url: Response} dict and recording the URLs asked for."""
    calls = []

    def fake(url):
        calls.append(url)
        return pages[url]

    fake.calls = calls
    return fake


def test_list_filings_takes_latest_10k_then_three_latest_10qs(monkeypatch, submissions):
    fake = _serve({"https://data.sec.gov/submissions/CIK0001234567.json": Response(200, submissions)})
    monkeypatch.setattr(edgar, "fetch", fake)
    filings = edgar.list_filings(1234567)
    assert [(f.form, f.filed) for f in filings] == [
        ("10-K", date(2024, 2, 16)),
        ("10-Q", date(2024, 10, 30)),
        ("10-Q", date(2024, 8, 2)),
        ("10-Q", date(2024, 5, 3)),
    ]  # no 8-K, no 10-Q/A amendment, not the older 10-K or the fourth 10-Q
    k = filings[0]
    assert k.ticker == "ACME" and k.cik == 1234567 and k.period == date(2023, 12, 30)
    assert k.accession == "0001234567-24-000003" and k.primary_doc == "acme-20231230.htm"
    assert k.url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000003/acme-20231230.htm"
    assert k.key == "ACME-10-K-20240216"


def test_list_filings_single_form_and_ticker_label(monkeypatch, submissions):
    monkeypatch.setattr(edgar, "fetch", lambda url: Response(200, submissions))
    filings = edgar.list_filings(1234567, forms=("10-Q",), limit=2, ticker="acme")
    assert [(f.form, f.filed) for f in filings] == [("10-Q", date(2024, 10, 30)), ("10-Q", date(2024, 8, 2))]
    assert {f.ticker for f in filings} == {"ACME"}


def test_list_filings_merges_a_predecessor_registrant(monkeypatch, submissions):
    old = json.loads(submissions)
    old["tickers"] = []  # the old registrant no longer lists the ticker
    new = json.loads(submissions)
    new["cik"] = "7654321"
    recent = new["filings"]["recent"]
    for column in recent:
        recent[column] = recent[column][:2]  # the new registrant has only the latest 8-K and 10-Q
    fake = _serve({
        "https://data.sec.gov/submissions/CIK0007654321.json": Response(200, json.dumps(new).encode()),
        "https://data.sec.gov/submissions/CIK0001234567.json": Response(200, json.dumps(old).encode()),
    })
    monkeypatch.setattr(edgar, "fetch", fake)
    filings = edgar.list_filings([7654321, 1234567])
    assert [(f.form, f.filed, f.cik) for f in filings] == [
        ("10-K", date(2024, 2, 16), 1234567),
        ("10-Q", date(2024, 10, 30), 7654321),  # listed under both registrants, taken once, from the first
        ("10-Q", date(2024, 8, 2), 1234567),
        ("10-Q", date(2024, 5, 3), 1234567),
    ]
    assert {f.ticker for f in filings} == {"ACME"}
    assert filings[1].url.startswith("https://www.sec.gov/Archives/edgar/data/7654321/")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (Response(403, UNDECLARED_TOOL_PAGE), ["HTTP 403", "Undeclared Automated Tool", "SEC_USER_AGENT"]),
        (Response(200, b"<html>maintenance</html>"), ["not JSON", "HTTP 200", "<html>maintenance"]),
        (Response(503, b""), ["HTTP 503"]),
    ],
)
def test_bad_responses_raise_with_status_and_body_start(monkeypatch, response, expected):
    monkeypatch.setattr(edgar, "fetch", lambda url: response)
    with pytest.raises(EdgarError) as info:
        edgar.list_filings(320193)
    assert info.value.status == response.status
    for text in expected:
        assert text in str(info.value)
    assert len(str(info.value)) < 700  # only the start of the body is quoted


def test_download_saves_once_and_skips_existing_file(monkeypatch, tmp_path, sample_filing):
    fake = _serve({sample_filing.url: Response(200, b"<html>filing</html>")})
    monkeypatch.setattr(edgar, "fetch", fake)
    path = edgar.download(sample_filing, tmp_path / "raw")
    assert path == tmp_path / "raw" / "ACME" / "0001234567-24-000012.htm"
    assert path.read_bytes() == b"<html>filing</html>"
    assert edgar.download(sample_filing, tmp_path / "raw") == path
    assert fake.calls == [sample_filing.url]  # the second call found the file and made no request


def test_download_failure_leaves_no_file(monkeypatch, tmp_path, sample_filing):
    monkeypatch.setattr(edgar, "fetch", lambda url: Response(404, b"Not Found"))
    with pytest.raises(EdgarError, match="HTTP 404"):
        edgar.download(sample_filing, tmp_path)
    assert not any(tmp_path.rglob("*.htm"))


def test_cik_for_uses_and_fills_the_cache(monkeypatch, tmp_path):
    table = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
             "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."}}
    fake = _serve({edgar.TICKERS_URL: Response(200, json.dumps(table).encode())})
    monkeypatch.setattr(edgar, "fetch", fake)
    cache = tmp_path / "cache" / "company_tickers.json"
    assert edgar.cik_for("aapl", cache_path=cache) == 320193
    assert edgar.cik_for("BRK.B", cache_path=cache) == 1067983
    assert fake.calls == [edgar.TICKERS_URL]  # second lookup read the cache
    with pytest.raises(EdgarError, match="not found"):
        edgar.cik_for("NOPE", cache_path=cache)


def test_fetch_sends_sec_headers_waits_between_requests_and_retries(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test Suite contact-address")
    sleeps: list[float] = []
    monkeypatch.setattr(edgar.time, "sleep", sleeps.append)
    monkeypatch.setattr(edgar.time, "monotonic", lambda: 1000.0)  # a frozen clock: every request looks immediate
    monkeypatch.setattr(edgar, "_last_request", 0.0)
    script = [requests.ConnectionError("reset"), Response(503, b""), Response(200, b"ok"), Response(403, b"no")]
    seen_headers = []

    def fake_get(url, headers):
        seen_headers.append(headers)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(edgar, "_http_get", fake_get)
    assert edgar.fetch("https://example.test/a") == Response(200, b"ok")  # after an error and a 503
    assert edgar.fetch("https://example.test/b").status == 403  # returned at once, not retried
    assert script == []
    assert seen_headers[0] == {"User-Agent": "Test Suite contact-address", "Accept-Encoding": "gzip, deflate"}
    # every request after the first waits the minimum interval
    assert sum(s == pytest.approx(edgar.MIN_INTERVAL_S) for s in sleeps) == 3
    assert [s for s in sleeps if s >= 1] == [1, 2]  # backoff after the two failures


def test_user_agent_defaults_without_env(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert edgar.user_agent() == edgar.DEFAULT_USER_AGENT
