"""SEC EDGAR access: ticker to CIK, the latest 10-K/10-Q filings of a company, and download of their main document.

EDGAR's fair-access policy (https://www.sec.gov/os/accessing-edgar-data) asks automated tools to declare who they
are in the User-Agent header, with a contact email, and to stay under 10 requests per second. The User-Agent comes
from the ``SEC_USER_AGENT`` environment variable; without it SEC usually answers HTTP 403.

All network traffic goes through :func:`fetch`, which tests replace with a fake.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import requests

USER_AGENT_ENV = "SEC_USER_AGENT"
DEFAULT_USER_AGENT = "filings-qa (https://github.com/jackieyangjq/filings-qa-agent)"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{doc}"
DEFAULT_TICKERS_CACHE = Path("data/cache/company_tickers.json")

MIN_INTERVAL_S = 0.15  # between requests; SEC allows at most 10 per second
RETRIES = 3  # extra attempts after a network error, HTTP 429 or HTTP 5xx
TIMEOUT_S = 60

_last_request = 0.0


class EdgarError(RuntimeError):
    """A failed or unusable EDGAR response. ``status`` is the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


class Response(NamedTuple):
    status: int
    body: bytes


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    form: str
    filed: date
    period: date | None
    accession: str
    primary_doc: str
    url: str

    @property
    def key(self) -> str:
        """Stable id of the filing, e.g. ``AAPL-10-K-20241101`` (ticker, form, filing date)."""
        return f"{self.ticker}-{self.form}-{self.filed:%Y%m%d}"


def user_agent() -> str:
    return os.environ.get(USER_AGENT_ENV, "").strip() or DEFAULT_USER_AGENT


def _http_get(url: str, headers: dict[str, str]) -> Response:
    """One GET request, no retries."""
    r = requests.get(url, headers=headers, timeout=TIMEOUT_S)
    return Response(r.status_code, r.content)


def fetch(url: str) -> Response:
    """GET ``url`` with the SEC headers, at least ``MIN_INTERVAL_S`` after the previous request.

    Network errors, HTTP 429 and HTTP 5xx are retried ``RETRIES`` times with backoff (1 s, 2 s, 4 s). Any other
    status, 403 included, is returned to the caller as is.
    """
    global _last_request
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"}
    for attempt in range(RETRIES + 1):
        wait = _last_request + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = _http_get(url, headers)
        except requests.RequestException as e:
            if attempt == RETRIES:
                raise EdgarError(f"GET {url} failed after {RETRIES + 1} attempts: {e}", url=url) from e
        else:
            if (resp.status != 429 and resp.status < 500) or attempt == RETRIES:
                return resp
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _snippet(body: bytes) -> str:
    return " ".join(body[:200].decode("utf-8", errors="replace").split())


def _http_error(url: str, resp: Response, problem: str) -> EdgarError:
    message = f"{problem}: HTTP {resp.status} from {url}; response starts with: {_snippet(resp.body)!r}"
    if resp.status == 403:
        message += (
            f". SEC rejects automated requests whose User-Agent has no contact details; set {USER_AGENT_ENV} to your"
            " name and email address separated by a space (see .env.example)"
        )
    return EdgarError(message, status=resp.status, url=url)


def _get_json(url: str) -> Any:
    resp = fetch(url)
    if resp.status != 200:
        raise _http_error(url, resp, "SEC request failed")
    try:
        return json.loads(resp.body)
    except ValueError:
        raise _http_error(url, resp, "SEC response is not JSON") from None


def cik_for(ticker: str, *, cache_path: Path | str | None = DEFAULT_TICKERS_CACHE) -> int:
    """CIK of ``ticker`` from SEC's company_tickers.json, cached at ``cache_path`` (delete the file to refresh;
    ``None`` skips the cache)."""
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        table = json.loads(cache.read_text(encoding="utf-8"))
    else:
        table = _get_json(TICKERS_URL)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(table), encoding="utf-8")
    wanted = ticker.strip().upper().replace(".", "-")
    rows = table.values() if isinstance(table, dict) else table
    for row in rows:
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    raise EdgarError(f"ticker {ticker!r} not found in SEC company_tickers.json")


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def list_filings(
    cik: int | Sequence[int], forms: Sequence[str] = ("10-K", "10-Q"), limit: int = 4, *, ticker: str = ""
) -> list[Filing]:
    """Latest filings of ``forms`` from ``filings.recent`` of the company's EDGAR submissions (which covers at least
    the last year).

    The first form gets one slot and the other forms share the remaining ``limit - 1``, each group newest first:
    with the defaults, the latest 10-K followed by the three latest 10-Qs. With a single form, its latest ``limit``
    filings. ``cik`` may list several CIKs, current registrant first, for a company whose earlier filings sit under
    a predecessor (e.g. after a holding-company reorganization); a filing listed under both is taken once.
    ``ticker`` labels the filings; by default the first ticker SEC lists for the first CIK.
    """
    ciks = [cik] if isinstance(cik, int) else list(cik)
    label = ticker.upper()
    found: dict[str, Filing] = {}
    for c in ciks:
        data = _get_json(SUBMISSIONS_URL.format(cik=c))
        try:
            recent = data["filings"]["recent"]
            columns = [recent[k] for k in ("form", "accessionNumber", "filingDate", "reportDate", "primaryDocument")]
        except (KeyError, TypeError):
            raise EdgarError(f"unexpected submissions JSON for CIK {c}: no filings.recent") from None
        label = label or next(iter(data.get("tickers") or []), "").upper()
        if not label:
            raise EdgarError(f"no ticker given and SEC lists none for CIK {c}")
        for form, accession, filed, period, doc in zip(*columns, strict=True):
            if form in forms and doc and accession not in found:
                url = ARCHIVES_URL.format(cik=c, folder=accession.replace("-", ""), doc=doc)
                found[accession] = Filing(label, c, form, date.fromisoformat(filed), _date(period), accession, doc, url)
    rows = sorted(found.values(), key=lambda f: f.filed, reverse=True)
    if len(forms) == 1:
        return rows[:limit]
    first = [f for f in rows if f.form == forms[0]][: min(1, limit)]
    rest = [f for f in rows if f.form != forms[0]][: max(limit - 1, 0)]
    return first + rest


def download(filing: Filing, dest: Path | str) -> Path:
    """Save the filing's main document as ``dest/<TICKER>/<accession>.htm`` (``dest`` is the raw-files folder, e.g.
    ``data/raw``) and return the path. An existing non-empty file is kept and no request is made."""
    path = Path(dest) / filing.ticker / f"{filing.accession}.htm"
    if path.exists() and path.stat().st_size > 0:
        return path
    resp = fetch(filing.url)
    if resp.status != 200:
        raise _http_error(filing.url, resp, f"download of {filing.key} failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(resp.body)
    partial.replace(path)
    return path
