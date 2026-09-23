from datetime import date
from pathlib import Path

import pytest

from filings_qa import edgar

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never reach SEC: a test that needs HTTP replaces ``edgar.fetch`` or ``edgar._http_get`` itself."""

    def refuse(url, headers):
        raise AssertionError(f"unexpected network request in a test: {url}")

    monkeypatch.setattr(edgar, "_http_get", refuse)


@pytest.fixture
def sample_html() -> bytes:
    return (FIXTURES / "filing_sample.htm").read_bytes()


@pytest.fixture
def submissions() -> bytes:
    return (FIXTURES / "submissions_sample.json").read_bytes()


@pytest.fixture
def sample_filing() -> edgar.Filing:
    return edgar.Filing(
        ticker="ACME",
        cik=1234567,
        form="10-Q",
        filed=date(2024, 8, 2),
        period=date(2024, 6, 29),
        accession="0001234567-24-000012",
        primary_doc="acme-20240629.htm",
        url="https://www.sec.gov/Archives/edgar/data/1234567/000123456724000012/acme-20240629.htm",
    )
