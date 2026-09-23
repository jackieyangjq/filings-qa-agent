import sys
import types
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from filings_qa import edgar
from filings_qa.chunk import Chunk
from filings_qa.embed import FakeEmbedder
from filings_qa.llm import Gemini, SetupError
from filings_qa.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never reach SEC: a test that needs HTTP replaces ``edgar.fetch`` or ``edgar._http_get`` itself."""

    def refuse(url, headers):
        raise AssertionError(f"unexpected network request in a test: {url}")

    monkeypatch.setattr(edgar, "_http_get", refuse)


@pytest.fixture(autouse=True)
def no_llm_requests(monkeypatch):
    """Tests never reach Gemini: they use FakeLLM, or replace ``_call`` on their own Gemini. The error is a SetupError
    so that it fails at once instead of being retried after 30 and 60 seconds."""

    def refuse(self, model, **kwargs):
        raise SetupError(f"unexpected Gemini request in a test: {model}")

    monkeypatch.setattr(Gemini, "_call", refuse)


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch):
    """Tests never load a real embedding model (it would be downloaded): ``import fastembed`` fails unless a test
    asks for the ``fake_fastembed`` fixture."""
    monkeypatch.setitem(sys.modules, "fastembed", None)


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


def _chunk(filing, item, ordinal, text):
    return Chunk(f"{filing.key}-{item}-{ordinal:03d}", filing.key, item, ordinal, text, len(text.split()))


@pytest.fixture
def store(tmp_path, sample_filing):
    """Seven chunks: six from an ACME 10-Q and one from an OTHR 10-K, in ``tmp_path/index/filings.sqlite`` (so
    ``tmp_path`` works as the ``--data`` folder)."""
    s = Store(tmp_path / "index" / "filings.sqlite")
    other = replace(sample_filing, ticker="OTHR", form="10-K", filed=date(2024, 2, 16), accession="0000000002-24-1")
    s.add_filing(sample_filing, 6)
    s.add_chunks(
        [
            _chunk(sample_filing, "2", 1, "Data center revenue growth was strong: data center revenue growth."),
            _chunk(sample_filing, "2", 2, "Gaming revenue declined because of weaker consumer demand."),
            _chunk(sample_filing, "1A", 1, "Supply chain disruptions could delay shipments of our products."),
            # unrelated chunks, so that the query terms are rare enough for BM25 to weigh them
            _chunk(sample_filing, "3", 1, "Legal proceedings arise in the ordinary course of business."),
            _chunk(sample_filing, "4", 1, "The company repurchased shares under its buyback program."),
            _chunk(sample_filing, "5", 1, "Employees work in plants located in several countries."),
        ]
    )
    s.add_filing(other, 1)
    s.add_chunks([_chunk(other, "7", 1, "Revenue growth at the data center business accelerated all year.")])
    yield s
    s.close()


@pytest.fixture
def fake_fastembed(monkeypatch, no_model_download):
    """Stands in for the fastembed package, so no model is downloaded: ``TextEmbedding(model_name, ...)`` with the
    ``embedding_size`` property and ``embed(documents, batch_size=256, parallel=None)`` generator of the real class.
    Its vectors are bag-of-words vectors times 3, as float64, i.e. not unit length. Returns the models created, which
    remember their ``model_name`` and ``cache_dir``. ``FASTEMBED_CACHE_PATH`` is unset for the test."""
    created = []
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)

    class TextEmbedding:
        def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.batch_sizes = []
            created.append(self)

        @property
        def embedding_size(self):
            return 8

        def embed(self, documents, batch_size=256, parallel=None, **kwargs):
            self.batch_sizes.append(batch_size)
            for vector in FakeEmbedder(8).embed(list(documents)):
                yield vector.astype(np.float64) * 3

    module = types.ModuleType("fastembed")
    module.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return created
