import json
import re

import pytest

from filings_qa import cli, edgar
from filings_qa.chunk import Chunk
from filings_qa.edgar import Response
from filings_qa.llm import FakeLLM
from filings_qa.store import Store, db_path

TICKERS = {"0": {"cik_str": 1234567, "ticker": "ACME", "title": "Acme Widgets, Inc."},
           "1": {"cik_str": 7654321, "ticker": "OTHR", "title": "Other Corp."}}


@pytest.fixture
def companies(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text("companies:\n  - {ticker: acme, name: Acme}\n  - OTHR\n")
    return path


@pytest.fixture
def fake_sec(monkeypatch, submissions, sample_html):
    """Serves the fixtures for every URL of the Acme filings; any other company's submissions answer 404."""
    calls = []

    def fake(url):
        calls.append(url)
        if url == edgar.TICKERS_URL:
            return Response(200, json.dumps(TICKERS).encode())
        if url == "https://data.sec.gov/submissions/CIK0001234567.json":
            return Response(200, submissions)
        if url.startswith("https://www.sec.gov/Archives/edgar/data/1234567/"):
            return Response(200, sample_html)
        return Response(404, b"Not Found")

    monkeypatch.setattr(edgar, "fetch", fake)
    monkeypatch.setenv("SEC_USER_AGENT", "Test Suite contact-address")
    return calls


def test_ingest_only_one_company_then_stats(tmp_path, companies, fake_sec, capsys):
    data = tmp_path / "data"
    assert cli.main(["ingest", "--companies", str(companies), "--data", str(data), "--only", "acme"]) == 0
    out = capsys.readouterr().out
    assert "ACME: 4 filings, 24 chunks" in out  # six sections of the sample, each one chunk, in four filings
    assert "[-, 1, 2, II-1, 1A, 7]" in out
    assert "total: 4 filings, 24 chunks" in out
    assert not any("7654321" in url for url in fake_sec)  # --only skipped the other company
    assert (data / "raw" / "ACME" / "0001234567-24-000003.htm").exists()
    assert (data / "text" / "ACME-10-K-20240216.txt").read_text().startswith("UNITED STATES")

    with Store(db_path(data)) as store:
        assert store.stats()["chunks"] == 24
        assert store.bm25("specialty steel suppliers", k=1)[0].chunk_id.endswith("-1A-001")

    assert cli.main(["stats", "--data", str(data)]) == 0
    assert "total: 4 filings, 24 chunks" in capsys.readouterr().out

    # a second run downloads nothing new and leaves the same index
    fake_sec.clear()
    assert cli.main(["ingest", "--companies", str(companies), "--data", str(data), "--only", "ACME"]) == 0
    assert not any("/Archives/" in url for url in fake_sec)
    with Store(db_path(data)) as store:
        assert store.stats()["chunks"] == 24


def test_ingest_reports_failed_company_and_returns_1(tmp_path, companies, fake_sec, capsys):
    assert cli.main(["ingest", "--companies", str(companies), "--data", str(tmp_path / "data")]) == 1
    captured = capsys.readouterr()
    assert "OTHR: failed: SEC request failed: HTTP 404" in captured.err
    assert "total: 4 filings, 24 chunks; failed: OTHR" in captured.out


def test_ingest_stops_at_403(tmp_path, companies, monkeypatch, capsys):
    calls = []

    def reject(url):
        calls.append(url)
        return Response(403, b"<title>Your Request Originates from an Undeclared Automated Tool</title>")

    monkeypatch.setattr(edgar, "fetch", reject)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert cli.main(["ingest", "--companies", str(companies), "--data", str(tmp_path / "data")]) == 1
    err = capsys.readouterr().err
    assert "SEC_USER_AGENT is not set" in err
    assert "HTTP 403" in err and "stopping" in err
    assert len(calls) == 1  # no request for the second company


def test_ciks_from_the_company_list_skip_the_ticker_lookup(tmp_path, fake_sec, capsys):
    companies = tmp_path / "companies.yaml"
    companies.write_text("limit: 2\ncompanies:\n  - {ticker: ACME, ciks: [1234567]}\n")
    assert cli.main(["ingest", "--companies", str(companies), "--data", str(tmp_path / "data")]) == 0
    assert edgar.TICKERS_URL not in fake_sec
    out = capsys.readouterr().out
    assert "ACME (CIK 1234567): 2 filings" in out and "ACME: 2 filings, 12 chunks" in out


def test_only_unknown_ticker_and_stats_without_index(tmp_path, companies, capsys):
    assert cli.main(["ingest", "--companies", str(companies), "--only", "NFLX"]) == 2
    assert cli.main(["stats", "--data", str(tmp_path / "empty")]) == 1
    assert not (tmp_path / "empty").exists()


_RESULT = re.compile(r"^\s*(\d+)\.\s+(-?[\d.]+)\s+(\S+)\s+item (\S+)$")


def _results(out):
    """(rank, chunk id, item, preview) of each result printed by `search`; the preview is the line after."""
    lines = out.splitlines()
    found = []
    for line, preview in zip(lines, lines[1:] + [""], strict=True):
        m = _RESULT.match(line)
        if m:
            found.append((int(m[1]), m[3], m[4], preview.strip()))
    return found


def test_index_then_search_with_the_fake_embedder(tmp_path, store, capsys):
    data = str(tmp_path)  # the `store` fixture is tmp_path/index/filings.sqlite
    assert cli.main(["index", "--fake", "--data", data]) == 0
    assert "7 chunks" in capsys.readouterr().out
    assert cli.main(["search", "data center revenue growth", "--k", "2", "--data", data]) == 0  # hybrid by default
    results = _results(capsys.readouterr().out)
    assert [r[0] for r in results] == [1, 2]
    assert results[0][1:3] == ("ACME-10-Q-20240802-2-001", "2")
    assert results[0][3] == "Data center revenue growth was strong: data center revenue growth."


def test_search_prints_the_first_200_characters_and_warns_about_a_stale_index(tmp_path, store, capsys):
    data = str(tmp_path)
    assert cli.main(["index", "--fake", "--data", data]) == 0
    text = "Tariffs\n" + " ".join(f"w{i:03d}" for i in range(80))  # 407 characters
    store.add_chunks([Chunk("OTHR-10-K-20240216-1A-001", "OTHR-10-K-20240216", "1A", 1, text, 81)])
    capsys.readouterr()

    assert cli.main(["search", "tariffs", "--strategy", "bm25", "--ticker", "othr", "--data", data]) == 0
    captured = capsys.readouterr()
    assert _results(captured.out) == [(1, "OTHR-10-K-20240216-1A-001", "1A", "Tariffs " + text[8:200] + "...")]
    assert captured.err == ""  # bm25 does not use the vectors

    assert cli.main(["search", "tariffs", "--data", data]) == 0
    assert "filings-qa index" in capsys.readouterr().err  # the new chunk has no vector yet


def test_search_needs_a_usable_dense_index_except_with_bm25(tmp_path, store, capsys):
    data = str(tmp_path)
    assert cli.main(["search", "revenue", "--strategy", "dense", "--data", data]) == 1
    assert "filings-qa index" in capsys.readouterr().err
    assert cli.main(["search", "revenue", "--strategy", "bm25", "--k", "1", "--data", data]) == 0
    assert len(_results(capsys.readouterr().out)) == 1

    # vectors made by an embedder that search cannot recreate: an error message, not a traceback
    assert cli.main(["index", "--fake", "--data", data]) == 0
    (tmp_path / "index" / "embedder.json").write_text('{"kind": "custom"}')
    assert cli.main(["search", "revenue", "--data", data]) == 1
    assert "filings-qa index" in capsys.readouterr().err


def test_index_without_fastembed_prints_the_install_command(tmp_path, store, capsys):
    assert cli.main(["index", "--data", str(tmp_path)]) == 1  # conftest makes `import fastembed` fail
    assert "pip install" in capsys.readouterr().err


def test_index_and_search_need_ingested_filings(tmp_path, capsys):
    assert cli.main(["index", "--fake", "--data", str(tmp_path / "empty")]) == 1
    assert cli.main(["search", "revenue", "--data", str(tmp_path / "empty")]) == 1
    assert "filings-qa ingest" in capsys.readouterr().err
    assert not (tmp_path / "empty").exists()


def test_search_embeds_the_query_with_the_model_that_built_the_index(tmp_path, store, fake_fastembed, capsys):
    data = str(tmp_path)
    assert cli.main(["index", "--model", "test/model-a", "--data", data]) == 0
    assert cli.main(["search", "data center revenue growth", "--strategy", "dense", "--k", "1", "--data", data]) == 0
    assert [m.model_name for m in fake_fastembed] == ["test/model-a", "test/model-a"]  # index, then search
    assert [m.cache_dir for m in fake_fastembed] == [str(tmp_path / "models")] * 2  # <data>/models
    assert len(_results(capsys.readouterr().out)) == 1


GROWTH = "ACME-10-Q-20240802-2-001"
USAGE_LINE = re.compile(r"^model=(\S+) tokens in/out=(\d+)/(\d+) latency=(\d+\.\d)s dropped=(\d+) uncited=(\d+)$")


def _reply(*sentences, abstained=False):
    return {"abstained": abstained, "sentences": [{"text": t, "citations": c} for t, c in sentences]}


@pytest.fixture
def fake_llm(monkeypatch):
    """Makes `ask` use a FakeLLM; call the fixture with the scripted replies. Returns the FakeLLM."""

    def install(*script, **kwargs):
        llm = FakeLLM(list(script), **kwargs)
        monkeypatch.setattr(cli, "_make_llm", lambda: llm)
        return llm

    return install


def test_ask_prints_each_sentence_with_its_citations_then_usage(tmp_path, store, fake_llm, capsys):
    data = str(tmp_path)
    assert cli.main(["index", "--fake", "--data", data]) == 0
    capsys.readouterr()
    llm = fake_llm(
        _reply(
            ("Data center revenue growth was strong.", [GROWTH, "OTHR-10-K-20240216-7-001"]),
            ("Gaming revenue doubled.", ["ACME-10-Q-20240802-8-008"]),  # not retrieved: dropped
            ("No growth rate is given.", []),
        ),
        tokens=(5321, 412),
        latency=1.25,
    )
    assert cli.main(["ask", "data center revenue growth", "--data", data]) == 0  # hybrid, k 8
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        f"Data center revenue growth was strong. [{GROWTH}, OTHR-10-K-20240216-7-001]",
        "No growth rate is given. [no citation]",
    ]
    m = USAGE_LINE.match(lines[-1])
    assert m and m.groups() == ("gemini-3.5-flash", "5321", "412", m[4], "1", "1") and 1.2 < float(m[4]) < 2
    assert len(lines) == 3
    assert llm.calls[0]["models"] == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


def test_ask_abstains_filters_retrieval_and_prints_json(tmp_path, store, fake_llm, capsys):
    data = str(tmp_path)
    llm = fake_llm(_reply(("The excerpts do not cover this.", []), abstained=True), _reply(("Growth.", [GROWTH])))
    args = ["ask", "revenue growth", "--strategy", "bm25", "--ticker", "othr", "--k", "2", "--data", data]
    assert cli.main(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Abstained: the retrieved excerpts do not answer this question."
    assert out[1] == "  The excerpts do not cover this."
    assert USAGE_LINE.match(out[2])[6] == "1"
    assert "OTHR-10-K-20240216-7-001" in llm.prompts[0] and "ACME-" not in llm.prompts[0]  # --ticker OTHR

    assert cli.main(["ask", "revenue growth", "--strategy", "bm25", "--json", "--models", "a, b", "--data", data]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sentences"] == [{"text": "Growth.", "citations": [GROWTH]}] and result["abstained"] is False
    assert result["model"] == "a" and llm.calls[1]["models"] == ["a", "b"]
    assert [h["rank"] for h in result["hits"]] == list(range(1, len(result["hits"]) + 1))


def test_ask_flags_advice_wording(tmp_path, store, fake_llm, capsys):
    fake_llm(_reply(("Growth was strong, so you should buy.", [GROWTH])))
    assert cli.main(["ask", "revenue growth", "--strategy", "bm25", "--data", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Note: the wording "should buy" reads like investment advice' in out


def test_ask_without_a_key_or_with_a_failing_model(tmp_path, store, fake_llm, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.main(["ask", "revenue", "--strategy", "bm25", "--data", str(tmp_path)]) == 1
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err

    error = Exception("quota")
    error.code = 429
    fake_llm(error)
    assert cli.main(["ask", "revenue", "--strategy", "bm25", "--data", str(tmp_path)]) == 1
    assert "no answer: quota or rate limit exceeded (HTTP 429)" in capsys.readouterr().err
    assert cli.main(["ask", "revenue", "--strategy", "dense", "--data", str(tmp_path)]) == 1  # no vectors yet
    assert "filings-qa index" in capsys.readouterr().err
