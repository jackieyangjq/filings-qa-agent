from filings_qa.chunk import Chunk
from filings_qa.store import fts_query


def _chunk(filing, item, ordinal, text):
    return Chunk(f"{filing.key}-{item}-{ordinal:03d}", filing.key, item, ordinal, text, len(text.split()))


def test_bm25_ranks_best_match_first_with_scores_descending(store):
    hits = store.bm25("data center revenue growth", k=10)
    ids = [h.chunk_id for h in hits]
    # every term twice, every term once, then one term once
    assert ids == ["ACME-10-Q-20240802-2-001", "OTHR-10-K-20240216-7-001", "ACME-10-Q-20240802-2-002"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
    assert hits[0].score > 0


def test_bm25_filters_by_ticker_and_form(store):
    assert [h.chunk_id for h in store.bm25("revenue growth", k=10, ticker="othr")] == ["OTHR-10-K-20240216-7-001"]
    assert {h.chunk_id for h in store.bm25("revenue", k=10, form="10-Q")} == {
        "ACME-10-Q-20240802-2-001",
        "ACME-10-Q-20240802-2-002",
    }
    assert len(store.bm25("revenue", k=1)) == 1


def test_bm25_survives_punctuation_and_query_syntax(store):
    for query in ['revenue AND "growth', "R&D: what's (NEAR) up?", "data-center* revenue^2", "", "?!"]:
        store.bm25(query, k=5)  # must not raise
    assert store.bm25("?!", k=5) == []
    assert fts_query("What is the data-center revenue?") == '"data" OR "center" OR "revenue"'


def test_get_chunks_keeps_requested_order(store):
    ids = ["ACME-10-Q-20240802-1A-001", "OTHR-10-K-20240216-7-001", "missing", "ACME-10-Q-20240802-2-001"]
    got = store.get_chunks(ids)
    assert [c.chunk_id for c in got] == [ids[0], ids[1], ids[3]]
    assert got[0].item == "1A" and got[0].ordinal == 1 and got[0].n_words == 9


def test_all_ids_stats_and_filing_lookup(store):
    assert len(store.all_chunk_ids()) == 7
    stats = store.stats()
    assert stats["filings"] == 2 and stats["chunks"] == 7
    assert stats["by_ticker"]["ACME"]["forms"] == {"10-Q": 1}
    assert stats["by_form"]["10-K"] == {"filings": 1, "chunks": 1}
    filing = store.get_filing("ACME-10-Q-20240802")
    assert filing["accession"] == "0001234567-24-000012" and filing["period"] == "2024-06-29"
    assert store.get_filing("nope") is None


def test_re_adding_a_filing_replaces_its_chunks_in_the_index(store, sample_filing):
    store.add_filing(sample_filing, 1)
    store.add_chunks([_chunk(sample_filing, "2", 1, "Inventory write-downs hit gross margin.")])
    assert store.bm25("gaming", k=5) == []  # old chunk text is gone from the full-text index
    assert [h.chunk_id for h in store.bm25("inventory", k=5)] == ["ACME-10-Q-20240802-2-001"]
    store.conn.execute("INSERT INTO chunks_fts (chunks_fts, rank) VALUES ('integrity-check', 1)")


def test_filters_by_filing_date_and_the_list_of_filings(store):
    assert [h.chunk_id for h in store.bm25("revenue growth", k=10, filed="2024-02-16")] == ["OTHR-10-K-20240216-7-001"]
    assert store.bm25("revenue growth", k=10, ticker="ACME", filed="2024-02-16") == []
    assert store.chunk_ids(filed="2024-02-16") == ["OTHR-10-K-20240216-7-001"]
    assert len(store.chunk_ids(ticker="ACME", filed="2024-08-02")) == 6
    assert store.filings() == [
        {"filing_key": "ACME-10-Q-20240802", "ticker": "ACME", "form": "10-Q", "filed": "2024-08-02",
         "period": "2024-06-29"},
        {"filing_key": "OTHR-10-K-20240216", "ticker": "OTHR", "form": "10-K", "filed": "2024-02-16",
         "period": "2024-06-29"},
    ]


def test_chunk_ids_by_ticker_and_form(store):
    acme = [f"ACME-10-Q-20240802-{item}" for item in ("2-001", "2-002", "1A-001", "3-001", "4-001", "5-001")]
    assert store.chunk_ids(ticker="acme") == acme  # in insertion order
    assert store.chunk_ids(form="10-k") == ["OTHR-10-K-20240216-7-001"]
    assert store.chunk_ids(ticker="ACME", form="10-K") == []
    assert len(store.chunk_ids()) == 7
