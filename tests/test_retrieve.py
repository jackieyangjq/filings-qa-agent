import numpy as np
import pytest

from filings_qa.embed import DenseIndex, FakeEmbedder
from filings_qa.retrieve import retrieve, rrf
from filings_qa.store import Hit

ACME = "ACME-10-Q-20240802-"  # chunk ids of the `store` fixture (conftest)
OTHR_7 = "OTHR-10-K-20240216-7-001"
QUERY = "data center revenue growth"  # BM25 ranks ACME 2-001, OTHR 7-001, ACME 2-002 and nothing else (test_store)

# Hand-set dense vectors for the seven chunks. Every query embeds to [1, 0], so the cosine similarity is the first
# coordinate and the dense ranking is the order below.
DENSE = {
    ACME + "1A-001": [1.0, 0.0],
    OTHR_7: [0.8, 0.6],
    ACME + "2-001": [0.6, 0.8],
    ACME + "4-001": [0.0, 1.0],
    ACME + "2-002": [-0.6, 0.8],
    ACME + "3-001": [-0.8, 0.6],
    ACME + "5-001": [-1.0, 0.0],
}


class QueryEmbedder:
    def embed(self, texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


@pytest.fixture
def indexes(store):
    dense = DenseIndex(list(DENSE), np.array(list(DENSE.values())))
    return {"store": store, "dense": dense, "embedder": QueryEmbedder()}


def _hits(*ids):
    return [Hit(chunk_id, 0.0, rank) for rank, chunk_id in enumerate(ids, start=1)]


def _ids(hits):
    return [h.chunk_id for h in hits]


def test_rrf_worked_example_with_ties():
    # score = sum over the lists of 1 / (k_const + rank), with k_const = 60:
    #   a: rank 1 in the first list, rank 2 in the second: 1/61 + 1/62 = (62 + 61) / (61 * 62) = 123/3782 ≈ 0.032522
    #   b: rank 2 in the first list, rank 1 in the second: 1/62 + 1/61 = 123/3782 ≈ 0.032522, a tie with a
    #   c: rank 3 in the first list, absent from the second: 1/63 ≈ 0.015873
    #   d: absent from the first list, rank 3 in the second: 1/63 ≈ 0.015873, a tie with c
    # Ties keep the order of first appearance, reading the first list before the second: a before b, c before d.
    fused = rrf([_hits("a", "b", "c"), _hits("b", "a", "d")])
    assert _ids(fused) == ["a", "b", "c", "d"]
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert [h.score for h in fused] == pytest.approx([123 / 3782, 123 / 3782, 1 / 63, 1 / 63])
    # the same tie with names in the other alphabetical order: first appearance still decides
    assert _ids(rrf([_hits("z", "y"), _hits("y", "z")])) == ["z", "y"]


def test_rrf_rewards_agreement_between_the_lists():
    # [a, b, c] and [c, a, d], k_const = 60:
    #   a: 1/61 + 1/62 ≈ 0.032522;  c: 1/63 + 1/61 ≈ 0.032266;  b: 1/62 ≈ 0.016129;  d: 1/63 ≈ 0.015873
    # c, last in the first list, overtakes b, second in the first list, because the second list ranks c first.
    lists = [_hits("a", "b", "c"), _hits("c", "a", "d")]
    assert _ids(rrf(lists)) == ["a", "c", "b", "d"]
    # with k_const = 10, a scores 1/11 + 1/12 ≈ 0.174242
    assert rrf(lists, k_const=10)[0].score == pytest.approx(1 / 11 + 1 / 12)
    assert rrf([]) == [] and rrf([[], []]) == []


def test_bm25_and_dense_strategies(indexes):
    assert _ids(retrieve(QUERY, strategy="bm25", k=2, **indexes)) == [ACME + "2-001", OTHR_7]
    hits = retrieve(QUERY, strategy="dense", k=2, **indexes)
    assert _ids(hits) == [ACME + "1A-001", OTHR_7]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.8])


def test_hybrid_fuses_two_k_candidates_from_each_list(indexes):
    # k = 1, so two candidates from each list. BM25 [ACME 2-001, OTHR 7-001], dense [ACME 1A-001, OTHR 7-001]:
    #   OTHR 7-001: 1/62 + 1/62 ≈ 0.032258;  ACME 2-001: 1/61 ≈ 0.016393;  ACME 1A-001: 1/61 ≈ 0.016393
    # With only one candidate from each list the answer would be ACME 2-001 (the tie goes to the first list).
    hits = retrieve(QUERY, strategy="hybrid", k=1, **indexes)
    assert [(h.chunk_id, h.rank) for h in hits] == [(OTHR_7, 1)]
    assert hits[0].score == pytest.approx(2 / 62)

    # k = 3, so six from each list; BM25 has only three matches. BM25 [ACME 2-001, OTHR 7-001, ACME 2-002],
    # dense [ACME 1A-001, OTHR 7-001, ACME 2-001, ACME 4-001, ACME 2-002, ACME 3-001]:
    #   ACME 2-001: 1/61 + 1/63 ≈ 0.032266;  OTHR 7-001: 1/62 + 1/62 ≈ 0.032258;  ACME 2-002: 1/63 + 1/65 ≈ 0.031258
    #   ACME 1A-001: 1/61 ≈ 0.016393 (dense's best, unknown to BM25);  ACME 4-001: 1/64;  ACME 3-001: 1/66
    hits = retrieve(QUERY, strategy="hybrid", k=3, **indexes)
    assert _ids(hits) == [ACME + "2-001", OTHR_7, ACME + "2-002"]
    assert [h.score for h in hits] == pytest.approx([1 / 61 + 1 / 63, 2 / 62, 1 / 63 + 1 / 65])
    assert [h.rank for h in hits] == [1, 2, 3]


def test_ticker_and_form_filters_apply_to_every_strategy(indexes):
    for strategy in ("bm25", "dense", "hybrid"):
        assert _ids(retrieve(QUERY, strategy=strategy, k=5, ticker="othr", **indexes)) == [OTHR_7]
        assert _ids(retrieve(QUERY, strategy=strategy, k=5, form="10-K", **indexes)) == [OTHR_7]
        acme = _ids(retrieve(QUERY, strategy=strategy, k=5, ticker="ACME", **indexes))
        assert acme and all(i.startswith(ACME) for i in acme)


def test_the_filing_date_filter_applies_to_every_strategy(indexes):
    for strategy in ("bm25", "dense", "hybrid"):
        assert _ids(retrieve(QUERY, strategy=strategy, k=5, filed="2024-02-16", **indexes)) == [OTHR_7]
        assert retrieve(QUERY, strategy=strategy, k=5, ticker="ACME", filed="2024-02-16", **indexes) == []


def test_hybrid_with_the_fake_embedder_returns_k_distinct_chunks(store, tmp_path):
    embedder = FakeEmbedder()
    dense = DenseIndex.build(store, embedder, tmp_path / "index")
    hits = retrieve(QUERY, store=store, dense=dense, embedder=embedder, strategy="hybrid", k=4)
    ids = _ids(hits)
    assert len(ids) == 4 == len(set(ids))
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
    assert set(ids[:2]) == {ACME + "2-001", OTHR_7}  # the two chunks that contain every query word


def test_unknown_strategy_and_missing_dense_index(store):
    with pytest.raises(ValueError, match="strategy"):
        retrieve(QUERY, store=store, strategy="sparse")
    with pytest.raises(ValueError, match="dense index"):
        retrieve(QUERY, store=store, strategy="hybrid")
    assert _ids(retrieve(QUERY, store=store, strategy="bm25", k=1)) == [ACME + "2-001"]  # bm25 needs no vectors
    assert retrieve(QUERY, store=store, strategy="bm25", k=0) == []
