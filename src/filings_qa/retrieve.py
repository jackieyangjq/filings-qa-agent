"""Three ways to find the chunks for a question: BM25 keyword search (SQLite FTS5), dense vector search, and a
hybrid that fuses the two rankings with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

from .embed import DenseIndex, Embedder
from .store import Hit, Store

Strategy = Literal["bm25", "dense", "hybrid"]
STRATEGIES: tuple[str, ...] = get_args(Strategy)
RRF_K = 60  # the constant of the original RRF paper (Cormack et al., 2009)


def rrf(rank_lists: Sequence[Sequence[Hit]], k_const: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion: a chunk scores the sum of 1 / (k_const + rank) over the lists it appears in, where
    rank is its 1-based position in that list. Returns every chunk, best first, ranked from 1. Ties keep the order
    in which chunks first appear, reading the lists in turn, so the first list wins them."""
    scores: dict[str, float] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k_const + rank)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)  # stable, so ties keep first appearance
    return [Hit(chunk_id, score, rank) for rank, (chunk_id, score) in enumerate(fused, start=1)]


def retrieve(
    query: str,
    *,
    store: Store,
    dense: DenseIndex | None = None,
    embedder: Embedder | None = None,
    strategy: Strategy,
    k: int = 8,
    ticker: str | None = None,
    form: str | None = None,
) -> list[Hit]:
    """Top ``k`` chunks for ``query``, optionally only from one ticker and/or form.

    ``bm25`` needs only the store. ``dense`` embeds the query with ``embedder`` (the one that built ``dense``) and
    searches ``dense``. ``hybrid`` takes the top ``2k`` of each and keeps the top ``k`` of their RRF fusion.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; use one of {', '.join(STRATEGIES)}")
    if k <= 0:
        return []
    if strategy == "bm25":
        return store.bm25(query, k, ticker=ticker, form=form)
    if dense is None or embedder is None:
        raise ValueError(f"the {strategy} strategy needs a dense index and its embedder")
    allowed = store.chunk_ids(ticker=ticker, form=form) if ticker or form else None
    query_vec = embedder.embed([query])[0]
    if strategy == "dense":
        return dense.search(query_vec, k, allowed_ids=allowed)
    keyword = store.bm25(query, 2 * k, ticker=ticker, form=form)
    vector = dense.search(query_vec, 2 * k, allowed_ids=allowed)
    return rrf([keyword, vector])[:k]
