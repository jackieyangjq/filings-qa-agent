"""Dense vectors for chunks and a brute-force index over them.

An embedder turns texts into float32 rows of unit length. ``FastEmbedEmbedder`` runs a small local ONNX model
(BAAI/bge-small-en-v1.5 by default, no API key); ``FakeEmbedder`` hashes words and serves tests and the offline demo.

The index lives in the index folder as ``embeddings.npy`` (one row per chunk), ``ids.json`` (the chunk id of each
row) and ``embedder.json`` (the embedder that made the vectors, so that queries are embedded the same way). The
command line keeps the downloaded model in ``<data>/models`` unless ``FASTEMBED_CACHE_PATH`` names another folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .store import STOPWORDS, Hit

if TYPE_CHECKING:
    from .store import Store

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
VECTORS_FILE = "embeddings.npy"
IDS_FILE = "ids.json"
EMBEDDER_FILE = "embedder.json"
MODELS_DIR = "models"  # model cache inside the data folder
CACHE_ENV = "FASTEMBED_CACHE_PATH"

_WORD = re.compile(r"[^\W_]+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """One float32 row of unit length per text."""
        ...


def normalize(vectors: Any) -> np.ndarray:
    """``vectors`` as float32 scaled to unit length along the last axis; all-zero vectors stay zero."""
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)


class FakeEmbedder:
    """Deterministic bag-of-words vectors for tests and the offline demo. Each word (lower-cased; stopwords dropped
    unless nothing else is left) adds +1 or -1 to one of ``dim`` coordinates, both picked by a hash of the word.
    Equal texts get equal vectors and texts that share words point the same way; meaning plays no part."""

    def __init__(self, dim: int = 16):
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim

    @property
    def spec(self) -> dict[str, Any]:
        return {"kind": "fake", "dim": self.dim}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = [w.lower() for w in _WORD.findall(text)]
            for word in [w for w in words if w not in STOPWORDS] or words:
                # a fixed hash: Python's hash() of a str changes from one process to the next
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return normalize(out)


def models_dir(data_dir: Path | str) -> Path:
    """Where the command line keeps downloaded embedding models: ``<data>/models``."""
    return Path(data_dir) / MODELS_DIR


class FastEmbedEmbedder:
    """A local embedding model run by fastembed (install with ``pip install 'filings-qa-agent[embed]'``). The model
    is loaded on the first ``embed`` call and downloaded then if needed, to the folder named by the environment
    variable ``FASTEMBED_CACHE_PATH`` if set, else to ``cache_dir``, else to fastembed's default (``fastembed_cache``
    in the system temp folder, which the system may clean, forcing a new download)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 64, cache_dir: Path | str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model: Any = None

    @property
    def spec(self) -> dict[str, Any]:
        return {"kind": "fastembed", "model": self.model_name}

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as e:
                raise RuntimeError("fastembed is not installed: pip install 'filings-qa-agent[embed]'") from e
            cache = os.environ.get(CACHE_ENV) or self.cache_dir
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache) if cache else None)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, model.embedding_size), dtype=np.float32)
        return normalize(np.stack(list(model.embed(list(texts), batch_size=self.batch_size))))


def embedder_from_spec(spec: dict[str, Any], *, cache_dir: Path | str | None = None) -> Embedder:
    """The embedder described by ``spec`` (the content of ``embedder.json``), to embed queries for that index.
    ``cache_dir`` is where a fastembed model is kept (see ``FastEmbedEmbedder``)."""
    kind = spec.get("kind")
    if kind == "fastembed":
        return FastEmbedEmbedder(spec["model"], cache_dir=cache_dir)
    if kind == "fake":
        return FakeEmbedder(int(spec["dim"]))
    raise ValueError(f"cannot recreate the embedder {spec!r}; rebuild the index with `filings-qa index`")


class DenseIndex:
    """Unit-length chunk vectors searched by a full scan: the dot product with a unit query vector is the cosine
    similarity. Thousands of chunks at 384 dimensions take about 10 MB and under a millisecond per query."""

    def __init__(self, ids: Sequence[str], vectors: Any, embedder_spec: dict[str, Any] | None = None):
        vectors = normalize(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(f"{len(ids)} ids do not match vectors of shape {vectors.shape}; run `filings-qa index`")
        self.ids = list(ids)
        self.vectors = vectors
        self.embedder_spec = dict(embedder_spec or {})
        self._row = {chunk_id: i for i, chunk_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @classmethod
    def build(
        cls,
        store: Store,
        embedder: Embedder,
        out_dir: Path | str,
        batch: int = 256,
        progress: Callable[[int, int], None] | None = None,
    ) -> DenseIndex:
        """Embed every chunk of ``store``, ``batch`` chunks per call to the embedder, save the index in ``out_dir``
        and return it. ``progress(done, total)`` is called after each batch."""
        ids = store.all_chunk_ids()
        if not ids:
            raise ValueError("the store has no chunks")
        parts = []
        for start in range(0, len(ids), batch):
            parts.append(embedder.embed([c.text for c in store.get_chunks(ids[start : start + batch])]))
            if progress:
                progress(min(start + batch, len(ids)), len(ids))
        index = cls(ids, np.vstack(parts), getattr(embedder, "spec", {"kind": type(embedder).__name__}))
        index.save(out_dir)
        return index

    def save(self, out_dir: Path | str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / VECTORS_FILE, self.vectors)
        (out / IDS_FILE).write_text(json.dumps(self.ids), encoding="utf-8")
        (out / EMBEDDER_FILE).write_text(json.dumps(self.embedder_spec) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path | str) -> DenseIndex:
        d = Path(index_dir)
        if not (d / VECTORS_FILE).exists():
            raise FileNotFoundError(f"no dense index in {d}; run `filings-qa index` first")
        vectors = np.load(d / VECTORS_FILE, allow_pickle=False)
        ids = json.loads((d / IDS_FILE).read_text(encoding="utf-8"))
        spec_path = d / EMBEDDER_FILE
        spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
        return cls(ids, vectors, spec)

    def search(self, query_vec: Any, k: int, allowed_ids: Iterable[str] | None = None) -> list[Hit]:
        """The ``k`` chunks most similar to ``query_vec``, optionally only among ``allowed_ids`` (ids that are not in
        the index are ignored). Equal scores keep index order. A zero query vector matches nothing."""
        query = normalize(np.asarray(query_vec).reshape(-1))
        if k <= 0 or not query.any():
            return []
        scores = self.vectors @ query
        if allowed_ids is not None:
            mask = np.zeros(len(self.ids), dtype=bool)
            mask[np.fromiter((self._row[i] for i in allowed_ids if i in self._row), dtype=np.intp)] = True
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.ids[i], float(scores[i]), rank) for rank, i in enumerate(top, start=1)]
