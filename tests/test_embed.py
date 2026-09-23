import json
import os
import subprocess
import sys

import numpy as np
import pytest

from filings_qa.chunk import Chunk
from filings_qa.embed import DenseIndex, FakeEmbedder, FastEmbedEmbedder, embedder_from_spec
from filings_qa.store import Store

# Six chunk texts and the vectors a stub embedder gives them. The comment is the cosine similarity with the query
# vector [2, 0, 0], i.e. the first coordinate of the unit-length vector.
VECTORS = {
    "north": [1.0, 0.0, 0.0],  # 1.0
    "north-east": [4.0, 3.0, 0.0],  # 0.8 once scaled to unit length (the stub returns length 5)
    "east-north": [0.6, 0.8, 0.0],  # 0.6
    "east": [0.0, 1.0, 0.0],  # 0.0
    "up": [0.0, 0.0, 1.0],  # 0.0, a tie with "east", which comes first in the index
    "south": [-1.0, 0.0, 0.0],  # -1.0
}
IDS = ["c1", "c2", "c3", "c4", "c5", "c6"]


class StubEmbedder:
    spec = {"kind": "stub"}

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([VECTORS[t] for t in texts], dtype=np.float32)


def _cos(a, b):
    return float(a @ b)  # rows are unit length


def test_fake_embedder_is_deterministic_float32_and_unit_length():
    texts = ["Data center revenue growth", "Supply chain disruptions", "Data center revenue growth"]
    vectors = FakeEmbedder().embed(texts)
    assert vectors.dtype == np.float32 and vectors.shape == (3, 16)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
    assert np.array_equal(vectors[0], vectors[2])
    assert np.array_equal(FakeEmbedder().embed(texts[:1])[0], vectors[0])  # a new instance agrees
    assert not np.array_equal(vectors[0], vectors[1])
    assert FakeEmbedder(dim=5).embed(texts).shape == (3, 5)


def test_fake_embedder_is_a_bag_of_words_without_case_or_stopwords():
    a, b = FakeEmbedder().embed(["Revenue of the company", "COMPANY revenue"])
    assert np.array_equal(a, b)


def test_fake_embedder_texts_sharing_words_are_closer():
    query, shared, unrelated = FakeEmbedder().embed(
        [
            "data center revenue growth",
            "Revenue growth at the data center business accelerated all year.",
            "Legal proceedings arise in the ordinary course of business.",
        ]
    )
    assert _cos(query, shared) > 0.5 > _cos(query, unrelated)


def test_fake_embedder_text_without_words_gives_a_zero_vector():
    vectors = FakeEmbedder().embed(["?!", ""])
    assert not vectors.any()  # zeros, not NaN from dividing by a zero length
    assert FakeEmbedder().embed([]).shape == (0, 16)


def test_fake_vectors_are_the_same_in_every_process():
    """``index --fake`` and ``search`` run in different processes, so the word hash must not be Python's ``hash``,
    which is salted per process."""
    script = "from filings_qa.embed import FakeEmbedder; print(FakeEmbedder().embed(['data center revenue']).tolist())"
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1] and outputs[0].startswith("[[")


def test_dense_index_build_save_load_search(tmp_path, sample_filing):
    with Store(tmp_path / "filings.sqlite") as store:
        store.add_filing(sample_filing, 6)
        store.add_chunks(Chunk(cid, sample_filing.key, "7", 1, text, 1) for cid, text in zip(IDS, VECTORS, strict=True))
        embedder = StubEmbedder()
        DenseIndex.build(store, embedder, tmp_path / "index", batch=4)
    assert embedder.calls == [["north", "north-east", "east-north", "east"], ["up", "south"]]  # 4 chunks per call

    saved = np.load(tmp_path / "index" / "embeddings.npy")
    assert saved.dtype == np.float32 and saved.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(saved, axis=1), 1.0, rtol=1e-6)
    assert json.loads((tmp_path / "index" / "ids.json").read_text()) == IDS

    index = DenseIndex.load(tmp_path / "index")
    assert index.embedder_spec == {"kind": "stub"}  # so a search can embed the query with the same embedder
    hits = index.search(np.array([2.0, 0.0, 0.0]), k=5)
    assert [h.chunk_id for h in hits] == ["c1", "c2", "c3", "c4", "c5"]
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.8, 0.6, 0.0, 0.0], abs=1e-6)
    assert [h.chunk_id for h in index.search(np.array([2.0, 0.0, 0.0]), k=10)][-1] == "c6"


def test_dense_search_only_among_allowed_ids():
    index = DenseIndex(IDS, np.array(list(VECTORS.values())))
    query = np.array([1.0, 0.0, 0.0])
    hits = index.search(query, k=5, allowed_ids=["c6", "c3", "not-indexed"])
    assert [(h.chunk_id, h.rank) for h in hits] == [("c3", 1), ("c6", 2)]
    assert [h.score for h in hits] == pytest.approx([0.6, -1.0])
    assert index.search(query, k=5, allowed_ids=[]) == []
    assert index.search(query, k=0) == []
    assert index.search(np.zeros(3), k=5) == []  # a query without a direction matches nothing


def test_load_refuses_a_missing_or_inconsistent_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="filings-qa index"):
        DenseIndex.load(tmp_path / "nowhere")
    DenseIndex(IDS, np.array(list(VECTORS.values())), {"kind": "stub"}).save(tmp_path)
    (tmp_path / "ids.json").write_text(json.dumps(IDS[:5]))  # e.g. a build that was interrupted
    with pytest.raises(ValueError, match="5 ids"):
        DenseIndex.load(tmp_path)


def test_fastembed_is_loaded_once_on_first_use_with_batches_of_64(fake_fastembed):
    embedder = FastEmbedEmbedder("test/model")
    assert fake_fastembed == []  # constructing loads (and downloads) nothing
    vectors = embedder.embed([f"filing text {i}" for i in range(130)])
    embedder.embed(["one more"])
    assert [m.model_name for m in fake_fastembed] == ["test/model"]
    assert fake_fastembed[0].batch_sizes == [64, 64]
    assert vectors.dtype == np.float32 and vectors.shape == (130, 8)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)  # the fake returns length 3


def test_fastembed_not_installed_gives_the_install_command():
    embedder = FastEmbedEmbedder()  # `import fastembed` fails in tests (conftest)
    with pytest.raises(RuntimeError, match=r"pip install"):
        embedder.embed(["text"])


def test_embedder_from_spec_rebuilds_the_embedder_of_an_index(fake_fastembed):
    assert embedder_from_spec(FakeEmbedder(dim=32).spec).embed(["x"]).shape == (1, 32)
    embedder_from_spec(FastEmbedEmbedder("test/model-b").spec).embed(["x"])
    assert [m.model_name for m in fake_fastembed] == ["test/model-b"]
    with pytest.raises(ValueError, match="stub"):
        embedder_from_spec({"kind": "stub"})
