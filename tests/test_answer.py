import json

import pytest

from filings_qa.answer import ANSWER_MODELS, SYSTEM_PROMPT, Answer, AnswerSchema, answer
from filings_qa.llm import FakeLLM
from filings_qa.store import Hit

GROWTH = "ACME-10-Q-20240802-2-001"  # "Data center revenue growth was strong: ..." (conftest's store)
GAMING = "ACME-10-Q-20240802-2-002"
OTHER = "OTHR-10-K-20240216-7-001"
HITS = [Hit(GROWTH, 0.031, 1), Hit(OTHER, 0.030, 2), Hit(GAMING, 0.016, 3)]


class FakeRetriever:
    """Returns ``hits`` and records how it was called."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def __call__(self, query, strategy, k):
        self.calls.append((query, strategy, k))
        return self.hits


def reply(*sentences, abstained=False):
    return {"abstained": abstained, "sentences": [{"text": t, "citations": c} for t, c in sentences]}


def test_sentence_citing_a_chunk_not_retrieved_is_dropped(store):
    llm = FakeLLM([reply(("Data center revenue grew strongly.", [GROWTH]), ("Gaming doubled.", ["ACME-10-Q-X-9"]))])
    retriever = FakeRetriever(HITS)
    result = answer("How did data center revenue do?", retriever=retriever, llm=llm, store=store, strategy="bm25", k=3)
    assert [(s.text, s.citations) for s in result.sentences] == [("Data center revenue grew strongly.", [GROWTH])]
    assert result.dropped_sentences == 1 and result.uncited_sentences == 0
    assert not result.abstained and result.advice_hits == []
    assert retriever.calls == [("How did data center revenue do?", "bm25", 3)]
    assert result.hits == HITS  # every retrieved chunk, with its score and rank
    assert (result.question, result.strategy) == ("How did data center revenue do?", "bm25")
    assert result.model == "gemini-3.5-flash"  # the first of the default models
    assert result.usage == {"in": 1000, "out": 200} and llm.usage == {"gemini-3.5-flash": [1, 1000, 200]}


def test_abstention_keeps_the_explanation(store):
    llm = FakeLLM([reply(("The excerpts do not give Netflix subscriber numbers.", []), abstained=True)])
    result = answer("How many subscribers does Netflix have?", retriever=FakeRetriever(HITS), llm=llm, store=store)
    assert result.abstained
    assert result.text == "The excerpts do not give Netflix subscriber numbers."
    assert (result.dropped_sentences, result.uncited_sentences) == (0, 1)


def test_prompt_labels_every_chunk_with_its_id_and_filing(store):
    llm = FakeLLM([reply(("Revenue grew.", [GROWTH, OTHER]))])
    answer("What drove revenue growth?", retriever=FakeRetriever(HITS), llm=llm, store=store)
    prompt = llm.prompts[0]
    for hit in HITS:
        assert f"[{hit.chunk_id}]" in prompt
    assert f"[{GROWTH}] ACME 10-Q filed=2024-08-02 item=2\nData center revenue growth was strong" in prompt
    assert f"[{OTHER}] OTHR 10-K filed=2024-02-16 item=7\n" in prompt
    assert prompt.index(GROWTH) < prompt.index(OTHER) < prompt.index(GAMING)  # best first
    assert prompt.endswith("Question: What drove revenue growth?")
    call = llm.calls[0]
    assert call["models"] == list(ANSWER_MODELS) and call["label"] == "answer:What drove revenue growth?"
    assert call["config"]["temperature"] == 0 and call["config"]["response_schema"] is AnswerSchema
    assert call["config"]["system_instruction"] == SYSTEM_PROMPT
    assert "investment advice" in SYSTEM_PROMPT and "abstained" in SYSTEM_PROMPT


def test_only_chunks_shown_to_the_model_can_be_cited(store):
    """A hit whose chunk is gone from the store (e.g. a stale vector index) is not shown, so citing it drops the
    sentence."""
    gone = "ACME-10-Q-20240802-9-001"
    llm = FakeLLM([reply(("Stale.", [gone]), ("Growth.", [GROWTH]))])
    result = answer("q", retriever=FakeRetriever([Hit(gone, 1.0, 1), *HITS]), llm=llm, store=store, models=["m"])
    assert gone not in llm.prompts[0]
    assert [s.text for s in result.sentences] == ["Growth."] and result.dropped_sentences == 1
    assert result.model == "m" and result.hits[0].chunk_id == gone


def test_nothing_retrieved_abstains_without_asking_the_model(store):
    llm = FakeLLM([])  # any call would fail
    result = answer("What did NFLX report?", retriever=FakeRetriever([]), llm=llm, store=store)
    assert result.abstained and result.model == "" and result.usage == {"in": 0, "out": 0}
    assert len(result.sentences) == 1 and result.uncited_sentences == 1
    assert llm.calls == []


def test_advice_wording_is_flagged_and_latency_includes_the_model(store):
    llm = FakeLLM([reply(("Revenue grew, so investors should buy ACME.", [GROWTH]))], latency=2.0)
    result = answer("q", retriever=FakeRetriever(HITS), llm=llm, store=store)
    assert result.advice_hits == ["should buy"]
    assert 2.0 <= result.latency_s < 2.5


def test_answer_round_trips_through_json(store):
    llm = FakeLLM([reply(("Growth.", [GROWTH]), ("No figure is given.", []))])
    result = answer("q", retriever=FakeRetriever(HITS), llm=llm, store=store)
    data = json.loads(json.dumps(result.to_dict()))
    assert set(data) >= {"question", "strategy", "sentences", "abstained", "dropped_sentences", "uncited_sentences",
                         "hits", "usage", "latency_s", "model"}
    assert data["hits"][0] == {"chunk_id": GROWTH, "score": 0.031, "rank": 1}
    assert data["sentences"][1] == {"text": "No figure is given.", "citations": []}
    assert Answer.from_dict(data) == result


def test_a_reply_that_is_not_the_schema_is_an_error(store):
    llm = FakeLLM(["Sure! Data center revenue grew."])
    with pytest.raises(ValueError, match="did not return a valid answer"):
        answer("q", retriever=FakeRetriever(HITS), llm=llm, store=store)
