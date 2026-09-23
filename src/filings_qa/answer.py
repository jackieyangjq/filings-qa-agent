"""Answer a question from retrieved filing chunks, with a citation on every sentence.

``answer`` retrieves chunks, shows them to the model labelled with their chunk ids, asks for JSON sentences that each
list the ids they rely on, then keeps only sentences whose citations are among the chunks shown (``guard``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .chunk import Chunk
from .guard import advice_check, verify_citations
from .llm import Gemini, token_counts
from .store import Hit, Store

ANSWER_MODELS: tuple[str, ...] = ("gemini-3.5-flash", "gemini-3.5-flash-lite")  # tried in this order

SYSTEM_PROMPT = """\
You are a research assistant answering questions only from the excerpts of SEC filings (10-K and 10-Q reports) \
given in the message.

Rules:
1. Use only what the excerpts state. Do not use outside knowledge, and do not guess or fill gaps.
2. Write the answer as short sentences. Every sentence must end with the ids of the excerpts it relies on: put them \
in the sentence's "citations" list, exactly as written in square brackets before each excerpt (for example \
AAPL-10-K-20251031-7-007), and do not write ids in the sentence text. Cite only excerpts that support the sentence. \
Refer to the company by name instead of copying "we" or "our" from the filing.
3. Give figures exactly as the excerpts state them, with units and the period they cover.
4. The header of each excerpt gives the company, the form and the date the filing was made. When the question asks \
about the latest or most recent period, answer from the most recently filed excerpts that address it; an older \
excerpt may add context only if its sentence names the period it covers.
5. If the excerpts do not contain the answer, set "abstained" to true and explain in one or two sentences what is \
missing, without citations. If they answer only part of the question, answer that part and say what is missing.
6. Never give investment advice or recommendations to buy, sell or hold, and do not predict prices."""


class SentenceSchema(BaseModel):
    text: str = Field(description="One sentence of the answer, without excerpt ids.")
    citations: list[str] = Field(description="Ids of the excerpts the sentence relies on, as given in brackets.")


class AnswerSchema(BaseModel):
    abstained: bool = Field(description="True when the excerpts do not contain the answer.")
    sentences: list[SentenceSchema]


@dataclass
class Sentence:
    text: str
    citations: list[str] = field(default_factory=list)

    @property
    def uncited(self) -> bool:
        return not self.citations


@dataclass
class Answer:
    """A checked answer. ``sentences`` are those left after ``verify_citations``: ``dropped_sentences`` were removed
    for citing a chunk the model was not shown, ``uncited_sentences`` of those left cite nothing. ``abstained`` is
    the model's own statement that the excerpts do not answer the question. ``hits`` are the chunks retrieved, best
    first. ``usage`` holds the input and output tokens of the model call; ``latency_s`` is the retrieval time plus
    the model's response time (waits between retries are not counted). ``advice_hits`` lists wording in the answer
    that reads as investment advice (see ``guard.advice_check``). ``model`` is the model that answered, empty when no
    chunk was retrieved and no model was asked."""

    question: str
    strategy: str
    sentences: list[Sentence]
    abstained: bool
    hits: list[Hit]
    dropped_sentences: int = 0
    uncited_sentences: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0})
    latency_s: float = 0.0
    model: str = ""
    advice_hits: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def cited_ids(self) -> list[str]:
        """Every chunk id cited by a sentence, in order of first citation."""
        return list(dict.fromkeys(c for s in self.sentences for c in s.citations))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Answer:
        return cls(
            **{
                **data,
                "sentences": [Sentence(**s) for s in data["sentences"]],
                "hits": [Hit(**h) for h in data["hits"]],
            }
        )


Retriever = Callable[..., list[Hit]]  # called as retriever(question, strategy=..., k=...)


def excerpt_header(chunk: Chunk, filing: dict[str, Any] | None) -> str:
    """``[chunk_id] TICKER FORM filed=YYYY-MM-DD item=N`` for one excerpt of the prompt."""
    ticker, form, filed = (filing["ticker"], filing["form"], filing["filed"]) if filing else ("?", "?", "?")
    return f"[{chunk.chunk_id}] {ticker} {form} filed={filed} item={chunk.item or '-'}"


def build_prompt(question: str, chunks: Sequence[Chunk], filings: dict[str, dict[str, Any] | None]) -> str:
    """The user message: the excerpts, each under its header, then the question."""
    excerpts = "\n\n".join(f"{excerpt_header(c, filings.get(c.filing_key))}\n{c.text}" for c in chunks)
    return f"Excerpts from SEC filings:\n\n{excerpts}\n\nQuestion: {question}"


def _parse(resp: Any) -> AnswerSchema:
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, AnswerSchema):
        return parsed
    try:
        if isinstance(parsed, dict):
            return AnswerSchema.model_validate(parsed)
        return AnswerSchema.model_validate_json(getattr(resp, "text", None) or "")
    except ValueError as e:  # pydantic's ValidationError is a ValueError
        text = " ".join((getattr(resp, "text", None) or "").split())
        raise ValueError(f"the model did not return a valid answer: {text[:200]!r}") from e


def answer(
    question: str,
    *,
    retriever: Retriever,
    llm: Gemini,
    store: Store,
    strategy: str = "hybrid",
    k: int = 8,
    models: Sequence[str] | None = None,
) -> Answer:
    """Answer ``question`` from the top ``k`` chunks that ``retriever`` finds with ``strategy``.

    The chunk texts and their filings come from ``store``. ``models`` are tried in order (default
    ``ANSWER_MODELS``), at temperature 0 with a JSON schema for the reply. When nothing is retrieved, the answer is
    an abstention and no model is asked.
    """
    started = time.perf_counter()
    hits = retriever(question, strategy=strategy, k=k)
    retrieval_s = time.perf_counter() - started
    chunks = store.get_chunks([h.chunk_id for h in hits])  # a chunk deleted after indexing is skipped
    if not chunks:
        empty = Answer(
            question,
            strategy,
            [Sentence("No filing excerpts were retrieved for this question, so it cannot be answered from them.")],
            abstained=True,
            hits=hits,
            latency_s=retrieval_s,
        )
        return verify_citations(empty, [])

    filings = {key: store.get_filing(key) for key in dict.fromkeys(c.filing_key for c in chunks)}
    config = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": AnswerSchema,
    }
    resp, model = llm.generate(
        list(models or ANSWER_MODELS),
        label=f"answer:{question}",
        contents=build_prompt(question, chunks, filings),
        config=config,
    )
    llm.add_usage(model, resp.usage_metadata)
    reply = _parse(resp)
    tokens_in, tokens_out = token_counts(resp.usage_metadata)
    raw = Answer(
        question,
        strategy,
        [Sentence(s.text, list(s.citations)) for s in reply.sentences],
        abstained=reply.abstained,
        hits=hits,
        usage={"in": tokens_in, "out": tokens_out},
        latency_s=retrieval_s + llm.last_latency,
        model=model,
    )
    checked = verify_citations(raw, [c.chunk_id for c in chunks])
    checked.advice_hits = advice_check(checked.text)
    return checked
