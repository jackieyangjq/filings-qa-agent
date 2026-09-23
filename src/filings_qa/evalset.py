"""Build the evaluation questions from the stored filings.

Answerable questions come from chunks sampled by company and section (see ``strata``). For each chunk a model writes
one question that the chunk alone answers, a reference answer of at most 30 words and a quote copied from the chunk.
A question is kept only if the quote is found in the chunk (differences of spacing, typographic quotes and dashes
aside) and the question passes a few checks: it names the company, it is not a yes/no question and it does not refer
to "the excerpt". A rejected chunk is replaced by another chunk of the same stratum. Unanswerable questions are
rewrites of kept questions: the same kind of fact about a company outside the corpus, or about fiscal 2019, which no
stored filing covers.

Every question starts with ``reviewed`` false; a person sets it to true after checking it against the filing.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .chunk import Chunk
from .llm import Gemini, with_quota_wait
from .store import Store

GENERATION_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
TRIES_PER_STRATUM = 3  # chunks tried in a stratum before it is given up
MIN_WORDS = 100  # shorter chunks (cover pages, one-line sections) rarely hold a fact worth a question
MAX_TABLE_SHARE = 0.5  # chunks that are mostly table rows are skipped: the column headers (periods) are often cut off
MAX_ANSWER_WORDS = 30
QUOTE_WORDS = (4, 80)  # a quote shorter than 4 words matches by chance; the prompt asks for at most 60

# JPMorgan's and Exxon Mobil's section labels cannot be trusted: their 10-Ks put the financial statements after Item
# 15 or 16, and JPMorgan's 10-Qs have no Item headings in the body. These companies are sampled by filing instead.
BY_FILING: tuple[str, ...] = ("JPM", "XOM")

# The sections sampled for the other companies, as (form, item) pairs.
SECTION_KINDS: dict[str, frozenset[tuple[str, str]]] = {
    "1A": frozenset({("10-K", "1A"), ("10-Q", "1A")}),  # risk factors
    "7": frozenset({("10-K", "7")}),  # management's discussion and analysis (MD&A) in a 10-K
    "2": frozenset({("10-Q", "2")}),  # MD&A in a 10-Q
    "fin": frozenset({("10-K", "8"), ("10-Q", "1")}),  # financial statements and their notes
}

# Names a question may use for each company, the first as the prompts write it. Matched as whole words, ignoring
# case; the ticker also counts, matched with its case (so COST is not "cost").
COMPANY_NAMES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "NVDA": ("NVIDIA",),
    "AMZN": ("Amazon",),
    "GOOGL": ("Alphabet", "Google"),
    "META": ("Meta Platforms", "Meta"),
    "TSLA": ("Tesla",),
    "AVGO": ("Broadcom",),
    "AMD": ("AMD", "Advanced Micro Devices"),
    "COST": ("Costco",),
    "JPM": ("JPMorgan Chase", "JPMorganChase", "JPMorgan"),
    "XOM": ("Exxon Mobil", "ExxonMobil", "Exxon"),
    "NFLX": ("Netflix",),
    "INTC": ("Intel",),
    "DIS": ("Disney",),
    "PFE": ("Pfizer",),
    "BA": ("Boeing",),
}
OTHER_COMPANIES: tuple[str, ...] = ("NFLX", "INTC", "DIS", "PFE", "BA")  # not in the corpus
OTHER_YEAR = "2019"  # no stored filing covers it

QUESTION_PROMPT = """\
You write test questions for a question-answering system over SEC filings (10-K and 10-Q reports). You get one \
excerpt of a filing, with the company, the form, the filing date and the period the filing covers.

Write one question that:
1. can be answered from this excerpt alone;
2. asks for one specific fact the excerpt states: a figure, a percentage, a date, a count, a name, or a stated term \
or policy. Not a yes/no question, not an opinion, not a summary or a list of everything the excerpt says;
3. makes sense to someone who has not seen the excerpt: it names the company (e.g. "Apple") and, when the fact \
belongs to a period, gives the period as the excerpt does (e.g. "for the three months ended June 27, 2026", "as of \
December 31, 2025", "in fiscal 2025"). Never refer to "the excerpt", "the passage", "this section", "the text" or \
"this filing";
4. has a single correct answer. Do not ask about a number in a table row when the excerpt does not show which \
period or column it belongs to.

Also give:
- "answer": the reference answer in at most 30 words, with units and the period;
- "quote": the sentence or table row of the excerpt that contains the answer, copied exactly, character for \
character: at most 60 words, no "...", no changes to numbers or punctuation.

If the excerpt holds no specific fact that suits such a question (for example it is a table of contents, a list of \
exhibits, signatures, or only general statements), set "skip" to true and leave the other fields empty."""

REWRITE_PROMPT = """\
You rewrite questions for a test of whether a question-answering system admits that its documents do not hold the \
answer. Its documents are the 10-K and 10-Q filings of 2025 and 2026 of these companies only: {companies}.

Rewrite the question as the instruction says. Keep the kind of fact it asks for (the same metric, term or detail) \
and its style, so that it reads like the original. Return only the new question."""

OTHER_COMPANY_INSTRUCTION = (
    "Ask about {new} instead of {old}. Replace names of products, segments or programs that belong to {old} with ones"
    " that fit {new}, or with generic words. Keep the period."
)
OTHER_PERIOD_INSTRUCTION = (
    "Ask about {old} in fiscal year 2019 (or the matching quarter or date in 2019) instead of the period in the"
    " question. Keep the company."
)

YES_NO = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|may|might|must)\b", re.I
)
REFERS_TO_EXCERPT = re.compile(
    r"\b(excerpts?|passages?|paragraphs?|above|this (section|text|filing|document|report)|the text)\b", re.I
)
LATER_YEAR = re.compile(r"\b20[2-9]\d\b")
_TYPOGRAPHIC = str.maketrans(  # curly quotes, en and em dashes, no-break space
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "}
)


@dataclass
class Question:
    """One evaluation question. For an answerable one, ``gold_chunk_id`` is the chunk it was written from and
    ``gold_chunk_ids`` every chunk of the company that contains ``gold_quote`` (chunks overlap, and 10-Qs repeat text
    of the 10-K), so that retrieving any of them counts. ``kind`` is "answerable", "other_company" (a company outside
    the corpus) or "other_period" (fiscal 2019); the last two have no gold chunk and name their ``source_qid``."""

    qid: str
    question: str
    gold_answer: str
    gold_quote: str = ""
    gold_chunk_id: str | None = None
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_filing_key: str | None = None
    gold_item: str | None = None
    ticker: str = ""
    stratum: str = ""
    kind: str = "answerable"
    source_qid: str | None = None
    answerable: bool = True
    reviewed: bool = False


@dataclass(frozen=True)
class Stratum:
    name: str  # "AAPL/1A", or "JPM/JPM-10-Q-20260806" for a company sampled by filing
    ticker: str
    chunk_ids: tuple[str, ...]


class QuestionSchema(BaseModel):
    skip: bool = Field(description="True when the excerpt holds no specific fact that suits a question.")
    question: str = Field(description="The question; empty when skip is true.")
    answer: str = Field(description="The reference answer, at most 30 words; empty when skip is true.")
    quote: str = Field(description="The sentence or table row with the answer, copied exactly; empty when skipping.")


class RewriteSchema(BaseModel):
    question: str = Field(description="The rewritten question.")


def save(questions: Sequence[Question], path: Path | str) -> None:
    """Write ``questions`` as JSON lines, one question per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")


def load(path: Path | str) -> list[Question]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def _flat(text: str) -> str:
    """``text`` with typographic quotes and dashes made plain and each run of whitespace made one space."""
    return " ".join(text.translate(_TYPOGRAPHIC).split())


def find_quote(quote: str, text: str) -> str | None:
    """The span of ``text`` that ``quote`` copies, as written in ``text``, or None when ``text`` does not contain it.
    Differences of spacing (a tab for a space, a line break), typographic quotes and dashes are ignored, and so are
    quotation marks around the whole quote; case, words, numbers and punctuation must match."""
    needle = _flat(quote).strip('"').strip()
    if not needle:
        return None
    chars: list[str] = []
    where: list[int] = []  # where[i]: position in ``text`` of chars[i]
    for i, ch in enumerate(text.translate(_TYPOGRAPHIC)):  # one character for one, so positions stay aligned
        if ch.isspace():
            if not chars or chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        where.append(i)
    start = "".join(chars).find(needle)
    if start < 0:
        return None
    return text[where[start] : where[start + len(needle) - 1] + 1]


def names_company(text: str, ticker: str, names: Mapping[str, Sequence[str]] = COMPANY_NAMES) -> bool:
    """Whether ``text`` names the company: one of its names as a whole word (any case) or its ticker (same case)."""
    if re.search(rf"\b{re.escape(ticker)}\b", text):
        return True
    return any(re.search(rf"\b{re.escape(name)}\b", text, re.I) for name in names.get(ticker, ()))


def eligible(chunk: Chunk, min_words: int = MIN_WORDS) -> bool:
    """Whether a question may be drawn from ``chunk``: at least ``min_words`` words, at most half of its lines table
    rows, and at least one number. Chunks without any number hold general statements (accounting policies, risk
    narratives): in a first trial the model found no fact to ask about in each of the five it was shown."""
    lines = [line for line in chunk.text.splitlines() if line.strip()]
    table_rows = sum("\t" in line for line in lines)
    has_number = any(ch.isdigit() for ch in chunk.text)
    return chunk.n_words >= min_words and table_rows <= MAX_TABLE_SHARE * len(lines) and has_number


def strata(
    store: Store, *, by_filing: Sequence[str] = BY_FILING, min_words: int = MIN_WORDS
) -> dict[str, list[Stratum]]:
    """The strata of each company, by ticker: one per section kind of ``SECTION_KINDS`` that has eligible chunks, or
    one per filing for the companies in ``by_filing``."""
    filings = {f["filing_key"]: f for f in store.filings()}
    groups: dict[tuple[str, str], list[str]] = {}
    for chunk in store.get_chunks(store.all_chunk_ids()):
        filing = filings.get(chunk.filing_key)
        if filing is None or not eligible(chunk, min_words):
            continue
        ticker = filing["ticker"]
        if ticker in by_filing:
            key: str | None = chunk.filing_key
        else:
            key = next((kind for kind, pairs in SECTION_KINDS.items() if (filing["form"], chunk.item) in pairs), None)
        if key is not None:
            groups.setdefault((ticker, key), []).append(chunk.chunk_id)
    out: dict[str, list[Stratum]] = {}
    for (ticker, key), ids in groups.items():
        out.setdefault(ticker, []).append(Stratum(f"{ticker}/{key}", ticker, tuple(ids)))
    return out


def plan(by_company: Mapping[str, Sequence[Stratum]], seed: int = 0) -> list[Stratum]:
    """Every stratum once, in the order questions are drawn: round after round over the companies (in a random order
    fixed by ``seed``), each company giving its next stratum. The section kinds are rotated from one company to the
    next, so any number of rounds spreads the questions evenly over the kinds. A company sampled by filing gives its
    10-K first, then its other filings in random order."""
    rng = random.Random(seed)
    companies = sorted(by_company)
    rng.shuffle(companies)
    kinds = list(SECTION_KINDS)
    orders: dict[str, list[Stratum]] = {}
    turn = 0
    for ticker in companies:
        own = {s.name.split("/", 1)[1]: s for s in by_company[ticker]}
        if set(own) <= set(kinds):
            rotated = kinds[turn % len(kinds) :] + kinds[: turn % len(kinds)]
            orders[ticker] = [own[k] for k in rotated if k in own]
            turn += 1
        else:
            own_filings = sorted(by_company[ticker], key=lambda s: s.name)
            rng.shuffle(own_filings)
            orders[ticker] = sorted(own_filings, key=lambda s: "-10-K-" not in s.name)  # stable: 10-K first
    rounds = max((len(o) for o in orders.values()), default=0)
    return [orders[t][r] for r in range(rounds) for t in companies if r < len(orders[t])]


def _ask(
    llm: Gemini,
    models: Sequence[str],
    *,
    label: str,
    system: str,
    contents: str,
    schema: type[BaseModel],
    cache: Path | None,
    sleep: Callable[[float], Any],
) -> dict[str, Any]:
    """The model's JSON reply, as a dict: from ``cache`` when it holds one, else asked (and then saved there)."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["reply"]
    config = {"system_instruction": system, "response_mime_type": "application/json", "response_schema": schema}
    resp = with_quota_wait(lambda: llm.ask(list(models), label=label, contents=contents, config=config), sleep=sleep)
    parsed = resp.parsed if isinstance(resp.parsed, schema) else schema.model_validate_json(resp.text or "")
    reply = parsed.model_dump()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"label": label, "reply": reply}, indent=1) + "\n", encoding="utf-8")
    return reply


def check_reply(
    reply: Mapping[str, Any], chunk: Chunk, ticker: str, names: Mapping[str, Sequence[str]] = COMPANY_NAMES
) -> tuple[str | None, str]:
    """(the quote as written in the chunk, "") when ``reply`` makes a usable question, else (None, the reason)."""
    if reply.get("skip"):
        return None, "the model found no fact to ask about"
    question, answer, quote = (str(reply.get(key) or "").strip() for key in ("question", "answer", "quote"))
    if not (question and answer and quote):
        return None, "incomplete reply"
    if YES_NO.match(question):
        return None, "yes/no question"
    if REFERS_TO_EXCERPT.search(question):
        return None, "refers to the excerpt"
    if not names_company(question, ticker, names):
        return None, "does not name the company"
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return None, f"answer longer than {MAX_ANSWER_WORDS} words"
    exact = find_quote(quote, chunk.text)
    if exact is None:
        return None, "quote not found in the chunk"
    if not QUOTE_WORDS[0] <= len(exact.split()) <= QUOTE_WORDS[1]:
        return None, "quote too short or too long"
    return exact, ""


def check_rewrite(
    question: str, kind: str, source_ticker: str, new_ticker: str, names: Mapping[str, Sequence[str]] = COMPANY_NAMES
) -> str:
    """Why a rewritten question is unusable, or "" when it is fine."""
    if not question:
        return "empty question"
    if YES_NO.match(question):
        return "yes/no question"
    if kind == "other_company":
        if not names_company(question, new_ticker, names):
            return f"does not name {new_ticker}"
        if names_company(question, source_ticker, names):
            return f"still names {source_ticker}"
        return ""
    if OTHER_YEAR not in question:
        return f"does not ask about {OTHER_YEAR}"
    if LATER_YEAR.search(question):
        return "still names a year after 2019"
    if not names_company(question, source_ticker, names):
        return "does not name the company"
    return ""


def _spread(questions: Sequence[Question]) -> list[Question]:
    """``questions`` reordered so that companies take turns, keeping the order within each company."""
    by_company: dict[str, list[Question]] = {}
    for q in questions:
        by_company.setdefault(q.ticker, []).append(q)
    columns = list(by_company.values())
    return [col[i] for i in range(max(map(len, columns), default=0)) for col in columns if i < len(col)]


def build(
    store: Store,
    llm: Gemini,
    n_answerable: int = 40,
    n_unanswerable: int = 10,
    seed: int = 0,
    *,
    models: Sequence[str] | None = None,
    cache_dir: Path | str | None = None,
    by_filing: Sequence[str] | None = None,
    names: Mapping[str, Sequence[str]] | None = None,
    tries: int = TRIES_PER_STRATUM,
    log: Callable[[str], Any] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> list[Question]:
    """``n_answerable`` questions drawn from the chunks of ``store`` by stratum (see ``plan``), then
    ``n_unanswerable`` rewrites: half of them (rounded up) about the companies of ``OTHER_COMPANIES`` in turn, the
    rest about fiscal 2019. Questions are numbered q01, q02, ... in that order.

    Each stratum tries up to ``tries`` of its chunks, in an order fixed by ``seed``, until one gives a usable
    question; the model is ``models`` (default ``GENERATION_MODELS``). ``by_filing`` (default ``BY_FILING``) are the
    companies sampled by filing; ``names`` adds to ``COMPANY_NAMES``. Replies are kept in ``cache_dir`` (one JSON
    file per chunk or rewrite) and reused, so a second run with the same seed asks nothing new. ``log`` receives one
    line per question and per rejected reply. Fewer questions are returned when the strata run out.
    """
    models = list(models or GENERATION_MODELS)
    by_filing = BY_FILING if by_filing is None else by_filing
    names = {**COMPANY_NAMES, **(names or {})}
    say = log or (lambda line: None)
    cache = Path(cache_dir) if cache_dir is not None else None
    filings = {f["filing_key"]: f for f in store.filings()}
    chunks = {c.chunk_id: c for c in store.get_chunks(store.all_chunk_ids())}
    flat_by_ticker: dict[str, list[tuple[str, str]]] = {}
    for chunk in chunks.values():
        ticker = filings[chunk.filing_key]["ticker"] if chunk.filing_key in filings else ""
        flat_by_ticker.setdefault(ticker, []).append((chunk.chunk_id, _flat(chunk.text)))

    answerable: list[Question] = []
    for stratum in plan(strata(store, by_filing=by_filing), seed):
        if len(answerable) >= n_answerable:
            break
        order = list(stratum.chunk_ids)
        random.Random(f"{seed}/{stratum.name}").shuffle(order)
        for chunk_id in order[:tries]:
            chunk = chunks[chunk_id]
            filing = filings[chunk.filing_key]
            name = names.get(stratum.ticker, (stratum.ticker,))[0]
            contents = (
                f"Company: {name} ({stratum.ticker})\nForm: {filing['form']}, filed {filing['filed']}, for the period"
                f" ended {filing['period'] or 'not given'}\nSection: {'Item ' + chunk.item if chunk.item else 'none'}"
                f"\n\nExcerpt:\n{chunk.text}"
            )
            reply = _ask(
                llm,
                models,
                label=f"question:{chunk_id}",
                system=QUESTION_PROMPT,
                contents=contents,
                schema=QuestionSchema,
                cache=cache / f"{chunk_id}.json" if cache else None,
                sleep=sleep,
            )
            quote, why = check_reply(reply, chunk, stratum.ticker, names)
            if quote is None:
                say(f"{stratum.name} {chunk_id}: rejected ({why})")
                continue
            needle = _flat(quote)
            q = Question(
                qid=f"q{len(answerable) + 1:02d}",
                question=reply["question"].strip(),
                gold_answer=reply["answer"].strip(),
                gold_quote=quote,
                gold_chunk_id=chunk_id,
                gold_chunk_ids=[cid for cid, flat in flat_by_ticker[stratum.ticker] if needle in flat],
                gold_filing_key=chunk.filing_key,
                gold_item=chunk.item,
                ticker=stratum.ticker,
                stratum=stratum.name,
            )
            answerable.append(q)
            say(f"{stratum.name} {chunk_id}: {q.qid} {q.question}")
            break

    unanswerable = _rewrites(
        answerable,
        n_unanswerable,
        seed=seed,
        llm=llm,
        models=models,
        names=names,
        companies=sorted({f["ticker"] for f in filings.values()}),
        cache=cache,
        tries=tries,
        say=say,
        sleep=sleep,
    )
    for i, q in enumerate(unanswerable, start=len(answerable) + 1):
        q.qid = f"q{i:02d}"
    return answerable + unanswerable


def _rewrites(
    sources: Sequence[Question],
    n: int,
    *,
    seed: int,
    llm: Gemini,
    models: Sequence[str],
    names: Mapping[str, Sequence[str]],
    companies: Sequence[str],
    cache: Path | None,
    tries: int,
    say: Callable[[str], Any],
    sleep: Callable[[float], Any],
) -> list[Question]:
    """``n`` unanswerable questions rewritten from ``sources``: questions whose answer holds a figure first, companies
    taking turns; for the fiscal 2019 ones, questions that name a year first. A rejected rewrite moves on to the next
    source, at most ``tries`` times per question."""
    rng = random.Random(f"{seed}/unanswerable")
    shuffled = list(sources)
    rng.shuffle(shuffled)
    with_figure = [q for q in shuffled if re.search(r"\d", q.gold_answer)]
    others = [q for q in shuffled if q not in with_figure]
    n_company = (n + 1) // 2
    targets = [("other_company", OTHER_COMPANIES[i % len(OTHER_COMPANIES)]) for i in range(n_company)]
    targets += [("other_period", "")] * (n - n_company)
    system = REWRITE_PROMPT.format(companies=", ".join(names.get(t, (t,))[0] for t in companies))
    used: set[str] = set()
    out: list[Question] = []
    pool = _spread(with_figure) + _spread(others)
    dated = [q for q in pool if LATER_YEAR.search(q.question)] + [q for q in pool if not LATER_YEAR.search(q.question)]
    for kind, new_ticker in targets:
        candidates = [q for q in (dated if kind == "other_period" else pool) if q.qid not in used][:tries]
        for source in candidates:
            used.add(source.qid)
            old = names.get(source.ticker, (source.ticker,))[0]
            if kind == "other_company":
                new = names.get(new_ticker, (new_ticker,))[0]
                instruction = OTHER_COMPANY_INSTRUCTION.format(new=f"{new} ({new_ticker})", old=old)
                key = f"rewrite-{source.gold_chunk_id}-{new_ticker}"
            else:
                instruction = OTHER_PERIOD_INSTRUCTION.format(old=old)
                key = f"rewrite-{source.gold_chunk_id}-FY{OTHER_YEAR}"
            reply = _ask(
                llm,
                models,
                label=f"rewrite:{source.qid}:{new_ticker or OTHER_YEAR}",
                system=system,
                contents=f"Question: {source.question}\nInstruction: {instruction}",
                schema=RewriteSchema,
                cache=cache / f"{key}.json" if cache else None,
                sleep=sleep,
            )
            question = str(reply.get("question") or "").strip()
            why = check_rewrite(question, kind, source.ticker, new_ticker, names)
            if why:
                say(f"rewrite of {source.qid} ({kind}): rejected ({why})")
                continue
            if kind == "other_company":
                gold = f"(none: {names.get(new_ticker, (new_ticker,))[0]} is not among the companies in the corpus)"
            else:
                gold = f"(none: no filing in the corpus covers {OTHER_YEAR})"
            q = Question(
                qid="",
                question=question,
                gold_answer=gold,
                ticker=new_ticker or source.ticker,
                kind=kind,
                source_qid=source.qid,
                answerable=False,
            )
            out.append(q)
            say(f"{kind} from {source.qid}: {question}")
            break
    return out
