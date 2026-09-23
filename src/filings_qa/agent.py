"""A research agent: Gemini chooses tools (filing search, prices, news), the loop runs them and sends the results
back, until the model answers or has asked for tools ``max_steps`` times.

``read_reply`` is the only code that knows the shape of a model response, so real google-genai responses and
FakeLLM's stand-ins take the same path. The model's own turn is sent back as it came, which keeps the thought
signatures Gemini 3 requires with function calling. Automatic function calling is off: this loop runs the tools. A run
keeps each tool call as a ``Step`` and each model reply as a ``Turn``, checks the answer for advice wording and for
chunk ids that no search returned, and can save everything as a JSON trace.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from itertools import count
from pathlib import Path
from typing import Any

from .guard import advice_check
from .llm import Gemini, token_counts
from .tools import SUMMARY_CHARS, TOOL_DECLARATIONS, clip, dispatch, summarize

AGENT_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
DEFAULT_MAX_STEPS = 6

SYSTEM_PROMPT = """\
You are a research assistant answering questions about public companies with three tools: a search of their SEC \
filings (10-K and 10-Q reports), their daily closing prices, and recent news headlines. Today is {today}.

Rules:
1. Before calling any tool, work out which facts the question needs and which tool gives each; to compare filings, \
search each one with its ticker and filed date. You can ask for tools at most {max_steps} times, so ask for all the \
calls that do not depend on each other at once.
2. Use only what the tools return in this conversation. Do not use outside knowledge, and do not guess figures, \
dates or prices. Work out periods such as "the past week" from today's date.
3. Give every fact its source in square brackets right after it: a filing passage by its chunk id, e.g. \
[NVDA-10-Q-20260826-2-004]; a price by tool, ticker and dates, e.g. [get_price NVDA 2026-08-26 to 2026-09-02]; a \
headline by tool, source and date, e.g. [get_news Reuters 2026-09-22].
4. For the move over N trading days after a date, call get_price with start set to that date and trading_days N, \
and compare the last close with the first; never count trading days by calendar days. State every move with its two \
closes and dates, e.g. "from 100.00 on 2026-01-02 to 104.00 on 2026-01-09, up 4.0%". A price with a note saying it \
is intraday is not a close: call it the latest price.
5. Report news as what the outlet wrote, not as established fact.
6. If the tools do not give something the question asks for, say what is missing instead of filling the gap.
7. Never give investment advice or recommendations to buy, sell or hold, and do not predict prices.
8. Refer to companies by name instead of copying "we" or "our" from a filing. Answer in short paragraphs or bullet \
points."""

FINAL_TURN_NOTE = """

You have asked for tools {max_steps} times, the limit, so no more tools can be called. Answer now from the tool \
results above, and say which parts of the question they leave unanswered."""

ADVICE_NOTE = (
    "Note: this answer only reports what the filings, prices and news say; nothing in it is a recommendation to buy,"
    " sell or hold."
)

# A chunk id: <ticker>-<form>-<yyyymmdd>-<item, "0" when none>-<nnn>, e.g. AAPL-10-Q-20260501-II-2-001.
CHUNK_ID = re.compile(r"\b[A-Z][A-Z0-9.]{0,9}-\d{1,2}-[A-Z]{1,2}-\d{8}-[0-9A-Z]+(?:-[0-9A-Z]+)*-\d{3}\b")


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class Reply:
    """What the loop needs from a model reply: the tool calls it asks for, in order; its text, without thoughts; its
    content, to send back as the model's turn; and why the model stopped (e.g. "STOP")."""

    calls: list[ToolCall]
    text: str
    content: Any
    finish_reason: str = ""


def read_reply(resp: Any) -> Reply:
    """Read a google-genai ``GenerateContentResponse`` or a ``fake_response``: the parts of the first candidate's
    content give the function calls and the text (parts marked as thoughts are left out)."""
    candidates = getattr(resp, "candidates", None) or []
    first = candidates[0] if candidates else None
    content = getattr(first, "content", None)
    calls: list[ToolCall] = []
    texts: list[str] = []
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        if call is not None:
            calls.append(
                ToolCall(
                    getattr(call, "name", None) or "",
                    dict(getattr(call, "args", None) or {}),
                    getattr(call, "id", None),
                )
            )
        elif isinstance(getattr(part, "text", None), str) and not getattr(part, "thought", None):
            texts.append(part.text)
    reason = getattr(first, "finish_reason", None)
    return Reply(calls, "".join(texts).strip(), content, str(getattr(reason, "value", reason) or ""))


@dataclass
class Step:
    """One tool call. ``result_summary`` says in one line (at most 300 characters) what the tool returned, or the
    error; ``latency_s`` is the time the tool took; ``turn`` is the model reply (counted from 1) that asked for it;
    ``ok`` is False when the call failed (the model was sent the error)."""

    tool: str
    args: dict[str, Any]
    result_summary: str
    latency_s: float
    turn: int = 0
    ok: bool = True


@dataclass
class Turn:
    """One model reply: the model that gave it, its input and output tokens (output includes thinking), its response
    time, how many tool calls it asked for, why the model stopped and the text it wrote (often none alongside tool
    calls)."""

    model: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    tool_calls: int = 0
    finish_reason: str = ""
    text: str = ""


@dataclass
class AgentResult:
    """A finished run. ``final_answer`` is the text the model wrote, joined over its replies (a reply that asks for
    tools may already hold part of the answer), followed by ``ADVICE_NOTE`` when ``advice_hits`` lists advice wording
    found in it. ``steps`` are the tool calls, in order. ``truncated`` is True when the model
    still asked for tools after ``max_steps`` rounds of them: those calls were refused and it was told to answer from
    what it had, without tools. ``unknown_citations`` are chunk ids cited in the answer that no search of this run
    returned. ``usage`` sums the tokens of all turns and counts them (``calls``). ``latency_s`` is the model and tool
    time; ``wall_s`` the elapsed time, including waits between retries. ``trace_path`` is where the JSON trace was
    saved, if it was."""

    question: str
    final_answer: str
    steps: list[Step] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0, "calls": 0})
    latency_s: float = 0.0
    wall_s: float = 0.0
    truncated: bool = False
    advice_hits: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS
    today: str = ""
    started_at: str = ""
    trace_path: str | None = None

    @property
    def models(self) -> list[str]:
        """The models that replied, in order of first reply."""
        return list(dict.fromkeys(t.model for t in self.turns))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rounds_text(n: int) -> str:
    return f"{n} round" + ("" if n == 1 else "s")


def _run_tool(call: ToolCall, tools: Mapping[str, Callable[..., Any]], turn: int) -> tuple[Step, dict[str, Any]]:
    """Run one call: the step to record and the function response for the model ({"output": ...} or {"error": ...})."""
    started = time.perf_counter()
    try:
        response = dispatch(call.name, call.args, tools)
        summary, ok = summarize(call.name, response["output"]), True
    except Exception as e:  # a failed call is reported to the model, which may try another way
        error = f"{type(e).__name__}: {e}"
        response, summary, ok = {"error": clip(error, 500)}, clip(f"error: {error}", SUMMARY_CHARS), False
    return Step(call.name, dict(call.args), summary, time.perf_counter() - started, turn, ok), response


def _function_response(call: ToolCall, response: dict[str, Any]) -> dict[str, Any]:
    reply: dict[str, Any] = {"name": call.name, "response": response}
    if call.id:
        reply["id"] = call.id
    return {"function_response": reply}


def write_trace(result: AgentResult, trace_dir: Path | str) -> Path:
    """Save ``result`` as ``<trace_dir>/<UTC start time>.json`` (with -1, -2, ... when that name is taken) and set its
    ``trace_path``."""
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.fromisoformat(result.started_at).strftime("%Y%m%dT%H%M%SZ")
    for n in count():
        path = directory / (f"{stem}.json" if n == 0 else f"{stem}-{n}.json")
        result.trace_path = str(path)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
            return path
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def run_agent(
    question: str,
    *,
    llm: Gemini,
    tools: Mapping[str, Callable[..., Any]],
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_dir: Path | str | None = None,
    models: Sequence[str] | None = None,
    declarations: Sequence[Mapping[str, Any]] | None = None,
    today: date | None = None,
    on_step: Callable[[Step], Any] | None = None,
) -> AgentResult:
    """Answer ``question`` with the tools in ``tools`` (name to implementation, see ``tools.default_tools``).

    ``models`` are tried in order (default ``AGENT_MODELS``). The tools are declared to the model with
    ``declarations`` (default: the ``TOOL_DECLARATIONS`` of the tools given). A step is one tool call. Every call of a
    reply is run and its result sent back, for at most ``max_steps`` replies that ask for tools (rounds); a call in a
    round beyond that is answered with an error, and the model is asked once more, with tools disabled, to answer from
    what it has (``truncated``). A call rejected for its name or arguments is a step too: the model sees the error and
    can try again. ``on_step`` is called after each step (the command line prints it). With ``trace_dir``, the result
    is saved there as JSON.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    started, started_at = time.perf_counter(), datetime.now(UTC)
    today = today or date.today()
    declared = list(declarations) if declarations is not None else [d for d in TOOL_DECLARATIONS if d["name"] in tools]
    system = SYSTEM_PROMPT.format(today=today.isoformat(), max_steps=max_steps)
    config: dict[str, Any] = {
        "system_instruction": system,
        "tools": [{"function_declarations": declared}],
        "automatic_function_calling": {"disable": True},
    }
    final_config = {
        **config,
        "system_instruction": system + FINAL_TURN_NOTE.format(max_steps=max_steps),
        "tool_config": {"function_calling_config": {"mode": "NONE"}},
    }
    contents: list[Any] = [{"role": "user", "parts": [{"text": question}]}]
    result = AgentResult(
        question, "", max_steps=max_steps, today=today.isoformat(), started_at=started_at.isoformat(timespec="seconds")
    )
    returned_ids: set[str] = set()
    tool_s = 0.0
    rounds = 0
    while True:
        resp, model = llm.generate(
            list(models or AGENT_MODELS),
            label=f"agent:{question}#{len(result.turns) + 1}",
            contents=list(contents),
            config=final_config if result.truncated else config,
        )
        llm.add_usage(model, resp.usage_metadata)
        reply = read_reply(resp)
        tokens_in, tokens_out = token_counts(resp.usage_metadata)
        result.turns.append(
            Turn(model, tokens_in, tokens_out, llm.last_latency, len(reply.calls), reply.finish_reason, reply.text)
        )
        if not reply.calls or result.truncated:  # the answer (tools are disabled once truncated)
            break
        contents.append(reply.content)
        result.truncated = rounds >= max_steps  # this round is one too many: its calls are refused
        rounds += 1
        responses = []
        for call in reply.calls:
            if result.truncated:
                response = {"error": f"not run: tools were already asked for {max_steps} times, the limit"}
            else:
                step, response = _run_tool(call, tools, len(result.turns))
                result.steps.append(step)
                tool_s += step.latency_s
                if step.ok and call.name == "search_filings":
                    returned_ids.update(str(r.get("chunk_id")) for r in response["output"])
                if on_step:
                    on_step(step)
            responses.append(_function_response(call, response))
        contents.append({"role": "user", "parts": responses})

    result.final_answer = "\n\n".join(t.text for t in result.turns if t.text)
    if not result.final_answer:
        last = result.turns[-1].finish_reason or "unknown"
        result.final_answer = (
            f"No answer: the model reached the limit of {rounds_text(max_steps)} of tool calls without writing one."
            if result.truncated
            else f"No answer: the model returned no text (finish reason {last})."
        )
    result.advice_hits = advice_check(result.final_answer)
    if result.advice_hits:
        result.final_answer += "\n\n" + ADVICE_NOTE
    cited = dict.fromkeys(CHUNK_ID.findall(result.final_answer))
    result.unknown_citations = [c for c in cited if c not in returned_ids]
    result.usage = {
        "in": sum(t.tokens_in for t in result.turns),
        "out": sum(t.tokens_out for t in result.turns),
        "calls": len(result.turns),
    }
    result.latency_s = sum(t.latency_s for t in result.turns) + tool_s
    result.wall_s = time.perf_counter() - started
    if trace_dir is not None:
        write_trace(result, trace_dir)
    return result
