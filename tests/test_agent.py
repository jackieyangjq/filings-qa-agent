import json
from datetime import date
from pathlib import Path

import pytest

from filings_qa.agent import ADVICE_NOTE, AGENT_MODELS, read_reply, run_agent
from filings_qa.llm import FakeLLM, fake_response

TODAY = date(2026, 9, 23)
GROWTH = "ACME-10-Q-20240802-2-001"
PASSAGE = {"chunk_id": GROWTH, "ticker": "ACME", "form": "10-Q", "filed": "2024-08-02", "item": "2",
           "text": "Data center revenue growth was strong."}
PRICES = [{"date": "2024-08-02", "close": 100.0}, {"date": "2024-08-09", "close": 104.0}]


def call(name, **args):
    """A scripted reply that asks for one tool call."""
    return {"function_call": {"name": name, "args": args}}


SEARCH = call("search_filings", query="data center demand", ticker="ACME")
PRICE = call("get_price", ticker="ACME", start="2024-08-02", end="2024-08-09")


def fake_tools():
    """Tools that return fixed results, and the log of their calls."""
    log = []

    def search_filings(query, ticker=None, form=None, k=6):
        log.append("search_filings")
        return [PASSAGE]

    def get_price(ticker, start, end):
        log.append("get_price")
        return PRICES

    def get_news(ticker, days=7):
        log.append("get_news")
        return []

    return {"search_filings": search_filings, "get_price": get_price, "get_news": get_news}, log


def responses(contents):
    """The function responses of the last user turn of ``contents``: [(name, response), ...]."""
    return [(p["function_response"]["name"], p["function_response"]["response"]) for p in contents[-1]["parts"]]


def test_three_turns_make_two_steps_in_order_and_a_trace(tmp_path):
    question = "How did ACME describe data center demand, and how did the stock move after the filing?"
    answer = (
        f"ACME said data center revenue growth was strong [{GROWTH}]. The stock went from 100.00 on 2024-08-02 to"
        " 104.00 on 2024-08-09, up 4.0% [get_price ACME 2024-08-02 to 2024-08-09]."
    )
    llm = FakeLLM([SEARCH, PRICE, answer], tokens=(2000, 50), latency=0.5)
    tools, log = fake_tools()
    shown = []
    result = run_agent(question, llm=llm, tools=tools, trace_dir=tmp_path / "traces", today=TODAY, on_step=shown.append)

    assert [s.tool for s in result.steps] == ["search_filings", "get_price"] and log == ["search_filings", "get_price"]
    assert result.steps[0].args == {"query": "data center demand", "ticker": "ACME"}
    assert result.steps[0].result_summary == f"1 passage: {GROWTH}"
    assert result.steps[1].result_summary == "2 closes: 2024-08-02 100.00 to 2024-08-09 104.00 (+4.0%)"
    assert [(s.turn, s.ok) for s in result.steps] == [(1, True), (2, True)] and shown == result.steps
    assert result.final_answer == answer and not result.truncated
    assert result.advice_hits == [] and result.unknown_citations == []
    assert result.usage == {"in": 6000, "out": 150, "calls": 3}
    assert llm.usage == {"gemini-3.5-flash-lite": [3, 6000, 150]} and result.models == list(AGENT_MODELS)
    assert [t.tool_calls for t in result.turns] == [1, 1, 0] and result.turns[0].finish_reason == "STOP"
    assert 1.5 <= result.latency_s < 2.0  # three replies of 0.5 s, plus the tools

    trace = Path(result.trace_path)
    assert trace.parent == tmp_path / "traces" and trace.name.startswith(result.started_at[:4])
    saved = json.loads(trace.read_text())
    assert saved["final_answer"] == answer and saved["trace_path"] == result.trace_path
    assert [(s["tool"], s["result_summary"]) for s in saved["steps"]] == [
        (s.tool, s.result_summary) for s in result.steps
    ]
    assert saved["today"] == "2026-09-23" and saved["max_steps"] == 6

    config = llm.calls[0]["config"]
    assert llm.calls[0]["models"] == list(AGENT_MODELS)
    assert config["automatic_function_calling"] == {"disable": True} and "tool_config" not in config
    assert [d["name"] for d in config["tools"][0]["function_declarations"]] == list(tools)
    assert "Today is 2026-09-23." in config["system_instruction"]
    assert "ask for tools at most 6 times" in config["system_instruction"]
    assert "buy, sell or hold" in config["system_instruction"]

    # each request carries the conversation so far: question, model turn, tool results, ...
    first, second, third = llm.prompts
    assert first == [{"role": "user", "parts": [{"text": question}]}]
    assert second[1].role == "model" and second[1].parts[0].function_call.name == "search_filings"  # as it came
    assert responses(second) == [("search_filings", {"output": [PASSAGE]})]
    assert len(third) == 5 and responses(third) == [("get_price", {"output": PRICES})]


def test_a_model_that_keeps_calling_tools_is_stopped_after_max_steps_rounds():
    llm = FakeLLM([PRICE, PRICE, PRICE, "From the two price lookups: the stock rose 4.0%."])
    tools, log = fake_tools()
    result = run_agent("q", llm=llm, tools=tools, max_steps=2, today=TODAY)
    assert len(result.steps) == 2 and log == ["get_price", "get_price"]  # the third call was not run
    assert result.truncated and result.final_answer == "From the two price lookups: the stock rose 4.0%."
    assert result.usage["calls"] == 4 and result.trace_path is None
    ((name, refused),) = responses(llm.prompts[3])
    assert name == "get_price" and "asked for 2 times, the limit" in refused["error"]
    assert "tool_config" not in llm.calls[2]["config"]  # the third reply could still have been the answer
    last = llm.calls[3]["config"]
    assert last["tool_config"] == {"function_calling_config": {"mode": "NONE"}}
    assert "You have asked for tools 2 times, the limit" in last["system_instruction"]


def test_using_exactly_max_steps_then_answering_is_not_truncation():
    llm = FakeLLM([SEARCH, PRICE, "Answer."])
    result = run_agent("q", llm=llm, tools=fake_tools()[0], max_steps=2, today=TODAY)
    assert len(result.steps) == 2 and not result.truncated and result.final_answer == "Answer."


def test_calls_asked_for_together_are_one_round_and_run_in_order():
    news = {"name": "get_news", "args": {"ticker": "ACME"}}
    together = {"function_calls": [SEARCH["function_call"], PRICE["function_call"], news]}
    llm = FakeLLM([together, together, call("get_news", ticker="ACME")])  # a fake ignores the disabled tools
    tools, log = fake_tools()
    result = run_agent("q", llm=llm, tools=tools, max_steps=1, today=TODAY)
    assert [(s.tool, s.turn) for s in result.steps] == [("search_filings", 1), ("get_price", 1), ("get_news", 1)]
    sent = responses(llm.prompts[1])  # one user turn answers every call, in order
    assert [name for name, response in sent] == ["search_filings", "get_price", "get_news"]
    assert all("output" in response for _, response in sent)
    refused = responses(llm.prompts[2])  # the second round is over the limit of one: none of it runs
    assert [name for name, _ in refused] == ["search_filings", "get_price", "get_news"]
    assert all("the limit" in response["error"] for _, response in refused)
    assert log == ["search_filings", "get_price", "get_news"] and result.truncated
    assert result.final_answer.startswith("No answer: the model reached the limit of 1 round of tool calls")


def test_text_written_alongside_tool_calls_stays_in_the_answer():
    """A reply may give most of the answer and ask for one more tool; the last reply then only adds to it."""
    first, rest = f"Demand was strong [{GROWTH}].", "The stock rose 4.0% [get_price ACME 2024-08-02 to 2024-08-09]."
    llm = FakeLLM([SEARCH, {"function_call": PRICE["function_call"], "text": first}, rest])
    result = run_agent("q", llm=llm, tools=fake_tools()[0], today=TODAY)
    assert result.final_answer == first + "\n\n" + rest
    assert [t.text for t in result.turns] == ["", first, rest]
    assert [s.tool for s in result.steps] == ["search_filings", "get_price"] and result.unknown_citations == []


def test_failed_calls_are_sent_back_to_the_model_and_recorded():
    def unreachable(ticker, days=7):
        raise ConnectionError("feed unreachable")

    tools = {**fake_tools()[0], "get_news": unreachable}
    asked = [{"name": "get_news", "args": {"ticker": "ACME"}}, {"name": "get_weather", "args": {}},
             {"name": "get_price", "args": {"ticker": "ACME"}}]
    llm = FakeLLM([{"function_calls": asked}, "The news feed could not be read."])
    result = run_agent("q", llm=llm, tools=tools, today=TODAY)
    assert [(s.tool, s.ok) for s in result.steps] == [("get_news", False), ("get_weather", False), ("get_price", False)]
    assert result.steps[0].result_summary == "error: ConnectionError: feed unreachable"
    errors = [response["error"] for _, response in responses(llm.prompts[1])]
    assert errors[0] == "ConnectionError: feed unreachable"
    assert "unknown tool 'get_weather'" in errors[1] and "missing start" in errors[2]
    assert result.final_answer == "The news feed could not be read." and not result.truncated


def test_advice_wording_gets_a_note_and_is_flagged():
    llm = FakeLLM(["Demand is strong, so you should buy ACME."])
    result = run_agent("Should I buy ACME?", llm=llm, tools=fake_tools()[0], today=TODAY)
    assert result.advice_hits == ["should buy"]
    assert result.final_answer == "Demand is strong, so you should buy ACME.\n\n" + ADVICE_NOTE
    assert result.steps == [] and not result.truncated


def test_chunk_ids_that_no_search_returned_are_flagged():
    other = "ACME-10-K-20240216-7-001"
    llm = FakeLLM([SEARCH, f"Demand grew [{GROWTH}] and margins rose [{other}, AAPL-10-Q-20260501-II-2-001]."])
    result = run_agent("q", llm=llm, tools=fake_tools()[0], today=TODAY)
    assert result.unknown_citations == [other, "AAPL-10-Q-20260501-II-2-001"]


def test_an_empty_reply_gives_a_stated_non_answer():
    llm = FakeLLM([""])
    result = run_agent("q", llm=llm, tools=fake_tools()[0], today=TODAY)
    assert result.final_answer == "No answer: the model returned no text (finish reason STOP)."
    with pytest.raises(ValueError, match="at least 1"):
        run_agent("q", llm=llm, tools=fake_tools()[0], max_steps=0)


def test_real_and_fake_responses_are_read_alike():
    genai_types = pytest.importorskip("google.genai.types")
    price_args = {"ticker": "NVDA", "start": "2026-08-26", "end": "2026-09-02"}
    parts = [
        genai_types.Part(text="Which tools do I need?", thought=True),
        genai_types.Part(
            function_call=genai_types.FunctionCall(name="get_price", args=price_args, id="c1"), thought_signature=b"s"
        ),
        genai_types.Part(function_call=genai_types.FunctionCall(name="get_news", args={"ticker": "NVDA"})),
    ]
    content = genai_types.Content(role="model", parts=parts)
    real = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=content, finish_reason=genai_types.FinishReason.STOP)]
    )
    fake = fake_response(
        {"function_calls": [{"name": "get_price", "args": price_args, "id": "c1"},
                            {"name": "get_news", "args": {"ticker": "NVDA"}}]}
    )
    real_reply, fake_reply = read_reply(real), read_reply(fake)
    assert real_reply.calls == fake_reply.calls and [c.id for c in real_reply.calls] == ["c1", None]
    assert (real_reply.text, real_reply.finish_reason) == (fake_reply.text, fake_reply.finish_reason) == ("", "STOP")
    assert real_reply.content is content  # sent back as it came, thought signature included

    answer = genai_types.Content(
        role="model", parts=[genai_types.Part(text="thinking...", thought=True), genai_types.Part(text="The answer.")]
    )
    real = genai_types.GenerateContentResponse(candidates=[genai_types.Candidate(content=answer)])
    assert read_reply(real).text == read_reply(fake_response("The answer.")).text == "The answer."
    blocked = read_reply(genai_types.GenerateContentResponse(candidates=[]))
    assert (blocked.calls, blocked.text) == ([], "")


def test_requests_are_valid_for_the_sdk():
    """What the loop sends (configs, and the turns it writes itself) passes the SDK's own validation."""
    genai_types = pytest.importorskip("google.genai.types")
    llm = FakeLLM([SEARCH, {"function_calls": [PRICE["function_call"], PRICE["function_call"]]}, "Answer."])
    run_agent("q", llm=llm, tools=fake_tools()[0], max_steps=1, today=TODAY)  # the last request is without tools
    for sent in llm.calls:
        config = genai_types.GenerateContentConfig.model_validate(sent["config"])
        assert config.automatic_function_calling.disable and config.tools[0].function_declarations
        for turn in sent["contents"]:
            if isinstance(turn, dict):  # the model's own turns are sent back as they came
                genai_types.Content.model_validate(turn)
    assert config.tool_config.function_calling_config.mode == genai_types.FunctionCallingConfigMode.NONE
