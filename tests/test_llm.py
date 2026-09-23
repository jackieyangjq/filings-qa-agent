import pytest
from pydantic import BaseModel

from filings_qa.llm import FakeLLM, Gemini, RecordedLLM, SetupError


class Err(Exception):
    def __init__(self, code):
        super().__init__(f"code {code}")
        self.code = code


class Meta:
    prompt_token_count, candidates_token_count, thoughts_token_count = 100, 20, 5


class Resp:
    text, parsed, usage_metadata = "ok", None, Meta()


def make(script):
    """A Gemini whose ``_call`` pops, per model, the next item of ``script`` ({model: [exception or Resp, ...]}).
    ``calls`` records the models asked and the sleeps between retries, in order."""
    calls = []
    llm = Gemini("k", sleep=lambda s: calls.append(("sleep", s)))

    def fake(model, **kwargs):
        calls.append(model)
        item = script[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm._call = fake  # every request goes through self._call(model, **kwargs)
    return llm, calls


def test_retries_on_503_then_succeeds():
    llm, calls = make({"m1": [Err(503), Resp()]})
    resp, model = llm.generate(["m1", "m2"], contents="x")
    assert model == "m1" and calls == ["m1", ("sleep", 30), "m1"]


def test_quota_error_skips_to_next_model_without_sleep():
    llm, calls = make({"m1": [Err(429)], "m2": [Resp()]})
    _, model = llm.generate(["m1", "m2"], contents="x")
    assert model == "m2" and calls == ["m1", "m2"]


def test_all_models_fail_raises_last_error():
    llm, calls = make({"m1": [Err(429)], "m2": [Err(400)]})
    with pytest.raises(Err) as e:
        llm.generate(["m1", "m2"], contents="x")
    assert e.value.code == 400 and calls == ["m1", "m2"]


def test_ask_records_usage():
    llm, _ = make({"m1": [Resp(), Resp()]})
    assert llm.ask(["m1"], contents="x").text == "ok"
    llm.ask(["m1"], contents="y")
    assert llm.usage["m1"] == [2, 200, 50]  # output counts thinking tokens: 2 x (20 + 5)
    assert llm.usage_text() == "Gemini usage: m1: 2 calls, 200 input and 50 output tokens"


def test_a_busy_model_gets_three_attempts_then_the_next_model():
    llm, calls = make({"m1": [ConnectionError("reset"), Err(503), Err(502)], "m2": [Resp()]})
    _, model = llm.generate(["m1", "m1", "m2"], contents="x")  # a model listed twice is asked once
    assert model == "m2"
    assert calls == ["m1", ("sleep", 30), "m1", ("sleep", 60), "m1", "m2"]


def test_last_latency_times_only_the_request_that_succeeded():
    now = [0.0]
    script = [(1.0, Err(503)), (2.5, Resp()), (0.75, Resp())]  # (seconds the request takes, its outcome)

    def fake(model, **kwargs):
        seconds, outcome = script.pop(0)
        now[0] += seconds
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    llm = Gemini("k", sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
    llm._call = fake
    assert llm.last_latency == 0.0
    llm.generate(["m1"], contents="x")
    assert now[0] == 33.5  # 1 s failed, 30 s wait, 2.5 s answer
    assert llm.last_latency == 2.5
    llm.generate(["m1"], contents="y")
    assert llm.last_latency == 0.75


def test_setup_problems_are_reported_at_once(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SetupError, match="GEMINI_API_KEY"):
        Gemini()
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    llm, calls = make({})

    def missing_sdk(model, **kwargs):
        calls.append(model)
        raise SetupError("google-genai is not installed")

    llm._call = missing_sdk
    with pytest.raises(SetupError):
        llm.generate(["m1", "m2"], contents="x")
    assert calls == ["m1"]  # no retry, no other model
    assert Gemini()._api_key == "from-env"


class Pick(BaseModel):
    ticker: str
    reasons: list[str]


def test_fake_llm_pops_scripted_replies_and_keeps_the_prompts():
    llm = FakeLLM([{"ticker": "ACME", "reasons": ["r"]}, "plain text", Err(429)], tokens=(10, 2), latency=1.5)
    resp, model = llm.generate(["m1", "m2"], label="a", contents="first", config={"response_schema": Pick})
    assert model == "m1" and resp.parsed == Pick(ticker="ACME", reasons=["r"])
    assert resp.text == '{"ticker": "ACME", "reasons": ["r"]}' and llm.last_latency == 1.5
    resp = llm.ask(["m2"], contents="second")
    assert (resp.text, resp.parsed) == ("plain text", None)
    with pytest.raises(Err):
        llm.generate(["m1"], contents="third")
    assert llm.prompts == ["first", "second", "third"]
    assert llm.calls[0]["label"] == "a" and llm.calls[0]["config"] == {"response_schema": Pick}
    assert llm.usage == {"m2": [1, 10, 2]}
    with pytest.raises(AssertionError, match="no scripted reply"):
        llm.generate(["m1"], contents="fourth")


def test_recorded_llm_replays_by_label():
    llm = RecordedLLM({"pick": {"ticker": "ACME", "reasons": []}, "note": "text"}, tokens=(1000, 200))
    resp, model = llm.generate(["m1", "m2"], label="pick", contents="x", config={"response_schema": Pick})
    assert model == "m1" and resp.parsed.ticker == "ACME"
    assert llm.ask(["m1"], label="note").text == "text"
    assert llm.ask(["m1"], label="pick").parsed == {"ticker": "ACME", "reasons": []}  # no schema: the dict itself
    assert llm.usage == {"m1": [2, 2000, 400]}
    with pytest.raises(KeyError):
        llm.generate(["m1"], label="never recorded")
