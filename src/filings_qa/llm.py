"""Gemini access: try the given models in order, retry while the service is busy, and keep token usage and latency.

Every request goes through ``Gemini._call(model, **kwargs)``, which tests replace. Two stand-ins need no network or
key: ``FakeLLM`` pops scripted replies in order (tests) and ``RecordedLLM`` replays replies recorded per call label
(the offline demo). The API key comes from the ``GEMINI_API_KEY`` environment variable unless one is passed.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

API_KEY_ENV = "GEMINI_API_KEY"
ATTEMPTS_PER_MODEL = 3  # the first request and two retries, after 30 s and 60 s
RETRY_BASE_S = 30
RETRY_CODES = (None, 500, 502, 503, 504)  # None: no HTTP status, e.g. a dropped connection

logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # the SDK's notes about response parts


class SetupError(RuntimeError):
    """Gemini cannot be used at all (no API key, SDK not installed). Never retried, and no other model is tried."""


def short_error(e: BaseException) -> str:
    code = getattr(e, "code", None)
    if code == 429:
        return "quota or rate limit exceeded (HTTP 429)"
    if code in (500, 502, 503, 504):
        return f"Gemini is busy (HTTP {code})"
    if code:
        return f"request rejected (HTTP {code}): {getattr(e, 'message', '') or str(e)[:150]}"
    return f"{type(e).__name__}: {str(e)[:150]}"


def with_retry(
    fn: Callable[[], Any], attempts: int = ATTEMPTS_PER_MODEL, *, sleep: Callable[[float], Any] = time.sleep
):
    """``fn()``, called again after 30 s, 60 s, ... while it fails with a busy server (HTTP 5xx) or no HTTP status,
    at most ``attempts`` times in all. Other errors (quota 429, bad request 400, ...) are raised at once."""
    for i in range(attempts):
        try:
            return fn()
        except SetupError:
            raise
        except Exception as e:
            code = getattr(e, "code", None)
            if i == attempts - 1 or code not in RETRY_CODES:
                raise
            wait = RETRY_BASE_S * 2**i
            print(f"    {short_error(e)}; retrying in {wait} s", file=sys.stderr, flush=True)
            sleep(wait)
    raise ValueError("attempts must be at least 1")


def model_list(models: str | Sequence[str]) -> list[str]:
    """``models`` as a list without repeats; a single name may be given as a str."""
    return [models] if isinstance(models, str) else list(dict.fromkeys(models))


def token_counts(meta: Any) -> tuple[int, int]:
    """(input, output) tokens of a response's ``usage_metadata``; output includes thinking tokens, which are billed
    as output."""
    if meta is None:
        return 0, 0
    out = (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
    return getattr(meta, "prompt_token_count", 0) or 0, out


class Gemini:
    """``generate`` tries each model in turn and returns (response, model used). ``usage`` counts, per model, the
    calls and input and output tokens recorded by ``ask``/``add_usage``. ``last_latency`` is how long the request
    that produced the last response took, in seconds: failed attempts and the waits between retries are left out, so
    it measures the model, not the retries."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        sleep: Callable[[float], Any] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._api_key = api_key or os.environ.get(API_KEY_ENV, "").strip()
        if not self._api_key:
            raise SetupError(f"{API_KEY_ENV} is not set; get a key at https://aistudio.google.com/apikey")
        self._client: Any = None
        self._sleep = sleep
        self._clock = clock
        self.usage: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # model: [calls, in, out]
        self.last_latency = 0.0

    def _call(self, model: str, **kwargs: Any) -> Any:
        """The only place that talks to Gemini; tests replace it."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise SetupError("google-genai is not installed: pip install 'filings-qa-agent[llm]'") from e
            self._client = genai.Client(api_key=self._api_key)
        return self._client.models.generate_content(model=model, **kwargs)

    def _timed_call(self, model: str, **kwargs: Any) -> Any:
        start = self._clock()
        resp = self._call(model, **kwargs)
        self.last_latency = self._clock() - start
        return resp

    def generate(self, models: Sequence[str], *, label: str = "", **kwargs: Any) -> tuple[Any, str]:
        """Ask each model in turn until one answers: up to ``ATTEMPTS_PER_MODEL`` attempts while the service is busy,
        then the next model; a quota or request error moves on at once. Returns (response, model used) or raises the
        last error. ``kwargs`` go to ``generate_content`` (``contents``, ``config``). ``label`` names the call (for
        example ``answer:<question>``); only ``RecordedLLM`` uses it."""
        last_error: Exception | None = None
        for model in model_list(models):
            try:
                call = functools.partial(self._timed_call, model, **kwargs)
                return with_retry(call, ATTEMPTS_PER_MODEL, sleep=self._sleep), model
            except SetupError:
                raise
            except Exception as e:
                last_error = e
                print(f"    model {model} failed: {short_error(e)}", file=sys.stderr, flush=True)
        if last_error is None:
            raise ValueError("no model given")
        raise last_error

    def ask(self, models: Sequence[str], *, label: str = "", **kwargs: Any) -> Any:
        """``generate``, recording the usage; returns only the response."""
        resp, model = self.generate(models, label=label, **kwargs)
        self.add_usage(model, resp.usage_metadata)
        return resp

    def add_usage(self, model: str, meta: Any) -> None:
        tokens_in, tokens_out = token_counts(meta)
        u = self.usage[model]
        u[0] += 1
        u[1] += tokens_in
        u[2] += tokens_out

    def usage_text(self) -> str:
        parts = [f"{m}: {n} calls, {i:,} input and {o:,} output tokens" for m, (n, i, o) in self.usage.items()]
        return "Gemini usage: " + "; ".join(parts) if parts else ""


def response_schema(config: Any) -> Any:
    """The ``response_schema`` of a ``generate_content`` config, given as a dict or as a GenerateContentConfig."""
    if isinstance(config, dict):
        return config.get("response_schema")
    return getattr(config, "response_schema", None)


def fake_response(value: Any, config: Any = None, tokens: tuple[int, int] = (1000, 200)) -> SimpleNamespace:
    """A stand-in for a Gemini response. A dict is the JSON the model returned: ``parsed`` is that dict validated by
    the config's Pydantic ``response_schema`` (the dict itself without one) and ``text`` its JSON. A str is plain
    text, with ``parsed`` None. ``usage_metadata`` reports ``tokens`` = (input, output)."""
    if isinstance(value, dict):
        schema = response_schema(config)
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        parsed, text = (schema.model_validate(value) if is_model else value), json.dumps(value)
    else:
        parsed, text = None, str(value)
    meta = SimpleNamespace(prompt_token_count=tokens[0], candidates_token_count=tokens[1], thoughts_token_count=0)
    return SimpleNamespace(parsed=parsed, text=text, usage_metadata=meta)


class FakeLLM(Gemini):
    """Scripted stand-in for tests, without network or key. Each call pops the next item of ``script``: a dict or str
    becomes the response (see ``fake_response``), an exception is raised. ``prompts`` keeps the ``contents`` of every
    call and ``calls`` all its arguments. Every call reports ``tokens`` and takes ``latency`` seconds."""

    def __init__(self, script: Sequence[Any], *, tokens: tuple[int, int] = (1000, 200), latency: float = 0.0):
        self.script = list(script)
        self.prompts: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self._tokens = tokens
        self._latency = latency
        self.usage = defaultdict(lambda: [0, 0, 0])
        self.last_latency = 0.0

    def generate(self, models: Sequence[str], *, label: str = "", **kwargs: Any) -> tuple[Any, str]:
        models = model_list(models)
        self.calls.append({"models": models, "label": label, **kwargs})
        self.prompts.append(kwargs.get("contents"))
        if not self.script:
            raise AssertionError(f"FakeLLM has no scripted reply left for call {len(self.calls)}")
        value = self.script.pop(0)
        if isinstance(value, BaseException):
            raise value
        self.last_latency = self._latency
        return fake_response(value, kwargs.get("config"), self._tokens), models[0]


class RecordedLLM(Gemini):
    """Replays replies recorded per call label, without network or key (the offline demo). ``responses`` maps a
    label to a reply, read as in ``fake_response``; a label that was not recorded raises KeyError. Every call reports
    ``tokens`` and takes ``latency`` seconds, so usage lines still appear."""

    def __init__(self, responses: dict[str, Any], *, tokens: tuple[int, int] = (1000, 200), latency: float = 0.0):
        self._responses = responses
        self._tokens = tokens
        self._latency = latency
        self.usage = defaultdict(lambda: [0, 0, 0])
        self.last_latency = 0.0

    def generate(self, models: Sequence[str], *, label: str = "", **kwargs: Any) -> tuple[Any, str]:
        value = self._responses[label]
        self.last_latency = self._latency
        return fake_response(value, kwargs.get("config"), self._tokens), model_list(models)[0]
