import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from bench.adapters.sse import iter_sse
from bench.config import OPENROUTER_CHAT_URL, ModelSpec, OpenRouterEnv, openrouter_api_key
from bench.llm import ChatModel, Completion, LLMAnswerRefused, LLMFormatError, LLMQuotaExhausted, LLMRequestRejected, LLMTimeoutError, LLMTransportError, OpenRouterApi

FAKE_KEY = "openrouter-test-key-not-real"
CLAUDE = ModelSpec(model_id="anthropic/claude-opus-5.5", transport="openrouter_api", provider=["Anthropic"], effort="medium", max_tokens=16000)
GEMINI = ModelSpec(model_id="google/gemini-4-argon", transport="openrouter_api", provider=["Google AI Studio"], effort="medium", temperature=1.0, seed=7)
USAGE = {"prompt_tokens": 120, "completion_tokens": 300, "total_tokens": 420, "cost": 0.0123, "completion_tokens_details": {"reasoning_tokens": 250}}


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str


def sse(*events: object, done: bool = True) -> bytes:
    frames = [": OPENROUTER PROCESSING\n\n"] + [f"data: {json.dumps(event)}\n\n" for event in events]
    if done:
        frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def delta(*, content: str | None = None, reasoning: str | None = None, finish_reason: str | None = None, usage: dict[str, object] | None = None) -> dict[str, object]:
    part: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        part["reasoning"] = reasoning
    event: dict[str, object] = {"id": "gen-abc123", "provider": "Anthropic", "model": "anthropic/claude-opus-5.5", "choices": [{"index": 0, "delta": part, "finish_reason": finish_reason}]}
    if usage is not None:
        event["usage"] = usage
    return event


def answer_stream(*, text: str = "White should castle.", finish_reason: str = "stop") -> bytes:
    return sse(delta(reasoning="thinking"), delta(content=text[:5]), delta(content=text[5:]), delta(content="", finish_reason=finish_reason, usage=USAGE))


Reply = Callable[[httpx.Request], httpx.Response]


def stream_reply(body: bytes) -> Reply:
    return lambda request: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def status_reply(status: int, *, message: str = "upstream says no", headers: dict[str, str] | None = None) -> Reply:
    return lambda request: httpx.Response(status, json={"error": {"code": status, "message": message}}, headers=headers or {})


class FakeOpenRouter:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.replies[min(len(self.requests), len(self.replies)) - 1](request)

    def model(self, *, spec: ModelSpec = CLAUDE, max_attempts: int = 4, timeout_s: float = 60.0) -> OpenRouterApi:
        return OpenRouterApi(spec=spec, api_key=SecretStr(FAKE_KEY), timeout_s=timeout_s, http_transport=httpx.MockTransport(self.handler), max_attempts=max_attempts, backoff_s=1.0, sleep=self.sleeps.append)

    def body(self, index: int = 0) -> dict[str, object]:
        return json.loads(self.requests[index].content)


def test_a_text_call_pins_the_provider_names_the_effort_and_asks_for_usage() -> None:
    api = FakeOpenRouter(stream_reply(answer_stream()))
    completion = api.model().complete_text(system="You coach chess.", user="What now?")
    assert completion.text == "White should castle."
    request = api.requests[0]
    assert request.method == "POST" and str(request.url) == OPENROUTER_CHAT_URL and request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert api.body() == {
        "model": "anthropic/claude-opus-5.5",
        "messages": [{"role": "system", "content": "You coach chess."}, {"role": "user", "content": "What now?"}],
        "stream": True,
        "provider": {"order": ["Anthropic"], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"},
        "reasoning": {"effort": "medium"},
        "usage": {"include": True},
        "max_tokens": 16000,
    }


def test_temperature_and_seed_are_sent_only_when_set() -> None:
    api = FakeOpenRouter(stream_reply(answer_stream()))
    api.model(spec=GEMINI).complete_text(system="s", user="u")
    body = api.body()
    assert body["temperature"] == 1.0 and body["seed"] == 7 and "max_tokens" not in body
    assert body["provider"] == {"order": ["Google AI Studio"], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}


def test_every_call_records_generation_provider_tokens_cost_finish_and_timing() -> None:
    model = FakeOpenRouter(stream_reply(answer_stream())).model()
    completion = model.complete_text(system="s", user="u")
    assert model.last_completion == completion
    assert (completion.generation_id, completion.provider, completion.finish_reason) == ("gen-abc123", "Anthropic", "stop")
    assert (completion.input_tokens, completion.output_tokens, completion.reasoning_tokens, completion.cost_usd) == (120, 300, 250, 0.0123)
    assert completion.transport == "openrouter_api" and completion.model_id == "anthropic/claude-opus-5.5" and completion.tool_calls == 0
    assert completion.first_token_s is not None and 0 <= completion.first_token_s <= completion.latency_s


def test_the_served_model_and_provider_are_recorded_apart_from_the_requested_alias() -> None:
    served = "anthropic/claude-opus-5.5-20261001"
    events = [{**delta(content="Castle."), "model": served}, {**delta(content="", finish_reason="stop", usage=USAGE), "model": served}]
    model = FakeOpenRouter(stream_reply(sse(*events))).model()
    completion = model.complete_text(system="s", user="u")
    assert (completion.model_id, completion.served_model, completion.provider) == ("anthropic/claude-opus-5.5", served, "Anthropic")
    assert model.usage.served_sorted() == [f"{served} via Anthropic"]


def test_stream_text_stamps_the_first_content_token_not_the_end() -> None:
    model = FakeOpenRouter(stream_reply(answer_stream())).model()
    chunks = list(model.stream_text(system="s", user="u"))
    assert [chunk.delta for chunk in chunks] == ["White should castle."]
    completion = model.last_completion
    assert completion is not None and completion.first_token_s is not None and chunks[0].at is not None


def test_a_json_call_asks_for_a_strict_json_schema_and_validates_the_answer() -> None:
    api = FakeOpenRouter(stream_reply(answer_stream(text='{"verdict": "good"}')))
    assert api.model().complete_json(system="s", user="u", schema=Verdict) == Verdict(verdict="good")
    assert api.body()["response_format"] == {"type": "json_schema", "json_schema": {"name": "Verdict", "strict": True, "schema": Verdict.model_json_schema()}}


def test_a_json_call_that_answers_garbage_is_a_format_error() -> None:
    api = FakeOpenRouter(stream_reply(answer_stream(text="not json at all")))
    with pytest.raises(LLMFormatError, match="Verdict"):
        api.model().complete_json(system="s", user="u", schema=Verdict)


def test_a_402_means_the_credit_is_exhausted_and_is_never_retried() -> None:
    api = FakeOpenRouter(status_reply(402, message="Insufficient credits"))
    with pytest.raises(LLMQuotaExhausted, match="credit") as caught:
        api.model().complete_text(system="s", user="u")
    assert caught.value.resets_at is None and len(api.requests) == 1


def test_a_429_waits_as_long_as_retry_after_says_then_answers() -> None:
    api = FakeOpenRouter(status_reply(429, headers={"retry-after": "3"}), stream_reply(answer_stream()))
    assert api.model().complete_text(system="s", user="u").text == "White should castle."
    assert api.sleeps == [3.0] and len(api.requests) == 2


def test_server_errors_back_off_exponentially_and_give_up_after_the_last_attempt() -> None:
    api = FakeOpenRouter(status_reply(502), status_reply(503), status_reply(500))
    with pytest.raises(LLMTransportError, match="HTTP 500.*3 attempts"):
        api.model(max_attempts=3).complete_text(system="s", user="u")
    assert api.sleeps == [1.0, 2.0] and len(api.requests) == 3


def test_a_retry_after_beyond_the_time_limit_gives_up_instead_of_waiting() -> None:
    api = FakeOpenRouter(status_reply(429, headers={"retry-after": "600"}))
    with pytest.raises(LLMTransportError, match="Retry-After|time limit"):
        api.model(timeout_s=30).complete_text(system="s", user="u")
    assert api.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_request_the_api_rejects_stops_the_run_and_never_shows_the_key(status: int) -> None:
    api = FakeOpenRouter(status_reply(status, message="No endpoints found that support the requested parameters"))
    with pytest.raises(LLMRequestRejected, match=f"HTTP {status}") as caught:
        api.model().complete_text(system="s", user="u")
    assert FAKE_KEY not in str(caught.value) and len(api.requests) == 1


def test_an_error_object_inside_a_200_body_names_its_code_and_message() -> None:
    api = FakeOpenRouter(lambda request: httpx.Response(200, json={"error": {"code": 400, "message": "Provider returned error"}}))
    with pytest.raises(LLMTransportError, match="400.*Provider returned error"):
        api.model().complete_text(system="s", user="u")


def test_a_mid_stream_server_error_is_retried() -> None:
    broken = sse(delta(content="White"), {"id": "gen-abc123", "error": {"code": 502, "message": "provider disconnected"}, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}, done=False)
    api = FakeOpenRouter(stream_reply(broken), stream_reply(answer_stream()))
    assert api.model().complete_text(system="s", user="u").text == "White should castle."
    assert api.sleeps == [1.0]


def test_reasoning_without_any_content_is_an_empty_answer() -> None:
    api = FakeOpenRouter(stream_reply(sse(delta(reasoning="long thoughts"), delta(content=None, finish_reason="stop", usage=USAGE))))
    with pytest.raises(LLMTransportError, match="empty_answer.*reasoning"):
        api.model().complete_text(system="s", user="u")


def test_an_answer_cut_off_at_max_tokens_is_refused_as_truncated_and_not_retried() -> None:
    api = FakeOpenRouter(stream_reply(answer_stream(finish_reason="length")))
    with pytest.raises(LLMAnswerRefused, match="truncated.*16000"):
        api.model().complete_json(system="s", user="u", schema=Verdict)
    assert len(api.requests) == 1


def test_a_stalled_stream_is_a_timeout_that_keeps_the_partial_text() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(LLMTimeoutError, match="timed out"):
        FakeOpenRouter(slow).model().complete_text(system="s", user="u")


def test_for_spec_builds_the_api_model_with_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    model = ChatModel.for_spec(spec=CLAUDE, timeout_s=30)
    assert isinstance(model, OpenRouterApi) and model.transport == "openrouter_api" and model.vendor == "anthropic"


def test_for_spec_refuses_tools_on_the_api_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    with pytest.raises(ValueError, match="openrouter_api.*tool loop"):
        ChatModel.for_spec(spec=CLAUDE.model_copy(update={"tools": True}), timeout_s=30)


def test_a_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY in .env"):
        openrouter_api_key(env_file=tmp_path / "missing.env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "  ")
    assert OpenRouterEnv(_env_file=None).openrouter_api_key is None  # type: ignore[call-arg]


def test_sse_skips_the_done_marker() -> None:
    lines = iter(answer_stream().decode().splitlines())
    assert [event["id"] for event in iter_sse(lines=lines)] == ["gen-abc123"] * 4


def test_the_completion_time_is_never_stamped_as_a_first_token() -> None:
    class Plain(ChatModel):
        def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
            return Completion(text="ok", model_id="m", transport="cursor_cli", latency_s=1.0, input_tokens=None, output_tokens=None, first_token_s=None)

    chunks = list(Plain(model_id="composer-2", vendor="cursor", transport="cursor_cli").stream_text(system="s", user="u"))
    assert [chunk.at for chunk in chunks] == [None]


def test_a_known_first_token_time_is_stamped_from_the_call_start() -> None:
    class Timed(ChatModel):
        def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
            return Completion(text="ok", model_id="m", transport="claude_cli", latency_s=9.0, input_tokens=None, output_tokens=None, first_token_s=5.0)

    before = datetime.now(UTC)
    [chunk] = list(Timed(model_id="claude-opus-5-5", vendor="anthropic", transport="claude_cli").stream_text(system="s", user="u"))
    assert chunk.at is not None and before + timedelta(seconds=5) <= chunk.at <= datetime.now(UTC) + timedelta(seconds=5)


class Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")
    side: str | None = None
    move: str


class WithOptional(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Nested]


def test_a_strict_schema_requires_every_property_and_drops_defaults_so_optional_fields_become_required_nullable() -> None:
    answer = '{"items": [{"side": null, "move": "Nf3"}]}'
    api = FakeOpenRouter(stream_reply(answer_stream(text=answer)))
    assert api.model().complete_json(system="s", user="u", schema=WithOptional) == WithOptional(items=[Nested(move="Nf3")])
    nested = api.body()["response_format"]["json_schema"]["schema"]["$defs"]["Nested"]  # type: ignore[index]
    assert sorted(nested["required"]) == ["move", "side"]
    assert "default" not in nested["properties"]["side"]


def test_every_judge_and_extractor_schema_is_strict_compatible() -> None:
    from bench.grading.claims import ClaimList
    from bench.llm import strict_json_schema

    def objects(node: object) -> list[dict[str, object]]:
        if isinstance(node, dict):
            own = [node] if "properties" in node else []
            return own + [o for value in node.values() for o in objects(value)]
        if isinstance(node, list):
            return [o for value in node for o in objects(value)]
        return []

    for found in objects(strict_json_schema(schema=ClaimList.model_json_schema())):
        assert set(found["required"]) == set(found["properties"])  # type: ignore[arg-type]
        assert found["additionalProperties"] is False
