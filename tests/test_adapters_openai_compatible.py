import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import SecretStr, ValidationError

from bench.adapters.openai_compatible import OpenAICompatibleAdapter
from bench.adapters.prompts import NEUTRAL_SYSTEM_PROMPT, render_user_message
from bench.adapters.registry import get_adapter, known_entrants
from bench.config import HttpEntrant, http_entrant_api_key, http_entrant_key_variable, load_bench, load_models, load_third_party
from bench.io_yaml import read_task
from bench.llm import LLMTransportError
from bench.runner import run_entrant
from bench.runs import read_answers

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"
POS_TASK = read_task(path=FIXTURES / "ap_9001.yaml")
GR_TASK = read_task(path=FIXTURES / "gr_9001.yaml")
FAKE_KEY = "test_key_not_real_0123456789"
SECRET_BODY = "upstream secret detail 42"


def _spec(**overrides: object) -> HttpEntrant:
    fields: dict[str, object] = {
        "label": "Acme Coach 2",
        "provider": "Acme",
        "declared_family": "undisclosed",
        "model_id": "acme/coach-2-large",
        "url": "https://api.acme.example/v1/chat/completions",
        "auth": "bearer",
        "service_tools": [],
        "affiliated_with_maintainer": False,
    }
    return HttpEntrant.model_validate({**fields, **overrides})


class FakeEndpoint:
    def __init__(self, *, status: int = 200, body: object = None, raw: bytes | None = None, failure: Exception | None = None) -> None:
        self.status = status
        self.body = body
        self.raw = raw
        self.failure = failure
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        if self.raw is not None:
            return httpx.Response(self.status, content=self.raw)
        return httpx.Response(self.status, json=self.body)

    def adapter(self, *, spec: HttpEntrant | None = None, key: str | None = FAKE_KEY) -> OpenAICompatibleAdapter:
        return OpenAICompatibleAdapter(name="acme_coach", spec=spec or _spec(), api_key=SecretStr(key) if key else None, transport=httpx.MockTransport(self.handler))


def _completion(*, content: object = "White should castle.", finish_reason: str = "stop", tool_calls: object = None) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"id": "x", "model": "acme/coach-2-large-0101", "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}], "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def test_answer_sends_the_neutral_prompt_and_the_shared_task_message_without_tools() -> None:
    endpoint = FakeEndpoint(body=_completion())
    record = endpoint.adapter().answer(task=POS_TASK, run_index=1, timeout_s=30)
    assert record.status == "ok" and record.error is None and record.text == "White should castle."
    request = endpoint.requests[0]
    assert request.method == "POST" and str(request.url) == "https://api.acme.example/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body == {
        "model": "acme/coach-2-large",
        "messages": [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}, {"role": "user", "content": render_user_message(task=POS_TASK)}],
        "stream": False,
    }
    assert "Audience:" in body["messages"][1]["content"]
    assert record.task_id == "ap_9001" and record.entrant == "acme_coach" and record.run_index == 1
    assert record.first_token_at is None and record.started_at <= record.finished_at
    assert record.raw == {
        "model_id": "acme/coach-2-large",
        "transport": "openai_compatible_http",
        "response_model": "acme/coach-2-large-0101",
        "finish_reason": "stop",
        "input_tokens": 120,
        "output_tokens": 30,
        "streamed": False,
        "timing_measured": True,
    }


def test_game_review_tasks_send_the_same_message_as_stock_entrants() -> None:
    endpoint = FakeEndpoint(body=_completion(content="Move 12 was the turning point."))
    endpoint.adapter().answer(task=GR_TASK, run_index=0, timeout_s=30)
    assert json.loads(endpoint.requests[0].content)["messages"][1]["content"] == render_user_message(task=GR_TASK)


def test_no_auth_sends_no_authorization_header_to_a_local_server() -> None:
    endpoint = FakeEndpoint(body=_completion())
    spec = _spec(url="http://127.0.0.1:8000/v1/chat/completions", auth="none")
    endpoint.adapter(spec=spec, key=None).answer(task=POS_TASK, run_index=0, timeout_s=30)
    assert "authorization" not in endpoint.requests[0].headers


def test_bearer_auth_without_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match="BENCH_HTTP_ACME_COACH_API_KEY"):
        FakeEndpoint(body=_completion()).adapter(key=None)


def test_card_declares_api_access_the_declared_family_and_only_service_tools() -> None:
    card = FakeEndpoint().adapter(spec=_spec(service_tools=["engine"], declared_family="google")).card()
    assert card.access_method == "api" and card.declared_family == "google" and card.version == "acme/coach-2-large" and card.provider == "Acme"
    assert card.tools == frozenset({"engine"}) and card.client_tools is False and card.label == "Acme Coach 2"
    assert card.answers_with_tools is True
    plain = FakeEndpoint().adapter().card()
    assert plain.tools == frozenset() and plain.answers_with_tools is False


def test_timeout_is_a_timeout_record() -> None:
    record = FakeEndpoint(failure=httpx.ReadTimeout("read timed out")).adapter().answer(task=POS_TASK, run_index=0, timeout_s=7)
    assert record.status == "timeout" and record.text == "" and record.error == "timeout: no complete response within 7s"


def test_connection_failure_is_a_retryable_transport_error_without_detail() -> None:
    record = FakeEndpoint(failure=httpx.ConnectError(f"refused {SECRET_BODY}")).adapter().answer(task=POS_TASK, run_index=0, timeout_s=7)
    assert record.status == "error" and record.error == "transport: ConnectError"


@pytest.mark.parametrize(("status", "error"), [(429, "transport: HTTP 429"), (503, "transport: HTTP 503"), (400, "http_status: 400"), (401, "http_status: 401"), (404, "http_status: 404")])
def test_http_errors_record_only_the_status_code(status: int, error: str) -> None:
    record = FakeEndpoint(status=status, body={"error": {"message": SECRET_BODY, "key": FAKE_KEY}}).adapter().answer(task=POS_TASK, run_index=0, timeout_s=7)
    assert record.status == "error" and record.error == error and record.text == ""
    assert SECRET_BODY not in record.model_dump_json() and FAKE_KEY not in record.model_dump_json()


@pytest.mark.parametrize(
    ("raw", "body", "error"),
    [
        (b"<html>" + SECRET_BODY.encode() + b"</html>", None, "malformed_response: the body is not JSON"),
        (None, [1, 2], "malformed_response: the body is not a JSON object"),
        (None, {"choices": []}, "malformed_response: no choices"),
        (None, {"choices": [{"message": "text"}]}, "malformed_response: choices[0].message is not an object"),
        (None, _completion(content=[{"type": "text", "text": "hi"}]), "unsupported_response: message content is not a string"),
        (None, _completion(content=None, finish_reason="tool_calls", tool_calls=[{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]), "unsupported_response: the model asked for tool calls, which this adapter never runs"),
        (None, _completion(content="partial", finish_reason="function_call"), "unsupported_response: the model asked for tool calls, which this adapter never runs"),
    ],
)
def test_malformed_and_unsupported_responses_are_errors(raw: bytes | None, body: object, error: str) -> None:
    record = FakeEndpoint(raw=raw, body=body).adapter().answer(task=POS_TASK, run_index=0, timeout_s=7)
    assert record.status == "error" and record.error == error and record.text == ""
    assert SECRET_BODY not in record.model_dump_json()


@pytest.mark.parametrize("content", ["", "   \n", None])
def test_empty_answers_are_errors(content: str | None) -> None:
    record = FakeEndpoint(body=_completion(content=content)).adapter().answer(task=POS_TASK, run_index=0, timeout_s=7)
    assert record.status == "error" and record.error == "empty_answer"


def test_check_available_raises_a_transport_error_without_the_body() -> None:
    adapter = FakeEndpoint(status=401, body={"error": SECRET_BODY}).adapter()
    with pytest.raises(LLMTransportError) as caught:
        adapter.check_available()
    assert str(caught.value) == "acme_coach answered the availability probe with http_status: 401"
    assert SECRET_BODY not in str(caught.value)


def test_check_available_passes_on_a_good_answer() -> None:
    FakeEndpoint(body=_completion(content="OK")).adapter().check_available()


def test_the_key_never_appears_in_the_adapter_repr() -> None:
    adapter = FakeEndpoint().adapter()
    assert FAKE_KEY not in repr(adapter) and FAKE_KEY not in repr(vars(adapter))


def test_the_family_must_be_declared_and_none_is_refused_for_a_model() -> None:
    fields = _spec().model_dump()
    del fields["declared_family"]
    with pytest.raises(ValidationError, match="declared_family"):
        HttpEntrant.model_validate(fields)
    with pytest.raises(ValidationError, match="declared_family"):
        _spec(declared_family="none")
    with pytest.raises(ValidationError, match="declared_family"):
        _spec(declared_family="mistral")


def test_arbitrary_model_ids_are_accepted_and_no_family_is_inferred_from_them() -> None:
    for model_id in ("olmo-3-32b", "openrouter/meta-llama/llama-4", "deepseek-v4", "gpt-oss:20b"):
        assert FakeEndpoint().adapter(spec=_spec(model_id=model_id)).card().declared_family == "undisclosed"


def test_client_side_tool_calling_cannot_be_configured() -> None:
    with pytest.raises(ValidationError, match="tools"):
        _spec(tools=True)
    with pytest.raises(ValidationError, match="client_tools"):
        _spec(client_tools=True)


@pytest.mark.parametrize(
    "url",
    ["http://api.acme.example/v1", "ftp://api.acme.example", "https://api.acme.example/v1?x=1", "https://api.acme.example/v1#a", "https://user:pw@api.acme.example/v1", "api.acme.example/v1"],
)
def test_the_url_must_be_https_or_a_local_http_server_without_credentials_or_query(url: str) -> None:
    with pytest.raises(ValidationError, match="url"):
        _spec(url=url)


def test_the_url_is_the_full_endpoint_kept_exactly_as_given_and_local_http_is_allowed() -> None:
    assert _spec(url="http://localhost:11434/v1/chat/completions").url == "http://localhost:11434/v1/chat/completions"
    assert _spec(url="https://acme.example/chess/ask/").url == "https://acme.example/chess/ask/"


def test_the_old_base_url_field_is_refused_by_name() -> None:
    entry = _spec().model_dump()
    entry["base_url"] = entry.pop("url")
    with pytest.raises(ValidationError, match="base_url"):
        HttpEntrant.model_validate(entry)


def test_the_key_is_read_from_the_entrant_variable_by_the_settings_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert http_entrant_key_variable(entrant="acme_coach") == "BENCH_HTTP_ACME_COACH_API_KEY"
    monkeypatch.setenv("BENCH_HTTP_ACME_COACH_API_KEY", FAKE_KEY)
    assert http_entrant_api_key(entrant="acme_coach", env_file=tmp_path / "missing.env").get_secret_value() == FAKE_KEY


def test_a_missing_key_names_the_variable_and_never_a_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BENCH_HTTP_ACME_COACH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BENCH_HTTP_ACME_COACH_API_KEY in .env"):
        http_entrant_api_key(entrant="acme_coach", env_file=tmp_path / "missing.env")


class _LocalServer:
    """A real HTTP server on 127.0.0.1 that answers like an OpenAI-compatible chat completions endpoint."""

    def __init__(self, *, required_key: str | None) -> None:
        self.required_key = required_key
        self.bodies: list[dict[str, object]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                if self.path != "/chess/ask" or (outer.required_key and self.headers.get("Authorization") != f"Bearer {outer.required_key}"):
                    self.send_response(401)
                    self.end_headers()
                    return
                payload = json.dumps(_completion(content=f"Answer for {body['model']}.")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_LocalServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/chess/ask"


def _third_party_file(tmp_path: Path, *, url: str, auth: str) -> Path:
    entry = {"label": "Local Coach", "provider": "tests", "declared_family": "undisclosed", "model_id": "local/coach-7b", "url": url, "auth": auth, "service_tools": [], "affiliated_with_maintainer": False}
    path = tmp_path / "third_party.yaml"
    path.write_text(yaml.safe_dump({"entrants": {"local_coach": entry}}), encoding="utf-8")
    return path


@pytest.mark.parametrize("auth", ["none", "bearer"])
def test_a_configured_entrant_runs_end_to_end_through_the_registry_and_the_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auth: str) -> None:
    monkeypatch.setenv("BENCH_HTTP_LOCAL_COACH_API_KEY", FAKE_KEY)
    with _LocalServer(required_key=FAKE_KEY if auth == "bearer" else None) as server:
        third_party = load_third_party(path=_third_party_file(tmp_path, url=server.url, auth=auth))
        models = load_models(profile=None)
        assert "local_coach" in known_entrants(models=models, third_party=third_party)
        adapter = get_adapter(name="local_coach", models=models, cfg=load_bench(), third_party=third_party)
        adapter.check_available()
        summary = run_entrant(adapter=adapter, tasks=[POS_TASK, GR_TASK], runs_per_task=2, timeout_s=30, concurrency=2, out_dir=tmp_path / "run")
    assert (summary.answers, summary.ok, summary.failed) == (4, 4, 0)
    records = read_answers(run_dir=tmp_path / "run")
    assert {r.text for r in records} == {"Answer for local/coach-7b."}
    assert all(r.raw["transport"] == "openai_compatible_http" and r.raw["timing_measured"] is True for r in records)
    assert {json.dumps(b["messages"][1]["content"]) for b in server.bodies[1:]} == {json.dumps(render_user_message(task=POS_TASK)), json.dumps(render_user_message(task=GR_TASK))}
    assert all("tools" not in body for body in server.bodies)
    assert adapter.card().access_method == "api" and adapter.card().declared_family == "undisclosed"


def test_a_configured_entrant_is_not_gated_by_the_claude_only_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_PROFILE", "claude_only")
    third_party = load_third_party(path=_third_party_file(tmp_path, url="http://127.0.0.1:9/v1/chat/completions", auth="none"))
    adapter = get_adapter(name="local_coach", models=load_models(), cfg=load_bench(), third_party=third_party)
    assert isinstance(adapter, OpenAICompatibleAdapter)


def test_a_third_party_name_may_not_reuse_a_built_in_entrant(tmp_path: Path) -> None:
    path = _third_party_file(tmp_path, url="http://127.0.0.1:9/v1/chat/completions", auth="none")
    path.write_text(path.read_text(encoding="utf-8").replace("local_coach", "claude_openrouter"), encoding="utf-8")
    with pytest.raises(ValueError, match="reuses the entrant name"):
        known_entrants(models=load_models(profile=None), third_party=load_third_party(path=path))


def test_the_shipped_third_party_file_is_valid_and_empty() -> None:
    assert load_third_party().entrants == {}


def test_a_bearer_entrant_without_its_key_stops_with_the_variable_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_HTTP_LOCAL_COACH_API_KEY", raising=False)
    third_party = load_third_party(path=_third_party_file(tmp_path, url="https://api.local.example/v1/chat/completions", auth="bearer"))
    with pytest.raises(RuntimeError, match="BENCH_HTTP_LOCAL_COACH_API_KEY"):
        get_adapter(name="local_coach", models=load_models(profile=None), cfg=load_bench(), third_party=third_party)


class _DribbleStream(httpx.SyncByteStream):
    def __iter__(self):  # type: ignore[no-untyped-def]
        yield b" "
        time.sleep(0.6)
        yield b" "
        time.sleep(0.6)
        yield json.dumps(_completion()).encode()


def test_the_timeout_is_a_wall_clock_deadline_even_when_bytes_keep_arriving() -> None:
    adapter = OpenAICompatibleAdapter(name="acme_coach", spec=_spec(), api_key=SecretStr(FAKE_KEY), transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=_DribbleStream())))
    record = adapter.answer(task=POS_TASK, run_index=0, timeout_s=1)
    assert record.status == "timeout" and record.text == "" and record.error == "timeout: no complete response within 1s"


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_key_is_refused_with_the_variable_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str) -> None:
    monkeypatch.setenv("BENCH_HTTP_ACME_COACH_API_KEY", value)
    with pytest.raises(RuntimeError, match="BENCH_HTTP_ACME_COACH_API_KEY in .env"):
        http_entrant_api_key(entrant="acme_coach", env_file=tmp_path / "missing.env")
