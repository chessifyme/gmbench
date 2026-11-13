import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml

from bench.adapters.http_entrant import HttpEntrantAdapter
from bench.adapters.prompts import NEUTRAL_SYSTEM_PROMPT, render_user_message
from bench.adapters.python_entrant import PythonEntrantAdapter
from bench.adapters.registry import entrant_file_adapter, get_adapter, known_entrants
from bench.config import load_bench, load_models, load_third_party
from bench.entrants import ENTRANTS_DIR, EntrantFileError, HttpEntrantFile, entrant_file, entrant_files, extract, fill, load_entrant_file
from bench.io_yaml import read_task
from bench.llm import LLMTransportError
from bench.runner import answer_with_retry, run_entrant
from bench.runs import read_answers
from tests.fake_agent_server import FAKE_ANSWER, fake_agent_server

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"
POS_TASK = read_task(path=FIXTURES / "ap_9001.yaml")
GR_TASK = read_task(path=FIXTURES / "gr_9001.yaml")
FAKE_KEY = "fake_entrant_key_0123456789abcdef"
SECRET_BODY = "upstream secret detail 42"
CARD = {"name": "acme_coach", "label": "Acme Coach", "version": "2.1", "provider": "Acme", "declared_family": "undisclosed", "tools": ["engine"], "affiliated_with_maintainer": False}
HTTP = {"url": "https://api.acme.example/ask", "headers": {"Authorization": "Bearer ${BENCH_ENTRANT_ACME_COACH_KEY}"}, "body": {"q": "{message}", "sys": "{system}", "session": "{fresh_id}", "n": 1, "stream": False}, "answer": "output.text"}


def _entrant(**http: object) -> HttpEntrantFile:
    return HttpEntrantFile.model_validate({**CARD, "http": {**HTTP, **http}})


def _write(tmp_path: Path, data: dict[str, object], *, name: str = "acme_coach") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _lookup(variable: str) -> str | None:
    return FAKE_KEY if variable == "BENCH_ENTRANT_ACME_COACH_KEY" else None


class Endpoint:
    """A fake endpoint behind httpx.MockTransport: a list of replies (status, body) served in turn, the last one repeated."""

    def __init__(self, *replies: tuple[int, object], delay_s: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay_s = delay_s
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if self.delay_s:
            time.sleep(self.delay_s)
        if isinstance(body, Exception):
            raise body
        if isinstance(body, bytes):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    def adapter(self, *, entrant: HttpEntrantFile | None = None, answer_timeout_s: float = 900) -> HttpEntrantAdapter:
        return HttpEntrantAdapter(entrant=entrant or _entrant(), answer_timeout_s=answer_timeout_s, lookup=_lookup, transport=httpx.MockTransport(self.handler))


OK = (200, {"output": {"text": "Black is better: the bishop pair."}, "id": "r1"})


# configuration


def test_the_shipped_examples_are_valid_entrant_files() -> None:
    files = entrant_files(directory=ENTRANTS_DIR)
    assert {"example_http", "example_python"} <= set(files)
    http = load_entrant_file(path=files["example_http"])
    assert isinstance(http, HttpEntrantFile) and http.http.answer == "output.text" and http.http.url.startswith("http://127.0.0.1")
    assert not isinstance(load_entrant_file(path=files["example_python"]), HttpEntrantFile)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"answer": ""}, "answer"),
        ({"answer": "output..text"}, "dotted path"),
        ({"body": {"q": "{message}", "id": "{task_id}"}}, "task_id"),
        ({"body": {"q": "{message}", "x": "{question}"}}, r"unknown placeholder\(s\) \{question\}"),
        ({"body": {"q": "{fen}"}}, "unknown placeholder"),
        ({"body": {"q": "{system}"}}, "never sends {message}"),
        ({"url": "http://api.acme.example/ask"}, "plain http only for localhost"),
        ({"url": "https://api.acme.example/ask?key=abc"}, "no credentials, query or fragment"),
        ({"headers": {"Authorization": "Bearer sk-literal-key"}}, "from .env"),
        ({"headers": {"X-Api-Key": "abc"}}, "from .env"),
        ({"headers": {"Authorization": "Bearer ${OPENROUTER_API_KEY}"}}, "BENCH_ENTRANT_ACME_COACH_<WORD>"),
        ({"headers": {"Authorization": "Bearer ${BENCH_HTTP_ACME_COACH_API_KEY}"}}, "BENCH_ENTRANT_ACME_COACH_<WORD>"),
        ({"headers": {"Authorization": "Bearer ${BENCH_ENTRANT_ACME_COACH_API_KEY}"}}, "one word"),
        ({"headers": {"X-Trace": "{message}"}}, "placeholder"),
        ({"method": "GET"}, "method"),
        ({"timeout_s": 0}, "timeout_s"),
    ],
)
def test_a_bad_entrant_file_is_refused_with_what_is_wrong(tmp_path: Path, change: dict[str, object], message: str) -> None:
    path = _write(tmp_path, {**CARD, "http": {**HTTP, **change}})
    with pytest.raises(EntrantFileError, match=message):
        load_entrant_file(path=path)


def test_an_entrant_cannot_name_a_longer_entrants_variable() -> None:
    with pytest.raises(ValueError, match="BENCH_ENTRANT_ACME_<WORD>"):
        HttpEntrantFile.model_validate({**CARD, "name": "acme", "http": {**HTTP, "headers": {"X-Key": "${BENCH_ENTRANT_ACME_COACH_KEY}"}}})


def test_a_missing_answer_path_and_an_unknown_field_are_refused(tmp_path: Path) -> None:
    without = {key: value for key, value in HTTP.items() if key != "answer"}
    with pytest.raises(EntrantFileError, match="answer"):
        load_entrant_file(path=_write(tmp_path, {**CARD, "http": without}))
    with pytest.raises(EntrantFileError, match="model_id"):
        load_entrant_file(path=_write(tmp_path, {**CARD, "model_id": "x", "http": HTTP}))


def test_the_name_must_be_the_file_name(tmp_path: Path) -> None:
    with pytest.raises(EntrantFileError, match="must be 'other_coach'"):
        load_entrant_file(path=_write(tmp_path, {**CARD, "http": HTTP}, name="other_coach"))


def test_a_timeout_over_the_benchmarks_limit_is_refused() -> None:
    with pytest.raises(EntrantFileError, match="over the benchmark's answer_timeout_s 900"):
        Endpoint(OK).adapter(entrant=_entrant(timeout_s=1200))


def test_a_missing_header_variable_names_the_variable_never_a_value() -> None:
    with pytest.raises(EntrantFileError, match="set BENCH_ENTRANT_ACME_COACH_OTHER in .env") as caught:
        HttpEntrantAdapter(entrant=_entrant(headers={"Authorization": "Bearer ${BENCH_ENTRANT_ACME_COACH_OTHER}"}), answer_timeout_s=900, lookup=_lookup)
    assert FAKE_KEY not in str(caught.value)


# template filling and the answer path


def test_fill_replaces_placeholders_in_nested_strings_and_keys_and_keeps_other_values() -> None:
    template = {"input": [{"role": "user", "content": "Q: {message}"}], "{fresh_id}": {"n": 2, "ok": True, "none": None}, "system": "{system}"}
    assert fill(template=template, values={"message": "m", "system": "s", "fresh_id": "id1"}) == {"input": [{"role": "user", "content": "Q: m"}], "id1": {"n": 2, "ok": True, "none": None}, "system": "s"}


def test_fill_does_not_expand_braces_in_the_message() -> None:
    assert fill(template={"q": "{message}"}, values={"message": "{system} {fresh_id}", "system": "S", "fresh_id": "F"}) == {"q": "{system} {fresh_id}"}


@pytest.mark.parametrize(
    ("reply", "path", "expected"),
    [
        ({"output": {"text": "A plan."}}, "output.text", ("ok", None, "A plan.")),
        ({"choices": [{"message": {"content": "Nf3."}}]}, "choices.0.message.content", ("ok", None, "Nf3.")),
        ([["a", "b"]], "0.1", ("ok", None, "b")),
        ({"output": {}}, "output.text", ("error", "malformed_response: the reply has no output.text", "")),
        ({"choices": []}, "choices.0.message", ("error", "malformed_response: the reply has no choices.0", "")),
        ({"choices": {"0": "x"}}, "choices.0", ("ok", None, "x")),
        ({"a": ["x"]}, "a.first", ("error", "malformed_response: the reply has no a.first", "")),
        ({"output": {"text": ["part"]}}, "output.text", ("error", "unsupported_response: output.text is not a string", "")),
        ({"output": {"text": 3}}, "output.text", ("error", "unsupported_response: output.text is not a string", "")),
        ({"output": {"text": None}}, "output.text", ("error", "unsupported_response: output.text is not a string", "")),
        ({"output": {"text": "  "}}, "output.text", ("error", "empty_answer", "  ")),
        ("plain", "output", ("error", "malformed_response: the reply has no output", "")),
    ],
)
def test_extract_reads_dicts_and_lists_and_names_what_is_missing(reply: object, path: str, expected: tuple[str, str | None, str]) -> None:
    assert extract(reply=reply, path=path) == expected


# the HTTP adapter


def test_one_request_per_answer_carries_the_shared_message_and_never_the_task_id() -> None:
    endpoint = Endpoint(OK)
    adapter = endpoint.adapter()
    first = adapter.answer(task=POS_TASK, run_index=0, timeout_s=30)
    adapter.answer(task=POS_TASK, run_index=1, timeout_s=30)
    assert (first.status, first.error, first.text) == ("ok", None, "Black is better: the bishop pair.")
    assert len(endpoint.requests) == 2
    request = endpoint.requests[0]
    assert request.method == "POST" and str(request.url) == "https://api.acme.example/ask" and request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    bodies = [json.loads(r.content) for r in endpoint.requests]
    assert bodies[0]["q"] == render_user_message(task=POS_TASK) and bodies[0]["sys"] == NEUTRAL_SYSTEM_PROMPT and bodies[0]["n"] == 1 and bodies[0]["stream"] is False
    assert bodies[0]["session"] != bodies[1]["session"]
    assert all(POS_TASK.task_id not in r.content.decode() for r in endpoint.requests)
    assert first.raw == {"transport": "http_entrant", "answer_path": "output.text", "reply": OK[1], "streamed": False, "timing_measured": True}
    assert first.first_token_at is None and first.finished_at >= first.started_at


def test_put_is_sent_when_configured() -> None:
    endpoint = Endpoint(OK)
    endpoint.adapter(entrant=_entrant(method="PUT")).answer(task=POS_TASK, run_index=0, timeout_s=30)
    assert endpoint.requests[0].method == "PUT"


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_a_retryable_status_is_retried_once_and_the_retry_recorded(status: int) -> None:
    endpoint = Endpoint((status, {"error": SECRET_BODY}), OK)
    record = answer_with_retry(adapter=endpoint.adapter(), task=POS_TASK, run_index=0, timeout_s=30)
    assert record.status == "ok" and record.raw["retried_after"] == f"transport: HTTP {status}" and len(endpoint.requests) == 2


def test_a_retry_that_fails_again_is_an_error_and_never_a_third_request() -> None:
    endpoint = Endpoint((502, {"error": SECRET_BODY}))
    record = answer_with_retry(adapter=endpoint.adapter(), task=POS_TASK, run_index=0, timeout_s=30)
    assert (record.status, record.error, len(endpoint.requests)) == ("error", "transport: HTTP 502", 2)


def test_a_connection_failure_is_a_transport_error() -> None:
    record = Endpoint((200, httpx.ConnectError("refused"))).adapter().answer(task=POS_TASK, run_index=0, timeout_s=30)
    assert (record.status, record.error) == ("error", "transport: ConnectError")


def test_another_status_is_an_http_status_error_and_not_retried() -> None:
    endpoint = Endpoint((400, {"error": SECRET_BODY}))
    record = answer_with_retry(adapter=endpoint.adapter(), task=POS_TASK, run_index=0, timeout_s=30)
    assert (record.status, record.error, len(endpoint.requests)) == ("error", "http_status: 400", 1)
    assert record.raw.get("reply") is None


def test_a_timeout_is_recorded_and_never_retried() -> None:
    endpoint = Endpoint(OK, delay_s=0.3)
    record = answer_with_retry(adapter=endpoint.adapter(entrant=_entrant(timeout_s=0.1)), task=POS_TASK, run_index=0, timeout_s=30)
    assert record.status == "timeout" and record.error is not None and record.error.startswith("timeout:") and len(endpoint.requests) == 1


def test_the_entrant_timeout_never_exceeds_the_run_timeout() -> None:
    endpoint = Endpoint(OK, delay_s=0.3)
    record = endpoint.adapter(entrant=_entrant(timeout_s=600)).answer(task=POS_TASK, run_index=0, timeout_s=0.1)
    assert record.status == "timeout"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((200, b"<html>oops</html>"), "malformed_response: the body is not JSON"),
        ((200, {"output": {}}), "malformed_response: the reply has no output.text"),
        ((200, {"output": {"text": {"parts": []}}}), "unsupported_response: output.text is not a string"),
        ((200, {"output": {"text": ""}}), "empty_answer"),
    ],
)
def test_a_reply_without_a_usable_answer_is_an_error_row(reply: tuple[int, object], error: str) -> None:
    record = Endpoint(reply).adapter().answer(task=POS_TASK, run_index=0, timeout_s=30)
    assert (record.status, record.error, record.text) == ("error", error, "")


def test_secrets_never_reach_records_errors_or_logs(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    caplog.set_level(logging.DEBUG)
    records = [Endpoint(reply).adapter().answer(task=POS_TASK, run_index=0, timeout_s=30) for reply in ((401, {"error": SECRET_BODY}), (500, SECRET_BODY.encode()), OK)]
    captured = capsys.readouterr()
    for text in [*(r.model_dump_json() for r in records), caplog.text, captured.out, captured.err]:
        assert FAKE_KEY not in text and SECRET_BODY not in text and "authorization" not in text.lower()


def test_the_probe_passes_on_an_answer_at_the_path_and_fails_otherwise() -> None:
    endpoint = Endpoint(OK)
    endpoint.adapter().check_available()
    assert json.loads(endpoint.requests[0].content)["q"] == "Reply with the single word OK."
    with pytest.raises(LLMTransportError, match="malformed_response: the reply has no output$"):
        Endpoint((200, {"result": "OK"})).adapter().check_available()
    with pytest.raises(LLMTransportError, match="transport: HTTP 503"):
        Endpoint((503, {})).adapter().check_available()


def test_the_card_comes_from_the_entrant_file() -> None:
    card = Endpoint(OK).adapter().card()
    assert (card.name, card.label, card.version, card.provider, card.declared_family, card.access_method, set(card.tools)) == ("acme_coach", "Acme Coach", "2.1", "Acme", "undisclosed", "api", {"engine"})


def test_an_http_entrant_runs_end_to_end_against_the_fake_agent_server(tmp_path: Path) -> None:
    server = fake_agent_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        entrants = tmp_path / "entrants"
        entrants.mkdir()
        example = yaml.safe_load((ENTRANTS_DIR / "example_http.yaml").read_text(encoding="utf-8"))
        example["http"]["url"] = f"http://127.0.0.1:{server.server_address[1]}/answer"
        (entrants / "example_http.yaml").write_text(yaml.safe_dump(example), encoding="utf-8")
        models = load_models(profile=None)
        assert "example_http" in known_entrants(models=models, entrants_dir=entrants)
        adapter = get_adapter(name="example_http", models=models, cfg=load_bench(), entrants_dir=entrants)
        adapter.check_available()
        summary = run_entrant(adapter=adapter, tasks=[POS_TASK, GR_TASK], runs_per_task=2, timeout_s=30, concurrency=2, out_dir=tmp_path / "run")
    finally:
        server.shutdown()
    assert (summary.answers, summary.ok) == (4, 4)
    assert {r.text for r in read_answers(run_dir=tmp_path / "run")} == {FAKE_ANSWER}
    bodies = [r["body"] for r in server.requests[1:]]  # type: ignore[attr-defined]
    assert {b["input"] for b in bodies} == {render_user_message(task=POS_TASK), render_user_message(task=GR_TASK)} and len({b["session_id"] for b in bodies}) == 4


# the Python function entrant

PY_ENTRANT = '''
import pathlib
pathlib.Path(__file__).with_name("imported.marker").write_text("x")
ENTRANT = {"name": "py_coach", "label": "Py Coach", "version": "0.1", "provider": "Me", "declared_family": "none", "tools": [], "affiliated_with_maintainer": False}
CALLS = []

def answer(message: str) -> str:
    CALLS.append(message)
    if "Reply with the single word OK." in message:
        return "OK"
    return BEHAVIOUR(message)
'''


def _python_entrant(tmp_path: Path, behaviour: str) -> PythonEntrantAdapter:
    path = tmp_path / "py_coach.py"
    path.write_text(PY_ENTRANT + f"\nBEHAVIOUR = {behaviour}\n", encoding="utf-8")
    adapter = entrant_file_adapter(path=path, cfg=load_bench())
    assert isinstance(adapter, PythonEntrantAdapter)
    return adapter


def test_the_python_card_is_read_without_running_the_file(tmp_path: Path) -> None:
    adapter = _python_entrant(tmp_path, "lambda m: 'Nf3 wins.'")
    assert not (tmp_path / "imported.marker").exists()
    assert adapter.card().declared_family == "none" and adapter.card().label == "Py Coach"
    adapter.check_available()
    assert (tmp_path / "imported.marker").exists()


def test_the_python_function_gets_the_shared_message_and_its_answer_is_recorded(tmp_path: Path) -> None:
    adapter = _python_entrant(tmp_path, "lambda m: 'Nf3 wins.'")
    record = adapter.answer(task=POS_TASK, run_index=2, timeout_s=5)
    assert (record.status, record.error, record.text, record.run_index) == ("ok", None, "Nf3 wins.", 2)
    assert record.raw == {"transport": "python_function", "reply": "Nf3 wins.", "timing_measured": True}


@pytest.mark.parametrize(
    ("behaviour", "status", "error"),
    [
        ("lambda m: 1 / 0", "error", "exception: ZeroDivisionError"),
        ("lambda m: ['Nf3']", "error", "unsupported_response: answer() did not return a string"),
        ("lambda m: ' '", "error", "empty_answer"),
        ("lambda m: __import__('time').sleep(2) or 'late'", "timeout", "timeout: no complete response within 0s"),
    ],
)
def test_python_failures_are_recorded_like_the_http_ones(tmp_path: Path, behaviour: str, status: str, error: str) -> None:
    record = answer_with_retry(adapter=_python_entrant(tmp_path, behaviour), task=POS_TASK, run_index=0, timeout_s=0.2)
    assert (record.status, record.error, record.text) == (status, error, "")
    assert "retried_after" not in record.raw


def test_a_python_file_without_answer_or_card_is_refused(tmp_path: Path) -> None:
    (tmp_path / "no_card.py").write_text("def answer(m):\n    return m\n", encoding="utf-8")
    with pytest.raises(EntrantFileError, match="ENTRANT"):
        load_entrant_file(path=tmp_path / "no_card.py")
    (tmp_path / "no_answer.py").write_text(PY_ENTRANT.replace("py_coach", "no_answer").replace("def answer", "def reply"), encoding="utf-8")
    with pytest.raises(EntrantFileError, match="defines no function answer"):
        entrant_file_adapter(path=tmp_path / "no_answer.py", cfg=load_bench()).check_available()


# the registry


def test_one_entrant_has_one_file_and_no_name_is_reused(tmp_path: Path) -> None:
    (tmp_path / "acme_coach.yaml").write_text(yaml.safe_dump({**CARD, "http": HTTP}), encoding="utf-8")
    (tmp_path / "acme_coach.py").write_text("ENTRANT = {}\n", encoding="utf-8")
    (tmp_path / "beta_coach.yaml").write_text(yaml.safe_dump({**CARD, "name": "beta_coach", "http": HTTP}), encoding="utf-8")
    assert {"acme_coach", "beta_coach"} <= set(entrant_files(directory=tmp_path))
    with pytest.raises(EntrantFileError, match="both acme_coach.py and acme_coach.yaml"):
        entrant_file(name="acme_coach", directory=tmp_path)
    assert entrant_file(name="beta_coach", directory=tmp_path) == tmp_path / "beta_coach.yaml"  # only the doubled entrant is refused
    (tmp_path / "acme_coach.py").unlink()
    built_in = next(iter(load_models(profile=None).entrants))
    (tmp_path / f"{built_in}.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=f"reuses the entrant name\\(s\\) {built_in}"):
        known_entrants(models=load_models(profile=None), third_party=load_third_party(), entrants_dir=tmp_path)


def test_a_variable_in_the_body_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EntrantFileError, match="never in the body"):
        load_entrant_file(path=_write(tmp_path, {**CARD, "http": {**HTTP, "body": {"q": "{message}", "key": "${BENCH_ENTRANT_ACME_COACH_KEY}"}}}))


@pytest.mark.parametrize("text", ["ENTRANT = {\n", "\x00\x01", "ENTRANT = {'name': f'{1}'}\n", "ENTRANT = {'name': 1 + 1j}\n", "ENTRANT = dict(name='x')\n"])
def test_any_python_card_that_is_not_a_plain_dict_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "py_coach.py"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EntrantFileError):
        load_entrant_file(path=path)


def test_a_self_run_entrant_never_answers_the_private_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bench.cli.run as run

    monkeypatch.setattr(run, "task_source", lambda **kwargs: run.TaskSource(task_set="private", tasks_dir=tmp_path))
    monkeypatch.setattr(run, "registration_for", lambda **kwargs: None)
    monkeypatch.setattr(run, "check_runs_dir", lambda **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run", "--entrant", "example_python", "--set", "private"])
    with pytest.raises(SystemExit, match="never answers the private set"):
        run.main()
