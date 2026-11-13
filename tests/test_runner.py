import json
import shutil
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench.adapters.base import ALL_FAMILIES, Adapter, AgentCard, AnswerRecord
from bench.llm import Completion
from bench.io_yaml import load_all_tasks
from bench.models import Family, Task
from bench.config import API_TRANSPORTS, OPENROUTER_CHAT_URL, load_bench, load_models
from bench.provenance import client_version, engine_identity, entrant_transports, git_head, temperature_label, uses_local_engine
from bench.runner import build_meta, run_entrant
from bench.runs import RunMeta, latest_run_dir, new_run_dir, read_answers, read_meta, write_meta

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"


class FakeAdapter(Adapter):
    def __init__(self, *, families: frozenset[Family] = ALL_FAMILIES, fail_first: bool = False, always_error: str | None = None) -> None:
        self.name = "fake"
        self.declared_family = "none"
        self.attempted_families = families
        self.fail_first = fail_first
        self.always_error = always_error
        self.calls: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def card(self) -> AgentCard:
        return AgentCard(name="fake", version="1", provider="tests", declared_family="none", access_method="library", attempted_families=self.attempted_families, tools=frozenset(), affiliated_with_maintainer=False)

    def answer(self, *, task: Task, run_index: int, timeout_s: float) -> AnswerRecord:
        with self._lock:
            self.calls.append((task.task_id, run_index))
            first = self.calls.count((task.task_id, run_index)) == 1
        now = datetime.now(UTC)
        error = self.always_error if self.always_error else ("transport: boom" if self.fail_first and first else None)
        return AnswerRecord(
            task_id=task.task_id,
            entrant=self.name,
            run_index=run_index,
            text="" if error else f"answer {task.task_id} {run_index}",
            status="error" if error else "ok",
            error=error,
            started_at=now,
            first_token_at=None if error else now,
            finished_at=now,
            raw={},
        )


def _tasks() -> list[Task]:
    return load_all_tasks(tasks_dir=FIXTURES)


def _run(tmp_path: Path, adapter: FakeAdapter, tasks: list[Task]):
    return run_entrant(adapter=adapter, tasks=tasks, runs_per_task=3, timeout_s=5, concurrency=2, out_dir=tmp_path)


def test_runner_writes_three_records_per_task_and_resumes(tmp_path: Path) -> None:
    tasks = _tasks()[:2]
    summary = _run(tmp_path, FakeAdapter(), tasks)
    assert summary.answers == 6 and summary.ok == 6 and summary.failed == 0
    records = read_answers(run_dir=tmp_path)
    assert {(r.task_id, r.run_index) for r in records} == {(t.task_id, i) for t in tasks for i in range(3)}
    second = FakeAdapter()
    assert _run(tmp_path, second, tasks).answers == 6 and second.calls == []
    lines = (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "answers.jsonl").write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    third = FakeAdapter()
    summary = _run(tmp_path, third, tasks)
    assert len(third.calls) == 2 and summary.answers == 6
    assert len({(r.task_id, r.run_index) for r in read_answers(run_dir=tmp_path)}) == 6


def test_runner_marks_unattempted_family(tmp_path: Path) -> None:
    tasks = [t for t in _tasks() if t.task_id in ("gr_9001", "ap_9001")]
    adapter = FakeAdapter(families=frozenset({"assessment_plan", "move_judgement", "move_choice"}))
    summary = _run(tmp_path, adapter, tasks)
    records = read_answers(run_dir=tmp_path)
    skipped = [r for r in records if r.task_id == "gr_9001"]
    assert len(skipped) == 3 and all(r.status == "error" and r.error == "not_attempted" for r in skipped)
    assert all(call[0] != "gr_9001" for call in adapter.calls)
    assert summary.failed == 3 and summary.ok == 3


def test_runner_retries_transport_once_then_fails(tmp_path: Path) -> None:
    tasks = _tasks()[:1]
    flaky = FakeAdapter(fail_first=True)
    assert _run(tmp_path, flaky, tasks).ok == 3
    assert len(flaky.calls) == 6
    records = read_answers(run_dir=tmp_path)
    assert all(r.status == "ok" and r.raw.get("retried_after") == "transport: boom" for r in records)
    dead = FakeAdapter(always_error="transport: down")
    summary = _run(tmp_path / "dead", dead, tasks)
    assert summary.failed == 3 and len(dead.calls) == 6
    assert all(r.status == "error" for r in read_answers(run_dir=tmp_path / "dead"))


def test_runner_does_not_retry_other_errors(tmp_path: Path) -> None:
    adapter = FakeAdapter(always_error="empty_answer")
    summary = _run(tmp_path, adapter, _tasks()[:1])
    assert summary.failed == 3 and len(adapter.calls) == 3


def test_runner_never_retries_a_refused_answer(tmp_path: Path) -> None:
    adapter = FakeAdapter(always_error="refused: agy used search_web outside the local command and file tools")
    summary = _run(tmp_path, adapter, _tasks()[:1])
    assert summary.failed == 3 and len(adapter.calls) == 3


def test_resuming_with_retry_failed_answers_failed_rows_again_and_keeps_the_failures_aside(tmp_path: Path) -> None:
    tasks = [t for t in _tasks() if t.task_id in ("gr_9001", "ap_9001")]
    families = frozenset({"assessment_plan", "move_judgement", "move_choice"})
    _run(tmp_path, FakeAdapter(families=families, always_error="transport: down"), tasks)
    untouched = FakeAdapter(families=families)
    _run(tmp_path, untouched, tasks)
    assert untouched.calls == []
    again = FakeAdapter(families=families)
    summary = run_entrant(adapter=again, tasks=tasks, runs_per_task=3, timeout_s=5, concurrency=2, out_dir=tmp_path, retry_failed=True)
    assert sorted(again.calls) == [("ap_9001", i) for i in range(3)] and summary.answers == 6 and summary.ok == 3
    rows = read_answers(run_dir=tmp_path)
    assert len(rows) == 6 and {r.error for r in rows if r.task_id == "gr_9001"} == {"not_attempted"}
    set_aside = [json.loads(line) for line in (tmp_path / "answers_failed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(set_aside) == 3 and {row["error"] for row in set_aside} == {"transport: down"}


def test_retry_failed_is_refused_once_the_run_is_graded(tmp_path: Path) -> None:
    tasks = _tasks()[:1]
    _run(tmp_path, FakeAdapter(always_error="transport: down"), tasks)
    (tmp_path / "grades.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="graded"):
        run_entrant(adapter=FakeAdapter(), tasks=tasks, runs_per_task=3, timeout_s=5, concurrency=2, out_dir=tmp_path, retry_failed=True)


def test_retry_failed_is_refused_on_a_registered_attempt(tmp_path: Path) -> None:
    tasks = _tasks()[:1]
    write_meta(meta=_meta_for(registration_sha256="r" * 64), run_dir=tmp_path)
    _run(tmp_path, FakeAdapter(always_error="transport: down"), tasks)
    before = (tmp_path / "answers.jsonl").read_bytes()
    again = FakeAdapter()
    with pytest.raises(ValueError, match="registered attempt"):
        run_entrant(adapter=again, tasks=tasks, runs_per_task=3, timeout_s=5, concurrency=2, out_dir=tmp_path, retry_failed=True)
    assert again.calls == [] and (tmp_path / "answers.jsonl").read_bytes() == before and not (tmp_path / "answers_failed.jsonl").exists()


def test_resuming_refuses_a_changed_card_models_config_profile_or_model_spec() -> None:
    from bench.runner import resume_drift

    meta = _meta_for(registration_sha256=None).model_copy(update={"entrant_details": {"route": "openrouter", "provider": "Anthropic"}})
    same = {"card": meta.card, "models_config_hash": "a", "profile": None, "entrant_config_hash": None, "entrant_details": {"route": "openrouter", "provider": "Anthropic"}}
    assert resume_drift(meta=meta, **same) == []
    changed_card = meta.card.model_copy(update={"effort": "high", "version": "claude-opus-5-5"})
    assert resume_drift(meta=meta, **{**same, "card": changed_card}) == ["the entrant card differs on effort (medium -> high), version (claude-sonnet-5-5 -> claude-opus-5-5)"]
    assert any("models config" in p for p in resume_drift(meta=meta, **{**same, "models_config_hash": "z"}))
    assert any("profile" in p for p in resume_drift(meta=meta, **{**same, "profile": "claude_only"}))
    assert any("config/third_party.yaml" in p for p in resume_drift(meta=meta, **{**same, "entrant_config_hash": "e"}))
    assert resume_drift(meta=meta, **{**same, "entrant_details": {"route": "openrouter", "provider": "Amazon Bedrock"}}) == ["the model spec differs on provider (Anthropic -> Amazon Bedrock)"]


def test_new_run_dir_uses_underscore_timestamp_and_meta_roundtrip(tmp_path: Path) -> None:
    run_dir = new_run_dir(entrant="gemini_default", runs_dir=tmp_path, now=datetime(2026, 10, 17, 12, 0, 5, tzinfo=UTC))
    assert run_dir == tmp_path / "gemini_default" / "20261017_120005" and run_dir.is_dir()
    meta = RunMeta(
        entrant="gemini_default",
        card=FakeAdapter().card(),
        dataset_version="pilot",
        task_set="public",
        grader_version=None,
        started_at=datetime(2026, 10, 17, 12, 0, 5, tzinfo=UTC),
        models_config_hash="abc",
        bench_config_hash="def",
        git_commit="0123456789abcdef0123456789abcdef01234567",
        client_versions={"claude_cli": "2.1.0 (Claude Code)"},
        engine_id="Stockfish 19",
        engine_sha256="a" * 64,
        temperature="client_default",
        registration_sha256=None,
    )
    write_meta(meta=meta, run_dir=run_dir)
    assert read_meta(run_dir=run_dir) == meta
    stored = json.loads((run_dir / "meta.json").read_text())
    assert stored["grader_version"] is None and stored["registration_sha256"] is None and stored["client_versions"] == {"claude_cli": "2.1.0 (Claude Code)"}
    later = new_run_dir(entrant="gemini_default", runs_dir=tmp_path, now=datetime(2026, 10, 17, 13, 0, 0, tzinfo=UTC))
    assert latest_run_dir(entrant="gemini_default", runs_dir=tmp_path) == later
    with pytest.raises(FileNotFoundError):
        latest_run_dir(entrant="nobody", runs_dir=tmp_path)


def test_an_unavailable_client_is_recorded_on_the_run_instead_of_failing_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.config import EntrantModel
    from bench.adapters.stock import StockAdapter
    from bench.llm import LLMTransportError
    from bench.runner import unavailable_reason
    from tests.grading_fakes import FakeJson

    class Down(FakeJson):
        def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
            raise LLMTransportError("codex exited with 1: You have hit your usage limit")

    spec = EntrantModel(family="openai", model_id="gpt-6.1-sol", transport="codex_cli", label="gpt-6.1-sol", engine_context=False)
    down = StockAdapter(entrant_key="openai_default", spec=spec, model_factory=lambda timeout_s: Down(payload=None))
    assert unavailable_reason(adapter=down) == "codex exited with 1: You have hit your usage limit"
    up = StockAdapter(entrant_key="openai_default", spec=spec, model_factory=lambda timeout_s: FakeJson(payload="OK"))
    assert unavailable_reason(adapter=up) is None
    assert unavailable_reason(adapter=FakeAdapter()) is None
    monkeypatch.setattr("bench.runner.client_version", lambda *, transport: "codex-cli 1.0")
    run_dir = new_run_dir(entrant="openai_default", runs_dir=tmp_path)
    meta = build_meta(
        entrant="openai_default", card=down.card(), models=load_models(), cfg=load_bench(), task_set="public", registration_sha256=None, engine_path="stockfish", profile=None, unavailable_reason="codex exited with 1", root=_git_repo_root(tmp_path / "repo")
    )
    write_meta(meta=meta, run_dir=run_dir)
    assert read_meta(run_dir=run_dir).unavailable_reason == "codex exited with 1"
    assert not (run_dir / "answers.jsonl").exists()


def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config").mkdir()
    (path / "config" / "bench.yaml").write_text("dataset_version: pilot\n", encoding="utf-8")
    (path / "config" / "models.yaml").write_text("judges: {}\n", encoding="utf-8")
    (path / "config" / "models_claude_only.yaml").write_text("judges: {anthropic_opus: {}}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "init"], cwd=path, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()


def _git_repo_root(path: Path) -> Path:
    _git_repo(path)
    return path


def test_build_meta_records_commit_hashes_and_defaults(tmp_path: Path) -> None:
    commit = _git_repo(tmp_path / "repo")
    card = AgentCard(name="sample_coach", version="1", provider="tests", declared_family="undisclosed", access_method="api", attempted_families=ALL_FAMILIES, tools=frozenset({"engine"}), affiliated_with_maintainer=True)
    meta = build_meta(entrant="sample_coach", card=card, models=load_models(), cfg=load_bench(), task_set="public", registration_sha256="r" * 64, engine_path="stockfish", profile=None, unavailable_reason=None, root=tmp_path / "repo")
    assert meta.git_commit == commit == git_head(root=tmp_path / "repo")
    assert meta.client_versions == {} and meta.engine_id is None and meta.engine_sha256 is None
    assert meta.temperature == "service_default" and meta.registration_sha256 == "r" * 64 and meta.grader_version is None
    assert len(meta.models_config_hash) == 16 and meta.models_config_hash != meta.bench_config_hash


def test_build_meta_records_the_models_profile_and_hashes_that_profiles_config(tmp_path: Path) -> None:
    _git_repo(tmp_path / "repo")
    card = AgentCard(name="sample_coach", version="1", provider="tests", declared_family="anthropic", access_method="cli", attempted_families=ALL_FAMILIES, tools=frozenset(), affiliated_with_maintainer=False)
    common = {"entrant": "sample_coach", "card": card, "models": load_models(), "cfg": load_bench(), "task_set": "public", "registration_sha256": None, "engine_path": "stockfish", "root": tmp_path / "repo"}
    default = build_meta(profile=None, unavailable_reason=None, **common)
    claude_only = build_meta(profile="claude_only", unavailable_reason="claude_default is not part of the claude_only profile", **common)
    assert default.models_profile is None and default.unavailable_reason is None
    assert claude_only.models_profile == "claude_only" and claude_only.unavailable_reason is not None
    assert claude_only.models_config_hash != default.models_config_hash and claude_only.bench_config_hash == default.bench_config_hash
    write_meta(meta=claude_only, run_dir=new_run_dir(entrant="sample_coach", runs_dir=tmp_path / "runs"))
    assert read_meta(run_dir=latest_run_dir(entrant="sample_coach", runs_dir=tmp_path / "runs")).models_profile == "claude_only"


def test_git_commit_that_cannot_be_read_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    bare = tmp_path / "not_a_repo"
    bare.mkdir()
    with pytest.raises(RuntimeError, match="rev-parse"):
        git_head(root=bare)


def test_entrant_provenance_follows_the_models_config() -> None:
    models = load_models()
    assert entrant_transports(entrant="gemini_default", models=models) == [models.entrants["gemini_default"].transport]
    assert entrant_transports(entrant="sample_coach", models=models) == []
    assert temperature_label(access_method="cli") == "client_default" and temperature_label(access_method="api") == "service_default"


@pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not installed")
def test_client_version_is_the_first_line_of_version_output() -> None:
    version = client_version(transport="claude_cli")
    assert version and "\n" not in version


def test_missing_client_or_engine_binary_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="cannot run"):
        client_version(transport="codex_cli")
    with pytest.raises(RuntimeError, match="not on PATH"):
        engine_identity(engine_path="no_such_engine_binary")


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish is not installed")
def test_engine_identity_reports_name_and_binary_hash() -> None:
    name, digest = engine_identity(engine_path="stockfish")
    assert "Stockfish" in name and len(digest) == 64


def test_every_transport_names_its_client_executable() -> None:
    from typing import get_args

    from bench.config import Transport
    from bench.provenance import TRANSPORT_EXECUTABLES

    assert set(TRANSPORT_EXECUTABLES) | API_TRANSPORTS == set(get_args(Transport)) and not set(TRANSPORT_EXECUTABLES) & API_TRANSPORTS
    assert TRANSPORT_EXECUTABLES["agy_cli"] == "agy"


def test_an_api_transport_records_its_endpoint_instead_of_a_client_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert client_version(transport="openrouter_api") == f"openrouter api {OPENROUTER_CHAT_URL}"


def test_a_set_temperature_is_recorded_and_an_unset_one_is_the_services_default() -> None:
    assert temperature_label(access_method="api", temperature=None) == "service_default"
    assert temperature_label(access_method="api", temperature=0.7) == "0.7"
    assert temperature_label(access_method="cli", temperature=None) == "client_default"


def test_build_meta_records_the_openrouter_endpoint_and_the_entrants_temperature(tmp_path: Path) -> None:
    _git_repo(tmp_path / "repo")
    models = load_models()
    models.entrants["gemini_openrouter"] = models.entrants["gemini_openrouter"].model_copy(update={"temperature": 1.0})
    card = AgentCard(name="gemini_openrouter", version="google/gemini-4-argon", provider="OpenRouter, routed to Google AI Studio", declared_family="google", access_method="api", attempted_families=ALL_FAMILIES, tools=frozenset(), affiliated_with_maintainer=False)
    meta = build_meta(entrant="gemini_openrouter", card=card, models=models, cfg=load_bench(), task_set="public", registration_sha256=None, engine_path="stockfish", profile=None, unavailable_reason=None, root=tmp_path / "repo")
    assert meta.client_versions == {"openrouter_api": f"openrouter api {OPENROUTER_CHAT_URL}"} and meta.temperature == "1.0"


class QuotaAdapter(FakeAdapter):
    def __init__(self, *, answers_before_quota: int) -> None:
        super().__init__()
        self.answers_before_quota = answers_before_quota

    def answer(self, *, task: Task, run_index: int, timeout_s: float) -> AnswerRecord:
        from bench.llm import LLMQuotaExhausted

        if len(self.calls) >= self.answers_before_quota:
            self.calls.append((task.task_id, run_index))
            raise LLMQuotaExhausted(message="agy quota reached for gemini-3.8-flash-high: resets in 166h59m5s", resets_at=None)
        return super().answer(task=task, run_index=run_index, timeout_s=timeout_s)


def test_a_quota_error_stops_the_run_without_recording_failed_answers(tmp_path: Path) -> None:
    from bench.llm import LLMQuotaExhausted

    adapter = QuotaAdapter(answers_before_quota=2)
    with pytest.raises(LLMQuotaExhausted, match="166h59m5s"):
        run_entrant(adapter=adapter, tasks=_tasks(), runs_per_task=3, timeout_s=5, concurrency=1, out_dir=tmp_path)
    assert len(adapter.calls) == 3
    records = read_answers(run_dir=tmp_path)
    assert len(records) == 2 and all(r.status == "ok" for r in records)


def test_an_entrant_with_local_tools_answers_one_question_at_a_time() -> None:
    from bench.adapters.base import AgentCard
    from bench.cli.run import answer_concurrency
    from bench.config import load_bench

    cfg = load_bench()
    common = {"version": "1", "provider": "anthropic", "declared_family": "anthropic", "access_method": "cli", "attempted_families": frozenset({"assessment_plan"}), "affiliated_with_maintainer": False}
    with_tools = AgentCard(name="claude_tools", tools=frozenset({"code"}), client_tools=True, **common)
    plain = AgentCard(name="claude_default", tools=frozenset(), **common)
    assert cfg.client_tools_concurrency == 1
    assert answer_concurrency(cfg=cfg, card=with_tools, requested=None) == 1
    assert answer_concurrency(cfg=cfg, card=plain, requested=None) == cfg.concurrency
    assert answer_concurrency(cfg=cfg, card=with_tools, requested=2) == 2


def test_build_meta_records_a_dirty_tree_of_hashed_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    (root / "bench").mkdir()
    (root / "bench" / "score.py").write_text("X = 1\n", encoding="utf-8")
    (root / "notes.txt").write_text("free\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "files"], cwd=root, check=True)
    card = AgentCard(name="sample_coach", version="1", provider="tests", declared_family="undisclosed", access_method="api", attempted_families=ALL_FAMILIES, tools=frozenset(), affiliated_with_maintainer=True)
    common = {"entrant": "sample_coach", "card": card, "models": load_models(), "cfg": load_bench(), "task_set": "public", "registration_sha256": None, "engine_path": "stockfish", "profile": None, "unavailable_reason": None, "root": root}
    (root / "notes.txt").write_text("changed\n", encoding="utf-8")
    assert build_meta(**common).git_dirty is False
    (root / "bench" / "score.py").write_text("X = 2\n", encoding="utf-8")
    assert build_meta(**common).git_dirty is True


def _meta_for(*, registration_sha256: str | None) -> RunMeta:
    card = AgentCard(name="claude_tools", version="claude-sonnet-5-5", provider="tests", declared_family="anthropic", access_method="cli", attempted_families=ALL_FAMILIES, tools=frozenset({"code"}), affiliated_with_maintainer=False, client_tools=True, effort="medium")
    return RunMeta(
        entrant="claude_tools", card=card, dataset_version="v1", task_set="public", grader_version=None, started_at=datetime(2026, 10, 20, 1, tzinfo=UTC), models_config_hash="a", bench_config_hash="b",
        git_commit="c" * 40, client_versions={}, engine_id=None, engine_sha256=None, temperature="client_default", registration_sha256=registration_sha256, git_dirty=False,
    )


def test_starting_a_registered_run_appends_one_attempt_line_with_repo_relative_dir_and_identity(tmp_path: Path) -> None:
    from bench.runs import ledger_path, read_ledger, start_run

    run_dir = start_run(meta=_meta_for(registration_sha256="r" * 64), runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")
    assert read_meta(run_dir=run_dir).entrant == "claude_tools"
    lines = read_ledger(ledger=ledger_path(dataset_version="v1", registrations_dir=tmp_path / "registrations"))
    assert len(lines) == 1 and lines[0].kind == "attempt" and lines[0].run_dir == run_dir.relative_to(tmp_path).as_posix()
    line = lines[0]
    assert (line.entrant, line.version, line.declared_family, line.access_method, line.tools, line.client_tools, line.affiliated_with_maintainer, line.effort) == ("claude_tools", "claude-sonnet-5-5", "anthropic", "cli", ["code"], True, False, "medium")
    assert (line.git_commit, line.registration_sha256, line.task_set, line.reason) == ("c" * 40, "r" * 64, "public", None)
    assert (tmp_path / "registrations" / "v1_attempts.jsonl").is_file()


def test_an_unregistered_run_writes_no_attempt_line(tmp_path: Path) -> None:
    from bench.runs import start_run

    start_run(meta=_meta_for(registration_sha256=None), runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")
    assert not (tmp_path / "registrations").exists()


def test_make_run_stops_a_with_tools_entrant_before_anything_runs_when_the_entrant_user_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli import run as run_cli
    from bench.config import DataEnv

    monkeypatch.setattr(run_cli, "DataEnv", lambda: DataEnv(_env_file=None, private_data_dir=None, entrant_user=None, entrant_work_dir=None))  # type: ignore[call-arg]
    models = load_models()
    assert run_cli.isolation_for(name="claude_default", models=models) is None
    with pytest.raises(SystemExit, match="claude_tools not started.*ENTRANT_USER"):
        run_cli.isolation_for(name="claude_tools", models=models)


def test_a_registered_run_from_a_dirty_tree_never_starts(tmp_path: Path) -> None:
    from bench.runs import start_run

    for dirty in (True, None):
        meta = _meta_for(registration_sha256="r" * 64).model_copy(update={"git_dirty": dirty})
        with pytest.raises(ValueError, match="uncommitted|clean"):
            start_run(meta=meta, runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")
    assert not (tmp_path / "runs").exists() and not (tmp_path / "registrations").exists()
    start_run(meta=_meta_for(registration_sha256=None).model_copy(update={"git_dirty": True}), runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")


def test_a_registered_run_resumes_only_on_its_registration_its_commit_and_a_clean_tree(tmp_path: Path) -> None:
    from bench.runner import resume_problems

    root = tmp_path / "repo"
    commit = _git_repo(root)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "configs"], cwd=root, check=True)
    commit = git_head(root=root)
    meta = _meta_for(registration_sha256="r" * 64).model_copy(update={"git_commit": commit})
    assert resume_problems(meta=meta, registration_sha256="r" * 64, root=root) == []
    assert resume_problems(meta=_meta_for(registration_sha256=None), registration_sha256=None, root=root) == []
    assert any("registration" in p for p in resume_problems(meta=meta, registration_sha256="s" * 64, root=root))
    assert any("registration" in p for p in resume_problems(meta=meta, registration_sha256=None, root=root))
    assert any("commit" in p for p in resume_problems(meta=meta.model_copy(update={"git_commit": "d" * 40}), registration_sha256="r" * 64, root=root))
    (root / "config" / "bench.yaml").write_text("dataset_version: other\n", encoding="utf-8")
    assert any("uncommitted" in p for p in resume_problems(meta=meta, registration_sha256="r" * 64, root=root))


def test_a_registered_run_recorded_as_not_available_is_never_resumed(tmp_path: Path) -> None:
    from bench.runner import resume_problems

    root = tmp_path / "repo"
    _git_repo(root)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "configs"], cwd=root, check=True)
    meta = _meta_for(registration_sha256="r" * 64).model_copy(update={"git_commit": git_head(root=root), "unavailable_reason": "probe failed"})
    assert any("not available" in p for p in resume_problems(meta=meta, registration_sha256="r" * 64, root=root))


def test_a_retried_answer_keeps_the_first_attempts_tool_trace() -> None:
    from bench.runner import answer_with_retry

    class _Retried(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def answer(self, *, task: Task, run_index: int, timeout_s: float) -> AnswerRecord:
            self.calls += 1
            first = self.calls == 1
            now = datetime.now(UTC)
            return AnswerRecord(
                task_id=task.task_id, entrant=self.name, run_index=run_index, text="" if first else "a4", status="error" if first else "ok", error="transport: codex exited with 1" if first else None,
                started_at=now, first_token_at=None, finished_at=now, raw={"tool_trace": [{"tool": "Bash", "input": "cat runs/x" if first else "ls /tmp"}]},
            )

    record = answer_with_retry(adapter=_Retried(), task=_tasks()[0], run_index=0, timeout_s=10)
    assert record.raw["tool_trace"] == [{"tool": "Bash", "input": "ls /tmp"}] and record.raw["retried_tool_trace"] == [{"tool": "Bash", "input": "cat runs/x"}]


def test_a_registered_run_recorded_as_not_available_asked_no_task_and_writes_no_attempt_line(tmp_path: Path) -> None:
    from bench.runs import ledger_path, read_ledger, start_run

    meta = _meta_for(registration_sha256="r" * 64).model_copy(update={"unavailable_reason": "codex exited with 1: usage limit"})
    start_run(meta=meta, runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")
    assert read_ledger(ledger=ledger_path(dataset_version="v1", registrations_dir=tmp_path / "registrations")) == []


def test_a_run_card_lists_only_the_families_of_its_task_set_and_resumes_against_a_card_that_listed_every_family() -> None:
    from bench.cli.run import run_families
    from bench.runner import card_for_families, resume_drift

    cfg = load_bench()
    families = run_families(tasks=load_all_tasks(tasks_dir=Path("data/public/tasks")), cfg=cfg)
    assert families == frozenset({"assessment_plan", "move_choice", "move_judgement"})
    wide = _meta_for(registration_sha256=None)
    narrow = card_for_families(card=wide.card, families=families)
    assert narrow.attempted_families == families and narrow.model_copy(update={"attempted_families": wide.card.attempted_families}) == wide.card
    same = {"models_config_hash": "a", "profile": None, "entrant_config_hash": None, "entrant_details": None, "families": families}
    assert resume_drift(meta=wide, card=wide.card, **same) == []
    assert resume_drift(meta=wide.model_copy(update={"card": narrow}), card=wide.card, **same) == []
    dropped = wide.card.model_copy(update={"attempted_families": frozenset({"assessment_plan", "move_choice"})})
    assert resume_drift(meta=wide, card=dropped, **same) == ["the entrant card differs on attempted_families (['assessment_plan', 'move_choice', 'move_judgement'] -> ['assessment_plan', 'move_choice'])"]


def test_a_runs_configuration_hash_covers_the_private_configuration_that_adds_to_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.runs import config_hash, run_config_hash

    models = tmp_path / "models.yaml"
    models.write_text("entrants: {}\n", encoding="utf-8")
    monkeypatch.setattr("bench.runs.private_config_files", lambda: ())
    assert run_config_hash(path=models) == config_hash(path=models)
    overlay = tmp_path / "private_config.yaml"
    overlay.write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setattr("bench.runs.private_config_files", lambda: (overlay,))
    before = run_config_hash(path=models)
    assert before != config_hash(path=models)
    overlay.write_text("models: {entrants: {}}\n", encoding="utf-8")
    assert run_config_hash(path=models) != before
