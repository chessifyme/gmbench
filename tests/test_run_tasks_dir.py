import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench.adapters.base import ALL_FAMILIES, Adapter, AgentCard, AnswerRecord
from bench.cli import run as run_cli
from bench.config import load_bench, load_models
from bench.models import Task
from bench.runner import build_meta
from bench.runs import RunMeta, folder_tasks_sha256, read_meta, write_meta

FIXTURES = Path(__file__).parent / "fixtures"


def _folder(tmp_path: Path, *, tasks: tuple[str, ...] = ("ap_9001", "mj_9001"), sheets: tuple[str, ...] = ("ap_9001",)) -> Path:
    folder = tmp_path / "picked"
    (folder / "tasks").mkdir(parents=True)
    (folder / "sheets").mkdir()
    (folder / "reference").mkdir()
    for task_id in tasks:
        shutil.copy(FIXTURES / "tasks" / f"{task_id}.yaml", folder / "tasks" / f"{task_id}.yaml")
    for task_id in sheets:
        shutil.copy(FIXTURES / "sheets" / f"{task_id}.yaml", folder / "sheets" / f"{task_id}.yaml")
    return folder


def test_a_tasks_dir_is_its_own_task_set_and_never_mixes_with_set_or_comparison(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    source = run_cli.task_source(task_set=None, tasks_dir=folder, comparison=None)
    assert source.task_set == "folder" and source.tasks_dir == folder / "tasks"
    assert run_cli.task_source(task_set=None, tasks_dir=None, comparison=None).task_set == "public"
    with pytest.raises(SystemExit, match="either --set or --tasks-dir"):
        run_cli.task_source(task_set="private", tasks_dir=folder, comparison=None)
    with pytest.raises(SystemExit, match="comparison"):
        run_cli.task_source(task_set=None, tasks_dir=folder, comparison="sample_head_to_head")


def test_a_tasks_dir_needs_the_private_data_layout(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no tasks/ folder"):
        run_cli.task_source(task_set=None, tasks_dir=tmp_path / "missing", comparison=None)
    (tmp_path / "empty" / "tasks").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no task files"):
        run_cli.folder_tasks(tasks_dir=tmp_path / "empty" / "tasks", allowed=load_bench().run_status_allowed)


def test_folder_tasks_load_after_the_same_validation_as_make_validate(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    tasks = run_cli.folder_tasks(tasks_dir=folder / "tasks", allowed=load_bench().run_status_allowed)
    assert [t.task_id for t in tasks] == ["ap_9001", "mj_9001"]
    (folder / "tasks" / "ap_9001.yaml").rename(folder / "tasks" / "ap_9002.yaml")
    with pytest.raises(SystemExit, match="ap_9002.yaml does not match task_id"):
        run_cli.folder_tasks(tasks_dir=folder / "tasks", allowed=load_bench().run_status_allowed)


def test_folder_tasks_refuse_a_reviewed_task_without_its_sheet(tmp_path: Path) -> None:
    folder = _folder(tmp_path, sheets=())
    with pytest.raises(SystemExit, match="reviewed but has no sheet"):
        run_cli.folder_tasks(tasks_dir=folder / "tasks", allowed=load_bench().run_status_allowed)


def test_tasks_dir_for_refuses_a_folder_run_whose_tasks_live_in_its_meta() -> None:
    with pytest.raises(SystemExit, match="tasks_dir"):
        run_cli.tasks_dir_for(task_set="folder")


class _ProductAdapter(Adapter):
    """An agent product reached through its website, recording its environment; it never answers."""

    name = "sample_coach"
    declared_family = "undisclosed"
    attempted_families = ALL_FAMILIES
    tools = frozenset({"analysis"})
    affiliated_with_maintainer = True

    def run_details(self) -> dict[str, str]:
        return {"environment": "test", "fresh_accounts": "false"}

    def card(self) -> AgentCard:
        return AgentCard(name=self.name, version="1.0", provider="Sample Labs", declared_family=self.declared_family, access_method="website", attempted_families=self.attempted_families, tools=self.tools, affiliated_with_maintainer=True)

    def answer(self, *, task: Task, run_index: int, timeout_s: float) -> AnswerRecord:
        raise AssertionError("no answer expected")


def _sample_coach() -> Adapter:
    return _ProductAdapter()


def test_a_folder_run_records_its_tasks_dir_and_the_entrants_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.runner.git_head", lambda *, root: "0" * 40)
    monkeypatch.setattr("bench.runner.git_dirty", lambda *, root, paths: False)
    adapter = _sample_coach()
    meta = build_meta(
        entrant="sample_coach",
        card=adapter.card(),
        models=load_models(),
        cfg=load_bench(),
        task_set="folder",
        registration_sha256=None,
        engine_path="stockfish",
        profile=None,
        unavailable_reason=None,
        tasks_dir=tmp_path / "picked" / "tasks",
        entrant_details=adapter.run_details(),
    )
    write_meta(meta=meta, run_dir=tmp_path)
    stored = read_meta(run_dir=tmp_path)
    assert stored.task_set == "folder" and stored.tasks_dir == str((tmp_path / "picked" / "tasks").resolve())
    assert stored.entrant_details is not None and stored.entrant_details["environment"] == "test"
    assert stored.started_at <= datetime.now(UTC)


def test_a_folder_run_is_graded_against_its_own_tasks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.runner.git_head", lambda *, root: "0" * 40)
    monkeypatch.setattr("bench.runner.git_dirty", lambda *, root, paths: False)
    adapter = _sample_coach()
    common = {"entrant": "sample_coach", "card": adapter.card(), "models": load_models(), "cfg": load_bench(), "registration_sha256": None, "engine_path": "stockfish", "profile": None, "unavailable_reason": None}
    (tmp_path / "tasks").mkdir()
    folder = build_meta(task_set="folder", tasks_dir=tmp_path / "tasks", **common)  # type: ignore[arg-type]
    assert run_cli.run_tasks_dir(meta=folder) == (tmp_path / "tasks").resolve()
    public = build_meta(task_set="public", **common)  # type: ignore[arg-type]
    assert public.tasks_dir is None and run_cli.run_tasks_dir(meta=public) == run_cli.tasks_dir_for(task_set="public")
    with pytest.raises(ValueError, match="tasks_dir"):
        build_meta(task_set="folder", **common)  # type: ignore[arg-type]


def _folder_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, folder: Path) -> RunMeta:
    monkeypatch.setattr("bench.runner.git_head", lambda *, root: "0" * 40)
    monkeypatch.setattr("bench.runner.git_dirty", lambda *, root, paths: False)
    adapter = _sample_coach()
    return build_meta(entrant="sample_coach", card=adapter.card(), models=load_models(), cfg=load_bench(), task_set="folder", registration_sha256=None, engine_path="stockfish", profile=None, unavailable_reason=None, tasks_dir=folder / "tasks")


def test_a_folder_run_records_the_hash_of_its_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _folder(tmp_path)
    meta = _folder_meta(tmp_path, monkeypatch, folder=folder)
    assert meta.tasks_sha256 == folder_tasks_sha256(tasks_dir=folder / "tasks") and len(meta.tasks_sha256) == 64
    (folder / "sheets" / "ap_9001.yaml").write_text((folder / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert folder_tasks_sha256(tasks_dir=folder / "tasks") == meta.tasks_sha256


def test_a_folder_run_whose_folder_is_gone_or_whose_tasks_changed_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _folder(tmp_path)
    meta = _folder_meta(tmp_path, monkeypatch, folder=folder)
    assert run_cli.run_tasks_dir(meta=meta) == (folder / "tasks").resolve()
    task = folder / "tasks" / "ap_9001.yaml"
    task.write_text(task.read_text(encoding="utf-8").replace("What is the best plan", "What is the plan"), encoding="utf-8")
    with pytest.raises(SystemExit, match="tasks changed since the run"):
        run_cli.run_tasks_dir(meta=meta)
    shutil.rmtree(folder)
    with pytest.raises(SystemExit, match="is gone"):
        run_cli.run_tasks_dir(meta=meta)


def test_grading_a_folder_run_whose_folder_is_gone_stops_before_any_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli import grade as grade_cli
    from bench.llm import ChatModel

    folder = _folder(tmp_path)
    meta = _folder_meta(tmp_path, monkeypatch, folder=folder)
    write_meta(meta=meta, run_dir=tmp_path)
    shutil.rmtree(folder)
    monkeypatch.setattr(ChatModel, "for_spec", lambda **kwargs: pytest.fail("no model before the folder check"))
    with pytest.raises(SystemExit, match="is gone"):
        grade_cli.grade_run(run_dir=tmp_path)


def test_a_run_resumes_only_on_the_task_source_it_started_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _folder(tmp_path)
    meta = _folder_meta(tmp_path, monkeypatch, folder=folder)
    same = run_cli.task_source(task_set=None, tasks_dir=folder, comparison=None)
    assert run_cli.source_problems(meta=meta, source=same) == []
    public = run_cli.task_source(task_set=None, tasks_dir=None, comparison=None)
    assert run_cli.source_problems(meta=meta, source=public) == [f"the run answers the folder tasks in {(folder / 'tasks').resolve()}, not the public tasks"]
    other = _folder(tmp_path / "other")
    assert "not the folder tasks in" in run_cli.source_problems(meta=meta, source=run_cli.task_source(task_set=None, tasks_dir=other, comparison=None))[0]
    task = folder / "tasks" / "ap_9001.yaml"
    task.write_text(task.read_text(encoding="utf-8").replace("What is the best plan", "What is the plan"), encoding="utf-8")
    assert run_cli.source_problems(meta=meta, source=same) == [f"the tasks in {(folder / 'tasks').resolve()} changed since the run started"]


def test_a_tasks_dir_inside_the_repository_must_be_ignored_by_git_and_never_the_public_data(tmp_path: Path) -> None:
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("private/\n", encoding="utf-8")
    for name in ("private/picked", "tracked/picked", "data/public"):
        (root / name / "tasks").mkdir(parents=True)
    outside = _folder(tmp_path / "outside")
    assert run_cli.task_source(task_set=None, tasks_dir=outside, comparison=None, root=root).task_set == "folder"
    assert run_cli.task_source(task_set=None, tasks_dir=root / "private" / "picked", comparison=None, root=root).task_set == "folder"
    with pytest.raises(SystemExit, match="inside the repository and not ignored by git"):
        run_cli.task_source(task_set=None, tasks_dir=root / "tracked" / "picked", comparison=None, root=root)
    with pytest.raises(SystemExit, match="is the public data"):
        run_cli.task_source(task_set=None, tasks_dir=root / "data" / "public", comparison=None, root=root)


def test_a_folder_board_is_refused_as_not_publishable() -> None:
    with pytest.raises(SystemExit, match="folder runs are not publishable"):
        run_cli.tasks_dir_for(task_set="folder")




def test_make_run_main_answers_a_valid_folder_and_refuses_one_that_does_not_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_runner import FakeAdapter

    folder = _folder(tmp_path)
    runs = tmp_path / "runs"
    monkeypatch.setattr(run_cli, "RUNS_DIR", runs)
    monkeypatch.setattr(run_cli, "get_adapter", lambda **kwargs: FakeAdapter())
    monkeypatch.setattr(run_cli, "isolation_for", lambda **kwargs: None)
    monkeypatch.setattr(run_cli, "models_profile", lambda: None)
    monkeypatch.setattr("bench.runner.git_head", lambda *, root: "0" * 40)
    monkeypatch.setattr("bench.runner.git_dirty", lambda *, root, paths: False)
    monkeypatch.setattr("sys.argv", ["run", "--entrant", "fake", "--tasks-dir", str(folder), "--runs", "1"])
    run_cli.main()
    run_dir = next((runs / "fake").iterdir())
    meta = read_meta(run_dir=run_dir)
    assert meta.task_set == "folder" and meta.tasks_dir == str((folder / "tasks").resolve()) and meta.registration_sha256 is None
    assert sorted(line.split('"task_id":"')[1].split('"')[0] for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == ["ap_9001", "mj_9001"]
    (folder / "sheets" / "ap_9001.yaml").unlink()
    with pytest.raises(SystemExit, match="does not validate"):
        run_cli.main()


def test_the_make_rule_passes_set_with_tasks_dir_so_the_run_refuses_both() -> None:
    import subprocess

    root = Path(__file__).parent.parent
    both = subprocess.run(["make", "-s", "-n", "run", "ENTRANT=fake", "TASKS_DIR=/somewhere", "SET=public"], cwd=root, check=True, capture_output=True, text=True).stdout
    assert "--tasks-dir /somewhere" in both and "--set public" in both
    folder_only = subprocess.run(["make", "-s", "-n", "run", "ENTRANT=fake", "TASKS_DIR=/somewhere"], cwd=root, check=True, capture_output=True, text=True, env={"PATH": __import__("os").environ["PATH"]}).stdout
    assert "--tasks-dir /somewhere" in folder_only and "--set" not in folder_only
    refused = subprocess.run(["make", "-s", "run", "ENTRANT=fake", "TASKS_DIR=/somewhere", "SET=public"], cwd=root, check=False, capture_output=True, text=True)
    assert refused.returncode != 0 and "pass either --set or --tasks-dir, not both" in refused.stderr


def test_runs_of_a_private_folder_stay_outside_the_repository_unless_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "runs").mkdir(parents=True)
    private = run_cli.task_source(task_set=None, tasks_dir=_folder(tmp_path / "outside"), comparison=None, root=root)
    run_cli.check_runs_dir(runs_dir=tmp_path / "private_runs", source=private, registered=False, allow_runs_in_repo=False, root=root)
    with pytest.raises(SystemExit, match="outside the repository.*RUNS_DIR.*ALLOW_RUNS_IN_REPO"):
        run_cli.check_runs_dir(runs_dir=root / "runs", source=private, registered=False, allow_runs_in_repo=False, root=root)
    run_cli.check_runs_dir(runs_dir=root / "runs", source=private, registered=False, allow_runs_in_repo=True, root=root)
    public = run_cli.task_source(task_set=None, tasks_dir=None, comparison=None, root=root)
    run_cli.check_runs_dir(runs_dir=root / "runs", source=public, registered=True, allow_runs_in_repo=False, root=root)
    run_cli.check_runs_dir(runs_dir=tmp_path / "elsewhere", source=public, registered=False, allow_runs_in_repo=False, root=root)
    with pytest.raises(SystemExit, match="registered run.*inside the repository"):
        run_cli.check_runs_dir(runs_dir=tmp_path / "elsewhere", source=public, registered=True, allow_runs_in_repo=False, root=root)


def test_make_run_writes_a_folder_run_into_the_runs_dir_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_runner import FakeAdapter

    folder = _folder(tmp_path)
    runs = tmp_path / "private_runs"
    monkeypatch.setattr(run_cli, "get_adapter", lambda **kwargs: FakeAdapter())
    monkeypatch.setattr(run_cli, "isolation_for", lambda **kwargs: None)
    monkeypatch.setattr(run_cli, "models_profile", lambda: None)
    monkeypatch.setattr("bench.runner.git_head", lambda *, root: "0" * 40)
    monkeypatch.setattr("bench.runner.git_dirty", lambda *, root, paths: False)
    monkeypatch.setattr("sys.argv", ["run", "--entrant", "fake", "--tasks-dir", str(folder), "--runs", "1"])
    with pytest.raises(SystemExit, match="outside the repository"):
        run_cli.main()
    monkeypatch.setattr("sys.argv", ["run", "--entrant", "fake", "--tasks-dir", str(folder), "--runs", "1", "--runs-dir", str(runs)])
    run_cli.main()
    run_dir = next((runs / "fake").iterdir())
    assert read_meta(run_dir=run_dir).task_set == "folder" and (run_dir / "answers.jsonl").is_file()


class _ReachedModels(Exception):
    pass


def test_grade_rescore_and_rejudge_read_a_run_outside_the_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli import grade as grade_cli
    from bench.cli import rejudge as rejudge_cli
    from bench.llm import ChatModel

    folder = _folder(tmp_path)
    run_dir = tmp_path / "private_runs" / "sample_coach" / "20261105_120000"
    run_dir.mkdir(parents=True)
    write_meta(meta=_folder_meta(tmp_path, monkeypatch, folder=folder), run_dir=run_dir)
    (run_dir / "answers.jsonl").write_text('{"task_id": "ap_9001", "entrant": "sample_coach", "run_index": 0, "text": "Be3.", "status": "ok", "error": null, "started_at": "2026-11-05T12:00:00Z", "first_token_at": null, "finished_at": "2026-11-05T12:00:00Z", "raw": {}}\n', encoding="utf-8")

    def reached(**kwargs: object) -> None:
        raise _ReachedModels

    monkeypatch.setattr(ChatModel, "for_spec", reached)
    with pytest.raises(_ReachedModels):
        grade_cli.grade_run(run_dir=run_dir)
    (run_dir / "grades.jsonl").write_text('{"task_id": "ap_9001"}\n', encoding="utf-8")
    monkeypatch.setattr(rejudge_cli, "load_grade_rows", lambda **kwargs: [])
    with pytest.raises(_ReachedModels):
        rejudge_cli.rejudge_run(run_dir=run_dir)
    from bench.cli import rescore as rescore_cli

    monkeypatch.setattr(rescore_cli, "load_grade_rows", lambda **kwargs: [])
    with pytest.raises(_ReachedModels):
        rescore_cli.rescore_run(run_dir=run_dir, rejudge_tasks={"ap_9001"}, drop_tasks=set())


class _NoEngine:
    @classmethod
    def from_config(cls, *, env: object, cfg: object) -> "_NoEngine":
        return cls()

    def __enter__(self) -> "_NoEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


