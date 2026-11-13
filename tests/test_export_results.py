import json
from pathlib import Path

import pytest

from bench.config import load_bench
from bench.cli.export_results import export_run, export_runs, leftover_paths, localise, refuse_unless_publishable
from tests.run_fakes import FIXTURE_TASK_FAMILY, make_run
from tests.run_fakes import tasks_for

HOME = "/Users/" + "someone"
ALL = set(FIXTURE_TASK_FAMILY.values())
SCORES = {"ap_9001": 60, "mj_9001": 50, "mc_9001": 70, "gr_9001": 40}


def _run(root: Path, entrant: str = "claude_default") -> Path:
    return make_run(root / "runs", entrant=entrant, families=ALL, scores=SCORES, task_family=FIXTURE_TASK_FAMILY, grader_version="g1")


def _patch_first_answer(run_dir: Path, *, text: str) -> None:
    rows = [json.loads(line) for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["raw"] = {"work_dir": text}
    (run_dir / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


COMMIT = "0123456789abcdef0123456789abcdef01234567"
VERSION = load_bench().dataset_version


def test_export_copies_answers_grades_and_meta_to_the_entrant_folder(tmp_path: Path) -> None:
    run = _run(tmp_path)
    written = export_runs(run_dirs=[run], results_dir=tmp_path / "data" / "public" / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)
    target = tmp_path / "data" / "public" / "results" / VERSION / "g1" / "claude_default"
    assert sorted(p.name for p in written) == ["answers.jsonl", "export_manifest.json", "grades.jsonl", "meta.json"]
    for name in ("answers.jsonl", "grades.jsonl", "meta.json"):
        assert (target / name).read_text(encoding="utf-8") == (run / name).read_text(encoding="utf-8")
    assert not (target / "engine_cache.json").exists()


def test_paths_under_the_repository_become_repo_relative(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _patch_first_answer(run, text=f"{tmp_path.resolve()}/runs/claude_default/work")
    export_run(run_dir=run, out_dir=tmp_path / "out", root=tmp_path)
    exported = (tmp_path / "out" / "answers.jsonl").read_text(encoding="utf-8")
    assert "runs/claude_default/work" in exported and str(tmp_path.resolve()) not in exported


def test_home_paths_outside_the_repository_fail_the_export(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _patch_first_answer(run, text=f"{HOME}/scratch/notes.txt")
    with pytest.raises(ValueError, match="absolute or home path"):
        export_run(run_dir=run, out_dir=tmp_path / "out", root=tmp_path)


def test_ungraded_run_cannot_be_exported(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "grades.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="grade the run"):
        export_run(run_dir=run, out_dir=tmp_path / "out", root=tmp_path)


def test_two_runs_of_one_entrant_are_refused(tmp_path: Path) -> None:
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")
    with pytest.raises(ValueError, match="claude_default"):
        export_runs(run_dirs=[first, second], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)


def test_path_helpers() -> None:
    assert localise(text="/repo/runs/x and /repo/runs/y", root=Path("/repo")) == "runs/x and runs/y"
    assert leftover_paths(text=f"fine, but {HOME}/x and C:\\Users\\bob\\y") != []
    assert leftover_paths(text="runs/x/answers.jsonl") == []


def test_export_refuses_results_that_fail_a_release_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli import export_results
    from bench.config import load_bench
    from bench.publish_check import GateFailure

    monkeypatch.setattr(export_results, "release_gates", lambda **kwargs: [GateFailure(gate="pre_registration", message="no manifest")])
    with pytest.raises(SystemExit, match=r"not publishable(.|\n)*pre_registration: no manifest"):
        refuse_unless_publishable(root=tmp_path, cfg=load_bench(), leaderboard_path=tmp_path / "leaderboard_pilot.json", grader_version="g1")
    monkeypatch.setattr(export_results, "release_gates", lambda **kwargs: [])
    refuse_unless_publishable(root=tmp_path, cfg=load_bench(), leaderboard_path=tmp_path / "leaderboard_pilot.json", grader_version="g1")


def test_export_writes_under_the_grader_version_with_a_sha256_manifest_and_never_overwrites(tmp_path: Path) -> None:
    import hashlib

    from bench.cli.export_results import ExportManifest

    runs = [_run(tmp_path, entrant="claude_default"), _run(tmp_path, entrant="sample_coach")]
    results = tmp_path / "data" / "public" / "results"
    export_runs(run_dirs=runs, results_dir=results, dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)
    folder = results / VERSION / "g1"
    manifest = ExportManifest.model_validate_json((folder / "export_manifest.json").read_text(encoding="utf-8"))
    assert (manifest.dataset_version, manifest.grader_version, manifest.registration_sha256, manifest.git_commit) == (VERSION, "g1", None, COMMIT)
    exported = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "export_manifest.json")
    assert sorted(manifest.files) == exported and len(exported) == 6
    for name, digest in manifest.files.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="already exported"):
        export_runs(run_dirs=runs, results_dir=results, dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)
    assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before


def test_export_refuses_runs_of_several_grader_versions(tmp_path: Path) -> None:
    first = _run(tmp_path, entrant="claude_default")
    second = make_run(tmp_path / "runs", entrant="sample_coach", families=ALL, scores=SCORES, task_family=FIXTURE_TASK_FAMILY, grader_version="g2")
    with pytest.raises(ValueError, match="g1.*g2"):
        export_runs(run_dirs=[first, second], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)


def test_export_ships_the_pinned_runs_the_attempt_ledger_and_the_registration_history(tmp_path: Path) -> None:
    from bench.cli.export_results import ExportManifest

    run = _run(tmp_path)
    ledger = tmp_path / "registrations" / "pilot_attempts.jsonl"
    history = tmp_path / "registrations" / "pilot_registrations.jsonl"
    ledger.parent.mkdir()
    ledger.write_text('{"kind": "attempt"}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="pilot_registrations.jsonl is missing"):
        export_runs(run_dirs=[run], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT, ledgers=(ledger, history))
    history.write_text('{"combined_sha256": "a"}\n{"combined_sha256": "b"}\n', encoding="utf-8")
    export_runs(run_dirs=[run], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT, ledgers=(ledger, history))
    folder = tmp_path / "results" / VERSION / "g1"
    assert (folder / "pilot_attempts.jsonl").read_bytes() == ledger.read_bytes()
    assert (folder / "pilot_registrations.jsonl").read_bytes() == history.read_bytes()
    assert {"pilot_attempts.jsonl", "pilot_registrations.jsonl"} <= set(ExportManifest.model_validate_json((folder / "export_manifest.json").read_text(encoding="utf-8")).files)


def test_a_run_whose_failed_answers_were_answered_again_is_never_exported(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / "answers_failed.jsonl").write_text('{"status": "error"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="answers_failed.jsonl"):
        export_runs(run_dirs=[run], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)
    assert not (tmp_path / "results").exists()


def test_the_export_takes_the_runs_of_the_official_board(tmp_path: Path) -> None:
    from bench.aggregate import ComparisonInfo, aggregate
    from bench.cli.export_results import official_board_run_dirs
    from bench.config import load_bench
    from tests.run_fakes import fake_sheets

    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    runs = [make_run(tmp_path / "runs", entrant=name, families=ALL, scores=SCORES, task_family=FIXTURE_TASK_FAMILY, sheets=sheets) for name in ("claude_default", "sample_coach")]
    board = aggregate(run_dirs=runs, cfg=load_bench(), sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, root=tmp_path, comparison=ComparisonInfo(name="official_pilot", description="d", entrants=["claude_default", "sample_coach"]))
    assert sorted(official_board_run_dirs(board=board, root=tmp_path)) == sorted(runs)


def test_an_export_that_fails_on_a_later_run_writes_nothing(tmp_path: Path) -> None:
    clean = _run(tmp_path, entrant="claude_default")
    leaking = _run(tmp_path, entrant="sample_coach")
    _patch_first_answer(leaking, text=f"{HOME}/scratch/notes.txt")
    results = tmp_path / "results"
    with pytest.raises(ValueError, match="absolute or home path"):
        export_runs(run_dirs=[clean, leaking], results_dir=results, dataset_version=VERSION, root=tmp_path, git_commit=COMMIT)
    assert not results.exists()


def test_the_entrant_work_folder_and_home_become_placeholders(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _patch_first_answer(run, text=f"/Users/Shared/bench_work/bench_llm_x/a.py and {HOME}_entrant/.local/lib")
    export_runs(run_dirs=[run], results_dir=tmp_path / "results", dataset_version=VERSION, root=tmp_path, git_commit=COMMIT, replacements={"/Users/Shared/bench_work": "entrant_work_dir", f"{HOME}_entrant": "entrant_home"})
    exported = (tmp_path / "results" / VERSION / "g1" / "claude_default" / "answers.jsonl").read_text(encoding="utf-8")
    assert "entrant_work_dir/bench_llm_x/a.py" in exported and "entrant_home/.local/lib" in exported and "/Users/" not in exported
