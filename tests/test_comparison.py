import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bench.adapters.registry import get_adapter
from bench.aggregate import aggregate
from bench.cli.aggregate import comparison_board
from bench.cli.aggregate import main as aggregate_main
from bench.cli.run import check_comparison_entrant
from bench.cli.site import assert_comparison_board
from bench.comparison import ComparisonError, identity_problems, manifest_run_dirs
from bench.config import BenchConfig, Comparison, load_bench, load_comparison, load_comparisons, load_models
from bench.models import Sheet
from bench.publish_check import development_results
from tests.conftest import DEV_COMPARISONS
from tests.run_fakes import fake_sheets, make_run
from tests.run_fakes import tasks_for

TASK_FAMILY = {"ap_0001": "assessment_plan", "ap_0002": "assessment_plan", "mj_0001": "move_judgement", "mc_0001": "move_choice", "gr_0001": "game_review"}
ALL = set(TASK_FAMILY.values())
SCORES = {"ap_0001": 60, "ap_0002": 80, "mj_0001": 50, "mc_0001": 70, "gr_0001": 40}
CLAUDE_JUDGES = {"anthropic_opus": "claude-opus-5-5"}
CLAUDE_EXTRACTORS = {"primary": "claude-sonnet-5-5", "secondary": "claude-haiku-5-5"}
IDENTITIES: dict[str, dict[str, object]] = {
    "sample_coach": {"version": "1.0", "declared_family": "undisclosed", "access_method": "website", "tools": frozenset({"analysis"}), "client_tools": False, "affiliated": True},
    "claude_tools": {"version": "claude-sonnet-5-5", "declared_family": "anthropic", "access_method": "cli", "tools": frozenset({"code"}), "client_tools": True, "affiliated": False},
    "claude_opus_tools": {"version": "claude-opus-5-5", "declared_family": "anthropic", "access_method": "cli", "tools": frozenset({"code"}), "client_tools": True, "affiliated": False},
    "openai_tools": {"version": "gpt-6-luna", "declared_family": "openai", "access_method": "cli", "tools": frozenset({"code"}), "client_tools": True, "affiliated": False},
}
NAME = "sample_head_to_head"


@pytest.fixture(autouse=True)
def _dev_comparisons(dev_comparisons: Path) -> None:
    """Every test here reads the development comparison from its fixture copy."""


@pytest.fixture(scope="module")
def cfg() -> BenchConfig:
    return load_bench()


@pytest.fixture
def sheets() -> dict[str, Sheet]:
    return fake_sheets(task_ids=list(TASK_FAMILY))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(root: Path, sheets: dict[str, Sheet], *, entrant: str, bump: float = 0.0, **overrides: object) -> Path:
    identity = {"judge_models": CLAUDE_JUDGES, "extractor_models": CLAUDE_EXTRACTORS, **IDENTITIES[entrant], **overrides}
    return make_run(
        root / "runs",
        entrant=entrant,
        families=ALL,
        scores={k: v + bump for k, v in SCORES.items()},
        task_family=TASK_FAMILY,
        sheets=sheets,
        version=str(identity.pop("version")),
        judge_panel=tuple(CLAUDE_JUDGES),
        grading_profile="claude_only",
        self_family=entrant in ("claude_tools", "claude_opus_tools"),
        **identity,  # type: ignore[arg-type]
    )


def _comparison(run_dirs: dict[str, Path], *, root: Path) -> Comparison:
    real = load_comparison(name=NAME)
    pinned = {name: {"run_dir": path.relative_to(root).as_posix(), "answers_sha256": _sha(path / "answers.jsonl"), "grades_sha256": _sha(path / "grades.jsonl")} for name, path in run_dirs.items()}
    return Comparison.model_validate({**real.model_dump(mode="json"), "runs": pinned, "stale": None})


def _pinned(root: Path, sheets: dict[str, Sheet]) -> dict[str, Path]:
    return {
        "sample_coach": _run(root, sheets, entrant="sample_coach", bump=10),
        "claude_opus_tools": _run(root, sheets, entrant="claude_opus_tools", bump=7),
        "claude_tools": _run(root, sheets, entrant="claude_tools", bump=5),
        "openai_tools": _run(root, sheets, entrant="openai_tools"),
    }


def _dirs(comparison: Comparison, root: Path) -> list[Path]:
    return manifest_run_dirs(name=NAME, comparison=comparison, root=root, task_ids=list(TASK_FAMILY), runs_per_task=3, grading_models=load_models(profile="claude_only"))


def test_the_selector_names_exactly_the_four_entrants_with_their_identities_and_the_claude_only_jury() -> None:
    comparison = load_comparison(name=NAME)
    assert set(comparison.entrants) == {"sample_coach", "claude_opus_tools", "claude_tools", "openai_tools"}
    assert comparison.grading_profile == "claude_only"
    sample_coach, claude, openai = comparison.entrants["sample_coach"], comparison.entrants["claude_tools"], comparison.entrants["openai_tools"]
    assert (sample_coach.version, sample_coach.declared_family, sample_coach.access_method, set(sample_coach.tools), sample_coach.client_tools, sample_coach.affiliated_with_maintainer) == ("1.0", "undisclosed", "website", {"analysis"}, False, True)
    assert (claude.version, claude.declared_family, claude.access_method, set(claude.tools), claude.client_tools) == ("claude-sonnet-5-5", "anthropic", "cli", {"code"}, True)
    assert (openai.version, openai.declared_family, openai.access_method, set(openai.tools), openai.client_tools) == ("gpt-6-luna", "openai", "cli", {"code"}, True)
    opus = comparison.entrants["claude_opus_tools"]
    assert (opus.version, opus.declared_family, opus.access_method, set(opus.tools), opus.client_tools, opus.affiliated_with_maintainer) == ("claude-opus-5-5", "anthropic", "cli", {"code"}, True, False)
    assert {name: run.run_dir for name, run in comparison.runs.items()} == {
        "sample_coach": "runs/sample_coach/20000101_000001",
        "claude_opus_tools": "runs/claude_opus_tools/20000101_000002",
        "claude_tools": "runs/claude_tools/20000101_000003",
        "openai_tools": "runs/openai_tools/20000101_000004",
    }


def test_the_selector_matches_the_cards_the_registry_builds_without_calling_any_model(cfg: BenchConfig) -> None:
    comparison = load_comparison(name=NAME)
    models = load_models(profile=None)
    for name in ("claude_tools", "claude_opus_tools", "openai_tools"):
        assert identity_problems(name=name, card=get_adapter(name=name, models=models, cfg=cfg).card(), expected=comparison.entrants[name]) == []


def test_the_selector_is_independent_of_the_grading_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_PROFILE", "claude_only")
    assert set(load_comparison(name=NAME).entrants) == {"sample_coach", "claude_opus_tools", "claude_tools", "openai_tools"}


def test_an_unknown_comparison_is_refused() -> None:
    with pytest.raises(ValueError, match="no comparison named 'nobody'"):
        load_comparison(name="nobody")


def test_a_manifest_run_outside_the_selector_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEV_COMPARISONS.read_text(encoding="utf-8"))
    raw["comparisons"][NAME]["runs"]["gemini_tools"] = {"run_dir": "runs/gemini_tools/x", "answers_sha256": "0" * 64, "grades_sha256": "0" * 64}
    path = tmp_path / "comparisons.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="gemini_tools"):
        load_comparisons(path=path)


def test_comparison_aggregation_uses_only_the_pinned_runs_and_filters_before_statistics(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    _run(tmp_path / "stray", sheets, entrant="openai_tools", bump=30).rename(tmp_path / "runs" / "openai_tools" / "20261018_000000")
    make_run(tmp_path / "runs", entrant="gemini_tools", families=ALL, scores=SCORES, task_family=TASK_FAMILY, sheets=sheets, client_tools=True, grading_profile="claude_only", judge_models=CLAUDE_JUDGES, extractor_models=CLAUDE_EXTRACTORS)
    comparison = _comparison(runs, root=tmp_path)
    board = comparison_board(name=NAME, comparison=comparison, cfg=cfg, root=tmp_path, sheets=sheets, task_ids=list(TASK_FAMILY), tasks=tasks_for(sheets=sheets), resamples=200)
    assert [e.name for e in board.entrants] == ["sample_coach", "claude_opus_tools", "claude_tools", "openai_tools"]
    assert board.comparison is not None and board.comparison.name == NAME and board.comparison.entrants == ["claude_opus_tools", "claude_tools", "openai_tools", "sample_coach"]
    ranked = ["sample_coach", "claude_opus_tools", "claude_tools", "openai_tools"]
    assert {(p.a, p.b) for p in board.pairwise} == {(a, b) for i, a in enumerate(ranked) for b in ranked[i + 1 :]}
    assert board.unavailable == [] and board.grading_profiles == ["claude_only"]
    alone = aggregate(run_dirs=list(runs.values()), cfg=cfg, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=200, root=tmp_path)
    assert [(e.name, e.headline, e.rank_interval) for e in board.entrants] == [(e.name, e.headline, e.rank_interval) for e in alone.entrants]
    assert board.pairwise == alone.pairwise
    gates = {failure.gate for failure in development_results(leaderboard=board)}
    assert "development_profile" in gates and "self_family_grader" not in gates


def test_a_missing_entrant_never_yields_a_complete_comparison(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    del runs["openai_tools"]
    with pytest.raises(ComparisonError, match="openai_tools: no run pinned in the manifest"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)


def test_an_identity_mismatch_is_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    runs["claude_tools"] = _run(tmp_path / "other", sheets, entrant="claude_tools", version="claude-haiku-5-5")
    with pytest.raises(ComparisonError, match="claude_tools: card version is 'claude-haiku-5-5', the selector requires 'claude-sonnet-5-5'"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)
    runs["claude_tools"] = _run(tmp_path / "third", sheets, entrant="claude_tools", client_tools=False)
    with pytest.raises(ComparisonError, match="claude_tools: card client_tools is False"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)


def test_a_run_dir_of_another_entrant_is_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    runs["openai_tools"] = runs["claude_tools"]
    with pytest.raises(ComparisonError, match="openai_tools: the pinned run belongs to entrant 'claude_tools'"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)


def test_changed_result_files_are_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    comparison = _comparison(runs, root=tmp_path)
    with (runs["sample_coach"] / "grades.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ComparisonError, match="sample_coach: grades.jsonl sha256 differs from the manifest"):
        _dirs(comparison, tmp_path)


def _rewrite(path: Path, *, keep: object) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.write_text("".join(json.dumps(row) + "\n" for row in keep(rows)), encoding="utf-8")  # type: ignore[operator]


def test_missing_and_duplicate_rows_are_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    _rewrite(runs["sample_coach"] / "grades.jsonl", keep=lambda rows: rows[1:])
    _rewrite(runs["openai_tools"] / "answers.jsonl", keep=lambda rows: rows + rows[:1])
    with pytest.raises(ComparisonError) as caught:
        _dirs(_comparison(runs, root=tmp_path), tmp_path)
    assert "sample_coach: grades.jsonl lacks ap_0001 run 0" in str(caught.value)
    assert "openai_tools: answers.jsonl repeats ap_0001 run 0" in str(caught.value)


def test_a_task_the_run_never_answered_is_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    comparison = _comparison(_pinned(tmp_path, sheets), root=tmp_path)
    with pytest.raises(ComparisonError, match="sample_coach: answers.jsonl lacks ap_0003 run 0"):
        manifest_run_dirs(name=NAME, comparison=comparison, root=tmp_path, task_ids=[*TASK_FAMILY, "ap_0003"], runs_per_task=3, grading_models=load_models(profile="claude_only"))


def test_grades_from_another_jury_or_profile_are_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    runs["sample_coach"] = _run(tmp_path / "other", sheets, entrant="sample_coach", judge_models={"cursor": "composer-2.5-fast", "anthropic": "claude-opus-5-5"})
    with pytest.raises(ComparisonError, match="sample_coach: ap_0001 run 0 was judged by composer-2.5-fast"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)
    runs["sample_coach"] = _run(tmp_path / "third", sheets, entrant="sample_coach", extractor_models={"primary": "composer-2.5-fast", "secondary": "claude-sonnet-5-5"})
    with pytest.raises(ComparisonError, match="sample_coach: ap_0001 run 0 was extracted by composer-2.5-fast"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)
    runs["sample_coach"] = _run(tmp_path / "fourth", sheets, entrant="sample_coach", claim_judge_model="claude-sonnet-5-5")
    with pytest.raises(ComparisonError, match=r"sample_coach: ap_0001 run 0 had its claims checked by claude-sonnet-5-5, outside the grading profile's claim judges \['claude-opus-5-5'\]"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)


def test_a_run_graded_under_another_profile_is_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    meta_path = runs["openai_tools"] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "grading_profile": "default"}), encoding="utf-8")
    with pytest.raises(ComparisonError, match="openai_tools: graded under profile 'default', the comparison requires 'claude_only'"):
        _dirs(_comparison(runs, root=tmp_path), tmp_path)


def test_check_comparison_entrant_admits_only_members_with_the_required_identity(cfg: BenchConfig) -> None:
    comparison = load_comparison(name=NAME)
    card = get_adapter(name="claude_tools", models=load_models(profile=None), cfg=cfg).card()
    check_comparison_entrant(name=NAME, comparison=comparison, entrant="claude_tools", card=card, gated_by_profile=None)
    gemini = get_adapter(name="gemini_tools", models=load_models(profile=None), cfg=cfg).card()
    with pytest.raises(SystemExit, match="gemini_tools is not part of the sample_head_to_head comparison"):
        check_comparison_entrant(name=NAME, comparison=comparison, entrant="gemini_tools", card=gemini, gated_by_profile=None)
    with pytest.raises(SystemExit, match="card version is 'gemini-3.8-flash-high'"):
        check_comparison_entrant(name=NAME, comparison=comparison, entrant="claude_tools", card=gemini.model_copy(update={"name": "claude_tools"}), gated_by_profile=None)


def test_a_comparison_entrant_gated_by_the_profile_stops_before_any_run_directory(cfg: BenchConfig) -> None:
    comparison = load_comparison(name=NAME)
    card = get_adapter(name="openai_tools", models=load_models(profile=None), cfg=cfg).card()
    with pytest.raises(SystemExit, match="answer it with BENCH_PROFILE unset, then grade it with BENCH_PROFILE=claude_only"):
        check_comparison_entrant(name=NAME, comparison=comparison, entrant="openai_tools", card=card, gated_by_profile="claude_only")


def test_the_dashboard_refuses_a_board_that_is_not_the_named_comparison(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    comparison = _comparison(runs, root=tmp_path)
    board = comparison_board(name=NAME, comparison=comparison, cfg=cfg, root=tmp_path, sheets=sheets, task_ids=list(TASK_FAMILY), tasks=tasks_for(sheets=sheets), resamples=50)
    assert_comparison_board(board=board, name=NAME, comparison=comparison, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)
    general = aggregate(run_dirs=list(runs.values()), cfg=cfg, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=50, root=tmp_path)
    with pytest.raises(SystemExit, match="is not the sample_head_to_head comparison"):
        assert_comparison_board(board=general, name=NAME, comparison=comparison, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)
    smaller = board.model_copy(update={"entrants": board.entrants[:3]})
    with pytest.raises(SystemExit, match="lacks openai_tools"):
        assert_comparison_board(board=smaller, name=NAME, comparison=comparison, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)


def test_the_dashboard_refuses_pinned_runs_that_changed_after_aggregation(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    comparison = _comparison(runs, root=tmp_path)
    board = comparison_board(name=NAME, comparison=comparison, cfg=cfg, root=tmp_path, sheets=sheets, task_ids=list(TASK_FAMILY), tasks=tasks_for(sheets=sheets), resamples=50)
    with (runs["openai_tools"] / "answers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(SystemExit, match="openai_tools: answers.jsonl sha256 differs from the manifest"):
        assert_comparison_board(board=board, name=NAME, comparison=comparison, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)


def test_the_dashboard_refuses_a_board_built_from_other_run_dirs(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    comparison = _comparison(runs, root=tmp_path)
    board = comparison_board(name=NAME, comparison=comparison, cfg=cfg, root=tmp_path, sheets=sheets, task_ids=list(TASK_FAMILY), tasks=tasks_for(sheets=sheets), resamples=50)
    moved = board.model_copy(update={"entrants": [e.model_copy(update={"run_dir": "runs/sample_coach/elsewhere"}) if e.name == "sample_coach" else e for e in board.entrants]})
    with pytest.raises(SystemExit, match="sample_coach: the board was built from runs/sample_coach/elsewhere"):
        assert_comparison_board(board=moved, name=NAME, comparison=comparison, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)


def test_an_unknown_comparison_stops_the_aggregate_cli_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["aggregate", "--comparison", "nobody"])
    with pytest.raises(SystemExit, match="no comparison named 'nobody'"):
        aggregate_main()


OFFICIAL = "official_pilot"


def _registered_attempt(root: Path, sheets: dict[str, Sheet], *, sub: str, hour: int, registrations: Path) -> Path:
    from datetime import UTC, datetime

    from bench.runs import append_attempt, ledger_path, read_meta, write_meta

    identity = {"judge_models": CLAUDE_JUDGES, "extractor_models": CLAUDE_EXTRACTORS, **IDENTITIES["sample_coach"]}
    run = make_run(root / "runs" / sub, entrant="sample_coach", families=ALL, scores=SCORES, task_family=TASK_FAMILY, sheets=sheets, version=str(identity.pop("version")), judge_panel=tuple(CLAUDE_JUDGES), grading_profile="claude_only", **identity)  # type: ignore[arg-type]
    meta = read_meta(run_dir=run).model_copy(update={"registration_sha256": "r" * 64, "git_dirty": False, "started_at": datetime(2026, 11, 1, hour, tzinfo=UTC)})
    write_meta(meta=meta, run_dir=run)
    append_attempt(ledger=ledger_path(dataset_version="pilot", registrations_dir=registrations), meta=meta, run_dir=run, root=root)
    return run


def _spreadsheet_attempt(root: Path, sheets: dict[str, Sheet], *, sub: str, hour: int, registrations: Path) -> Path:
    from datetime import UTC, datetime

    from bench.runs import append_attempt, ledger_path, read_meta, write_meta

    run = make_run(
        root / "runs" / sub, entrant="coach", families=ALL, scores=SCORES, task_family=TASK_FAMILY, sheets=sheets, version=f"Coach, Pro plan, answers entered 2026-11-0{hour}", access_method="spreadsheet",
        declared_family="undisclosed", judge_models=CLAUDE_JUDGES, extractor_models=CLAUDE_EXTRACTORS, judge_panel=tuple(CLAUDE_JUDGES), grading_profile="claude_only",
    )
    meta = read_meta(run_dir=run).model_copy(update={"registration_sha256": "r" * 64, "git_dirty": False, "started_at": datetime(2026, 11, hour, tzinfo=UTC)})
    write_meta(meta=meta, run_dir=run)
    append_attempt(ledger=ledger_path(dataset_version="pilot", registrations_dir=registrations), meta=meta, run_dir=run, root=root)
    return run


def test_a_spreadsheet_entrant_entered_again_on_a_later_date_is_still_held_to_its_first_attempt(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    registrations = tmp_path / "registrations"
    first = _spreadsheet_attempt(tmp_path, sheets, sub="first", hour=1, registrations=registrations)
    second = _spreadsheet_attempt(tmp_path, sheets, sub="second", hour=2, registrations=registrations)
    identity = {"version": "Coach, Pro plan, answers entered 2026-11-02", "declared_family": "undisclosed", "access_method": "spreadsheet", "tools": [], "client_tools": False, "affiliated_with_maintainer": False}
    run = {"run_dir": second.relative_to(tmp_path).as_posix(), "answers_sha256": _sha(second / "answers.jsonl"), "grades_sha256": _sha(second / "grades.jsonl")}
    comparison = Comparison.model_validate({"description": "official", "grading_profile": "claude_only", "stale": None, "entrants": {"coach": identity}, "runs": {"coach": run}})
    with pytest.raises(ComparisonError, match="is an earlier attempt"):
        _official_dirs(comparison, tmp_path)
    assert first.exists()


def _official(pinned: Path, *, root: Path) -> Comparison:
    real = load_comparison(name=NAME)
    run = {"run_dir": pinned.relative_to(root).as_posix(), "answers_sha256": _sha(pinned / "answers.jsonl"), "grades_sha256": _sha(pinned / "grades.jsonl")}
    return Comparison.model_validate({"description": "official", "grading_profile": "claude_only", "stale": None, "entrants": {"sample_coach": real.entrants["sample_coach"].model_dump(mode="json")}, "runs": {"sample_coach": run}})


def _official_dirs(comparison: Comparison, root: Path) -> list[Path]:
    return manifest_run_dirs(name=OFFICIAL, comparison=comparison, root=root, task_ids=list(TASK_FAMILY), runs_per_task=3, grading_models=load_models(profile="claude_only"), registrations_dir=root / "registrations")


def test_the_official_manifest_refuses_a_pin_when_an_earlier_complete_attempt_exists(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    registrations = tmp_path / "registrations"
    first = _registered_attempt(tmp_path, sheets, sub="first", hour=1, registrations=registrations)
    second = _registered_attempt(tmp_path, sheets, sub="second", hour=2, registrations=registrations)
    assert _official_dirs(_official(first, root=tmp_path), tmp_path) == [first]
    with pytest.raises(ComparisonError) as refused:
        _official_dirs(_official(second, root=tmp_path), tmp_path)
    assert "first started attempt" in str(refused.value) and first.relative_to(tmp_path).as_posix() in str(refused.value)
    assert manifest_run_dirs(name=NAME + "_draft", comparison=_official(second, root=tmp_path), root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3, grading_models=load_models(profile="claude_only"), registrations_dir=registrations) == [second]


def test_a_void_attempt_is_skipped_by_the_first_attempt_rule(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    from bench.runs import ledger_path, void_attempt

    registrations = tmp_path / "registrations"
    first = _registered_attempt(tmp_path, sheets, sub="first", hour=1, registrations=registrations)
    second = _registered_attempt(tmp_path, sheets, sub="second", hour=2, registrations=registrations)
    (first / "grades.jsonl").rename(tmp_path / "grades_moved_for_test.jsonl")
    void_attempt(ledger=ledger_path(dataset_version="pilot", registrations_dir=registrations), run_dir=first, root=tmp_path, reason="harness fault: the engine was missing")
    assert _official_dirs(_official(second, root=tmp_path), tmp_path) == [second]
    (tmp_path / "grades_moved_for_test.jsonl").rename(first / "grades.jsonl")
    with pytest.raises(ComparisonError, match="void"):
        _official_dirs(_official(first, root=tmp_path), tmp_path)


def test_an_earlier_attempt_blocks_a_later_pin_even_when_incomplete_until_it_is_declared_void(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    from bench.runs import ledger_path, void_attempt

    registrations = tmp_path / "registrations"
    first = _registered_attempt(tmp_path, sheets, sub="first", hour=1, registrations=registrations)
    lines = (first / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    (first / "answers.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    second = _registered_attempt(tmp_path, sheets, sub="second", hour=2, registrations=registrations)
    with pytest.raises(ComparisonError, match="is an earlier attempt") as refused:
        _official_dirs(_official(second, root=tmp_path), tmp_path)
    assert first.relative_to(tmp_path).as_posix() in str(refused.value)
    (first / "grades.jsonl").unlink()
    void_attempt(ledger=ledger_path(dataset_version="pilot", registrations_dir=registrations), run_dir=first, root=tmp_path, reason="harness fault: the run stopped on a full disk")
    assert _official_dirs(_official(second, root=tmp_path), tmp_path) == [second]


def test_an_earlier_attempt_whose_run_directory_is_gone_still_blocks_a_later_pin(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    registrations = tmp_path / "registrations"
    first = _registered_attempt(tmp_path, sheets, sub="first", hour=1, registrations=registrations)
    second = _registered_attempt(tmp_path, sheets, sub="second", hour=2, registrations=registrations)
    first.rename(first.parent / "moved_for_test")
    with pytest.raises(ComparisonError, match="is an earlier attempt"):
        _official_dirs(_official(second, root=tmp_path), tmp_path)


def test_an_official_pin_outside_the_ledger_is_refused(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    stray = _run(tmp_path, sheets, entrant="sample_coach")
    with pytest.raises(ComparisonError, match="not in the attempt ledger"):
        _official_dirs(_official(stray, root=tmp_path), tmp_path)


def test_the_head_to_head_is_marked_stale_until_regraded_and_re_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_comparison(name=NAME).stale is not None
    monkeypatch.setattr("sys.argv", ["aggregate", "--comparison", NAME])
    with pytest.raises(SystemExit, match="sample_head_to_head is stale: the pins are synthetic test values"):
        aggregate_main()


def test_every_reader_of_the_pins_refuses_a_stale_comparison(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    runs = _pinned(tmp_path, sheets)
    fresh = _comparison(runs, root=tmp_path)
    board = comparison_board(name=NAME, comparison=fresh, cfg=cfg, root=tmp_path, sheets=sheets, task_ids=list(TASK_FAMILY), tasks=tasks_for(sheets=sheets), resamples=50)
    stale = fresh.model_copy(update={"stale": "regrade and re-pin after a grader change"})
    with pytest.raises(ComparisonError, match="comparison sample_head_to_head is stale: regrade and re-pin after a grader change"):
        manifest_run_dirs(name=NAME, comparison=stale, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3, grading_models=load_models(profile="claude_only"))
    with pytest.raises(SystemExit, match="is stale: regrade") as refused:
        assert_comparison_board(board=board, name=NAME, comparison=stale, root=tmp_path, task_ids=list(TASK_FAMILY), runs_per_task=3)
    assert "make aggregate" not in str(refused.value)


def test_the_site_cli_names_the_stale_comparison_instead_of_asking_for_make_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli.site import main as site_main

    monkeypatch.setattr("sys.argv", ["site", "--comparison", NAME])
    with pytest.raises(SystemExit, match="comparison sample_head_to_head is stale") as refused:
        site_main()
    assert "make aggregate" not in str(refused.value)
