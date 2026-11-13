import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from bench import GRADER_VERSION
from bench.grading.claims import CLAIMS_PROMPT_VERSION
from bench.grading.grade import StatedNumberRule
from bench.aggregate import JURY_RATING_LABEL, aggregate, answer_pair_share
from bench.config import BenchConfig, load_bench
from bench.grading.grade import ExtractionAgreement
from bench.models import Sheet
from tests.audience_fakes import audience_verdict, fake_tasks, rate_audience
from tests.run_fakes import fake_sheets, make_run
from tests.run_fakes import summarise_run, tasks_for

TASK_FAMILY = {"ap_0001": "assessment_plan", "ap_0002": "assessment_plan", "mj_0001": "move_judgement", "mc_0001": "move_choice", "gr_0001": "game_review"}
ALL = set(TASK_FAMILY.values())
HEADLINE = ("assessment_plan", "move_choice", "move_judgement", "game_review")
SCORES = {"ap_0001": 60, "ap_0002": 80, "mj_0001": 50, "mc_0001": 70, "gr_0001": 40}


@pytest.fixture(scope="module")
def cfg() -> BenchConfig:
    """The configuration with game_review back in the headline and the head-to-head rating on: v0 leaves both out, the code still
    supports them."""
    bench = load_bench()
    return bench.model_copy(update={"headline_families": list(HEADLINE), "calibration": bench.calibration.model_copy(update={"head_to_head_rating": True})})


@pytest.fixture
def sheets() -> dict[str, Sheet]:
    return fake_sheets(task_ids=list(TASK_FAMILY))


def _run(tmp_path: Path, sheets: dict[str, Sheet], *, entrant: str, families: set[str] | None = None, scores: dict[str, float] | None = None, **kwargs) -> Path:
    return make_run(tmp_path, entrant=entrant, families=families or ALL, scores=scores or SCORES, task_family=TASK_FAMILY, sheets=sheets, **kwargs)


def _board(cfg: BenchConfig, sheets: dict[str, Sheet], root: Path, run_dirs: list[Path], resamples: int = 300, seed: int = 3):
    return aggregate(run_dirs=run_dirs, cfg=cfg, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=resamples, seed=seed, root=root)


def test_headline_averages_every_family_including_game_review(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.aggregate.VERDICT_ONLY_ENTRANTS", ("verdict_bot",))
    full = _run(tmp_path, sheets, entrant="claude_default")
    partial = _run(tmp_path, sheets, entrant="review_only_bot", families={"game_review"}, scores={"gr_0001": 90})
    verdict_only = _run(tmp_path, sheets, entrant="verdict_bot", families={"move_judgement", "game_review"}, scores={"mj_0001": 30, "gr_0001": 20})
    board = _board(cfg, sheets, tmp_path, [partial, full, verdict_only])
    by_name = {e.name: e for e in board.entrants}
    assert [e.name for e in board.entrants] == ["claude_default", "review_only_bot", "verdict_bot"]
    assert by_name["claude_default"].group == "full" and by_name["review_only_bot"].group == "partial" and by_name["verdict_bot"].group == "verdict_only"
    assert by_name["review_only_bot"].headline is None and by_name["verdict_bot"].headline is None
    assert by_name["review_only_bot"].rank_interval is None
    claude = by_name["claude_default"]
    assert claude.headline is not None and claude.headline.mean == pytest.approx((71 + 51 + 71 + 41) / 4, abs=1e-3)
    assert claude.headline.n == 5 and claude.family_scores["game_review"].mean == 41.0
    assert claude.family_scores["assessment_plan"].mean == 71.0 and claude.family_scores["assessment_plan"].n == 2
    assert "assessment_plan" not in by_name["review_only_bot"].family_scores and "game_review" in by_name["review_only_bot"].family_scores
    assert claude.failed == 0 and by_name["review_only_bot"].failed == 0 and by_name["review_only_bot"].answers == 3
    assert claude.total_median_s["game_review"] == 10.0
    assert board.dataset_version == cfg.dataset_version == "v0" and board.grader_version == GRADER_VERSION and board.task_set == "public"
    assert board.headline_families == list(HEADLINE)
    assert json.loads(board.model_dump_json())["entrants"][0]["name"] == "claude_default"


def test_a_family_outside_the_headline_does_not_move_it_and_game_review_now_does(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    high = _run(tmp_path / "a", sheets, entrant="claude_default", scores={**SCORES, "gr_0001": 100})
    low = _run(tmp_path / "b", sheets, entrant="openai_default", scores={**SCORES, "gr_0001": 0})
    three = cfg.model_copy(update={"headline_families": ["assessment_plan", "move_judgement", "move_choice"]})
    by_name = {e.name: e for e in _board(three, sheets, tmp_path, [high, low]).entrants}
    assert by_name["claude_default"].headline.mean == by_name["openai_default"].headline.mean  # type: ignore[union-attr]
    assert by_name["claude_default"].family_scores["game_review"].mean > by_name["openai_default"].family_scores["game_review"].mean
    by_name = {e.name: e for e in _board(cfg, sheets, tmp_path, [high, low]).entrants}
    assert by_name["claude_default"].headline.mean - by_name["openai_default"].headline.mean == pytest.approx(25.0, abs=1e-3)  # type: ignore[union-attr]


def test_run_dir_is_relative_to_the_repository_root(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path / "runs", sheets, entrant="claude_default")
    entrant = _board(cfg, sheets, tmp_path, [run]).entrants[0]
    assert entrant.run_dir == "runs/claude_default/20261017_120000" and not entrant.run_dir.startswith("/")
    with pytest.raises(ValueError, match="outside the repository"):
        _board(cfg, sheets, tmp_path / "elsewhere", [run])


def test_joint_bootstrap_shares_tasks_so_a_constant_gap_has_a_point_interval(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    varied = {"ap_0001": 20, "ap_0002": 80, "mj_0001": 10, "mc_0001": 60, "gr_0001": 40}
    ahead = _run(tmp_path, sheets, entrant="claude_default", scores={k: v + 10 for k, v in varied.items()}, run_spread=False)
    behind = _run(tmp_path, sheets, entrant="openai_default", scores=varied, run_spread=False)
    board = _board(cfg, sheets, tmp_path, [behind, ahead])
    assert len(board.pairwise) == 1
    pair = board.pairwise[0]
    assert (pair.a, pair.b) == ("claude_default", "openai_default")
    assert pair.mean == pytest.approx(10.0) and pair.ci_low == pytest.approx(10.0) and pair.ci_high == pytest.approx(10.0)
    assert pair.n_tasks == 5
    ahead_result, behind_result = board.entrants
    assert ahead_result.rank_interval == (1, 1) and behind_result.rank_interval == (2, 2)
    assert ahead_result.headline is not None and ahead_result.headline.ci_low < ahead_result.headline.mean < ahead_result.headline.ci_high


def test_equal_entrants_are_not_separable(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    varied = {"ap_0001": 20, "ap_0002": 80, "mj_0001": 10, "mc_0001": 60, "gr_0001": 40}
    first = _run(tmp_path, sheets, entrant="claude_default", scores=varied)
    second = _run(tmp_path, sheets, entrant="openai_default", scores=varied)
    pair = _board(cfg, sheets, tmp_path, [first, second]).pairwise[0]
    assert pair.ci_low <= 0.0 <= pair.ci_high and abs(pair.mean) < 1e-6


def test_aggregation_is_deterministic_and_independent_of_run_order(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    first = _run(tmp_path, sheets, entrant="claude_default")
    second = _run(tmp_path, sheets, entrant="openai_default", scores={**SCORES, "ap_0001": 10})
    one = _board(cfg, sheets, tmp_path, [first, second])
    two = _board(cfg, sheets, tmp_path, [second, first])
    assert [e.model_dump() for e in one.entrants] == [e.model_dump() for e in two.entrants]
    assert one.pairwise == two.pairwise
    assert one.jury_rating == two.jury_rating and one.jury_rating is not None


def test_joint_comparison_needs_the_same_tasks(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    first = _run(tmp_path, sheets, entrant="claude_default")
    narrower = make_run(tmp_path, entrant="openai_default", families=ALL, scores=SCORES, task_family={k: v for k, v in TASK_FAMILY.items() if k != "ap_0002"}, sheets=sheets)
    with pytest.raises(ValueError, match="different assessment_plan tasks"):
        _board(cfg, sheets, tmp_path, [first, narrower])


def test_failed_answers_count_and_score_zero(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="openai_default", scores={k: 60 for k in SCORES}, failed_task="gr_0001")
    entrant = _board(cfg, sheets, tmp_path, [run], resamples=50).entrants[0]
    assert entrant.failed == 3 and entrant.family_scores["game_review"].mean == 0.0 and entrant.headline is not None


def test_incomplete_or_stale_grades_are_refused(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from bench.cli.aggregate import graded_latest_runs

    run = _run(tmp_path, sheets, entrant="claude_default")
    assert graded_latest_runs(runs_dir=tmp_path) == [run]
    _board(cfg, sheets, tmp_path, [run], resamples=20)
    changed = {**sheets, "ap_0001": sheets["ap_0001"].model_copy(update={"notes": "edited after grading"})}
    with pytest.raises(ValueError, match="ap_0001"):
        _board(cfg, changed, tmp_path, [run], resamples=20)
    lines = (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "grades.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="grades for"):
        graded_latest_runs(runs_dir=tmp_path)


def test_coverage_and_precision_are_means_of_family_means_and_uncertainty_counts_answered_items_only(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["task_id"].startswith("gr_"):
            row["score"]["coverage"] = 0.0
            row["score"]["precision"] = 0.0
            row["score"]["uncertain_items"] = 0
            row["items_total"] = 0
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = summarise_run(stated_rule=StatedNumberRule.from_config(cfg=load_bench()), tasks=tasks_for(sheets=sheets), run_dir=run, resamples=10, seed=1, headline_families=HEADLINE, runs_per_task=3, root=tmp_path)
    assert result.precision_mean == 0.5  # game reviews have no claim precision
    assert result.coverage_by_family == {"assessment_plan": 0.5, "game_review": 0.0, "move_choice": 0.5, "move_judgement": 0.5}
    assert result.coverage_mean == 0.375  # the mean of the four family means, as the headline weighs families
    assert result.uncertain_rate == 0.25  # the reviews here carry no items; with items they would count
    with_items = _run(tmp_path / "items", sheets, entrant="claude_default")
    assert summarise_run(stated_rule=StatedNumberRule.from_config(cfg=load_bench()), tasks=tasks_for(sheets=sheets), run_dir=with_items, resamples=10, seed=1, headline_families=HEADLINE, runs_per_task=3, root=tmp_path / "items").uncertain_rate == 0.25


def test_illegal_move_rate_and_count_per_answer(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default", illegal_moves={"ap_0002": 2})
    entrant = _board(cfg, sheets, tmp_path, [run], resamples=20).entrants[0]
    assert entrant.illegal_move_rate == round(3 / 15, 4) and entrant.illegal_moves_per_answer == round(6 / 15, 4)


def test_stated_scores_are_counted_with_the_share_the_code_check_confirms_whatever_the_claim_judge_said(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    from tests.verify_fakes import claim_result

    def stated(verdict: str, cp: int | None, engine_cp: int | None) -> dict[str, object]:
        return claim_result(kind="eval_band", text="+0.6", band="equal", about="task_position", checkable=True, verdict=verdict, detail="d", stated_cp=cp, engine_cp_white=engine_cp).model_dump()

    # judged false but within 5 win points of the engine: holds; judged true but 3 pawns off: does not hold
    rows[0]["claims"] = [stated("false", 60, 50), stated("true", 200, -100)]
    # no stated number, or no engine number for its position: not counted
    rows[1]["claims"] = [stated("true", 10, 0), stated("true", None, 0), stated("true", 60, None)]
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    entrant = _board(cfg, sheets, tmp_path, [run], resamples=20).entrants[0]
    assert (entrant.stated_evals, entrant.stated_eval_accuracy) == (3, round(2 / 3, 4))
    plain = _run(tmp_path / "plain", sheets, entrant="claude_default")
    silent = _board(cfg, sheets, tmp_path / "plain", [plain], resamples=20).entrants[0]
    assert (silent.stated_evals, silent.stated_eval_accuracy) == (0, None)


def test_judge_panels_and_extractor_disagreement(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    agreement = ExtractionAgreement(secondary_family="cursor", verdict_match=False, turning_point_jaccard=None, claim_count_primary=3, claim_count_secondary=3)
    run = _run(tmp_path, sheets, entrant="claude_default", judge_panel=("xai", "cursor"), agreement=agreement)
    entrant = _board(cfg, sheets, tmp_path, [run], resamples=20).entrants[0]
    assert entrant.judge_panels == {"cursor+xai": 15}
    assert entrant.extractor_disagreement == 1.0
    agreed = ExtractionAgreement(secondary_family="cursor", verdict_match=True, turning_point_jaccard=None, claim_count_primary=3, claim_count_secondary=3)
    again = _run(tmp_path / "b", sheets, entrant="claude_default", agreement=agreed)
    assert _board(cfg, sheets, tmp_path, [again], resamples=20).entrants[0].extractor_disagreement == 0.0
    none = _run(tmp_path / "c", sheets, entrant="claude_default")
    assert _board(cfg, sheets, tmp_path, [none], resamples=20).entrants[0].extractor_disagreement is None


def test_run_sd_mean_is_the_within_task_spread(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    spread = _board(cfg, sheets, tmp_path, [_run(tmp_path / "a", sheets, entrant="claude_default")], resamples=20).entrants[0]
    flat = _board(cfg, sheets, tmp_path, [_run(tmp_path / "b", sheets, entrant="claude_default", run_spread=False)], resamples=20).entrants[0]
    assert spread.run_sd_mean == pytest.approx(1.0) and flat.run_sd_mean == 0.0


def test_first_token_time_only_for_api_access(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    cli = _board(cfg, sheets, tmp_path, [_run(tmp_path / "a", sheets, entrant="claude_default")], resamples=20).entrants[0]
    api = _board(cfg, sheets, tmp_path, [_run(tmp_path / "b", sheets, entrant="sample_coach", access_method="api")], resamples=20).entrants[0]
    assert cli.ttft_median_s is None
    assert api.ttft_median_s is not None and api.ttft_median_s["assessment_plan"] == 2.0


def test_task_set_signed_sheets_governance_and_small_families(tmp_path: Path, cfg: BenchConfig) -> None:
    signed = fake_sheets(task_ids=list(TASK_FAMILY), signed=2)
    run = _run(tmp_path, signed, entrant="claude_default")
    board = _board(cfg, signed, tmp_path, [run], resamples=50)
    assert (board.sheets_signed, board.sheets_total) == (2, 5)
    assert board.governance == cfg.governance
    assert set(board.small_families) == set(TASK_FAMILY.values())
    assert "50 draws" in board.stats_method and "joint hierarchical bootstrap" in board.stats_method


def test_public_and_private_runs_never_share_a_leaderboard(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    public = _run(tmp_path, sheets, entrant="claude_default")
    private = _run(tmp_path, sheets, entrant="openai_default", task_set="private")
    with pytest.raises(ValueError, match="task sets"):
        _board(cfg, sheets, tmp_path, [public, private])


def test_unfinished_runs_can_be_skipped_for_an_interim_leaderboard(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    from bench.cli.aggregate import graded_latest_runs

    done = _run(tmp_path, sheets, entrant="claude_default")
    partial = _run(tmp_path, sheets, entrant="sample_coach")
    lines = (partial / "grades.jsonl").read_text(encoding="utf-8").splitlines()
    (partial / "grades.jsonl").write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    assert graded_latest_runs(runs_dir=tmp_path, skip_incomplete=True) == [done]
    with pytest.raises(ValueError, match="sample_coach"):
        graded_latest_runs(runs_dir=tmp_path)


def test_entrants_without_a_graded_run_whose_last_run_was_unavailable_are_listed(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from datetime import UTC, datetime

    from bench.adapters.base import ALL_FAMILIES, AgentCard
    from bench.aggregate import unavailable_entrants
    from bench.runs import RunMeta, new_run_dir, write_meta

    full = _run(tmp_path, sheets, entrant="claude_default")
    card = AgentCard(name="openai_default", version="gpt-6.1-sol", provider="openai", declared_family="openai", access_method="cli", attempted_families=ALL_FAMILIES, tools=frozenset(), affiliated_with_maintainer=False)
    run_dir = new_run_dir(entrant="openai_default", runs_dir=tmp_path)
    meta = RunMeta(
        entrant="openai_default",
        card=card,
        dataset_version="pilot",
        task_set="public",
        grader_version=None,
        started_at=datetime(2026, 10, 17, 23, 2, tzinfo=UTC),
        models_config_hash="a",
        bench_config_hash="b",
        git_commit="0" * 40,
        client_versions={},
        engine_id=None,
        engine_sha256=None,
        temperature="client_default",
        registration_sha256=None,
        unavailable_reason="codex exited with 1: usage limit until 01:52",
    )
    write_meta(meta=meta, run_dir=run_dir)
    listed = unavailable_entrants(runs_dir=tmp_path)
    assert [(u.name, u.version) for u in listed] == [("openai_default", "gpt-6.1-sol")]
    assert "usage" not in listed[0].note and "codex" not in listed[0].note.lower() and listed[0].attempted_at.year == 2026
    board = aggregate(run_dirs=[full], cfg=cfg, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=50, seed=1, root=tmp_path, unavailable=listed)
    assert [u.name for u in board.unavailable] == ["openai_default"] and [e.name for e in board.entrants] == ["claude_default"]
    assert unavailable_entrants(runs_dir=tmp_path / "claude_default") == []


def test_the_leaderboard_names_its_grading_profile_instruments_judge_families_and_the_own_family_roles_of_each_entrant(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    panel = ("anthropic_opus",)
    claude = _run(tmp_path, sheets, entrant="claude_default", grading_profile="claude_only", self_family=True, judge_panel=panel, claim_judge_model="claude-opus-5-5")
    other = _run(tmp_path, sheets, entrant="sample_coach", declared_family="undisclosed", grading_profile="claude_only", judge_panel=panel, claim_judge_model="claude-opus-5-5")
    board = _board(cfg, sheets, tmp_path, [claude, other], resamples=50)
    by_name = {e.name: e for e in board.entrants}
    assert board.grading_profiles == ["claude_only"]
    assert board.grading_instruments == {"claim_judge": "claude-opus-5-5", "claims_prompt": CLAIMS_PROMPT_VERSION, "extractor": "m1", "jury": "m", "secondary_extractor": "m2"}
    assert by_name["claude_default"].self_family_roles == ["claim_judge", "extractor", "judge_anthropic_opus", "secondary_extractor"] and by_name["sample_coach"].self_family_roles == []
    assert by_name["claude_default"].judge_vendors == ["anthropic"] and by_name["sample_coach"].judge_vendors == ["anthropic"]
    assert by_name["sample_coach"].grading_transports == ["openrouter_api"] and by_name["sample_coach"].answer_transports == []
    default = _board(cfg, sheets, tmp_path, [_run(tmp_path / "d", sheets, entrant="claude_default")], resamples=50)
    assert default.grading_profiles == ["default"] and default.entrants[0].self_family_roles == []
    assert default.entrants[0].judge_vendors == ["cursor", "xai"]


def test_a_board_never_mixes_a_one_judge_panel_with_a_three_judge_panel(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    one = _run(tmp_path / "a", sheets, entrant="claude_default", claim_judge_model="openai/gpt-6.1-terra", judge_panel=("openai",), judge_models={"openai": "openai/gpt-6.1-terra"})
    three_models = {"openai": "openai/gpt-6.1-terra", "anthropic": "anthropic/claude-sonnet-5.5", "google": "google/gemini-3.8-flash"}
    three = _run(tmp_path / "b", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="openai/gpt-6.1-terra", judge_panel=tuple(three_models), judge_models=three_models)
    with pytest.raises(ValueError, match=r"different instruments(.|\n)*jury: anthropic/claude-sonnet-5.5\+google/gemini-3.8-flash\+openai/gpt-6.1-terra \(sample_coach\); openai/gpt-6.1-terra \(claude_default\)"):
        _board(cfg, sheets, tmp_path, [one, three], resamples=20)
    alone = _board(cfg, sheets, tmp_path, [one], resamples=20)
    assert alone.grading_instruments["jury"] == alone.grading_instruments["claim_judge"] == "openai/gpt-6.1-terra"


def test_a_board_whose_runs_were_graded_by_different_instruments_is_refused(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    same = _run(tmp_path / "a", sheets, entrant="claude_default", claim_judge_model="claude-opus-5-5")
    other_claim_judge = _run(tmp_path / "b", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="claude-sonnet-5-5")
    with pytest.raises(ValueError, match=r"different instruments(.|\n)*claim_judge: claude-opus-5-5 \(claude_default\); claude-sonnet-5-5 \(sample_coach\)"):
        _board(cfg, sheets, tmp_path, [same, other_claim_judge], resamples=20)
    other_extractor = _run(tmp_path / "c", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="claude-opus-5-5", extractor_models={"primary": "x1", "secondary": "m2"})
    with pytest.raises(ValueError, match="extractor: m1 \\(claude_default\\); x1 \\(sample_coach\\)"):
        _board(cfg, sheets, tmp_path, [same, other_extractor], resamples=20)
    other_jury = _run(tmp_path / "d", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="claude-opus-5-5", judge_models={"cursor": "j2", "xai": "m"})
    with pytest.raises(ValueError, match="jury: m\\+m \\(claude_default\\); j2\\+m \\(sample_coach\\)|jury: j2\\+m \\(sample_coach\\); m\\+m \\(claude_default\\)"):
        _board(cfg, sheets, tmp_path, [same, other_jury], resamples=20)
    alike = _run(tmp_path / "e", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="claude-opus-5-5")
    newer_prompt = _run(tmp_path / "f", sheets, entrant="sample_coach", declared_family="undisclosed", claim_judge_model="claude-opus-5-5")
    grades = newer_prompt / "grades.jsonl"
    grades.write_text("".join(json.dumps({**json.loads(line), "claims_prompt": "v1_rc28"}) + "\n" for line in grades.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
    with pytest.raises(ValueError, match="claims_prompt: v1_rc28 \\(sample_coach\\); v1_rc34 \\(claude_default\\)"):
        _board(cfg, sheets, tmp_path, [same, newer_prompt], resamples=20)
    assert _board(cfg, sheets, tmp_path, [same, alike], resamples=20).grading_instruments["claim_judge"] == "claude-opus-5-5"


def test_a_board_mixing_entrants_answered_through_clients_and_through_apis_is_refused(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    cli = _run(tmp_path / "a", sheets, entrant="claude_default", answer_transport="claude_cli")
    api = _run(tmp_path / "b", sheets, entrant="openai_default", declared_family="openai", answer_transport="openrouter_api")
    with pytest.raises(ValueError, match="command-line clients \\(claude_default \\(claude_cli\\)\\) and through APIs \\(openai_default \\(openrouter_api\\)\\)"):
        _board(cfg, sheets, tmp_path, [cli, api], resamples=20)
    product = _run(tmp_path / "c", sheets, entrant="sample_coach", declared_family="undisclosed")
    assert {e.name for e in _board(cfg, sheets, tmp_path, [api, product], resamples=20).entrants} == {"openai_default", "sample_coach"}


def test_grades_of_an_older_grader_or_without_grading_roles_are_refused_until_rescored(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    old = _run(tmp_path / "a", sheets, entrant="claude_default", grader_version="v1_rc15")
    with pytest.raises(ValueError, match="make rescore"):
        _board(cfg, sheets, tmp_path, [old], resamples=20)
    run = _run(tmp_path / "b", sheets, entrant="claude_default")
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0].pop("grading_roles")
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="record no grading roles.*make rescore"):
        _board(cfg, sheets, tmp_path, [run], resamples=20)


def _with_claim_judges(run: Path, *, task_id: str, stops: dict[str, str]) -> None:
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["task_id"] == task_id and row["run_index"] == 0:
            row["claim_judge_runs"] = {key: {"model_id": f"m/{key}", "tool_calls": [], "stopped": stop} for key, stop in stops.items()}
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_a_grade_whose_claim_judges_mostly_did_not_answer_is_a_grading_failure_the_board_refuses(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    _with_claim_judges(run, task_id="mj_0001", stops={"a": "answered", "b": "call_cap", "c": "answered"})
    assert _board(cfg, sheets, tmp_path, [run], resamples=20).entrants
    _with_claim_judges(run, task_id="mj_0001", stops={"a": "timeout", "b": "call_cap", "c": "answered"})
    message = re.escape(f"{run} holds 1 grading failure(s)") + ".*" + re.escape("mj_0001 run 0: 1 of 3 claim judges answered, a verdict needs 2: a timeout, b call_cap, c answered") + ".*" + re.escape(f'make rescore RUN={run} DROP="mj_0001"`, then `make grade RUN={run}`')
    with pytest.raises(ValueError, match=re.compile(message, re.DOTALL)):
        _board(cfg, sheets, tmp_path, [run], resamples=20)


def test_the_uncertain_rate_leaves_failed_answers_out_of_its_denominator(tmp_path: Path, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default", failed_task="ap_0001")
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["status"] != "ok":
            row["items_total"] = 4
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = summarise_run(stated_rule=StatedNumberRule.from_config(cfg=load_bench()), tasks=tasks_for(sheets=sheets), run_dir=run, resamples=10, seed=1, headline_families=HEADLINE, runs_per_task=3, root=tmp_path)
    assert result.failed == 3 and result.uncertain_rate == 0.25


def test_a_run_whose_grades_record_no_profile_is_marked_unrecorded(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    board = _board(cfg, sheets, tmp_path, [_run(tmp_path, sheets, entrant="claude_default", grading_profile=None)], resamples=50)
    assert board.grading_profiles == ["unrecorded"]


def test_runs_graded_under_different_profiles_never_share_a_leaderboard(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    development = _run(tmp_path, sheets, entrant="claude_default", grading_profile="claude_only")
    regular = _run(tmp_path, sheets, entrant="sample_coach", declared_family="undisclosed")
    with pytest.raises(ValueError, match="grading profiles.*claude_only.*default"):
        _board(cfg, sheets, tmp_path, [development, regular])


AUDIENCES = {"ap_0001": "beginner", "ap_0002": "club", "mj_0001": "club", "mc_0001": "advanced", "gr_0001": "club"}


def _rated_run(tmp_path: Path, sheets: dict[str, Sheet]) -> Path:
    run = _run(tmp_path, sheets, entrant="claude_default")
    fits, partly, none = ("fits", "fits"), ("partly", "partly"), ("does_not_fit", "does_not_fit")
    rate_audience(
        run_dir=run,
        verdicts={
            **{("ap_0001", 0): audience_verdict(votes=fits), ("ap_0001", 1): audience_verdict(votes=fits), ("ap_0001", 2): audience_verdict(votes=partly)},
            **{("ap_0002", 0): audience_verdict(votes=partly), ("ap_0002", 1): audience_verdict(votes=("fits", "invalid")), ("ap_0002", 2): audience_verdict(votes=none)},
            **{("mj_0001", i): audience_verdict(votes=fits) for i in range(3)},
            **{("mc_0001", i): audience_verdict(votes=none) for i in range(3)},
        },
    )
    return run


def test_audience_fit_is_the_mean_of_settled_credits_overall_and_per_level(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _rated_run(tmp_path, sheets)
    tasks = fake_tasks(audiences=AUDIENCES, families=TASK_FAMILY)
    entrant = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path).entrants[0]
    assert entrant.audience_fit_mean == pytest.approx(6.0 / 11, abs=1e-4)
    assert entrant.audience_fit_by_level == {"beginner": pytest.approx(2.5 / 3, abs=1e-4), "club": pytest.approx(0.7, abs=1e-4), "advanced": 0.0}
    assert entrant.audience_uncertain_rate == pytest.approx(1 / 12, abs=1e-4)


def test_levels_without_a_rated_answer_are_left_out_and_game_review_is_not_rated(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    rate_audience(run_dir=run, verdicts={("ap_0001", 0): audience_verdict(votes=("fits", "partly")), ("ap_0001", 1): audience_verdict(votes=("fits", "partly")), ("ap_0001", 2): audience_verdict(votes=("invalid", "fits"))})
    tasks = fake_tasks(audiences=AUDIENCES, families=TASK_FAMILY)
    entrant = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path).entrants[0]
    assert entrant.audience_fit_by_level == {"beginner": 0.75} and entrant.audience_fit_mean == 0.75
    assert entrant.audience_uncertain_rate == pytest.approx(1 / 3, abs=1e-4)  # of the three answers rated; the other nine are not measured


def test_an_entrant_with_no_audience_credit_has_no_audience_fit(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    rate_audience(run_dir=run, verdicts={})
    tasks = fake_tasks(audiences=AUDIENCES, families=TASK_FAMILY)
    entrant = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path).entrants[0]
    assert entrant.audience_fit_mean is None and entrant.audience_fit_by_level == {} and entrant.audience_uncertain_rate == 0.0
    only_game_review = _run(tmp_path / "gr", sheets, entrant="review_only_bot", families={"game_review"}, scores={"gr_0001": 90})
    lucid = aggregate(run_dirs=[only_game_review], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path / "gr").entrants[0]
    assert lucid.audience_fit_mean is None and lucid.audience_uncertain_rate == 0.0


def test_a_rated_grade_for_an_unknown_task_is_refused(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _rated_run(tmp_path, sheets)
    tasks = fake_tasks(audiences={k: v for k, v in AUDIENCES.items() if k != "mj_0001"}, families=TASK_FAMILY)
    with pytest.raises(ValueError, match="mj_0001"):
        aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path)


def test_every_full_coverage_entrant_is_ranked_and_compared_in_one_table_on_a_development_board(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    development = {"grading_profile": "claude_only"}
    sample_coach = _run(tmp_path, sheets, entrant="sample_coach", declared_family="undisclosed", tools=frozenset({"analysis"}), access_method="website", scores={k: v + 20 for k, v in SCORES.items()}, **development)
    with_tools = _run(tmp_path, sheets, entrant="claude_tools", client_tools=True, tools=frozenset({"code"}), scores={k: v + 10 for k, v in SCORES.items()}, **development)
    engine_in_prompt = _run(tmp_path, sheets, entrant="claude_engine", tools=frozenset({"engine"}), **development)
    board = _board(cfg, sheets, tmp_path, [engine_in_prompt, with_tools, sample_coach], resamples=50)
    assert [e.name for e in board.entrants] == ["sample_coach", "claude_tools", "claude_engine"]
    assert {e.group for e in board.entrants} == {"full"}
    assert [e.rank_interval for e in board.entrants] == [(1, 1), (2, 2), (3, 3)]
    assert [(p.a, p.b) for p in board.pairwise] == [("sample_coach", "claude_tools"), ("sample_coach", "claude_engine"), ("claude_tools", "claude_engine")]


def test_on_a_publishable_board_an_undisclosed_family_entrant_is_ranked_and_marked(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    sample_coach = _run(tmp_path, sheets, entrant="sample_coach", declared_family="undisclosed", tools=frozenset({"analysis"}), access_method="website", scores={k: v + 20 for k, v in SCORES.items()})
    claude = _run(tmp_path, sheets, entrant="claude_openrouter", scores={k: v + 10 for k, v in SCORES.items()})
    gemini = _run(tmp_path, sheets, entrant="gemini_openrouter", declared_family="google")
    board = _board(cfg, sheets, tmp_path, [sample_coach, claude, gemini], resamples=50)
    assert [e.name for e in board.entrants] == ["sample_coach", "claude_openrouter", "gemini_openrouter"]
    assert {e.group for e in board.entrants} == {"full"}
    assert [e.family_disclosed for e in board.entrants] == [False, True, True]
    assert [e.rank_interval for e in board.entrants] == [(1, 1), (2, 2), (3, 3)]
    assert [(p.a, p.b) for p in board.pairwise] == [("sample_coach", "claude_openrouter"), ("sample_coach", "gemini_openrouter"), ("claude_openrouter", "gemini_openrouter")]
    assert all(e.coverage_headline is not None and e.precision_headline is not None for e in board.entrants)
    assert board.jury_rating is not None and {r.name for r in board.jury_rating.entrants} == {"sample_coach", "claude_openrouter", "gemini_openrouter"}


def test_the_main_board_ranks_entrants_with_and_without_tools_in_one_table(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from bench.cli import aggregate as aggregate_cli

    plain = _run(tmp_path, sheets, entrant="claude_default")
    engine_in_prompt = _run(tmp_path, sheets, entrant="claude_engine", tools=frozenset({"engine"}), scores={k: v + 10 for k, v in SCORES.items()})
    board = _board(cfg, sheets, tmp_path, [plain, engine_in_prompt], resamples=50)
    assert [(e.name, sorted(e.card.tools)) for e in board.entrants] == [("claude_engine", ["engine"]), ("claude_default", [])]
    assert {e.group for e in board.entrants} == {"full"}
    assert not hasattr(aggregate_cli, "entrants_with_tools")


def test_every_entrant_needs_the_configured_number_of_runs_per_task(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    once = _run(tmp_path, sheets, entrant="manual_test", runs=1)
    with pytest.raises(ValueError, match=r"manual_test has \[1\] graded run\(s\) per .* 3 time"):
        _board(cfg, sheets, tmp_path, [once], resamples=20)
    assert _board(cfg.model_copy(update={"runs_per_task": 1}), sheets, tmp_path, [once], resamples=20).entrants[0].name == "manual_test"


def _with_calibration(cfg: BenchConfig, **changes: float) -> BenchConfig:
    return cfg.model_copy(update={"calibration": cfg.calibration.model_copy(update=changes)})


def _with_margin(cfg: BenchConfig, margin: float) -> BenchConfig:
    return _with_calibration(cfg, pair_tie_margin=margin)


def test_answer_pair_share_counts_wins_by_more_than_the_margin_and_halves_gaps_within_it() -> None:
    share = answer_pair_share(a=np.array([60.0, 61.0, 62.0]), b=np.array([55.0, 70.0, 50.0]), margin=5.0)
    assert share == pytest.approx(5.5 / 9)
    stacked = answer_pair_share(a=np.array([[60.0, 61.0, 62.0], [10.0, 10.0, 10.0]]), b=np.array([[55.0, 70.0, 50.0], [10.0, 14.9, 5.1]]), margin=5.0)
    assert stacked == pytest.approx([5.5 / 9, 0.5])


def test_jury_rating_counts_how_often_one_answer_outscores_the_other_and_names_a_different_order(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    behind_on_jury = {"ap_0001": 70, "ap_0002": 70, "mj_0001": 50, "mc_0001": 50, "gr_0001": 50}
    ahead_on_jury = {"ap_0001": 10, "ap_0002": 10, "mj_0001": 60, "mc_0001": 53, "gr_0001": 60}
    first = _run(tmp_path, sheets, entrant="openai_default", scores=behind_on_jury)
    second = _run(tmp_path, sheets, entrant="claude_default", scores=ahead_on_jury)
    board = _board(_with_margin(cfg, 5.0), sheets, tmp_path, [second, first])
    assert [e.name for e in board.entrants] == ["openai_default", "claude_default"]
    assert board.entrants[0].rank_interval == (1, 1) and board.pairwise[0].mean == pytest.approx(9.25)
    rating = board.jury_rating
    assert rating is not None
    assert rating.label == JURY_RATING_LABEL == "derived from jury scores; not validated by players; not a chess rating"
    assert rating.tie_margin == 5.0 and rating.field_mean == 1500.0 and rating.anchor == "1500 is the mean of the 2 rated entrants on this board"
    assert [r.name for r in rating.entrants] == ["openai_default", "claude_default"]
    by_name = {r.name: r for r in rating.entrants}
    won = 1.25 + 0.5 * 1.25 + 1.25
    assert by_name["claude_default"].rating - by_name["openai_default"].rating == pytest.approx(400 * math.log10((won + 0.1) / (5 - won + 0.1)), abs=0.2)
    assert by_name["claude_default"].rating + by_name["openai_default"].rating == pytest.approx(3000.0, abs=0.1)
    for entry in rating.entrants:
        assert entry.ci_low == pytest.approx(entry.rating, abs=0.1) and entry.ci_high == pytest.approx(entry.rating, abs=0.1)
    assert [(d.headline_ahead, d.rating_ahead) for d in rating.different_order] == [("openai_default", "claude_default")]
    difference = rating.different_order[0]
    assert difference.headline_diff == pytest.approx(9.25) and difference.rating_diff == pytest.approx(by_name["claude_default"].rating - by_name["openai_default"].rating, abs=0.1)
    assert "headline" in rating.method and "5" in rating.method


def test_jury_rating_intervals_come_from_the_headline_draws(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    varied = {"ap_0001": 20, "ap_0002": 80, "mj_0001": 10, "mc_0001": 60, "gr_0001": 40}
    first = _run(tmp_path, sheets, entrant="claude_default", scores={k: v + 4 * (k == "ap_0002") for k, v in varied.items()})
    second = _run(tmp_path, sheets, entrant="openai_default", scores={**varied, "mj_0001": 30})
    board = _board(_with_margin(cfg, 1.0), sheets, tmp_path, [first, second], resamples=200)
    rating = board.jury_rating
    assert rating is not None
    for entry in rating.entrants:
        assert entry.ci_low <= entry.rating <= entry.ci_high and entry.ci_low < entry.ci_high
    assert "200" in rating.method


def test_the_head_to_head_rating_is_off_by_default_and_the_board_then_has_none(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    assert load_bench().calibration.head_to_head_rating is False
    full = _run(tmp_path, sheets, entrant="claude_default")
    twin = _run(tmp_path, sheets, entrant="openai_default")
    assert _board(_with_calibration(cfg, head_to_head_rating=False), sheets, tmp_path, [full, twin]).jury_rating is None
    assert _board(cfg, sheets, tmp_path, [full, twin]).jury_rating is not None


def test_jury_rating_rates_only_entrants_with_a_headline_and_needs_two_of_them(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    full = _run(tmp_path, sheets, entrant="claude_default")
    twin = _run(tmp_path, sheets, entrant="openai_default")
    partial = _run(tmp_path, sheets, entrant="review_only_bot", families={"game_review"}, scores={"gr_0001": 90})
    board = _board(cfg, sheets, tmp_path, [full, twin, partial])
    assert board.jury_rating is not None
    assert [(r.name, r.rating) for r in board.jury_rating.entrants] == [("claude_default", 1500.0), ("openai_default", 1500.0)]
    assert board.jury_rating.different_order == []
    elsewhere = _board(_with_calibration(cfg, rating_anchor=1200.0), sheets, tmp_path, [full, twin]).jury_rating
    assert elsewhere is not None and [r.rating for r in elsewhere.entrants] == [1200.0, 1200.0]
    assert elsewhere.field_mean == 1200.0 and elsewhere.anchor == "1200 is the mean of the 2 rated entrants on this board"
    assert _board(cfg, sheets, tmp_path, [full, partial]).jury_rating is None


def test_the_board_counts_graded_tasks_per_family_and_warns_when_a_family_is_incomplete(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from bench.aggregate import FamilyTasks, completeness_warnings
    from tests.run_fakes import fake_task_list

    run = _run(tmp_path, sheets, entrant="claude_default")
    tasks = fake_task_list(task_family={**TASK_FAMILY, "mj_0002": "move_judgement"})
    board = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=tasks, resamples=20, seed=1, root=tmp_path)
    claude = board.entrants[0]
    assert claude.family_tasks["move_judgement"] == FamilyTasks(graded=1, total=2) and claude.family_tasks["assessment_plan"] == FamilyTasks(graded=2, total=2)
    assert completeness_warnings(board=board) == ["claude_default: move_judgement graded on 1 of 2 tasks"]
    complete = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=fake_task_list(task_family=TASK_FAMILY), resamples=20, seed=1, root=tmp_path)
    assert completeness_warnings(board=complete) == []


def test_the_board_records_how_many_answer_keys_the_grandmaster_has_not_reviewed(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from tests.run_fakes import fake_task_list

    run = _run(tmp_path, sheets, entrant="claude_default")
    board = aggregate(run_dirs=[run], cfg=cfg, sheets=sheets, tasks=fake_task_list(task_family=TASK_FAMILY, reviewed=frozenset({"ap_0001"})), resamples=20, seed=1, root=tmp_path)
    assert (board.answer_keys_unreviewed, board.answer_keys_total) == (4, 5)


def test_a_grade_for_a_task_outside_the_set_does_not_hide_a_missing_task_of_its_family(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    from bench.aggregate import FamilyTasks, completeness_warnings
    from tests.run_fakes import fake_task_list

    run = _run(tmp_path, sheets, entrant="claude_default")
    families = {**TASK_FAMILY, "mj_0002": "move_judgement"}
    drafts_only = cfg.model_copy(update={"run_status_allowed": ["draft"]})
    board = aggregate(run_dirs=[run], cfg=drafts_only, sheets=sheets, tasks=fake_task_list(task_family=families, reviewed=frozenset({"mj_0001"})), resamples=20, seed=1, root=tmp_path)
    assert board.entrants[0].family_tasks["move_judgement"] == FamilyTasks(graded=0, total=1)
    assert completeness_warnings(board=board) == ["claude_default: move_judgement graded on 0 of 1 tasks"]


def _set_components(*, run: Path, coverage: dict[str, float] | None = None, precision_from_score: bool = False) -> None:
    """Coverage per task (or the score / 100), precision the score / 100 when asked; a failed answer gets 0 for both, as the grader gives it."""
    rows = [json.loads(line) for line in (run / "grades.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["score"]["coverage"] = row["score"]["score"] / 100 if coverage is None else coverage[row["task_id"]]
        if precision_from_score:
            row["score"]["precision"] = row["score"]["score"] / 100
        if row["status"] != "ok":
            row["score"]["coverage"] = row["score"]["precision"] = 0.0
    (run / "grades.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_coverage_and_precision_intervals_come_from_the_headlines_own_draws(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    varied = {"ap_0001": 20, "ap_0002": 80, "mj_0001": 10, "mc_0001": 60, "gr_0001": 40}
    run = _run(tmp_path, sheets, entrant="claude_default", scores=varied)
    _set_components(run=run, precision_from_score=True)
    entrant = _board(cfg, sheets, tmp_path, [run]).entrants[0]
    headline, coverage, precision = entrant.headline, entrant.coverage_headline, entrant.precision_headline
    assert headline is not None and coverage is not None and precision is not None
    assert headline.ci_low < headline.ci_high
    assert (coverage.mean * 100, coverage.ci_low * 100, coverage.ci_high * 100) == pytest.approx((headline.mean, headline.ci_low, headline.ci_high), abs=1e-2)
    for family, stat in entrant.family_scores.items():
        assert (entrant.coverage_scores[family].ci_low * 100, entrant.coverage_scores[family].ci_high * 100) == pytest.approx((stat.ci_low, stat.ci_high), abs=1e-2)
        if family != "game_review":
            assert (entrant.precision_scores[family].ci_low * 100, entrant.precision_scores[family].ci_high * 100) == pytest.approx((stat.ci_low, stat.ci_high), abs=1e-2)
    assert "game_review" not in entrant.precision_scores and "game_review" not in entrant.precision_by_family
    assert coverage.n == 5 and precision.n == 4  # game reviews have no precision, so its mean of family means leaves them out
    assert precision.mean == pytest.approx(float(np.mean([entrant.precision_scores[f].mean for f in ("assessment_plan", "move_choice", "move_judgement")])), abs=1e-4)


def test_identical_seeds_give_identical_coverage_and_precision_intervals_and_constant_values_give_a_point_interval(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default")
    _set_components(run=run)
    one, two = (_board(cfg, sheets, tmp_path, [run], seed=7).entrants[0] for _ in range(2))
    assert one.coverage_headline == two.coverage_headline and one.coverage_scores == two.coverage_scores and one.precision_scores == two.precision_scores
    assert one.coverage_headline != _board(cfg, sheets, tmp_path, [run], seed=8).entrants[0].coverage_headline
    precision = one.precision_headline  # the fake grades give every answer precision 0.5
    assert precision is not None and precision.mean == precision.ci_low == precision.ci_high == 0.5
    assert all(s.ci_low == s.ci_high == 0.5 for s in one.precision_scores.values())


def test_coverage_and_precision_by_family_are_the_means_of_their_intervals_and_failed_answers_count_zero(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    run = _run(tmp_path, sheets, entrant="claude_default", failed_task="ap_0001")
    _set_components(run=run, coverage={"ap_0001": 0.0, "ap_0002": 0.9, "mj_0001": 0.3, "mc_0001": 0.6, "gr_0001": 0.4})
    entrant = _board(cfg, sheets, tmp_path, [run]).entrants[0]
    assert entrant.coverage_by_family == {f: s.mean for f, s in entrant.coverage_scores.items()} == {"assessment_plan": 0.45, "game_review": 0.4, "move_choice": 0.6, "move_judgement": 0.3}
    assert entrant.precision_by_family == {f: s.mean for f, s in entrant.precision_scores.items()} == {"assessment_plan": 0.25, "move_choice": 0.5, "move_judgement": 0.5}
    assert entrant.coverage_headline is not None and entrant.coverage_headline.mean == entrant.coverage_mean == pytest.approx(0.4375)
    assert entrant.precision_headline is not None and entrant.precision_headline.mean == entrant.precision_mean == pytest.approx(0.4167, abs=1e-4)


def test_an_entrant_without_a_headline_has_family_coverage_and_precision_but_no_overall(tmp_path: Path, cfg: BenchConfig, sheets: dict[str, Sheet]) -> None:
    full = _run(tmp_path, sheets, entrant="claude_default")
    partial = _run(tmp_path, sheets, entrant="review_only_bot", families={"game_review"}, scores={"gr_0001": 90})
    lucid = next(e for e in _board(cfg, sheets, tmp_path, [full, partial]).entrants if e.name == "review_only_bot")
    assert lucid.coverage_headline is None and lucid.precision_headline is None
    assert set(lucid.coverage_scores) == {"game_review"} and lucid.precision_scores == {}
