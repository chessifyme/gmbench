import json
import re
import shutil
from pathlib import Path

import pytest
from markupsafe import escape

from bench.aggregate import ComparisonInfo, aggregate
from bench.config import BenchConfig, load_bench, load_models_file
from bench.io_yaml import load_all_tasks
from bench.runs import TaskSet
from bench.site.boards import svg_for_fen, svg_for_pgn_at_ply
from bench.site.build import build_site
from tests.audience_fakes import audience_verdict, rate_audience
from tests.run_fakes import FIXTURE_TASK_FAMILY, fake_sheets, make_run, tasks_for

FIXTURES = Path(__file__).parent / "fixtures"
ALL = {"assessment_plan", "move_choice", "move_judgement", "game_review"}
UNRATED = {"ap_9001": None, "mj_9001": None, "mc_9001": None}
SCORES = {"ap_9001": 60, "mj_9001": 50, "mc_9001": 70, "gr_9001": 40}
# v0 leaves game_review out of the headline (config/bench.yaml); these boards put it back, as the code still supports
FOUR_FAMILIES = load_bench().model_copy(update={"headline_families": ["assessment_plan", "move_choice", "move_judgement", "game_review"]})
WITH_RATING = FOUR_FAMILIES.model_copy(update={"calibration": FOUR_FAMILIES.calibration.model_copy(update={"head_to_head_rating": True})})


def _build(tmp_path: Path, *, task_set: TaskSet = "public", signed: int = 1, api_entrant: bool = False, cfg: BenchConfig = FOUR_FAMILIES, audience_measured: bool = True) -> Path:
    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY), signed=signed)
    common = {"task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "task_set": task_set}
    full = make_run(runs, entrant="claude_default", families=ALL, scores=SCORES, declared_family="anthropic", with_trace=True, illegal_moves={"ap_9001": 2}, **common)
    rate_audience(
        run_dir=full,
        verdicts={
            **{("ap_9001", i): audience_verdict(votes=("fits", "fits"), quotes=("short everyday words", "no jargon")) for i in range(3)},
            **{("mj_9001", i): audience_verdict(votes=("partly", "does_not_fit"), quotes=("uses the word zugzwang", "assumes opening theory")) for i in range(3)},
        },
    )
    sample_coach = make_run(runs, entrant="sample_coach", families=ALL, scores={k: v + 5 for k, v in SCORES.items()}, declared_family="undisclosed", tools=frozenset({"analysis"}), affiliated=True, access_method="api" if api_entrant else "cli", audience_credit=UNRATED, **common)
    gemini = make_run(runs, entrant="gemini_default", families=ALL, scores={k: v + 5 for k, v in SCORES.items()}, declared_family="google", audience_credit=UNRATED, **common)
    partial = make_run(runs, entrant="review_only_bot", families={"game_review"}, scores={"gr_9001": 90}, declared_family="openai", tools=frozenset({"engine"}), **common)
    if not audience_measured:
        for run in (full, sample_coach, gemini, partial):
            rate_audience(run_dir=run, verdicts={})
    board = aggregate(run_dirs=[full, gemini, sample_coach, partial], cfg=cfg, sheets=sheets, tasks=load_all_tasks(tasks_dir=FIXTURES / "tasks"), resamples=50, seed=1, root=tmp_path)
    leaderboard = runs / "leaderboard_pilot.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    out = tmp_path / "site"
    build_site(leaderboard_path=leaderboard, run_dirs=[full, gemini, sample_coach, partial], tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=cfg, root=tmp_path, registrations_dir=tmp_path / "registrations")
    return out


def _page(site: Path, name: str) -> str:
    return (site / name).read_text(encoding="utf-8")


@pytest.fixture
def small_site(tmp_path: Path) -> Path:
    return _build(tmp_path)


def test_board_svgs_render() -> None:
    svg = svg_for_fen(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lastmove="e4")
    assert svg.startswith("<svg") and "lastmove" in svg
    assert "lastmove" not in svg_for_fen(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lastmove=None)
    pgn = "1. e4 e5 2. Nf3 Nc6 1-0"
    assert svg_for_pgn_at_ply(pgn=pgn, ply=2).startswith("<svg")
    with pytest.raises(ValueError):
        svg_for_pgn_at_ply(pgn=pgn, ply=9)


def test_site_builds_all_pages(small_site: Path) -> None:
    expected = ["index.html", "methodology.html", "grading_rules.html", "agents/claude_default.html", "agents/sample_coach.html", "agents/review_only_bot.html"] + [f"tasks/{t}.html" for t in FIXTURE_TASK_FAMILY]
    for name in expected:
        assert (small_site / name).exists(), name
    for path in small_site.rglob("*"):
        assert re.fullmatch(r"[a-z0-9_.]+", path.name), path.name
    index = _page(small_site, "index.html")
    assert "claude_default" in index and "review_only_bot" in index and "How to read" in index
    assert not re.search(r"<(script|link)[^>]+(src|href)=\"https?://", index)
    assert "<svg" in _page(small_site, "tasks/ap_9001.html")


def test_banner_on_every_page_while_public_or_unsigned(small_site: Path) -> None:
    banner = "Preview · public set · 1 of 4 answer keys signed · not an official result"
    for page in small_site.rglob("*.html"):
        assert banner in page.read_text(encoding="utf-8"), page.name


def test_banner_stays_on_every_board_that_is_not_an_official_result(tmp_path: Path) -> None:
    signed_private = _build(tmp_path / "a", task_set="private", signed=4)
    assert "Preview · 4 of 4 answer keys signed · not an official result" in _page(signed_private, "index.html")
    unsigned_private = _build(tmp_path / "b", task_set="private", signed=3)
    assert "Preview · 3 of 4 answer keys signed · not an official result" in _page(unsigned_private, "index.html")
    signed_public = _build(tmp_path / "c", task_set="public", signed=4)
    assert "Preview · public set · 4 of 4 answer keys signed · not an official result" in _page(signed_public, "index.html")


def test_an_official_result_has_no_banner(tmp_path: Path) -> None:
    from bench.aggregate import read_board
    from bench.site.build import banner_text

    board = read_board(path=_build(tmp_path, signed=4).parent / "runs" / "leaderboard_pilot.json")
    assert banner_text(board=board, official=True) is None
    assert banner_text(board=board, official=False) == "Preview · public set · 4 of 4 answer keys signed · not an official result"


def test_lead_names_only_the_headline_families(tmp_path: Path) -> None:
    v0 = _page(_build(tmp_path / "a", cfg=load_bench()), "index.html")
    assert "assess a position and give the plan, choose a move, judge a move." in v0 and "review a whole game" not in v0
    four = _page(_build(tmp_path / "b"), "index.html")
    assert "assess a position and give the plan, choose a move, judge a move, review a whole game." in four


def test_lead_never_claims_signed_sheets_unless_all_are(tmp_path: Path) -> None:
    draft = _page(_build(tmp_path / "a", signed=1), "index.html")
    assert "graded against grandmaster-signed sheets" not in draft and "1 of 4 answer keys are signed" in draft
    signed = _page(_build(tmp_path / "b", signed=4), "index.html")
    assert "All 4 answer keys are signed by expert chess players" in signed and "grandmaster" not in signed


def test_how_to_read_is_accurate_about_uncertain_items_and_discloses_the_reference(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert "never help or hurt" not in index and "when two intervals overlap" not in index
    assert "penalty" not in index and "must-not-claim" not in index and "leaves the coverage denominator" not in index
    assert "earns nothing, like an absent one" in _page(small_site, "grading_rules.html")
    assert "Stockfish 19" in index
    assert "joint hierarchical bootstrap shared by the score, coverage and precision" in index
    assert "The assessment plan, game review, move choice and move judgement families have fewer than 15 tasks each, so read those columns as indicative." in index
    assert 'href="grading_rules.html#what-the-board-shows"' in index and 'id="what-the-board-shows"' in _page(small_site, "grading_rules.html")


def test_how_to_read_gives_three_definitions_with_examples_worked_out_by_the_grader() -> None:
    from bench.site.build import score_definitions

    defs = score_definitions(cfg=load_bench())
    assert (defs.key_weight, defs.earned_weight, defs.coverage_example, defs.saturation_percent) == (10, 6, 100.0, 60.0)
    assert (defs.checked, defs.true, defs.precision_example, defs.min_checked, defs.few_checked, defs.few_checked_precision) == (10, 8, 80.0, 3, 2, 66.7)
    assert (defs.coverage_weight, defs.precision_weight, defs.score_example) == (0.5, 0.5, 90.0)
    scored = load_bench().model_copy(update={"weights": load_bench().weights.model_copy(update={"verdict": 0.1, "coverage": 0.45})})
    with pytest.raises(ValueError, match="weigh"):
        score_definitions(cfg=scored)


def test_how_to_read_is_three_definitions_and_a_few_short_lines(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    how = index[index.index('<div class="how definitions">') : index.index("</ul></div>")]
    text = re.sub(r"<[^>]+>", " ", how)
    assert "How much of what the expert answer key says matters the answer actually says." in text
    assert "a key worth 10, an answer making points worth 6 = 60% of the key, which already counts as full coverage, 100% (coverage saturates at 60% of the key)" in text
    assert "How much of what the answer states is true." in text
    assert "10 checkable claims, 8 true and 2 false (an illegal move counts as false) = 80%; with fewer than 3 checked claims precision is scaled down (2 true of 2 checked = 66.7%)" in text
    assert "0.5 × coverage + 0.5 × precision" in text and "0.5 × 100 + 0.5 × 80 = 90." in text
    assert how.count("<dt>") == 3 and 3 <= how.count("<li>") <= 5
    assert len(text.split()) < 400


def test_table_separates_what_each_entrant_is(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert "<th>Tools</th>" in index and "<th>Judges</th>" in index
    sample_coach_row = re.search(r'<tr data-entrant="sample_coach".*?</tr>', index, re.DOTALL)
    assert sample_coach_row is not None and "maintained by the benchmark owner" in sample_coach_row.group(0) and "analysis tools or services" in sample_coach_row.group(0) and "cursor + xai" in sample_coach_row.group(0)
    claude_row = re.search(r'<tr data-entrant="claude_default".*?</tr>', index, re.DOTALL)
    assert claude_row is not None and "maintained by the benchmark owner" not in claude_row.group(0) and "none" in claude_row.group(0)
    models = load_models_file(profile=None)
    extractor = next(iter(models.extractors.values())).model_id
    judge = next(iter(models.judges.values())).model_id
    coi = str(escape(load_bench().governance.conflict_of_interest.format(extractor=extractor, judge=judge)))
    assert f"one extractor model, {extractor}," in coi and f"one judge model, {judge}, checks those claims" in coi
    assert "a single judge from one model family grades every entrant" in coi and "no single family" not in coi and "two of the three" not in coi
    for name in ("index.html", "agents/claude_default.html", "tasks/ap_9001.html", "methodology.html"):
        assert coi in _page(small_site, name), name
    assert index.count(coi) >= 2
    assert "Maintainer: Chessify</span>" in index and not re.search(r"custodian|appointed|not yet published", index, re.IGNORECASE)


def test_columns_are_renamed_and_first_token_is_dropped_without_api_entrants(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert "Illegal moves" in index and "Hallucination" not in index
    assert ">game review<" in index and "moments only" not in index and "not in headline" not in index
    assert "a game review, whose claims are not checked yet, is scored on coverage alone" in index and "measured and shown but not scored yet" in index
    assert "First token" not in index and "First token" not in _page(small_site, "agents/claude_default.html")


def test_the_v0_board_shows_game_review_outside_the_headline(tmp_path: Path) -> None:
    index = _page(_build(tmp_path, cfg=load_bench()), "index.html")
    assert ">game review (not in headline)<" in index
    assert ">assessment plan<" in index and ">move choice<" in index and ">move judgement<" in index


def test_first_token_column_appears_when_an_entrant_has_api_timing(tmp_path: Path) -> None:
    site = _build(tmp_path, api_entrant=True)
    assert "First token s" in _page(site, "index.html") and "First token s" in _page(site, "agents/sample_coach.html")
    assert "First token" not in _page(site, "agents/claude_default.html")


def test_rank_interval_pairwise_table_and_answer_examples(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert "rank 1" in index or "rank 2" in index
    assert "Who is ahead of whom" in index and "Difference (a minus b)" in index
    assert re.search(r"<td>gemini_default</td><td>claude_default</td><td class=\"num\">\+5\.0</td>", index)
    section = index[index.index("What an answer looks like") : index.index("<h2>Tasks</h2>")]
    for task_id in FIXTURE_TASK_FAMILY:
        assert f'href="tasks/{task_id}.html"' in section
    assert "claude_default <span class=\"score\">60.0</span>" in section and "sample_coach <span class=\"score\">65.0</span>" in section
    assert "review_only_bot" not in section


def test_partial_group_has_no_headline_cell(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert 'data-entrant="review_only_bot" data-headline="none"' in index
    assert re.search(r'data-entrant="claude_default" data-headline="[0-9.]+"', index)
    assert "not attempted" in index


def test_agent_page_shows_provenance_and_noise(small_site: Path) -> None:
    sample_coach = _page(small_site, "agents/sample_coach.html")
    assert "Model family:</strong> not disclosed" in sample_coach and "Tools:</strong> analysis tools or services" in sample_coach
    assert "maintained by the benchmark owner" in sample_coach and "Judge panels:</strong> cursor + xai (12)" in sample_coach
    assert "Extractor disagreement:</strong> <span class=\"na\">not measured</span>" in sample_coach
    assert "Run-to-run spread:</strong> 1.0 points" in sample_coach
    claude = _page(small_site, "agents/claude_default.html")
    assert "Tools:</strong> none" in claude and "independent of the benchmark owner" in claude and "Answers with an illegal move:</strong> 25%" in claude


def test_task_page_shows_answers_sheet_and_trace(small_site: Path) -> None:
    page = _page(small_site, "tasks/ap_9001.html")
    assert "What is the best plan for White here?" in page
    assert "White&#39;s main plan is Be3" in page or "White's main plan is Be3" in page
    assert "Answer C14 for ap_9001 run 0" in page
    assert 'class="claim true"' in page and 'class="claim false"' in page
    assert 'class="item present"' in page and 'class="item contradicted"' in page
    assert "should play Be3" in page and "<details" in page
    assert "2 illegal moves mentioned" in page
    agent = _page(small_site, "agents/claude_default.html")
    assert "ap_9001" in agent and "anthropic" in agent and "../tasks/ap_9001.html" in agent


def test_index_lists_entrants_that_were_not_available(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from bench.aggregate import Unavailable

    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    full = make_run(runs, entrant="claude_default", families=ALL, scores=SCORES, task_family=FIXTURE_TASK_FAMILY, declared_family="anthropic", sheets=sheets)
    board = aggregate(run_dirs=[full], cfg=FOUR_FAMILIES, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path, unavailable=[Unavailable(name="openai_default", version="gpt-6.1-sol", note="the entrant's client did not answer when the run started", attempted_at=datetime(2026, 10, 17, 23, 2, tzinfo=UTC))])
    leaderboard = runs / "leaderboard_pilot.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    out = tmp_path / "site"
    build_site(leaderboard_path=leaderboard, run_dirs=[full], tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=load_bench(), root=tmp_path, registrations_dir=tmp_path / "registrations")
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Not available" in index and "openai_default" in index and "did not answer" in index and "2026-10-17" in index


def test_engine_words_are_no_longer_hidden_anywhere(small_site: Path) -> None:
    from bench.site import build

    assert not hasattr(build, "scan_for_engine_words")
    assert "Stockfish" in _page(small_site, "index.html")

def _build_profile(tmp_path: Path, *, grading_profile: str | None, waived: bool, panel: tuple[str, ...] = ("anthropic_opus",)) -> Path:
    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    common = {"task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "grading_profile": grading_profile, "judge_panel": panel}
    claude = make_run(runs, entrant="claude_default", families=ALL, scores=SCORES, declared_family="anthropic", self_family=waived, **common)
    other = make_run(runs, entrant="sample_coach", families=ALL, scores=SCORES, declared_family="undisclosed", **common)
    board = aggregate(run_dirs=[claude, other], cfg=FOUR_FAMILIES, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path)
    leaderboard = runs / "leaderboard_pilot.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    out = tmp_path / "site"
    build_site(leaderboard_path=leaderboard, run_dirs=[claude, other], tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=load_bench(), root=tmp_path, registrations_dir=tmp_path / "registrations")
    return out


def test_development_banner_and_own_family_mark_appear_for_the_claude_only_profile(tmp_path: Path) -> None:
    site = _build_profile(tmp_path, grading_profile="claude_only", waived=True)
    notice = "Development result: the board was graded under the Claude-only profile. Not publishable."
    for page in site.rglob("*.html"):
        assert notice in page.read_text(encoding="utf-8"), page.name
    index = _page(site, "index.html")
    row = re.search(r'<tr data-entrant="claude_default".*?</tr>', index, re.DOTALL)
    assert row is not None and "graded by its own family as: extractor" in row.group(0)
    assert index.count("graded by its own family as:") == 1


def test_no_development_banner_for_neutral_instruments_under_the_default_profile(tmp_path: Path) -> None:
    for panel in (("cursor", "xai"), ("openai",)):
        site = _build_profile(tmp_path / panel[0], grading_profile="default", waived=False, panel=panel)
        for page in site.rglob("*.html"):
            text = page.read_text(encoding="utf-8")
            assert "Development result:" not in text and "graded by its own family as:" not in text, page.name


def test_an_own_family_grader_is_named_on_its_row_without_a_development_banner(tmp_path: Path) -> None:
    index = _page(_build_profile(tmp_path / "own", grading_profile="default", waived=True, panel=("cursor", "xai")), "index.html")
    assert "Development result" not in index
    row = re.search(r'<tr data-entrant="claude_default".*?</tr>', index, re.DOTALL)
    assert row is not None and "graded by its own family as: extractor, judge_cursor, judge_xai, secondary_extractor" in row.group(0)


def test_index_shows_audience_fit_with_the_level_means_and_explains_it(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert '<th class="num">Audience fit</th>' in index
    claude_row = re.search(r'<tr data-entrant="claude_default".*?</tr>', index, re.DOTALL)
    assert claude_row is not None and re.search(r"6[23]%", claude_row.group(0)) and re.search(r"B — / C 6[23] / A —", claude_row.group(0))
    sample_coach_row = re.search(r'<tr data-entrant="sample_coach".*?</tr>', index, re.DOTALL)
    assert sample_coach_row is not None and "not rated" in sample_coach_row.group(0)
    rules = _page(small_site, "grading_rules.html")
    assert "audience criterion" in rules and "shown but not scored" in rules and "B / C / A" in rules
    assert "audience fit" in index and "measured and shown but not scored yet" in index and "15%" not in index


def test_agent_page_shows_audience_levels_and_votes_with_quotes(small_site: Path) -> None:
    claude = _page(small_site, "agents/claude_default.html")
    assert re.search(r"Audience fit:</strong> 6[23]%", claude) and "<h2>Audience fit</h2>" in claude
    assert "short everyday words" in claude and "uses the word zugzwang" in claude
    sample_coach = _page(small_site, "agents/sample_coach.html")
    assert "<h2>Audience fit</h2>" in sample_coach and "No answer of this entrant was rated" in sample_coach and "no valid vote" in sample_coach


def test_task_page_shows_each_runs_audience_verdict_and_every_judges_vote(small_site: Path) -> None:
    fits = _page(small_site, "tasks/ap_9001.html")
    assert "audience fits" in fits and "opus (opus_model): fits" in fits and "sonnet (sonnet_model): fits" in fits and "short everyday words" in fits
    mixed = _page(small_site, "tasks/mj_9001.html")
    assert "audience partly fits (0.25)" in mixed and "sonnet (sonnet_model): does not fit" in mixed and "assumes opening theory" in mixed
    assert "audience" not in _page(small_site, "tasks/gr_9001.html").split("<h2>Answers</h2>")[1]


def _build_runs(tmp_path: Path, run_dirs: list[Path], sheets: dict, cfg: BenchConfig = FOUR_FAMILIES) -> Path:
    board = aggregate(run_dirs=run_dirs, cfg=cfg, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path)
    leaderboard = tmp_path / "runs" / "leaderboard_pilot.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    out = tmp_path / "site"
    build_site(leaderboard_path=leaderboard, run_dirs=run_dirs, tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=load_bench(), root=tmp_path, registrations_dir=tmp_path / "registrations")
    return out


def test_task_page_shows_answers_as_plain_text(tmp_path: Path) -> None:
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    full = make_run(tmp_path / "runs", entrant="sample_coach", families=ALL, scores=SCORES, task_family=FIXTURE_TASK_FAMILY, sheets=sheets, declared_family="undisclosed", with_trace=True)
    answers_path = full / "answers.jsonl"
    rows = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = 'Play <move path="0.0.1">21. a4</move> hitting <square>b5</square>. **Strong.**'
    answers_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = _build_runs(tmp_path, [full], sheets)
    page = (out / "tasks" / f"{rows[0]['task_id']}.html").read_text(encoding="utf-8")
    assert "Play 21. a4 hitting b5. Strong." in page
    assert "move path" not in page and "&lt;square" not in page


def test_index_shows_labels_and_every_entrant_in_one_table(tmp_path: Path) -> None:
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    common = {"families": ALL, "scores": SCORES, "task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "declared_family": "anthropic"}
    plain = make_run(tmp_path / "runs", entrant="claude_default", label="Claude Opus 5.5", **common)
    with_tools = make_run(tmp_path / "runs", entrant="claude_tools", client_tools=True, tools=frozenset({"code"}), label="Claude Sonnet 5.5, with tools", **common)
    out = _build_runs(tmp_path, [plain, with_tools], sheets)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Claude Opus 5.5" in index and "Claude Sonnet 5.5, with tools" in index
    leaderboard = index[: index.index("</table>")]
    assert 'data-entrant="claude_default"' in leaderboard and 'data-entrant="claude_tools"' in leaderboard
    assert "with tools</h2>" not in index and 'class="group"' not in index
    agent = (out / "agents" / "claude_tools.html").read_text(encoding="utf-8")
    assert "Claude Sonnet 5.5, with tools" in agent and "local command and file tools on" in agent


def test_a_product_with_analysis_tools_is_described_without_naming_them(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    row = re.search(r'<tr data-entrant="sample_coach".*?</tr>', index, re.DOTALL)
    assert row is not None and "analysis tools or services" in row.group(0) and "engine" not in row.group(0).lower()
    assert "Ziuk may use analysis tools or services" in index and "Ziuk uses" not in index
    assert "Tools:</strong> analysis tools or services" in _page(small_site, "agents/sample_coach.html")


def test_a_comparison_board_names_its_selection_on_every_page_and_shows_with_tools_examples(tmp_path: Path) -> None:
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    common = {"families": ALL, "scores": SCORES, "task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "grading_profile": "claude_only"}
    sample_coach = make_run(tmp_path / "runs", entrant="sample_coach", declared_family="undisclosed", tools=frozenset({"analysis"}), affiliated=True, access_method="website", **common)
    claude = make_run(tmp_path / "runs", entrant="claude_tools", declared_family="anthropic", client_tools=True, tools=frozenset({"code"}), self_family=True, **common)
    info = ComparisonInfo(name="sample_head_to_head", description="Sample Coach against two stock models with tools.", entrants=["claude_tools", "sample_coach"])
    board = aggregate(run_dirs=[sample_coach, claude], cfg=FOUR_FAMILIES, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path, comparison=info)
    leaderboard = tmp_path / "runs" / "leaderboard_pilot_sample_head_to_head.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    out = tmp_path / "site"
    build_site(leaderboard_path=leaderboard, run_dirs=[sample_coach, claude], tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=load_bench(), root=tmp_path, registrations_dir=tmp_path / "registrations")
    for page in ("index.html", "methodology.html", "agents/sample_coach.html", "tasks/ap_9001.html"):
        text = (out / page).read_text(encoding="utf-8")
        assert "Comparison sample_head_to_head: claude_tools, sample_coach only" in text and "Sample Coach against two stock models with tools." in text
        assert "Development result" in text
    index = (out / "index.html").read_text(encoding="utf-8")
    examples = index[index.index("<h2>What an answer looks like") :] if "<h2>What an answer looks like" in index else ""
    assert 'href="tasks/ap_9001.html"' in examples and "sample_coach" in examples and "claude_tools" in examples


def _stored_board(tmp_path: Path, *, comparison: ComparisonInfo | None = None) -> tuple[Path, Path, Path]:
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    common = {"families": ALL, "task_family": FIXTURE_TASK_FAMILY, "sheets": sheets}
    first = make_run(tmp_path / "runs", entrant="claude_tools", scores=SCORES, tools=frozenset({"code"}), client_tools=True, **common)
    second = make_run(tmp_path / "runs", entrant="sample_coach", scores={k: v + 5 for k, v in SCORES.items()}, declared_family="undisclosed", tools=frozenset({"analysis"}), affiliated=True, **common)
    board = aggregate(run_dirs=[first, second], cfg=FOUR_FAMILIES, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path, comparison=comparison)
    path = tmp_path / "runs" / "leaderboard_pilot.json"
    path.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    return path, first, second


def test_the_dashboard_refuses_a_main_board_whose_grades_changed_after_aggregation(tmp_path: Path) -> None:
    from bench.cli.site import load_current_board

    path, _, second = _stored_board(tmp_path)
    assert load_current_board(path=path, root=tmp_path).entrants
    with (second / "grades.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(SystemExit) as stopped:
        load_current_board(path=path, root=tmp_path)
    message = str(stopped.value)
    assert "sample_coach" in message and "grades.jsonl" in message and "make aggregate" in message
    assert "claude_tools" not in message and "COMPARISON" not in message


def test_the_dashboard_refuses_a_board_whose_answers_changed_after_aggregation(tmp_path: Path) -> None:
    from bench.cli.site import load_current_board

    path, first, _ = _stored_board(tmp_path)
    with (first / "answers.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(SystemExit) as stopped:
        load_current_board(path=path, root=tmp_path)
    assert "claude_tools" in str(stopped.value) and "answers.jsonl" in str(stopped.value)


def test_the_dashboard_refuses_a_board_whose_meta_changed_after_aggregation(tmp_path: Path) -> None:
    from bench.cli.site import load_current_board

    info = ComparisonInfo(name="sample_head_to_head", description="d", entrants=["claude_tools", "sample_coach"])
    path, first, _ = _stored_board(tmp_path, comparison=info)
    (first / "meta.json").write_text((first / "meta.json").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        load_current_board(path=path, root=tmp_path)
    assert "claude_tools" in str(stopped.value) and "meta.json" in str(stopped.value) and "make aggregate COMPARISON=sample_head_to_head" in str(stopped.value)


def test_a_missing_run_file_makes_the_board_stale(tmp_path: Path) -> None:
    from bench.aggregate import Leaderboard, board_changes

    path, first, _ = _stored_board(tmp_path)
    (first / "grades.jsonl").unlink()
    changes = board_changes(board=Leaderboard.model_validate_json(path.read_text(encoding="utf-8")), root=tmp_path)
    assert len(changes) == 1 and "claude_tools" in changes[0] and "grades.jsonl" in changes[0] and "missing" in changes[0]


def test_an_old_board_file_that_no_longer_validates_asks_for_a_rebuild(tmp_path: Path) -> None:
    from bench.cli.site import load_current_board

    path, _, _ = _stored_board(tmp_path)
    old = json.loads(path.read_text(encoding="utf-8"))
    for entrant in old["entrants"]:
        del entrant["grades_sha256"]
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        load_current_board(path=path, root=tmp_path)
    assert "make aggregate" in str(stopped.value) and "leaderboard_pilot.json" in str(stopped.value)


def test_a_board_from_before_the_coverage_and_precision_columns_asks_for_a_rebuild_wherever_it_is_read(tmp_path: Path) -> None:
    from bench.aggregate import OldBoard, read_board
    from bench.cli.export_results import refuse_unless_publishable

    path, _, _ = _stored_board(tmp_path)
    old = json.loads(path.read_text(encoding="utf-8"))
    for entrant in old["entrants"]:
        for field in ("precision_by_family", "coverage_headline", "precision_headline", "coverage_scores", "precision_scores"):
            del entrant[field]
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(OldBoard, match="leaderboard_pilot.json was built by an older version of the aggregate .* rebuild it with `make aggregate`"):
        read_board(path=path)
    with pytest.raises(SystemExit, match="rebuild it with `make aggregate`"):
        refuse_unless_publishable(root=tmp_path, cfg=load_bench(), leaderboard_path=path, grader_version="g1")


def _main_table(index: str) -> str:
    return index[: index.index("</table>")]


def _jury_note(index: str) -> str:
    after = index[index.index("</table>") :]
    return after[: after.index("</p>") + len("</p>")] if "<p class=\"how jury\">" in after[:200] else ""


def _row(index: str, name: str) -> str:
    row = index[index.index(f'<tr data-entrant="{name}"') :]
    return row[: row.index("</tr>")]


def test_methodology_links_its_details_to_the_grading_rules_page(small_site: Path) -> None:
    methodology = _page(small_site, "methodology.html")
    rules = _page(small_site, "grading_rules.html")
    anchors = re.findall(r'href="grading_rules\.html#([a-z0-9-]+)"', methodology)
    assert anchors and 'href="grading_rules.md' not in methodology and 'href="methodology.html#tasks"' in rules
    for anchor in anchors:
        assert f'id="{anchor}"' in rules, anchor


def test_index_shows_the_stated_scores_column_after_illegal_moves(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    table = _main_table(index)
    assert table.index('<th class="num">Illegal moves</th>') < table.index('<th class="num">Stated scores</th>') < table.index('<th class="num">Uncertain</th>')
    assert "the share of the evaluations an answer states as a number" in _page(small_site, "grading_rules.html")


def test_index_shows_coverage_and_precision_beside_the_headline_and_under_each_family_score(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    table = _main_table(index)
    assert table.index('<th class="num">Headline</th>') < table.index('<th class="num">Coverage</th>') < table.index('<th class="num">Precision</th>')
    assert "score · C · P" in table
    claude = _row(index, "claude_default")
    cells = re.findall(r'<td class="num">(.*?)</td>', claude, re.DOTALL)
    assert cells[1] == '<span class="score">50%</span><br><span class="ci">[50, 50]</span>' and cells[2] == '<span class="score">50%</span><br><span class="ci">[50, 50]</span>'
    assert '<span class="ci cp">C 50% · P 50%</span>' in claude and '<span class="ci cp">C 50%</span>' in claude  # game review has no precision
    lucid = _row(index, "review_only_bot")
    assert re.findall(r'<td class="num">(.*?)</td>', lucid, re.DOTALL)[1:3] == ['<span class="na">–</span>', '<span class="na">–</span>']
    assert '<div class="scroll"><table>' in index


def test_agent_page_shows_coverage_and_precision_with_intervals_overall_and_per_family(small_site: Path) -> None:
    claude = _page(small_site, "agents/claude_default.html")
    assert "<strong>Coverage:</strong> 50% [50, 50]" in claude and "<strong>Precision:</strong> 50% [50, 50]" in claude
    assert '<th class="num">Coverage</th><th class="num">Precision</th>' in claude
    review = claude[claude.index('<tr data-family="game_review">') :]
    review = review[: review.index("</tr>")]
    assert '50% <span class="ci">[50, 50]</span>' in review and "not checked" in review


def test_the_v0_board_has_no_head_to_head_rating_column(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    assert "Head-to-head rating" not in index and 'class="num jury"' not in index and "Elo" not in index


def test_index_shows_the_jury_rating_as_a_column_beside_the_headline_with_its_label_and_anchor(tmp_path: Path) -> None:
    index = _page(_build(tmp_path, cfg=WITH_RATING), "index.html")
    table = _main_table(index)
    assert table.index('<th class="num">Headline</th>') < table.index('<th class="num">Head-to-head rating') < table.index("<th class=\"num\">Audience fit</th>")
    assert "field mean = 1500" in table
    cells = {}
    for name in ("gemini_default", "sample_coach", "claude_default"):
        cell = re.search(r'<td class="num jury"><span class="score">(\d{4})</span><br><span class="ci">\[(\d{4}), (\d{4})\]</span></td>', _row(index, name))
        assert cell is not None
        cells[name] = [int(value) for value in cell.groups()]
        assert cells[name][1] <= cells[name][0] <= cells[name][2]
    assert cells["gemini_default"][0] > 1500 > cells["claude_default"][0] and cells["sample_coach"][0] > 1500 and sum(c[0] for c in cells.values()) in (4499, 4500, 4501)
    assert '<td class="num jury"><span class="na">not rated</span></td>' in _row(index, "review_only_bot")
    assert [m for m in re.findall(r'data-entrant="(\w+)" data-headline', table)] == ["gemini_default", "sample_coach", "claude_default", "review_only_bot"]
    assert "rank" not in _row(index, "gemini_default").split('class="num jury"')[1]
    note = _jury_note(index)
    assert "derived from jury scores; not validated by players; not a chess rating" in note
    assert "1500 is the mean of the 3 rated entrants on this board" in note and "the headline stays the only rank" in note
    assert "not a playing strength" in note
    assert "It orders every pair of entrants the way the headline does." in note
    assert "<h2>Jury rating</h2>" not in index and "Jury rating" not in index and "Community" not in index
    assert index.count("Elo") == index.count("Elo-style") > 0
    assert "the higher score wins and scores within 5 points are a draw" in note


def test_index_marks_every_pair_the_jury_rating_orders_differently_from_the_headline(tmp_path: Path) -> None:
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    common = {"families": ALL, "task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "declared_family": "openai", "tools": frozenset({"engine"})}
    steady = make_run(tmp_path / "runs", entrant="openai_engine", scores={"ap_9001": 90, "mj_9001": 50, "mc_9001": 50, "gr_9001": 50}, **common)
    uneven = make_run(tmp_path / "runs", entrant="gemini_engine", scores={"ap_9001": 20, "mj_9001": 70, "mc_9001": 70, "gr_9001": 70}, **common)
    index = (_build_runs(tmp_path, [steady, uneven], sheets, cfg=WITH_RATING) / "index.html").read_text(encoding="utf-8")
    assert '<span class="badge marker">headline and head-to-head rating disagree with gemini_engine</span>' in _row(index, "openai_engine")
    assert '<span class="badge marker">headline and head-to-head rating disagree with openai_engine</span>' in _row(index, "gemini_engine")
    note = _jury_note(index)
    assert re.search(r"openai_engine is ahead of gemini_engine on the headline \(\+2\.5\) and behind it on the head-to-head rating \(−\d+\)", note)
    assert "It orders every pair of entrants the way the headline does." not in note


def test_every_page_flags_unreviewed_answer_keys_and_incomplete_families(tmp_path: Path) -> None:
    site = _build(tmp_path)
    for name in ("index.html", "agents/claude_default.html", "tasks/ap_9001.html"):
        page = _page(site, name)
        assert "answer keys not yet expert-reviewed: 4 of 5 tasks" in page, name
        assert "claude_default: move_judgement graded on 1 of 2 tasks" in page, name


def test_the_task_page_marks_only_guarded_claims_as_checked_against_the_reference(tmp_path: Path) -> None:
    site = _build(tmp_path)
    sheet_path = tmp_path / "data" / "sheets" / "ap_9001.yaml"
    sheet_path.write_text(sheet_path.read_text(encoding="utf-8").replace("  move: Be3\n  band: null\n  line: null\n  outcome: null\n  mate_in: null\n  side: null\n  about: task_position", "  move: Be3\n  band: null\n  line: null\n  outcome: null\n  mate_in: null\n  side: null\n  about: after_move"), encoding="utf-8")
    runs = sorted(p for p in (tmp_path / "runs").glob("*/*") if p.is_dir())
    out = tmp_path / "site_after"
    build_site(leaderboard_path=tmp_path / "runs" / "leaderboard_pilot.json", run_dirs=runs, tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=out, cfg=load_bench(), root=tmp_path, registrations_dir=tmp_path / "registrations")
    before, after = _page(site, "tasks/ap_9001.html"), _page(out, "tasks/ap_9001.html")
    marked = "Be3 is the strongest and most common continuation <span class=\"ci\">weight 2, "
    assert f"{marked}checked against the reference and judged" in before
    assert f"{marked}judged</span>" in after
    assert "The position is roughly balanced <span class=\"ci\">weight 1, checked against the reference and judged" in after


def test_a_board_built_for_a_stale_comparison_is_refused_even_without_the_comparison_flag(tmp_path: Path) -> None:
    from bench.cli.site import assert_board_comparison_not_stale

    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    run = make_run(tmp_path / "runs", entrant="claude_default", families=ALL, scores={task: float(score) for task, score in SCORES.items()}, task_family=FIXTURE_TASK_FAMILY, sheets=sheets)
    info = ComparisonInfo(name="head_to_head", description="two entrants", entrants=["claude_default"])
    board = aggregate(run_dirs=[run], cfg=FOUR_FAMILIES, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, seed=1, root=tmp_path, comparison=info)
    comparisons = tmp_path / "comparisons.yaml"

    def write(stale: str | None) -> None:
        identity = {"version": "1", "declared_family": "undisclosed", "access_method": "cli", "tools": [], "client_tools": False, "affiliated_with_maintainer": False}
        pinned = {"run_dir": "runs/x", "answers_sha256": "a" * 64, "grades_sha256": "g" * 64}
        comparisons.write_text(json.dumps({"comparisons": {"head_to_head": {"description": "d", "grading_profile": "default", "stale": stale, "entrants": {"claude_default": identity}, "runs": {"claude_default": pinned}}}}), encoding="utf-8")

    write(None)
    assert_board_comparison_not_stale(board=board, comparisons_path=comparisons)
    assert_board_comparison_not_stale(board=board.model_copy(update={"comparison": None}), comparisons_path=comparisons)
    write("the sheets changed")
    with pytest.raises(SystemExit, match="comparison head_to_head is stale: the sheets changed"):
        assert_board_comparison_not_stale(board=board, comparisons_path=comparisons)


def test_the_site_command_refuses_a_stale_comparison_board_passed_by_path_without_the_comparison_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_comparisons: Path) -> None:
    import sys

    import bench.cli.site as site_cli

    info = ComparisonInfo(name="sample_head_to_head", description="d", entrants=["claude_tools", "sample_coach"])
    path, _, _ = _stored_board(tmp_path, comparison=info)
    monkeypatch.setattr(site_cli, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["site", "--leaderboard", str(path), "--out", str(tmp_path / "site")])
    with pytest.raises(SystemExit, match="sample_head_to_head is stale"):
        site_cli.main()
    assert not (tmp_path / "site").exists()


def test_the_board_says_one_judge_grades_every_answer_when_its_grades_record_one(tmp_path: Path) -> None:
    from bench.aggregate import Leaderboard
    from bench.site.build import judging_sentence

    path, _, _ = _stored_board(tmp_path)
    board = Leaderboard.model_validate_json(path.read_text(encoding="utf-8"))
    one = board.model_copy(update={"grading_instruments": {**board.grading_instruments, "jury": "openai/gpt-6.1-terra", "claim_judge": "openai/gpt-6.1-terra"}})
    sentence = judging_sentence(board=one)
    assert sentence.startswith("Every answer is graded by one judge model, openai/gpt-6.1-terra, which checks its claims with chess tools")
    assert "A single judge from one model family grades every entrant." in sentence and "two of three" not in sentence
    three = board.model_copy(update={"grading_instruments": {**board.grading_instruments, "jury": "a+b+c", "claim_judge": "a+b+c"}})
    assert judging_sentence(board=three) == "Every answer is graded by a panel of 3 judges (a, b, c), and more than half of the panel decides."


def test_an_undisclosed_family_entrant_is_ranked_without_a_family_mark(small_site: Path) -> None:
    index = _page(small_site, "index.html")
    row = re.search(r'<tr data-entrant="sample_coach".*?</tr>', index, re.S)
    assert row is not None and "rank " in row.group(0)
    table = index.split('<div class="scroll"><table>')[1].split("</table>")[0]
    assert "not disclosed" not in table and "model family" not in table
    assert "unranked" not in index.lower()
    assert re.search(r"<tr><td>(sample_coach</td><td>\w+|\w+</td><td>sample_coach)</td><td class=\"num\">", index)
    how = index.split('<p class="how">Entrants with and without tools')[1].split("</p>")[0]
    assert "Every answer is graded by a panel of 2 judges (m, m), and more than half of the panel decides." in how and "undisclosed" not in how
    agent = _page(small_site, "agents/sample_coach.html")
    assert "Model family:</strong> not disclosed" in agent and "badge marker\">model family" not in agent and "unranked" not in agent.lower()
    claude = _page(small_site, "agents/claude_default.html")
    assert "Model family:</strong> anthropic" in claude


def test_audience_fit_is_hidden_when_no_grade_on_the_board_has_it_measured(tmp_path: Path) -> None:
    site = _build(tmp_path, audience_measured=False)
    index = _page(site, "index.html")
    assert "Audience fit" not in index and "audience fit, illegal moves" not in index and "illegal moves, stated scores" in index
    claude = _page(site, "agents/claude_default.html")
    assert "Audience fit" not in claude and "Audience not settled" not in claude and "Audience votes" not in claude and "not rated" not in claude
    assert "audience" not in _page(site, "tasks/ap_9001.html").split("<h2>Answers</h2>")[1]


def test_the_grading_trace_shows_each_idea_group_with_its_claims_and_what_it_earned() -> None:
    from types import SimpleNamespace

    from bench.grading.score import ClaimOutcome
    from bench.site.build import _environment

    outcomes = [
        ClaimOutcome(claim_id="ap_9001_c001", weight=3, jury="present", guard=None, credited=True, reason="jury"),
        ClaimOutcome(claim_id="ap_9001_c002", weight=2, jury="present", guard=False, credited=False, reason="guard_failed", idea="ap_9001_c002"),
        ClaimOutcome(claim_id="ap_9001_c003", weight=1, jury="present", guard=True, credited=True, reason="jury", idea="ap_9001_c002"),
    ]
    grade = SimpleNamespace(claim_outcomes=outcomes, item_verdicts=[SimpleNamespace(item_text=t) for t in ("White plans Be3", "Be3 is best", "The position is balanced")])
    template = _environment().from_string('{% for idea in grade | idea_lines %}{{ idea.earned }}/{{ idea.weight }}:{% for text, weight, reason in idea.members %} {{ reason }} {{ weight }} {{ text }};{% endfor %}{% endfor %}')
    assert template.render(grade=grade) == "1/2: guard_failed 2 Be3 is best; jury 1 The position is balanced;"
    assert "idea_lines" in (Path(__file__).parents[1] / "bench" / "site" / "templates" / "task.html").read_text(encoding="utf-8")


def test_every_relative_link_and_image_of_the_site_resolves_inside_it(small_site: Path) -> None:
    """The docs pages carry the documents and images they link to, so the built site has no link out to a missing file."""
    import html as html_lib

    broken: list[str] = []
    for page in sorted(small_site.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        for value in re.findall(r'(?:href|src|srcset)="([^"]+)"', text):
            value = html_lib.unescape(value)
            if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", value, re.IGNORECASE):
                continue
            file_part, _, anchor = value.partition("#")
            target = (page.parent / file_part).resolve()
            if not target.is_file():
                broken.append(f"{page.relative_to(small_site)}: {value}")
            elif anchor and target.suffix == ".html" and f'id="{anchor}"' not in target.read_text(encoding="utf-8"):
                broken.append(f"{page.relative_to(small_site)}: {value} (no such anchor)")
    assert broken == []
    assert (small_site / "doc_dataset_card.html").is_file() and (small_site / "img" / "scores_light.svg").is_file()
