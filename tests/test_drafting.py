from pathlib import Path

import pytest

from bench.config import EvalBands, LabelThresholds, ReferenceLabelling, load_bench
from bench.drafting import band_for_cp, draft_sheet, outcome_for, verdict_from_reference
from bench.engine import GameAnalysis, Line, PlyEval, PositionAnalysis, label_for_drop
from bench.io_yaml import load_all_sheets, load_all_tasks, read_task
from bench.models import single_claim_problem
from bench.paths import DATA_PUBLIC_REFERENCE, DATA_PUBLIC_SHEETS, DATA_PUBLIC_TASKS
from bench.reference import Material, Reference, StaleReferenceLabels, read_reference, reference_path
from bench.validate import errors_of, sheet_problems
from tests.fixture_bars import fixture_cfg

FIXTURES = Path(__file__).parent / "fixtures"
FEN_MV = "r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"
BANDS = load_bench().eval_bands
CFG = fixture_cfg()
LICHESS = LabelThresholds(blunder_drop=15.0, mistake_drop=10.0, inaccuracy_drop=5.0, best_gap=2.0)


def _line(*, rank: int, pv: list[str], cp: int, mate: int | None = None) -> Line:
    return Line(rank=rank, move_san=pv[0], pv_san=pv, cp_white=cp, mate=mate, depth=20)


def _position(*, fen: str, lines: list[Line]) -> PositionAnalysis:
    return PositionAnalysis(fen=fen, version="Stockfish 19", nodes=5_000_000, multipv=3, lines=lines)


def mv_reference() -> Reference:
    return Reference(
        task_id="mj_9001",
        kind="position",
        position=_position(fen=FEN_MV, lines=[_line(rank=1, pv=["Na5", "Bb5+", "c6"], cp=-20), _line(rank=2, pv=["Nd4", "c3"], cp=40), _line(rank=3, pv=["Nxd5", "Nxf7", "Kxf7", "Qf3+"], cp=260)]),
        after_move=_position(fen="r1bqkb1r/ppp2ppp/2n5/3np1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 6", lines=[_line(rank=1, pv=["Nxf7", "Kxf7", "Qf3+", "Ke6"], cp=260)]),
        game=None,
        material=Material(white={"pawn": 7, "knight": 2, "bishop": 2, "rook": 2, "queen": 1}, black={"pawn": 7, "knight": 2, "bishop": 2, "rook": 2, "queen": 1}, balance_cp=0),
        legal_moves=["Na5", "Nxd5", "Nd4"],
        turning_point_candidates=[],
    )


def test_band_for_cp() -> None:
    assert band_for_cp(cp_white=0, bands=BANDS) == "equal"
    assert band_for_cp(cp_white=50, bands=BANDS) == "white_slightly_better"
    assert band_for_cp(cp_white=-120, bands=BANDS) == "black_better"
    assert band_for_cp(cp_white=-180, bands=BANDS) == "black_large_edge"
    assert band_for_cp(cp_white=400, bands=BANDS) == "white_winning"
    assert band_for_cp(cp_white=-29, bands=BANDS) == "equal"


def test_outcome_for_rules() -> None:
    win, level = CFG.eval_bands.winning_cp, CFG.eval_bands.slightly_better_cp
    assert outcome_for(cp_end_mover=50, mate_mover=2, material_delta_mover=0, winning_cp=win, equal_cp=level) == "mates"
    assert outcome_for(cp_end_mover=-900, mate_mover=-1, material_delta_mover=0, winning_cp=win, equal_cp=level) == "gets_mated"
    assert outcome_for(cp_end_mover=250, mate_mover=None, material_delta_mover=300, winning_cp=win, equal_cp=level) == "wins_material"
    assert outcome_for(cp_end_mover=-200, mate_mover=None, material_delta_mover=-100, winning_cp=win, equal_cp=level) == "loses_material"
    assert outcome_for(cp_end_mover=win, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) == "wins"
    assert outcome_for(cp_end_mover=-win, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) == "loses"
    assert outcome_for(cp_end_mover=10, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) == "equalises"
    assert outcome_for(cp_end_mover=level - 1, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) == "equalises"
    assert outcome_for(cp_end_mover=level, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) is None, "equalises ends where the equal band does"
    assert outcome_for(cp_end_mover=90, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) is None


def test_wins_and_loses_start_at_the_winning_band_not_at_one_and_a_half_pawns() -> None:
    """mj_0029_c001 said "leads to a lost position" at +2.15 for the opponent while the same card said "clearly better": a line
    outcome says wins or loses only from the configured winning band on (2.00 on the benchmark's scale)."""
    win, level = CFG.eval_bands.winning_cp, CFG.eval_bands.slightly_better_cp
    assert win == 200
    for cp in (150, 170, -170, -199, win - 1, 1 - win):
        assert outcome_for(cp_end_mover=cp, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) is None, cp
    assert outcome_for(cp_end_mover=-215, mate_mover=None, material_delta_mover=0, winning_cp=win, equal_cp=level) == "loses"
    assert outcome_for(cp_end_mover=-215, mate_mover=None, material_delta_mover=0, winning_cp=300, equal_cp=level) is None, "the band is read from what it is given"


def test_verdict_from_reference_uses_movers_view() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    assert verdict_from_reference(task=task, reference=mv_reference(), thresholds=CFG.labels) == "mistake"


def test_verdict_from_reference_labels_with_the_thresholds_it_is_given() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    reference = mv_reference()
    thinner = reference.model_copy(update={"after_move": reference.after_move.model_copy(update={"lines": [reference.after_move.lines[0].model_copy(update={"cp_white": 110})]})})  # type: ignore[union-attr]
    assert verdict_from_reference(task=task, reference=thinner, thresholds=CFG.labels) == "inaccuracy"
    assert verdict_from_reference(task=task, reference=thinner, thresholds=LICHESS) == "mistake"
    with pytest.raises(TypeError, match="thresholds"):
        verdict_from_reference(task=task, reference=thinner)  # type: ignore[call-arg]


def _lost_after_the_move(*, cp_white: int) -> Reference:
    """mv_reference with White a given amount up after the judged 5...Nxd5."""
    reference = mv_reference()
    assert reference.after_move is not None
    return reference.model_copy(update={"after_move": reference.after_move.model_copy(update={"lines": [_line(rank=1, pv=["Nxf7", "Kxf7", "Qf3+", "Ke6"], cp=cp_white)]})})


def test_a_draft_holds_the_engines_claims_only_weighted_and_true_by_source() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    sheet = draft_sheet(task=task, reference=mv_reference(), cfg=CFG)
    assert sheet.verdict == "mistake"
    large = draft_sheet(task=task, reference=_lost_after_the_move(cp_white=CFG.eval_bands.winning_cp - 1), cfg=CFG)
    assert not any(k.kind == "line_outcome" for k in large.claims), "+1.99 for White is a large advantage, not lost for Black"
    assert any(k.kind == "best_move" and k.move == "Na5" and k.weight == 3 for k in large.claims)
    lost = draft_sheet(task=task, reference=_lost_after_the_move(cp_white=CFG.eval_bands.winning_cp), cfg=CFG)
    assert any(k.kind == "line_outcome" and k.outcome == "loses" and k.weight == 3 and k.line.startswith("Nxd5 Nxf7") and k.text.endswith("leads to a lost position") for k in lost.claims)
    assert any(k.kind == "best_move" and k.move == "Na5" and k.weight == 2 for k in lost.claims)
    assert any(k.kind == "eval_band" for k in sheet.claims)
    assert sheet.author is None and sheet.notes == "" and sheet.turning_points == []
    assert sheet.expert_points == sheet.claims and [c.claim_id for c in sheet.claims] == [f"mj_9001_c{n:03d}" for n in range(1, len(sheet.claims) + 1)]
    assert {(c.source.type, c.source.name, c.source.answer_ref) for c in sheet.claims} == {("engine", "stockfish_19", None)} and all(c.labels == [] for c in sheet.claims)


def test_a_move_judgement_on_a_move_not_played_drafts_its_verdict_and_line() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9002.yaml")
    reference = mv_reference().model_copy(
        update={
            "task_id": "mj_9002",
            "after_move": _position(fen="r1bqkb1r/ppp2ppp/5n2/n2Pp1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 1 6", lines=[_line(rank=1, pv=["Bb5+", "c6", "dxc6", "bxc6"], cp=-25)]),
        }
    )
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    assert sheet.verdict is not None
    alternative = [k for k in sheet.claims if k.kind == "line_outcome"]
    assert alternative and alternative[0].weight == 3 and alternative[0].line.startswith("Na5 Bb5+")
    assert alternative[0].outcome == "equalises"


def _gr_plies() -> list[PlyEval]:
    plies = [
        PlyEval(ply=p, move_san=san, cp_white_before=0, cp_white_after=0, best_san=san, win_pct_drop=0.0, label="best")
        for p, san in enumerate(["e4", "e5", "Nf3", "d6", "d4", "Bg4", "dxe5", "Bxf3", "Qxf3", "dxe5", "Bc4", "Nf6", "Qb3", "Qe7", "Nc3", "c6", "Bg5", "b5"], start=1)
    ]
    plies[5] = plies[5].model_copy(update={"cp_white_before": 20, "cp_white_after": 140, "best_san": "Nf6", "win_pct_drop": 12.0, "label": "inaccuracy"})
    plies[17] = plies[17].model_copy(update={"cp_white_before": 150, "cp_white_after": 900, "best_san": "Qc5", "win_pct_drop": 45.0, "label": "blunder"})
    plies[11] = plies[11].model_copy(update={"cp_white_after": 350, "win_pct_drop": 24.0, "best_san": "Nd7", "label": "mistake"})
    plies[12] = plies[12].model_copy(update={"cp_white_before": 350, "win_pct_drop": 11.0, "best_san": "Bxf7+", "label": "inaccuracy"})
    return plies


def _gr_reference(*, plies: list[PlyEval] | None = None, labelled_with: ReferenceLabelling | None = None) -> Reference:
    return Reference(
        task_id="gr_9001",
        kind="game",
        position=None,
        after_move=None,
        game=GameAnalysis(version="Stockfish 19", nodes_per_ply=1_000_000, plies=plies or _gr_plies()),
        material=Material(white={}, black={}, balance_cp=0),
        legal_moves=[],
        turning_point_candidates=[{"ply": 6, "move": "Bg4", "label": "inaccuracy"}, {"ply": 12, "move": "Nf6", "label": "mistake"}, {"ply": 18, "move": "b5", "label": "blunder"}],
        labelled_with=labelled_with,
    )


def _with_turning_points(**changes: float):
    return CFG.model_copy(update={"turning_points": CFG.turning_points.model_copy(update=changes)})


def test_draft_sheet_game_review_turning_points_become_engine_claims() -> None:
    task = read_task(path=FIXTURES / "tasks" / "gr_9001.yaml")
    reference = _gr_reference()
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    assert task.user_color == "black" and [(tp.ply, tp.label) for tp in sheet.turning_points] == [(12, "mistake"), (18, "blunder")]
    texts = [k.text for k in sheet.claims]
    assert "9...b5 was a blunder" in texts and not any("7.Qb3" in t or "better" in t for t in texts if "was a " in t)
    white = draft_sheet(task=task.model_copy(update={"user_color": "white"}), reference=reference, cfg=CFG)
    assert [(tp.ply, tp.label) for tp in white.turning_points] == [(13, "missed_win")]
    assert "7.Qb3 was a missed win" in [k.text for k in white.claims]
    assert all(k.kind == "other" and not k.checkable and k.source.type == "engine" for k in sheet.claims)
    assert [(k.weight, k.move) for k in sheet.claims if "b5" in k.text] == [(3, "b5")]


def _turning_points(*, cfg, reference: Reference | None = None, user_color: str = "black") -> list[tuple[int, str]]:
    task = read_task(path=FIXTURES / "tasks" / "gr_9001.yaml").model_copy(update={"user_color": user_color})
    sheet = draft_sheet(task=task, reference=reference or _gr_reference(), cfg=cfg)
    return [(tp.ply, tp.label) for tp in sheet.turning_points]


def test_the_sheet_bar_and_the_missed_win_bar_come_from_the_config() -> None:
    assert _turning_points(cfg=CFG) == [(12, "mistake"), (18, "blunder")]
    assert _turning_points(cfg=_with_turning_points(sheet_min_drop=10.0)) == [(6, "inaccuracy"), (12, "mistake"), (18, "blunder")]
    assert _turning_points(cfg=_with_turning_points(sheet_min_drop=30.0)) == [(18, "blunder")]
    assert _turning_points(cfg=CFG, user_color="white") == [(13, "missed_win")]
    assert _turning_points(cfg=_with_turning_points(sheet_min_drop=30.0), user_color="white") == [(13, "missed_win")]
    assert _turning_points(cfg=_with_turning_points(missed_win_min_drop=12.0), user_color="white") == []


def test_the_label_thresholds_and_the_sheet_bar_decide_the_sheet_together() -> None:
    plies = [ply.model_copy(update={"label": label_for_drop(drop=ply.win_pct_drop, is_top_move=ply.best_san == ply.move_san, thresholds=LICHESS)}) for ply in _gr_plies()]
    labelling = ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0, winning_cp=CFG.eval_bands.winning_cp)
    lichess = CFG.model_copy(update={"labels": LICHESS, "turning_points": CFG.turning_points.model_copy(update={"sheet_min_drop": 10.0, "candidate_min_drop": 5.0, "missed_win_min_drop": 5.0})})
    assert lichess.reference_labelling() == labelling
    reference = _gr_reference(plies=plies, labelled_with=labelling)
    assert _turning_points(cfg=lichess, reference=reference) == [(6, "mistake"), (12, "blunder"), (18, "blunder")]
    assert _turning_points(cfg=lichess, reference=reference, user_color="white") == [(13, "missed_win")]


def test_drafting_from_a_reference_labelled_under_other_thresholds_stops() -> None:
    stale = _gr_reference(labelled_with=ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0))
    with pytest.raises(StaleReferenceLabels, match=r"gr_9001.*labels\.blunder_drop 15\.0 in the reference, 30\.0 in config/bench\.yaml"):
        _turning_points(cfg=CFG, reference=stale)
    unrecorded_and_inconsistent = _gr_reference(plies=[ply.model_copy(update={"label": "good"}) if ply.ply == 18 else ply for ply in _gr_plies()])
    with pytest.raises(StaleReferenceLabels, match="ply 18"):
        _turning_points(cfg=CFG, reference=unrecorded_and_inconsistent)


def test_published_verdicts_and_turning_points_are_what_the_configured_thresholds_give_from_the_published_references() -> None:
    sheets = load_all_sheets(sheets_dir=DATA_PUBLIC_SHEETS)
    checked = {"move_judgement": 0, "game_review": 0}
    for task in load_all_tasks(tasks_dir=DATA_PUBLIC_TASKS):
        if task.family not in checked:
            continue
        reference = read_reference(path=reference_path(reference_dir=DATA_PUBLIC_REFERENCE, task_id=task.task_id))
        drafted = draft_sheet(task=task, reference=reference, cfg=load_bench())
        sheet = sheets[task.task_id]
        if sheet.author is not None:
            continue
        assert (drafted.verdict, drafted.turning_points) == (sheet.verdict, sheet.turning_points), task.task_id
        checked[task.family] += 1
    # every v0 sheet is signed, so none is left as drafted: a signed verdict is the reviewer's, not the thresholds'
    assert checked == {"move_judgement": 0, "game_review": 0}


def test_every_drafted_checkable_claim_says_what_position_it_is_about() -> None:
    mv = draft_sheet(task=read_task(path=FIXTURES / "tasks" / "mj_9001.yaml"), reference=mv_reference(), cfg=CFG)
    checks = [k for k in mv.claims if k.checkable]
    assert checks and all(c.about in ("task_position", "after_move") for c in checks)
    band = next(c for c in checks if c.kind == "eval_band")
    assert band.about == "after_move" and band.band == "white_winning", "+2.60 is winning on the benchmark's scale (from 2.00)"
    assert next(c for c in checks if c.kind == "best_move").about == "task_position"
    assert band.text == "After 5...Nxd5, White is winning"


def test_move_judgement_band_is_the_evaluation_after_the_move() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9002.yaml")
    reference = mv_reference().model_copy(
        update={
            "task_id": "mj_9002",
            "after_move": _position(fen="r1bqkb1r/ppp2ppp/5n2/n2Pp1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 1 6", lines=[_line(rank=1, pv=["Bb5+", "c6", "dxc6", "bxc6"], cp=-25)]),
        }
    )
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    band = next(k for k in sheet.claims if k.kind == "eval_band")
    assert band.about == "after_move" and band.band == "equal"
    assert band.text == "After 5...Na5, the position is roughly equal"


def test_position_task_band_is_about_the_task_position() -> None:
    task = read_task(path=FIXTURES / "tasks" / "ap_9001.yaml")
    reference = Reference(
        task_id="ap_9001",
        kind="position",
        position=_position(fen=task.fen or "", lines=[_line(rank=1, pv=["Be3", "e5", "Nb3", "Be6"], cp=-60)]),
        after_move=None,
        game=None,
        material=Material(white={}, black={}, balance_cp=150),
        legal_moves=["Be3"],
        turning_point_candidates=[],
    )
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    band = next(k for k in sheet.claims if k.kind == "eval_band")
    assert band.about == "task_position" and band.text == "Black is slightly better"
    assert all(k.about == "task_position" for k in sheet.claims if k.checkable)


def test_drafted_sheet_passes_the_validator() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    reference = mv_reference()
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    assert errors_of(problems=sheet_problems(sheet=sheet, task=task, path=FIXTURES / "x.yaml", reference=reference, bands=BANDS)) == []
    assert all(single_claim_problem(text=k.text) is None for k in sheet.claims)


def test_beginner_tasks_get_no_line_outcome_over_four_plies() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml").model_copy(update={"audience": "beginner"})
    lost = _lost_after_the_move(cp_white=400)
    sheet = draft_sheet(task=task, reference=lost, cfg=CFG)
    lines = [k.line for k in sheet.claims if k.kind == "line_outcome"]
    assert lines and all(line is not None and len(line.split()) <= 4 for line in lines)
    club = draft_sheet(task=read_task(path=FIXTURES / "tasks" / "mj_9001.yaml"), reference=lost, cfg=CFG)
    assert any(k.checkable and k.line and len(k.line.split()) > 4 for k in club.claims)


def _cf_reference(*, reply_cp: int, reply_pv: list[str]) -> Reference:
    return mv_reference().model_copy(
        update={
            "task_id": "mj_9002",
            "after_move": _position(fen="r1bqkb1r/ppp2ppp/5n2/n2Pp1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 1 6", lines=[_line(rank=1, pv=reply_pv, cp=reply_cp)]),
        }
    )


def test_a_slight_edge_after_the_judged_move_gets_no_line_outcome_point() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9002.yaml")
    reference = _cf_reference(reply_cp=-50, reply_pv=["Bb5+", "c6", "dxc6", "bxc6"])
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    assert not [k for k in sheet.claims if k.kind == "line_outcome"]
    band = next(k for k in sheet.claims if k.kind == "eval_band")
    assert band is not None and band.band == "black_slightly_better"


def test_a_winning_edge_after_the_judged_move_still_earns_a_wins_or_loses_outcome() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9002.yaml")
    reference = _cf_reference(reply_cp=400, reply_pv=["Bb5+", "c6", "dxc6", "bxc6"])
    sheet = draft_sheet(task=task, reference=reference, cfg=CFG)
    outcomes = [k.outcome for k in sheet.claims if k.kind == "line_outcome"]
    assert outcomes == ["loses"]


def _position_reference(*, task_id: str, fen: str) -> Reference:
    return Reference(task_id=task_id, kind="position", position=_position(fen=fen, lines=[_line(rank=1, pv=["Be3", "e5"], cp=20), _line(rank=2, pv=["f3", "e5"], cp=10)]), after_move=None, game=None, material=Material(white={}, black={}, balance_cp=0), legal_moves=["Be3"], turning_point_candidates=[])


def test_only_a_move_choice_draft_names_the_strongest_move() -> None:
    plan = read_task(path=FIXTURES / "tasks" / "ap_9001.yaml")
    choice = read_task(path=FIXTURES / "tasks" / "mc_9001.yaml")
    assert not any(c.kind == "best_move" for c in draft_sheet(task=plan, reference=_position_reference(task_id="ap_9001", fen=plan.fen or ""), cfg=CFG).claims)
    best = [c for c in draft_sheet(task=choice, reference=_position_reference(task_id="mc_9001", fen=choice.fen or ""), cfg=CFG).claims if c.kind == "best_move"]
    assert [(c.move, c.weight, c.text) for c in best] == [("Be3", 3, "6.Be3 is the strongest move")]


def test_a_best_verdict_on_a_move_other_than_the_engines_first_choice_names_no_better_move() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    close = mv_reference().model_copy(update={"after_move": _position(fen="r1bqkb1r/ppp2ppp/2n5/3np1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 6", lines=[_line(rank=1, pv=["Nxf7", "Kxf7", "Qf3+", "Ke6"], cp=-18)])})
    sheet = draft_sheet(task=task, reference=close, cfg=CFG)
    assert sheet.verdict == "best"
    best = [c for c in sheet.claims if c.kind == "best_move"]
    assert [(c.move, c.text) for c in best] == [("Nxd5", "5...Nxd5 is the strongest move")]
    assert not [c for c in sheet.claims if "better move" in c.text]


def test_every_published_best_verdict_sheet_carries_the_drafters_strongest_move_claim() -> None:
    """mj_0008 lost it to the old drafter rule; it is back as an unweighted engine claim until the grandmaster weights it (weighting it changes the judged texts).
    A signed sheet carries the grandmaster's verdict, not the drafter's."""
    tasks = {t.task_id: t for t in load_all_tasks(tasks_dir=DATA_PUBLIC_TASKS)}
    for task_id, sheet in load_all_sheets(sheets_dir=DATA_PUBLIC_SHEETS).items():
        if sheet.verdict != "best" or sheet.author is not None:
            continue
        named = [c for c in sheet.claims if c.kind == "best_move" and c.about == "task_position" and c.source.type == "engine"]
        assert [c.move for c in named] == [tasks[task_id].move], task_id


def test_a_task_without_moves_so_far_numbers_its_moves_from_the_fen() -> None:
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml").model_copy(update={"pgn": None})
    sheet = draft_sheet(task=task, reference=mv_reference(), cfg=CFG)
    better = next(c for c in sheet.claims if c.kind == "best_move")
    assert better.text == "5...Na5 was the better move"
