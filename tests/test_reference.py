import json
import shutil
from pathlib import Path

import chess
import chess.engine
import pytest
from pydantic import ValidationError

from bench.config import LabelThresholds, ReferenceLabelling
from bench.engine import Engine, GameAnalysis, PlyEval, PositionAnalysis, label_for_drop
from bench.io_yaml import read_task
from bench.reference import (
    Material,
    Reference,
    StaleReferenceLabels,
    assert_labels_current,
    build_reference,
    is_current,
    label_problems,
    material_for_fen,
    moment_label,
    read_reference,
    reference_path,
    turning_point_candidates,
    write_reference,
)
from tests.engine_fakes import ScriptedUci, engine_over
from tests.fixture_bars import FIXTURE_BARS, fixture_cfg

FIXTURES = Path(__file__).parent / "fixtures"
STOCKFISH = shutil.which("stockfish")
needs_engine = pytest.mark.skipif(STOCKFISH is None, reason="stockfish binary not installed")


def _small_cfg():
    cfg = fixture_cfg()
    return cfg.model_copy(update={"reference_nodes_position": 60_000, "reference_nodes_ply": 20_000})


def test_material_counts_from_fen() -> None:
    material = material_for_fen(fen="rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6")
    assert material.white == {"pawn": 7, "knight": 2, "bishop": 2, "rook": 2, "queen": 1}
    assert material.black["pawn"] == 7
    assert material.balance_cp == 0
    up = material_for_fen(fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert up.balance_cp == 100 and up.white == {"pawn": 1}


@needs_engine
@pytest.mark.engine
def test_build_reference_position_has_lines_material_and_legal_moves() -> None:
    from bench.reference import build_reference

    task = read_task(path=FIXTURES / "tasks" / "ap_9001.yaml")
    with Engine.open(path=STOCKFISH or "", version_expected="Stockfish", threads=1, hash_mb=16, thresholds=FIXTURE_BARS) as engine:
        ref = build_reference(task=task, engine=engine, cfg=_small_cfg())
    assert ref.kind == "position" and ref.position is not None and ref.game is None
    assert len(ref.position.lines) == 3 and ref.position.nodes == 60_000
    assert "Be3" in ref.legal_moves
    assert ref.material.balance_cp == 0
    assert ref.turning_point_candidates == []


@needs_engine
@pytest.mark.engine
def test_build_reference_game_has_turning_point_candidates(tmp_path: Path) -> None:
    from bench.reference import build_reference

    task = read_task(path=FIXTURES / "tasks" / "gr_9001.yaml")
    with Engine.open(path=STOCKFISH or "", version_expected="Stockfish", threads=1, hash_mb=16, thresholds=FIXTURE_BARS) as engine:
        ref = build_reference(task=task, engine=engine, cfg=_small_cfg())
    assert ref.kind == "game" and ref.game is not None and ref.position is None
    assert len(ref.game.plies) == 33
    assert any(c.ply == 18 and c.move == "b5" for c in ref.turning_point_candidates)
    assert all(c.label in ("inaccuracy", "mistake", "blunder") for c in ref.turning_point_candidates)
    path = reference_path(reference_dir=tmp_path, task_id=task.task_id)
    write_reference(reference=ref, path=path)
    assert read_reference(path=path) == ref
    assert is_current(reference=ref, cfg=_small_cfg())
    assert not is_current(reference=ref, cfg=fixture_cfg())


def test_reference_json_roundtrip_without_engine(tmp_path: Path) -> None:
    ref = Reference.model_validate(
        {
            "task_id": "ap_9001",
            "kind": "position",
            "position": {
                "fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "version": "Stockfish 19",
                "nodes": 10,
                "multipv": 1,
                "lines": [{"rank": 1, "move_san": "e4", "pv_san": ["e4", "Kd7"], "cp_white": 120, "mate": None, "depth": 20, "bound": None}],
            },
            "after_move": None,
            "game": None,
            "material": {"white": {"pawn": 1}, "black": {}, "balance_cp": 100},
            "legal_moves": ["e4", "e3", "Kd1"],
            "turning_point_candidates": [],
        }
    )
    path = reference_path(reference_dir=tmp_path, task_id="ap_9001")
    write_reference(reference=ref, path=path)
    assert path.name == "ap_9001.json"
    assert read_reference(path=path) == ref


@needs_engine
@pytest.mark.engine
def test_build_reference_analyses_position_after_the_task_move() -> None:
    from bench.reference import build_reference, is_complete

    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    with Engine.open(path=STOCKFISH or "", version_expected="Stockfish", threads=1, hash_mb=16, thresholds=FIXTURE_BARS) as engine:
        ref = build_reference(task=task, engine=engine, cfg=_small_cfg())
    assert ref.after_move is not None and ref.after_move.lines
    assert ref.after_move.fen.split(" ")[1] == "w"
    assert is_complete(reference=ref, task=task)
    assert not is_complete(reference=ref.model_copy(update={"after_move": None}), task=task)


LABELS = FIXTURE_BARS
LICHESS = LabelThresholds(blunder_drop=15.0, mistake_drop=10.0, inaccuracy_drop=5.0, best_gap=2.0)
WINNING = 300


def _plies(*, drops: list[float], thresholds: LabelThresholds, before: dict[int, int] | None = None) -> list[PlyEval]:
    """Plies with the given drops; before gives the evaluation (White's view) before a ply, 0 for the others."""
    return [
        PlyEval(ply=i, move_san=f"m{i}", cp_white_before=(before or {}).get(i, 0), cp_white_after=0, best_san=None if d else f"m{i}", win_pct_drop=d, label=label_for_drop(drop=d, is_top_move=d == 0, thresholds=thresholds))
        for i, d in enumerate(drops, start=1)
    ]


def _game_reference(*, drops: list[float], thresholds: LabelThresholds = LABELS, labelled_with: ReferenceLabelling | None = None) -> Reference:
    plies = _plies(drops=drops, thresholds=thresholds)
    return Reference(
        task_id="gr_9001",
        kind="game",
        position=None,
        after_move=None,
        game=GameAnalysis(version="Stockfish 19", nodes_per_ply=1_000_000, plies=plies),
        material=Material(white={}, black={}, balance_cp=0),
        legal_moves=[],
        turning_point_candidates=turning_point_candidates(game=GameAnalysis(version="Stockfish 19", nodes_per_ply=1_000_000, plies=plies), labelling=ReferenceLabelling(labels=thresholds, candidate_min_drop=10.0), winning_cp=WINNING),
        labelled_with=labelled_with,
    )


def test_moment_label_marks_a_lost_opportunity_after_the_opponents_error() -> None:
    drops = [0.0, 25.0, 12.0, 0.0, 35.0, 0.0, 8.0, 22.0]
    plies = _plies(drops=drops, thresholds=LABELS, before={3: 450, 8: -500})
    labels = [moment_label(plies=plies, index=i, thresholds=LABELS, winning_cp=WINNING) for i in range(len(plies))]
    assert labels == ["best", "mistake", "missed_win", "best", "blunder", "best", "good", "mistake"]


def test_a_move_after_the_opponents_mistake_is_a_missed_win_only_when_the_mover_was_winning() -> None:
    """The audit's cases: the opponent erred and the mover dropped again, but was never winning (gr_0008: White at -1.69 lets
    it go to -3.15; gr_0013: +2.42 is only better), so the move is labelled by its own drop."""
    drops = [0.0, 25.0, 12.0]
    for before, winning in ((-169, False), (128, False), (3, False), (242, False), (299, False), (300, True), (900, True)):
        plies = _plies(drops=drops, thresholds=LABELS, before={3: before})
        assert moment_label(plies=plies, index=2, thresholds=LABELS, winning_cp=WINNING) == ("missed_win" if winning else plies[2].label), before
    black = _plies(drops=[25.0, 12.0], thresholds=LABELS, before={2: -350})
    assert moment_label(plies=black, index=1, thresholds=LABELS, winning_cp=WINNING) == "missed_win", "Black's view: -3.50 is winning for Black"
    black_white_up = _plies(drops=[25.0, 12.0], thresholds=LABELS, before={2: 350})
    assert moment_label(plies=black_white_up, index=1, thresholds=LABELS, winning_cp=WINNING) == black_white_up[1].label != "missed_win"


def test_moment_label_takes_the_thresholds_it_is_given() -> None:
    drops = [0.0, 12.0, 7.0, 0.0]
    plies = _plies(drops=drops, thresholds=LICHESS, before={3: 400})
    assert [moment_label(plies=plies, index=i, thresholds=LICHESS, winning_cp=WINNING) for i in range(len(plies))] == ["best", "mistake", "missed_win", "best"]
    plies_now = _plies(drops=drops, thresholds=LABELS, before={3: 400})
    assert [moment_label(plies=plies_now, index=i, thresholds=LABELS, winning_cp=WINNING) for i in range(len(plies_now))] == ["best", "inaccuracy", "good", "best"]
    with pytest.raises(TypeError, match="thresholds"):
        moment_label(plies=plies, index=1)  # type: ignore[call-arg]


def test_turning_point_candidates_include_missed_wins() -> None:
    drops = [0.0, 25.0, 12.0, 0.0]
    game = GameAnalysis(version="v", nodes_per_ply=1, plies=_plies(drops=drops, thresholds=LABELS, before={3: 400}))
    candidates = turning_point_candidates(game=game, labelling=ReferenceLabelling(labels=LABELS, candidate_min_drop=10.0), winning_cp=WINNING)
    assert [(c.ply, c.label) for c in candidates] == [(2, "mistake"), (3, "missed_win")]


def test_turning_point_candidates_follow_the_labelling_they_are_given() -> None:
    drops = [0.0, 12.0, 7.0, 0.0]
    lichess_game = GameAnalysis(version="v", nodes_per_ply=1, plies=_plies(drops=drops, thresholds=LICHESS, before={3: 400}))
    now_game = GameAnalysis(version="v", nodes_per_ply=1, plies=_plies(drops=drops, thresholds=LABELS, before={3: 400}))
    assert [(c.ply, c.label) for c in turning_point_candidates(game=lichess_game, labelling=ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0), winning_cp=WINNING)] == [(2, "mistake"), (3, "missed_win")]
    assert [(c.ply, c.label) for c in turning_point_candidates(game=now_game, labelling=ReferenceLabelling(labels=LABELS, candidate_min_drop=10.0), winning_cp=WINNING)] == [(2, "inaccuracy")]
    assert turning_point_candidates(game=now_game, labelling=ReferenceLabelling(labels=LABELS, candidate_min_drop=13.0), winning_cp=WINNING) == []


def test_build_reference_records_the_thresholds_it_labelled_with() -> None:
    cfg = _small_cfg()
    ref = build_reference(task=read_task(path=FIXTURES / "tasks" / "gr_9001.yaml"), engine=engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS), cfg=cfg)
    assert ref.kind == "game" and ref.labelled_with == cfg.reference_labelling()
    assert json.loads(ref.model_dump_json())["labelled_with"] == {"labels": {"blunder_drop": 30.0, "mistake_drop": 20.0, "inaccuracy_drop": 10.0, "best_gap": 2.0}, "candidate_min_drop": 10.0, "winning_cp": 200}
    position = build_reference(task=read_task(path=FIXTURES / "tasks" / "ap_9001.yaml"), engine=engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS), cfg=cfg)
    assert position.labelled_with is None


def test_a_position_reference_has_no_labels_to_record() -> None:
    base = build_reference(task=read_task(path=FIXTURES / "tasks" / "ap_9001.yaml"), engine=engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS), cfg=_small_cfg())
    with pytest.raises(ValidationError, match="labelled_with"):
        Reference.model_validate({**base.model_dump(), "labelled_with": fixture_cfg().reference_labelling().model_dump()})


def test_a_reference_written_before_the_record_existed_still_loads(tmp_path: Path) -> None:
    ref = _game_reference(drops=[0.0, 25.0, 12.0])
    path = reference_path(reference_dir=tmp_path, task_id="gr_9001")
    without_record = {key: value for key, value in json.loads(ref.model_dump_json()).items() if key != "labelled_with"}
    path.write_text(json.dumps(without_record), encoding="utf-8")
    assert read_reference(path=path).labelled_with is None


def test_reference_labelled_under_the_configured_thresholds_has_no_label_problems() -> None:
    labelling = fixture_cfg().reference_labelling()
    assert label_problems(reference=_game_reference(drops=[0.0, 25.0, 12.0, 33.0], labelled_with=labelling), labelling=labelling) == []
    assert label_problems(reference=_game_reference(drops=[0.0, 25.0, 12.0, 33.0]), labelling=labelling) == []


def test_a_recorded_threshold_that_differs_from_the_config_is_named() -> None:
    now = fixture_cfg().reference_labelling()
    reference = _game_reference(drops=[0.0, 25.0, 12.0], labelled_with=now)
    lichess = ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0)
    (problem,) = label_problems(reference=reference, labelling=lichess)
    for named in ("gr_9001", "labels.blunder_drop 30.0 in the reference, 15.0 in config/bench.yaml", "labels.mistake_drop 20.0 in the reference, 10.0 in config/bench.yaml", "labels.inaccuracy_drop 10.0 in the reference, 5.0 in config/bench.yaml", "turning_points.candidate_min_drop 10.0 in the reference, 5.0 in config/bench.yaml"):
        assert named in problem
    assert "labels.best_gap" not in problem and "make reference rebuilds it under the current thresholds" in problem


def test_a_reference_labelled_under_another_winning_edge_is_named() -> None:
    """A missed win depends on the winning edge (it was 300 cp before it became 200 cp), so a game reference records it and
    one labelled under another edge, or before the edge was recorded, is stale."""
    now = fixture_cfg().reference_labelling()
    assert now.winning_cp == fixture_cfg().eval_bands.winning_cp == 200
    (problem,) = label_problems(reference=_game_reference(drops=[0.0, 25.0, 12.0], labelled_with=now.model_copy(update={"winning_cp": 300})), labelling=now)
    assert "eval_bands.winning_cp 300 in the reference, 200 in config/bench.yaml" in problem and "make reference rebuilds it" in problem
    (unrecorded,) = label_problems(reference=_game_reference(drops=[0.0, 25.0, 12.0], labelled_with=now.model_copy(update={"winning_cp": None})), labelling=now)
    assert "eval_bands.winning_cp None in the reference, 200" in unrecorded


def test_stored_labels_that_the_config_would_not_give_are_flagged_when_nothing_was_recorded() -> None:
    reference = _game_reference(drops=[0.0, 25.0, 12.0, 33.0])
    assert reference.labelled_with is None
    (problem,) = label_problems(reference=reference, labelling=ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0))
    for named in ("gr_9001", "2 stored labels", "ply 2 (m2) is stored as mistake", "gives blunder", "make reference rebuilds it under the current thresholds"):
        assert named in problem


def _relabelled(*, reference: Reference, labels: dict[int, str]) -> Reference:
    assert reference.game is not None
    plies = [ply.model_copy(update={"label": labels.get(ply.ply, ply.label)}) for ply in reference.game.plies]
    return reference.model_copy(update={"game": reference.game.model_copy(update={"plies": plies})})


def test_a_stored_label_is_consistent_when_a_drop_that_rounds_to_the_stored_one_gives_it() -> None:
    labelling = fixture_cfg().reference_labelling()
    reference = _game_reference(drops=[0.0, 2.0, 10.0, 20.0, 30.0])
    assert [p.label for p in reference.game.plies] == ["best", "best", "inaccuracy", "mistake", "blunder"]  # type: ignore[union-attr]
    assert label_problems(reference=_relabelled(reference=reference, labels={2: "good", 3: "good", 4: "inaccuracy", 5: "mistake"}), labelling=labelling) == []
    assert label_problems(reference=_relabelled(reference=reference, labels={4: "blunder"}), labelling=labelling)
    assert label_problems(reference=_relabelled(reference=reference, labels={2: "inaccuracy"}), labelling=labelling)


def test_a_position_reference_has_no_label_problems() -> None:
    position = build_reference(task=read_task(path=FIXTURES / "tasks" / "ap_9001.yaml"), engine=engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS), cfg=_small_cfg())
    assert label_problems(reference=position, labelling=ReferenceLabelling(labels=LICHESS, candidate_min_drop=5.0)) == []


def test_is_current_fails_a_reference_labelled_under_other_thresholds() -> None:
    cfg = fixture_cfg()
    ref = build_reference(task=read_task(path=FIXTURES / "tasks" / "gr_9001.yaml"), engine=engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS), cfg=cfg)
    version_matching = ref.model_copy(update={"game": ref.game.model_copy(update={"version": "Stockfish 19.0", "nodes_per_ply": cfg.reference_nodes_ply})})  # type: ignore[union-attr]
    assert is_current(reference=version_matching, cfg=cfg)
    stricter = cfg.model_copy(update={"labels": cfg.labels.model_copy(update={"inaccuracy_drop": 5.0})})
    assert not is_current(reference=version_matching, cfg=stricter)
    later_bar = cfg.model_copy(update={"turning_points": cfg.turning_points.model_copy(update={"candidate_min_drop": 15.0})})
    assert not is_current(reference=version_matching, cfg=later_bar)


def test_assert_labels_current_names_the_mismatch() -> None:
    now = fixture_cfg().reference_labelling()
    reference = _game_reference(drops=[0.0, 25.0], labelled_with=now)
    assert_labels_current(reference=reference, labelling=now)
    with pytest.raises(StaleReferenceLabels, match=r"gr_9001.*labels\.blunder_drop 30\.0 in the reference, 15\.0 in config/bench\.yaml"):
        assert_labels_current(reference=reference, labelling=ReferenceLabelling(labels=LICHESS, candidate_min_drop=10.0))


def _position(*, fen: str, scores: list[tuple[str, int]]) -> PositionAnalysis:
    from bench.engine import Line

    return PositionAnalysis(fen=fen, version="Stockfish 19", nodes=5_000_000, multipv=3, lines=[Line(rank=n, move_san=san, pv_san=[san], cp_white=cp, mate=None, depth=20) for n, (san, cp) in enumerate(scores, start=1)])


WHITE_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_a_reference_whose_lines_are_out_of_order_does_not_load() -> None:
    def reference(analysis: object) -> Reference:
        return Reference.model_validate({"task_id": "ap_9001", "kind": "position", "position": analysis, "after_move": None, "game": None, "material": {"white": {}, "black": {}, "balance_cp": 0}, "legal_moves": [], "turning_point_candidates": []})

    assert reference(_position(fen=WHITE_TO_MOVE, scores=[("e4", 30), ("d4", 30), ("c4", 10)])).position is not None
    assert reference(_position(fen=BLACK_TO_MOVE, scores=[("c5", 20), ("e5", 30)])).position is not None
    with pytest.raises(ValidationError, match=r"reference ap_9001: position line 2 \(d4 \+71 cp for White\) beats line 1 \(e4 \+54\)"):
        reference(_position(fen=WHITE_TO_MOVE, scores=[("e4", 54), ("d4", 71)]))
    with pytest.raises(ValidationError, match=r"line 2 \(e5 \+10 cp for White\) beats line 1"):
        reference(_position(fen=BLACK_TO_MOVE, scores=[("c5", 20), ("e5", 10)]))
    with pytest.raises(ValidationError, match=r"ranked \[2\], not 1 to 1"):
        reference(_position(fen=WHITE_TO_MOVE, scores=[("e4", 54)]).model_copy(update={"lines": [_position(fen=WHITE_TO_MOVE, scores=[("x", 0), ("e4", 54)]).lines[1]]}))


def test_reference_lines_come_from_one_completed_depth_with_exact_scores() -> None:
    def reference(analysis: PositionAnalysis) -> Reference:
        return Reference.model_validate({"task_id": "ap_9001", "kind": "position", "position": analysis, "after_move": None, "game": None, "material": {"white": {}, "black": {}, "balance_cp": 0}, "legal_moves": [], "turning_point_candidates": []})

    settled = _position(fen=WHITE_TO_MOVE, scores=[("Kg2", 87), ("Be2", 77), ("h5", 69)])
    settled = settled.model_copy(update={"lines": [line.model_copy(update={"depth": 23}) for line in settled.lines]})
    assert reference(settled).position is not None
    bound = settled.model_copy(update={"lines": [settled.lines[0].model_copy(update={"bound": "upperbound"}), *settled.lines[1:]]})
    with pytest.raises(ValidationError, match="line 1 \\(Kg2\\) is an upperbound, not an exact score"):
        reference(bound)
    mixed = settled.model_copy(update={"lines": [*settled.lines[:2], settled.lines[2].model_copy(update={"depth": 22})]})
    with pytest.raises(ValidationError, match="lines mix depths \\[22, 23\\]"):
        reference(mixed)
    partial = settled.model_copy(update={"lines": [*settled.lines[:2], settled.lines[2].model_copy(update={"depth": None})]})
    with pytest.raises(ValidationError, match="lines \\[3\\] record no depth, so they are not shown to come from one completed depth"):
        reference(partial)


def test_a_settled_analysis_takes_the_last_completed_depth_and_never_a_bound() -> None:
    from tests.engine_fakes import ScriptedUci

    engine = engine_over(uci=ScriptedUci())
    analysis = engine.analyse_position_settled(fen=WHITE_TO_MOVE, nodes=1000, multipv=3)
    assert [(line.rank, line.depth, line.bound) for line in analysis.lines] == [(1, 2, None), (2, 2, None), (3, 2, None)]


def test_a_search_that_completes_no_depth_fails_loudly() -> None:
    from tests.engine_fakes import ScriptedAnalysis, ScriptedUci

    class Unfinished(ScriptedUci):
        def analysis(self, board, limit, multipv=None, game=None):  # type: ignore[no-untyped-def]
            move = next(iter(board.legal_moves))
            return ScriptedAnalysis(infos=[{"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE), "multipv": 1, "depth": 9, "lowerbound": True}])

    with pytest.raises(RuntimeError, match="completed no depth with 3 exact lines within 1000 nodes"):
        engine_over(uci=Unfinished()).analyse_position_settled(fen=WHITE_TO_MOVE, nodes=1000, multipv=3)


def _burst(*, depth: int | tuple[int, ...], cps: tuple[int, int, int], bound: dict[int, str] | None = None) -> list[dict[str, object]]:
    """One report of the three lines Nf3, e4, d4 at `depth` (one per line when a tuple), with the bound named per rank."""
    depths = depth if isinstance(depth, tuple) else (depth,) * 3
    moves = [chess.Move.from_uci(uci) for uci in ("g1f3", "e2e4", "d2d4")]
    return [
        {"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE), "multipv": rank, "depth": d, **({(bound or {})[rank]: True} if rank in (bound or {}) else {})}
        for rank, (move, cp, d) in enumerate(zip(moves, cps, depths, strict=True), start=1)
    ]


def _settled_lines(*bursts: list[dict[str, object]]) -> list[tuple[str, int, int | None]]:
    from tests.engine_fakes import ScriptedAnalysis

    class Streamed(ScriptedUci):
        def analysis(self, board, limit, multipv=None, game=None):  # type: ignore[no-untyped-def]
            return ScriptedAnalysis(infos=[info for burst in bursts for info in burst])

    analysis = engine_over(uci=Streamed()).analyse_position_settled(fen=chess.STARTING_FEN, nodes=1000, multipv=3)
    return [(line.move_san, line.cp_white, line.depth) for line in analysis.lines]


@pytest.mark.parametrize("bound", ["lowerbound", "upperbound"])
@pytest.mark.parametrize("rank", [1, 3])
def test_a_complete_report_with_a_bound_line_is_never_the_settled_depth(bound: str, rank: int) -> None:
    exact = _burst(depth=20, cps=(30, 20, 10))
    bounded = _burst(depth=21, cps=(90, 80, 70), bound={rank: bound})
    assert _settled_lines(exact, bounded) == [("Nf3", 30, 20), ("e4", 20, 20), ("d4", 10, 20)]


def test_a_complete_report_whose_lines_come_from_two_depths_is_never_the_settled_depth() -> None:
    exact = _burst(depth=20, cps=(30, 20, 10))
    mixed = _burst(depth=(21, 21, 20), cps=(90, 80, 70))
    assert _settled_lines(exact, mixed) == [("Nf3", 30, 20), ("e4", 20, 20), ("d4", 10, 20)]


def test_of_two_completed_depths_the_deepest_is_used() -> None:
    assert _settled_lines(_burst(depth=20, cps=(30, 20, 10)), _burst(depth=21, cps=(35, 25, 15))) == [("Nf3", 35, 21), ("e4", 25, 21), ("d4", 15, 21)]
