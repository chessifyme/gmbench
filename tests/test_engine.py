import shutil
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import pytest

from bench import config as config_module
from bench import engine as engine_module
from bench.config import EngineContextRules, EngineEnv, load_bench
from bench.engine import Engine, EngineVersionMismatch, SharedEngineContext, cp_from_score, label_for_drop, win_pct
from bench.io_yaml import read_task
from bench.models import Task
from tests.engine_fakes import ScriptedUci, engine_over
from tests.fixture_bars import FIXTURE_BARS

STOCKFISH = shutil.which("stockfish")
needs_engine = pytest.mark.skipif(STOCKFISH is None, reason="stockfish binary not installed")


def test_win_pct_is_monotone_and_clamped() -> None:
    assert win_pct(cp=0) == 50.0
    assert win_pct(cp=5000) == win_pct(cp=1000)
    assert win_pct(cp=-300) < 30
    assert win_pct(cp=100) > win_pct(cp=50) > win_pct(cp=0)


def test_win_pct_is_symmetric() -> None:
    assert abs(win_pct(cp=200) + win_pct(cp=-200) - 100.0) < 1e-9


def test_label_thresholds() -> None:
    labels = FIXTURE_BARS
    assert label_for_drop(drop=31.0, is_top_move=False, thresholds=labels) == "blunder"
    assert label_for_drop(drop=30.0, is_top_move=False, thresholds=labels) == "blunder"
    assert label_for_drop(drop=25.0, is_top_move=False, thresholds=labels) == "mistake"
    assert label_for_drop(drop=12.0, is_top_move=False, thresholds=labels) == "inaccuracy"
    assert label_for_drop(drop=5.0, is_top_move=False, thresholds=labels) == "good"
    assert label_for_drop(drop=0.5, is_top_move=False, thresholds=labels) == "best"
    assert label_for_drop(drop=5.0, is_top_move=True, thresholds=labels) == "best"


def test_label_for_drop_follows_the_thresholds_it_is_given() -> None:
    labels = FIXTURE_BARS
    lichess = labels.model_copy(update={"blunder_drop": 15.0, "mistake_drop": 10.0, "inaccuracy_drop": 5.0})
    assert [label_for_drop(drop=drop, is_top_move=False, thresholds=lichess) for drop in (4.0, 7.0, 12.0, 16.0)] == ["good", "inaccuracy", "mistake", "blunder"]
    assert [label_for_drop(drop=drop, is_top_move=False, thresholds=labels) for drop in (4.0, 7.0, 12.0, 16.0)] == ["good", "good", "inaccuracy", "inaccuracy"]


def test_no_label_threshold_has_a_default_that_could_differ_from_the_config() -> None:
    assert engine_module.LabelThresholds is config_module.LabelThresholds
    with pytest.raises(TypeError, match="thresholds"):
        label_for_drop(drop=1.0, is_top_move=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="thresholds"):
        Engine.open(path="stockfish", version_expected="Stockfish", threads=1, hash_mb=16)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        config_module.LabelThresholds()  # type: ignore[call-arg]


def test_engine_from_config_labels_with_the_configured_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_bench()
    opened: dict[str, object] = {}

    def fake_open(cls: type[Engine], **kwargs: object) -> str:
        opened.update(kwargs)
        return "engine"

    monkeypatch.setattr(Engine, "open", classmethod(fake_open))
    assert Engine.from_config(env=SimpleNamespace(engine_path="stockfish"), cfg=cfg) == "engine"  # type: ignore[arg-type]
    assert opened["thresholds"] == cfg.labels


def test_mate_score_parsing_never_raises() -> None:
    assert cp_from_score(score=chess.engine.Mate(3)) == 1000
    assert cp_from_score(score=chess.engine.Mate(-3)) == -1000
    assert cp_from_score(score=chess.engine.MateGiven) == 1000
    assert cp_from_score(score=chess.engine.Cp(45000)) == 1000
    assert cp_from_score(score=chess.engine.Cp(-45000)) == -1000
    assert win_pct(cp=cp_from_score(score=chess.engine.Mate(1))) == win_pct(cp=1000)


@needs_engine
def test_engine_open_checks_version() -> None:
    with pytest.raises(EngineVersionMismatch):
        Engine.open(path=STOCKFISH or "", version_expected="Stockfish 99", threads=1, hash_mb=16, thresholds=load_bench().labels)


@needs_engine
@pytest.mark.engine
def test_mate_in_one_reference_does_not_raise() -> None:
    with Engine.open(path=STOCKFISH or "", version_expected="Stockfish", threads=1, hash_mb=16, thresholds=load_bench().labels) as engine:
        ref = engine.analyse_position(fen="6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", nodes=200_000, multipv=3)
    assert ref.lines[0].move_san == "Ra8#" and ref.lines[0].mate == 1
    assert ref.lines[0].cp_white == 1000
    assert ref.version.startswith("Stockfish")


def test_analyse_move_searches_only_that_move_at_the_given_budget() -> None:
    uci = ScriptedUci(override=lambda board, nodes: [("e2e4", 35), ("g2g4", -90)])
    engine = engine_over(uci=uci)
    analysis = engine.analyse_move(fen=chess.STARTING_FEN, move_uci="g2g4", nodes=5_000_000)
    assert [(line.move_san, line.cp_white) for line in analysis.lines] == [("g4", -90)]
    assert (analysis.nodes, analysis.multipv) == (5_000_000, 1) and uci.calls[-1][1] == 5_000_000
    with pytest.raises(ValueError, match="not legal"):
        engine.analyse_move(fen=chess.STARTING_FEN, move_uci="e2e5", nodes=10)


@needs_engine
@pytest.mark.engine
def test_analyse_game_labels_the_losing_blunder() -> None:
    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    with Engine.open(path=STOCKFISH or "", version_expected="Stockfish", threads=1, hash_mb=16, thresholds=load_bench().labels) as engine:
        analysis = engine.analyse_game(pgn=pgn, nodes_per_ply=50_000)
    assert len(analysis.plies) == 7
    nf6 = analysis.plies[5]
    assert nf6.ply == 6 and nf6.move_san == "Nf6"
    assert nf6.label == "blunder" and nf6.win_pct_drop >= 30
    assert analysis.plies[6].cp_white_after == 1000
    assert analysis.plies[0].label in ("best", "good")


FEN_MV = "r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"


def _rules() -> EngineContextRules:
    return EngineContextRules(nodes=1_000_000, multipv=3, nodes_per_ply=200_000)


def _task(name: str) -> Task:
    return read_task(path=Path(__file__).parent / "fixtures" / "tasks" / f"{name}.yaml")


def test_label_for_uses_the_engine_thresholds() -> None:
    engine = engine_over(uci=ScriptedUci(), thresholds=FIXTURE_BARS)
    assert engine.label_for(drop=25.0, is_top_move=False) == "mistake"
    assert engine.label_for(drop=25.0, is_top_move=True) == "best"
    lichess = load_bench().labels.model_copy(update={"blunder_drop": 15.0, "mistake_drop": 10.0, "inaccuracy_drop": 5.0})
    assert engine_over(uci=ScriptedUci(), thresholds=lichess).label_for(drop=25.0, is_top_move=False) == "blunder"


def _losing_first_move(board: chess.Board, nodes: int) -> list[tuple[str, int]] | None:
    scripted = {0: [("d2d4", 0)], 1: [("e7e5", -60)]}
    return scripted.get(len(board.move_stack))


def test_analyse_game_labels_every_ply_with_the_engines_thresholds() -> None:
    labels = FIXTURE_BARS
    lichess = labels.model_copy(update={"blunder_drop": 15.0, "mistake_drop": 10.0, "inaccuracy_drop": 5.0})
    now = engine_over(uci=ScriptedUci(override=_losing_first_move), thresholds=labels).analyse_game(pgn="1. e4 *", nodes_per_ply=1_000)
    stricter = engine_over(uci=ScriptedUci(override=_losing_first_move), thresholds=lichess).analyse_game(pgn="1. e4 *", nodes_per_ply=1_000)
    assert now.plies[0].win_pct_drop == stricter.plies[0].win_pct_drop and 5.0 < now.plies[0].win_pct_drop < 10.0
    assert (now.plies[0].label, stricter.plies[0].label) == ("good", "inaccuracy")


def test_engine_context_for_a_position_task_analyses_the_task_position_only() -> None:
    uci = ScriptedUci()
    context = engine_over(uci=uci).engine_context_for(task=_task("ap_9001"), rules=_rules())
    assert context.position is not None and context.after_move is None and context.game is None
    assert len(context.position.lines) == 3 and context.position.nodes == 1_000_000
    assert [(nodes, multipv) for _, nodes, multipv in uci.calls] == [(1_000_000, 3)]
    assert context.budget() == {"engine": "Stockfish 19 scripted", "nodes": 1_000_000, "multipv": 3}


def test_engine_context_for_a_move_task_adds_the_position_after_the_move() -> None:
    uci = ScriptedUci()
    task = _task("mj_9001")
    context = engine_over(uci=uci).engine_context_for(task=task, rules=_rules())
    after = chess.Board(FEN_MV)
    after.push_san(task.move or "")
    assert context.move == task.move
    assert context.after_move is not None and context.after_move.fen == after.fen()
    assert [fen for fen, _, _ in uci.calls] == [FEN_MV, after.fen()]


def test_engine_context_for_a_game_uses_the_per_ply_budget() -> None:
    uci = ScriptedUci()
    context = engine_over(uci=uci).engine_context_for(task=_task("gr_9001"), rules=_rules())
    assert context.game is not None and context.position is None and context.game.plies
    assert {nodes for _, nodes, _ in uci.calls} == {200_000}
    assert context.budget() == {"engine": "Stockfish 19 scripted", "nodes_per_ply": 200_000}


def test_shared_engine_context_analyses_each_task_once(monkeypatch: pytest.MonkeyPatch) -> None:
    uci = ScriptedUci()
    opened: list[int] = []

    def fake_from_config(*, env: EngineEnv, cfg: object) -> Engine:
        opened.append(1)
        return engine_over(uci=uci)

    monkeypatch.setattr(Engine, "from_config", classmethod(lambda cls, **kwargs: fake_from_config(**kwargs)))
    source = SharedEngineContext(env=EngineEnv(), cfg=load_bench())
    first = source.context_for(task=_task("ap_9001"))
    second = source.context_for(task=_task("ap_9001"))
    assert first is second and opened == [1] and len(uci.calls) == 1


def test_labels_follow_lichess_5_10_15_with_the_bar_itself_taking_the_higher_label() -> None:
    thresholds = load_bench().labels
    cases = {1.9: "best", 2.1: "good", 4.99: "good", 5.0: "inaccuracy", 9.99: "inaccuracy", 10.0: "mistake", 14.99: "mistake", 15.0: "blunder", 15.03: "blunder"}
    assert {drop: label_for_drop(drop=drop, is_top_move=False, thresholds=thresholds) for drop in cases} == cases


def _drop_just_under_the_blunder_bar(board: chess.Board, nodes: int) -> list[tuple[str, int]] | None:
    return {0: [("d2d4", -300)], 1: [("e7e5", -600)]}.get(len(board.move_stack))


