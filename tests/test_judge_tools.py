import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import pytest

from bench.config import EngineEnv, LabelThresholds, load_bench
from bench.engine import Engine
from bench.engine import PositionAnalysis
from bench.judge_tools import GM_TOOLS, GmClaim, reference_positions, EngineCache, JudgeToolbox, TaskPositions, ToolInputError, eval_band, moment_ply, move_tokens, parse_reference, PositionReference
from tests.grading_fakes import GR_TASK, MV_TASK
from tests.judge_tool_fakes import BUDGET, CFG, ScriptedAnalyser, fen_after, line, toolbox

MV_FEN = MV_TASK.fen or ""
START_TASK = MV_TASK.model_copy(update={"fen": chess.STARTING_FEN, "side_to_move": "white", "move": None})


# position references


def test_references_name_the_task_moves_from_it_a_game_moment_or_a_moment_and_moves() -> None:
    assert parse_reference(text="task") == PositionReference(base="task", moment=None, moves=())
    assert parse_reference(text="after: Nf3 d5") == PositionReference(base="task", moment=None, moves=("Nf3", "d5"))
    assert parse_reference(text="after: 5...Nxd5 6.Nxf7 Kxf7") == PositionReference(base="task", moment=None, moves=("Nxd5", "Nxf7", "Kxf7"))
    assert parse_reference(text="game: 23...") == PositionReference(base="game", moment="23...", moves=())
    assert parse_reference(text="game: 23... then: Qh5 g6") == PositionReference(base="game", moment="23...", moves=("Qh5", "g6"))
    assert parse_reference(text="  game:   45  ") == PositionReference(base="game", moment="45", moves=())


@pytest.mark.parametrize("text", ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "after:", "the position after Nf3", "game: 23... then:"])
def test_raw_fens_and_unknown_references_are_refused(text: str) -> None:
    with pytest.raises(ToolInputError):
        parse_reference(text=text)


def test_game_moments_are_white_and_black_moves_or_plies() -> None:
    assert (moment_ply(moment="23."), moment_ply(moment="23..."), moment_ply(moment="45"), moment_ply(moment="ply 7")) == (45, 46, 45, 7)
    with pytest.raises(ToolInputError):
        moment_ply(moment="0.")


def test_a_raw_fen_is_refused_by_every_tool_and_the_judge_is_told_how_to_name_a_position() -> None:
    result = toolbox(task=MV_TASK).call(name="board", arguments={"position": MV_FEN})
    assert "never as a FEN" in str(result["error"])


def test_a_reference_through_an_illegal_move_says_which_move() -> None:
    result = toolbox(task=MV_TASK).call(name="board", arguments={"position": "after: Nxd5 Qh3"})
    assert "move 2 (Qh3) is illegal" in str(result["error"])


# play


def test_play_replays_a_legal_line_with_pieces_captures_checks_and_the_position_after_each_move() -> None:
    result = toolbox(task=MV_TASK).call(name="play", arguments={"position": "task", "moves": "5...Nxd5 6.Nxf7 Kxf7 7.Qf3+"})
    moves = result["moves"]
    assert isinstance(moves, list) and [m["san"] for m in moves] == ["Nxd5", "Nxf7", "Kxf7", "Qf3+"]
    assert [m["captures"] for m in moves] == ["pawn", "pawn", "knight", None]
    assert [m["check"] for m in moves] == [False, False, False, True] and moves[0]["piece"] == "knight" and moves[2]["piece"] == "king"
    assert moves[3]["fen_after"] == fen_after(fen=MV_FEN, sans=["Nxd5", "Nxf7", "Kxf7", "Qf3+"])
    assert result["illegal"] is None and result["end"] == {"fen": moves[3]["fen_after"], "check": True, "checkmate": False, "stalemate": False}
    assert all(m["notation_problem"] is None for m in moves)


def test_play_stops_at_the_first_illegal_move_and_says_why() -> None:
    result = toolbox(task=MV_TASK).call(name="play", arguments={"position": "task", "moves": "Nxd5 Nxf7 Qh3 Kxf7"})
    assert [m["san"] for m in result["moves"]] == ["Nxd5", "Nxf7"]  # type: ignore[union-attr]
    assert result["illegal"] == {"index": 3, "written": "Qh3", "legal": False, "reason": "no Black move Qh3 exists in this position (no such piece can reach that square, or the move would leave the king in check)"}


def test_a_move_of_the_wrong_side_is_named_as_such() -> None:
    result = toolbox(task=MV_TASK).call(name="play", arguments={"position": "task", "moves": "Nxf7"})
    assert result["illegal"]["reason"] == "it is Black to move, and Nxf7 is a move for the other side"  # type: ignore[index]


def test_a_capture_sign_on_an_empty_square_and_a_false_check_sign_are_notation_problems_of_a_legal_move() -> None:
    capture = toolbox(task=START_TASK).call(name="play", arguments={"position": "task", "moves": "e4 e5 Bb5 Nf6 Bxc6"})
    last = capture["moves"][-1]  # type: ignore[index]
    assert last["san"] == "Bc6" and last["legal"] is True and last["captures"] is None
    assert last["notation_problem"] == "written as a capture, but c6 holds no piece to take"
    check = toolbox(task=MV_TASK).call(name="play", arguments={"position": "task", "moves": "Bb4+"})
    assert check["moves"][0]["notation_problem"] == "written with + but it gives no check"  # type: ignore[index]


def test_mate_and_its_sign() -> None:
    result = toolbox(task=START_TASK).call(name="play", arguments={"position": "task", "moves": "1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#"})
    last = result["moves"][-1]  # type: ignore[index]
    assert last["mate"] is True and last["notation_problem"] is None and result["end"]["checkmate"] is True  # type: ignore[index]
    false_mate = toolbox(task=START_TASK).call(name="play", arguments={"position": "task", "moves": "e4 e5 Qh5#"})
    assert false_mate["moves"][-1]["notation_problem"] == "written with # but it is not mate"  # type: ignore[index]


# board


def test_board_lists_pieces_by_square_and_counts_material_one_three_three_five_nine() -> None:
    result = toolbox(task=MV_TASK).call(name="board", arguments={"position": "task"})
    assert result["side_to_move"] == "black" and result["pieces"]["white"]["d5"] == "pawn" and result["pieces"]["black"]["f6"] == "knight"  # type: ignore[index]
    assert result["material"] == {
        "white": {"pawns": 8, "knights": 2, "bishops": 2, "rooks": 2, "queens": 1},
        "black": {"pawns": 7, "knights": 2, "bishops": 2, "rooks": 2, "queens": 1},
        "white_points": 39,
        "black_points": 38,
        "balance": 1,
    }
    after = toolbox(task=MV_TASK).call(name="board", arguments={"position": "after: Nxd5 Nxf7 Kxf7"})
    assert after["material"]["balance"] == -2  # type: ignore[index]


# evaluate


def test_evaluate_gives_the_eval_from_whites_side_its_band_and_the_top_moves_with_short_lines() -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20, pv=["Na5", "Bb5+", "c6", "dxc6", "bxc6"]), line(rank=2, san="Nd4", cp=10), line(rank=3, san="Nxd5", cp=180)]})
    result = toolbox(task=MV_TASK, analyser=analyser).call(name="evaluate", arguments={"position": "task"})
    assert result["eval_cp_white"] == -20 and result["band"] == "equal" and result["mate"] is None
    assert result["top_moves"][0] == {"move": "Na5", "eval_cp_white": -20, "mate": None, "pv": ["Na5", "Bb5+", "c6", "dxc6"]}  # type: ignore[index]
    assert result["engine"] == {"source": "search", "nodes": BUDGET.nodes, "multipv": BUDGET.multipv}


def test_evaluate_reports_a_forced_mate_in_n() -> None:
    after = fen_after(fen=MV_FEN, sans=["Nxd5", "Nxf7", "Kxf7", "Qf3+"])
    analyser = ScriptedAnalyser(lines={after: [line(rank=1, san="Ke6", cp=1000, mate=7)]})
    result = toolbox(task=MV_TASK, analyser=analyser).call(name="evaluate", arguments={"position": "after: Nxd5 Nxf7 Kxf7 Qf3+"})
    assert (result["mate"], result["eval_cp_white"], result["band"]) == (7, 1000, "white_winning")


def test_evaluate_on_a_mated_board_needs_no_engine() -> None:
    analyser = ScriptedAnalyser()
    result = toolbox(task=START_TASK, analyser=analyser).call(name="evaluate", arguments={"position": "after: e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#"})
    assert result["checkmate"] is True and result["eval_cp_white"] == 1000 and result["top_moves"] == [] and analyser.position_calls == []


@pytest.mark.parametrize(("cp", "band"), [(0, "equal"), (29, "equal"), (30, "white_slightly_better"), (-149, "black_better"), (-150, "black_large_edge"), (-199, "black_large_edge"), (200, "white_winning")])
def test_eval_bands_are_the_tasks_bands(cp: int, band: str) -> None:
    assert eval_band(cp_white=cp, bands=CFG.eval_bands) == band


# judge_move


def test_judge_move_gives_a_blunder_with_the_task_win_formula_and_the_rank() -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20), line(rank=2, san="Nd4", cp=10), line(rank=3, san="h6", cp=60)]}, moves={(MV_FEN, "f6d5"): line(rank=1, san="Nxd5", cp=480, pv=["Nxd5", "Nxf7"])})
    result = toolbox(task=MV_TASK, analyser=analyser).call(name="judge_move", arguments={"position": "task", "move": "Nxd5"})
    before = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * 20)) - 1)
    after = 50 + 50 * (2 / (1 + math.exp(-0.00368208 * -480)) - 1)
    assert result["win_pct_before"] == round(before, 2) and result["win_pct_after"] == round(after, 2)
    assert result["win_pct_drop"] == round(before - after, 2) and result["verdict"] == "blunder"
    assert (result["move"], result["best_move"], result["rank"], result["ranked_moves"]) == ("Nxd5", "Na5", None, 3)
    assert (result["eval_before_cp_white"], result["eval_after_cp_white"]) == (-20, 480)


def test_judge_move_reads_a_top_move_from_the_position_search_and_calls_the_best_move_best() -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20), line(rank=2, san="Nd4", cp=10)]})
    box = toolbox(task=MV_TASK, analyser=analyser)
    best = box.call(name="judge_move", arguments={"position": "task", "move": "Na5"})
    second = box.call(name="judge_move", arguments={"position": "task", "move": "Nd4"})
    assert (best["verdict"], best["rank"], best["win_pct_drop"]) == ("best", 1, 0.0)
    assert (second["rank"], second["verdict"], second["win_pct_drop"]) == (2, "good", 2.76) and analyser.move_calls == []


def test_judge_move_refuses_an_illegal_move() -> None:
    result = toolbox(task=MV_TASK).call(name="judge_move", arguments={"position": "task", "move": "Qh3"})
    assert "no Black move Qh3" in str(result["error"])


# game_position


def test_game_position_gives_the_board_where_a_game_move_was_played() -> None:
    box = toolbox(task=GR_TASK)
    black = box.call(name="game_position", arguments={"move": "9..."})
    white = box.call(name="game_position", arguments={"move": "9."})
    by_ply = box.call(name="game_position", arguments={"move": "17"})
    assert (black["ply"], black["played"], black["side_to_move"]) == (18, "b5", "black")
    assert white == by_ply and (white["ply"], white["played"], white["reference"]) == (17, "Bg5", "game: 9.")
    assert black["fen"] == white["fen_after"]
    end = box.call(name="game_position", arguments={"move": "34"})
    assert end["played"] is None and end["checkmate"] is True
    assert "past its end" in str(box.call(name="game_position", arguments={"move": "40."})["error"])


def test_a_game_reference_continues_with_moves_and_needs_a_game() -> None:
    box = toolbox(task=GR_TASK)
    board = box.call(name="board", arguments={"position": "game: 10. then: Nxb5"})
    after_ply_19 = box.call(name="game_position", arguments={"move": "10..."})
    assert board["fen"] == after_ply_19["fen"]
    assert "game positions exist only for game review tasks" in str(toolbox(task=MV_TASK).call(name="board", arguments={"position": "game: 3."})["error"])
    assert "no single task position" in str(box.call(name="board", arguments={"position": "task"})["error"])


# the toolbox


def test_engine_searches_are_cached_per_position_and_calls_per_arguments() -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20)]})
    box = toolbox(task=MV_TASK, analyser=analyser)
    first = box.call(name="evaluate", arguments={"position": "task"})
    assert box.call(name="evaluate", arguments={"position": "task"}) == first
    box.call(name="judge_move", arguments={"position": "task", "move": "Na5"})
    assert analyser.position_calls == [MV_FEN]


def test_unknown_tools_and_wrong_arguments_are_answered_with_an_error() -> None:
    box = toolbox(task=MV_TASK)
    assert "unknown tool" in str(box.call(name="fen", arguments={})["error"])
    assert "exactly the string arguments position" in str(box.call(name="board", arguments={"position": "task", "fen": "x"})["error"])


needs_engine = pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish binary not installed")


@needs_engine
def test_the_real_engine_finds_mate_in_one_at_the_fixed_budget() -> None:
    cfg = load_bench()
    with Engine.from_config(env=EngineEnv(), cfg=cfg) as engine:
        box = JudgeToolbox(positions=TaskPositions.for_task(task=START_TASK), engine=EngineCache(analyser=engine, budget=BUDGET.model_copy(update={"nodes": 20000})), bands=cfg.eval_bands, thresholds=cfg.labels)
        result = box.call(name="evaluate", arguments={"position": "after: e4 e5 Bc4 Nc6 Qh5 Nf6"})
        assert result["mate"] == 1 and result["top_moves"][0]["move"] == "Qxf7#"  # type: ignore[index]
        assert box.call(name="judge_move", arguments={"position": "after: e4 e5 Bc4 Nc6 Qh5 Nf6", "move": "Qxf7#"})["verdict"] == "best"


def test_the_thresholds_are_the_repositorys() -> None:
    assert CFG.labels == LabelThresholds(blunder_drop=15, mistake_drop=10, inaccuracy_drop=5, best_gap=2)


# review fixes: null moves, notation, the engine


@pytest.mark.parametrize("null", ["--", "0000", "Z0"])
def test_a_null_move_is_refused_by_every_tool_and_never_reaches_the_engine(null: str) -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20)]})
    box = toolbox(task=MV_TASK, analyser=analyser)
    assert "null move" in str(box.call(name="play", arguments={"position": "task", "moves": f"Nxd5 {null}"})["illegal"]["reason"])  # type: ignore[index]
    assert "null move" in str(box.call(name="judge_move", arguments={"position": "task", "move": null})["error"])
    assert "null move" in str(box.call(name="evaluate", arguments={"position": f"after: {null}"})["error"])
    assert analyser.position_calls == [] and analyser.move_calls == []


def test_a_uci_move_is_played_but_flagged_and_an_annotation_before_the_check_sign_is_read() -> None:
    uci = toolbox(task=START_TASK).call(name="play", arguments={"position": "task", "moves": "e2e4"})
    assert uci["moves"][0]["san"] == "e4" and uci["moves"][0]["notation_problem"] == "written as e2e4, a computer (UCI) coordinate move, not SAN (e4)"  # type: ignore[index]
    annotated = toolbox(task=START_TASK).call(name="play", arguments={"position": "task", "moves": "e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7!#"})
    assert annotated["illegal"] is None and annotated["moves"][-1]["mate"] is True and annotated["moves"][-1]["notation_problem"] is None  # type: ignore[index]
    checked = toolbox(task=MV_TASK).call(name="play", arguments={"position": "task", "moves": "Nxd5 Nxf7 Kxf7 Qf3!+"})
    assert checked["illegal"] is None and checked["moves"][-1]["notation_problem"] is None  # type: ignore[index]


class OneAtATime(ScriptedAnalyser):
    """Fails if two searches ever run at once, as one engine process cannot take two."""

    def __init__(self, *, lines: dict[str, list]) -> None:  # type: ignore[type-arg]
        super().__init__(lines=lines)
        self.running = 0
        self.most = 0
        self.guard = threading.Lock()

    def analyse_position(self, *, fen: str, nodes: int, multipv: int) -> object:  # type: ignore[override]
        with self.guard:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(0.02)
        try:
            return super().analyse_position(fen=fen, nodes=nodes, multipv=multipv)
        finally:
            with self.guard:
                self.running -= 1


def test_engine_searches_run_one_at_a_time_and_a_cached_result_needs_no_wait() -> None:
    fens = [fen_after(fen=MV_FEN, sans=sans) for sans in ([], ["Nxd5"], ["Na5"], ["h6"])]
    analyser = OneAtATime(lines={fen: [line(rank=1, san="Kf1", cp=0)] for fen in fens})
    cache = EngineCache(analyser=analyser, budget=BUDGET)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda fen: cache.position(fen=fen), fens * 4))
    assert analyser.most == 1 and sorted(analyser.position_calls) == sorted(fens)


@needs_engine
def test_the_claim_checks_engine_answers_concurrent_searches_as_it_answers_them_alone() -> None:
    """Two searches at once on one Stockfish process stop each other; the cache runs them one at a time, so answers searched from
    many threads are exactly the answers searched alone."""
    cfg = load_bench()
    fens = [chess.STARTING_FEN, MV_FEN, fen_after(fen=MV_FEN, sans=["Nxd5"])]
    budget = BUDGET.model_copy(update={"nodes": 300000})
    with Engine.from_config(env=EngineEnv(), cfg=cfg) as engine:
        alone = [engine.analyse_position(fen=fen, nodes=budget.nodes, multipv=budget.multipv) for fen in fens]
        cache = EngineCache(analyser=engine, budget=budget)
        with ThreadPoolExecutor(max_workers=8) as pool:
            together = list(pool.map(lambda fen: cache.position(fen=fen), fens * 3))[: len(fens)]
    assert together == alone


# the published reference answers its own positions


def _published(*, fen: str, lines: list) -> PositionAnalysis:  # type: ignore[type-arg]
    return PositionAnalysis(fen=fen, version="Stockfish 19", nodes=5000000, multipv=3, lines=lines)


def test_a_position_of_the_tasks_reference_is_answered_from_the_reference_and_others_by_a_search() -> None:
    after = fen_after(fen=MV_FEN, sans=["Nxd5"])
    published = reference_positions(analyses=[_published(fen=MV_FEN, lines=[line(rank=1, san="Na5", cp=-35), line(rank=2, san="Nd4", cp=-5)]), _published(fen=after, lines=[line(rank=1, san="Nxf7", cp=260, pv=["Nxf7", "Kxf7"])])])
    elsewhere = fen_after(fen=MV_FEN, sans=["h6"])
    analyser = ScriptedAnalyser(lines={elsewhere: [line(rank=1, san="Nf3", cp=40)]})
    box = toolbox(task=MV_TASK, analyser=analyser)
    box.reference = published
    task = box.call(name="evaluate", arguments={"position": "task"})
    assert (task["eval_cp_white"], task["engine"]) == (-35, {"source": "reference", "nodes": 5000000, "multipv": 3})
    other = box.call(name="evaluate", arguments={"position": "after: h6"})
    assert (other["eval_cp_white"], other["engine"]) == (40, {"source": "search", "nodes": BUDGET.nodes, "multipv": BUDGET.multipv})
    judged = box.call(name="judge_move", arguments={"position": "task", "move": "Nxd5"})
    assert (judged["eval_before_cp_white"], judged["eval_after_cp_white"], judged["pv"]) == (-35, 260, ["Nxd5", "Nxf7", "Kxf7"])
    assert judged["engine"] == {"before": {"source": "reference", "nodes": 5000000, "multipv": 3}, "after": {"source": "reference", "nodes": 5000000, "multipv": 3}}
    assert analyser.position_calls == [elsewhere] and analyser.move_calls == []


def test_a_move_outside_the_reference_is_searched_and_says_so() -> None:
    published = reference_positions(analyses=[_published(fen=MV_FEN, lines=[line(rank=1, san="Na5", cp=-35)])])
    analyser = ScriptedAnalyser(moves={(MV_FEN, "h7h6"): line(rank=1, san="h6", cp=90)})
    box = toolbox(task=MV_TASK, analyser=analyser)
    box.reference = published
    judged = box.call(name="judge_move", arguments={"position": "task", "move": "h6"})
    assert judged["engine"] == {"before": {"source": "reference", "nodes": 5000000, "multipv": 3}, "after": {"source": "search", "nodes": BUDGET.nodes, "multipv": 1}}


# grandmaster labels


GM = (GmClaim(claim_id="mj_9001_c001", text="5...Nxd5 is a mistake", label="true", comment="Nxf7 wins"), GmClaim(claim_id="mj_9001_c002", text="Black should castle", label="partly_true", comment=""))


def test_gm_claims_lists_ids_and_texts_without_labels_or_comments_and_gm_label_gives_one_label() -> None:
    box = toolbox(task=MV_TASK)
    box.gm = GM
    assert box.call(name="gm_claims", arguments={}) == {"claims": [{"claim_id": "mj_9001_c001", "text": "5...Nxd5 is a mistake"}, {"claim_id": "mj_9001_c002", "text": "Black should castle"}]}
    assert box.call(name="gm_label", arguments={"claim_id": " mj_9001_c001 "}) == {"claim_id": "mj_9001_c001", "text": "5...Nxd5 is a mistake", "label": "true", "comment": "Nxf7 wins"}
    assert "no grandmaster-labelled claim has the id 'mj_9001_c009'" in str(box.call(name="gm_label", arguments={"claim_id": "mj_9001_c009"})["error"])
    assert box.call(name="gm_claims", arguments={"position": "task"})["error"] == 'gm_claims takes exactly the string arguments ; got {"position":"task"}'
    assert GM_TOOLS == ("gm_claims", "gm_label")


def test_gm_claims_shows_the_structured_fields_a_claim_has() -> None:
    box = toolbox(task=MV_TASK)
    box.gm = (GmClaim(claim_id="mj_9001_c004", text="Na5 was best", label="true", comment="", kind="best_move", move="Na5", band=None, about="task_position"),)
    assert box.call(name="gm_claims", arguments={}) == {"claims": [{"claim_id": "mj_9001_c004", "text": "Na5 was best", "kind": "best_move", "move": "Na5", "about": "task_position"}]}


def test_a_task_without_labelled_claims_lists_none() -> None:
    assert toolbox(task=MV_TASK).call(name="gm_claims", arguments={}) == {"claims": []}


# re-checks from stored calls


def test_seeded_calls_are_answered_without_the_engine_but_grandmaster_tools_are_read_as_they_stand_now() -> None:
    analyser = ScriptedAnalyser()
    box = toolbox(task=MV_TASK, analyser=analyser)
    box.gm = (GmClaim(claim_id="mj_9001_c001", text="5...Nxd5 is a mistake", label="false", comment="relabelled"),)
    stored_eval: dict[str, object] = {"eval_cp_white": -20, "stored": True}
    box.seed(calls=[("evaluate", {"position": "task"}, stored_eval), ("gm_label", {"claim_id": "mj_9001_c001"}, {"label": "true"}), ("gm_claims", {}, {"claims": []})])
    assert box.call(name="evaluate", arguments={"position": "task"}) == stored_eval and analyser.position_calls == []
    assert box.call(name="gm_label", arguments={"claim_id": "mj_9001_c001"})["label"] == "false"
    assert box.call(name="gm_claims", arguments={}) == {"claims": [{"claim_id": "mj_9001_c001", "text": "5...Nxd5 is a mistake"}]}


# boards code names directly


def test_evaluate_board_and_judge_move_on_answer_for_a_board_as_the_tools_answer_for_its_reference() -> None:
    analyser = ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20), line(rank=2, san="Nd4", cp=10)]}, moves={(MV_FEN, "f6d5"): line(rank=1, san="Nxd5", cp=480)})
    box = toolbox(task=MV_TASK, analyser=analyser)
    board = chess.Board(MV_FEN)
    assert box.evaluate_board(board=board, position="task") == box.call(name="evaluate", arguments={"position": "task"})
    assert box.judge_move_on(board=board, position="task", move="Nxd5") == box.call(name="judge_move", arguments={"position": "task", "move": "Nxd5"})
    assert board.fen() == MV_FEN
    mated = chess.Board(fen_after(fen=chess.STARTING_FEN, sans=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]))
    with pytest.raises(ToolInputError, match="the game is over"):
        box.judge_move_on(board=mated, position="the claim's position", move="Ke7")


# task positions and move tokens


def test_task_positions_of_a_position_task_hold_its_fen_and_of_a_game_review_its_game() -> None:
    assert TaskPositions.of(task_id="mj_9001", family="move_judgement", fen=MV_FEN, pgn="1. e4") == TaskPositions(fen=MV_FEN, game_start=None, game_moves=None)
    game = TaskPositions.of(task_id="gr_9001", family="game_review", fen=None, pgn=GR_TASK.pgn)
    assert game == TaskPositions.for_task(task=GR_TASK) and game.fen is None and len(game.game_moves or ()) == 33
    with pytest.raises(ValueError, match="gr_9002: its PGN does not parse"):
        TaskPositions.of(task_id="gr_9002", family="game_review", fen=None, pgn="1. e4 e5 2. Ke3")


def test_move_tokens_drop_move_numbers_commas_and_figurines() -> None:
    assert move_tokens(text="5...Nxd5 6.Nxf7, Kxf7 7. ♕f3+") == ("Nxd5", "Nxf7", "Kxf7", "Qf3+")
    assert move_tokens(text="  ") == ()
