from pathlib import Path

import pytest

from bench.grading.moves import check_legality, find_lines, find_standalone_moves
from bench.io_yaml import read_task
from bench.models import Source, Task

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"
POS_9001 = read_task(path=FIXTURES / "ap_9001.yaml")
MV_9001 = read_task(path=FIXTURES / "mj_9001.yaml")
GR_9001 = read_task(path=FIXTURES / "gr_9001.yaml")
FEN12 = "r1bq1rk1/pppnbppp/4p3/3p4/3PP3/2PB1N2/PP3PPP/R1BQ1RK1 b - - 0 12"
FEN_CASTLE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _task(fen: str, pgn: str | None = None) -> Task:
    side = "white" if fen.split(" ")[1] == "w" else "black"
    return Task(
        task_id="ap_9100",
        status="draft",
        family="assessment_plan",
        version="test",
        fen=fen,
        side_to_move=side,
        pgn=pgn,
        move=None,
        user_color=None,
        question="q",
        audience="club",
        focus="plan",
        source=Source(origin="lichess_db", game_id="x", month="2026-08", players_stripped=True, ply=None),
        tags=[],
    )


def test_find_lines_recognises_numbered_sequences() -> None:
    lines = find_lines(text="After 12...Nf6 13.Bg5 h6 White is fine. Then 1. e4 e5 2. Nf3 Nc6 later.")
    assert [(l.start_move_number, l.black_starts, l.sans) for l in lines] == [(12, True, ["Nf6", "Bg5", "h6"]), (1, False, ["e4", "e5", "Nf3", "Nc6"])]
    assert lines[0].span == (6, 24)


def test_find_lines_handles_spaced_dots_and_annotations() -> None:
    lines = find_lines(text="12... Nf6!? 13. Bg5+ h6# wins")
    assert lines[0].sans == ["Nf6", "Bg5", "h6"] and lines[0].black_starts


def test_find_standalone_moves_maps_variants_and_skips_bare_squares() -> None:
    mentions = find_standalone_moves(text="♘f3!? keeps options; castle (0-0) or 0-0-0, exd5, e8=Q, the b6 pawn, g1f3 later.")
    assert [m.san for m in mentions] == ["Nf3", "O-O", "O-O-O", "exd5", "e8=Q", "g1f3"]


def test_find_standalone_moves_ignores_moves_inside_lines() -> None:
    mentions = find_standalone_moves(text="12...Nf6 13.Bg5 h6 and later Nd5")
    assert [m.san for m in mentions] == ["Nd5"]


@pytest.mark.parametrize(
    "fen,text,expected_illegal",
    [
        (FEN12, "After 12...Nf6 13.Bg5 h6 White is fine.", None),
        (FEN12, "After 12...Nf6 13.Bg5 Qh9 White is fine.", "Qh9"),
        (FEN_CASTLE, "White should castle (0-0) soon.", None),
        (POS_9001.fen or "", "♘f3!? keeps options.", None),
        (FEN_START, "The engine line g1f3 then e7e5 is fine.", None),
        (FEN12, "12...Nf6 is natural.", None),
        (FEN12, "12...Nf6 13.Bg5 h6 14.Bxh9 wins.", "Bxh9"),
        (FEN_CASTLE, "Bb5!? is possible but Nf3 is already there.", None),
        (FEN_CASTLE, "Nh9 would be a dream.", "Nh9"),
    ],
)
def test_check_legality_examples(fen: str, text: str, expected_illegal: str | None) -> None:
    result = check_legality(task=_task(fen), text=text)
    illegal = [c.illegal_san for c in result.lines if c.illegal_san] + [s.san for s in result.standalone if s.impossible]
    assert (illegal[0] if illegal else None) == expected_illegal
    assert result.hallucination == (expected_illegal is not None)


def test_prose_move_is_not_checked() -> None:
    assert not check_legality(task=_task(FEN12), text="the knight goes to h9 and the bishop to z3").hallucination


def test_line_with_foreign_move_number_is_skipped() -> None:
    result = check_legality(task=_task(FEN12), text="Later 20.Qh9 Qh8 21.Rz1 decides.")
    assert result.lines == [] and len(result.skipped) == 1 and not result.hallucination


def test_line_inside_pgn_mainline_is_replayed_from_there() -> None:
    assert check_legality(task=MV_9001, text="After 2.Nf3 Nc6 3.Bb5 the game is different.").hallucination is False
    result = check_legality(task=MV_9001, text="After 2.Nf3 Qh9 the game is different.")
    assert result.hallucination and result.lines[0].illegal_san == "Qh9" and result.lines[0].legal_prefix_len == 1


def test_line_from_task_position() -> None:
    result = check_legality(task=MV_9001, text="5...Nxd5 6.Nxf7 Kxf7 7.Qf3+ is the point.")
    assert result.lines[0].legal_prefix_len == 4 and not result.hallucination


def test_game_review_checks_lines_and_off_board_squares_only() -> None:
    result = check_legality(task=GR_9001, text="9...b5 10.Nxb5 cxb5 11.Bxb5+ was decisive; Rd8# ends it; Qh7 is prose-like.")
    assert result.lines[0].legal_prefix_len == 4 and result.standalone == [] and not result.hallucination
    off_board = check_legality(task=GR_9001, text="9...b5 10.Nxb5 cxb5 11.Bxb5+ was decisive; Qh9 is prose-like.")
    assert [(s.san, s.impossible) for s in off_board.standalone] == [("Qh9", True)] and off_board.illegal_moves == 1
    bad = check_legality(task=GR_9001, text="9...b5 10.Nxb5 Qh9")
    assert bad.hallucination and bad.lines[0].illegal_san == "Qh9"


def test_standalone_move_legal_for_the_other_side_is_tolerated() -> None:
    result = check_legality(task=_task(FEN_START), text="e7e5 and Nf6 are natural replies.")
    assert not result.hallucination


def test_standalone_moves_illegal_in_the_task_position_are_ambiguous_not_hallucinations() -> None:
    fen = "rn2k2r/ppq1bpp1/2p1pn1p/5b2/2B5/2NP1N1P/PPPB1PP1/R2QR1K1 w kq - 2 11"
    result = check_legality(task=_task(fen), text="With the Bc4 and Re1 lined up on e6, ideas like Nxf5 and cxd3 hang in the air.")
    assert [s.san for s in result.standalone] == ["Bc4", "Re1", "Nxf5", "cxd3"]
    assert not any(s.legal for s in result.standalone) and not any(s.impossible for s in result.standalone)
    assert not result.hallucination


def test_markdown_list_items_are_ambiguous_lines_unless_numbered_from_the_task_position() -> None:
    fen = "rn1r2k1/pp1bppb1/2p3p1/q5Pp/P1pPPN1P/2P1BP2/3Q4/R3KB1R w KQ - 7 15"
    pgn = "1. d4 c6 2. c4 d5 3. Nc3 Nf6 4. f3 dxc4 5. a4 Nd5 6. e4 Nxc3 7. bxc3 Be6 8. Be3 g6 9. h4 Bg7 10. g4 h5 11. g5 Qa5 12. Qd2 O-O 13. Ne2 Rd8 14. Nf4 Bd7"
    task = _task(fen, pgn)
    result = check_legality(task=task, text="Two ideas:\n1. Bxc4 restores material.\n2. Kf2 gives Black a tempo.\n")
    # "1. Bxc4" reads as numbered from the task position, White to move there; "2. Kf2" continues nothing and stays ambiguous
    assert [line.sans for line in result.lines] == [["Bxc4"]] and len(result.skipped) == 1 and not result.hallucination
    illegal_first = check_legality(task=task, text="Two ideas:\n1. Nxe5 is not possible.\n2. Kf2 gives Black a tempo.\n")
    assert illegal_first.lines == [] and len(illegal_first.skipped) == 2 and not illegal_first.hallucination


def test_pgn_line_at_line_start_keeps_its_move_numbers() -> None:
    lines = find_lines(text="1. e4 e5 2. Nf3 Nc6 is the game.")
    assert lines[0].sans == ["e4", "e5", "Nf3", "Nc6"] and lines[0].start_move_number == 1
    assert find_lines(text="15.Kf2 is a sensible plan.")[0].sans == ["Kf2"]


def test_list_item_move_is_not_a_hallucination() -> None:
    fen = "rn1r2k1/pp1bppb1/2p3p1/q5Pp/P1pPPN1P/2P1BP2/3Q4/R3KB1R w KQ - 7 15"
    pgn = "1. d4 c6 2. c4 d5 3. Nc3 Nf6 4. f3 dxc4 5. a4 Nd5 6. e4 Nxc3 7. bxc3 Be6 8. Be3 g6 9. h4 Bg7 10. g4 h5 11. g5 Qa5 12. Qd2 O-O 13. Ne2 Rd8 14. Nf4 Bd7"
    task = _task(fen, pgn)
    result = check_legality(task=task, text="Taking is safe.\n\n2. Kf2 gives Black a tempo to save the pawn.")
    assert result.lines == [] and not result.hallucination


def test_repeated_move_number_starts_a_new_line() -> None:
    lines = find_lines(text="After 23...g6\n\n23...g6 is safe but slow.")
    assert [(l.start_move_number, l.black_starts, l.sans) for l in lines] == [(23, True, ["g6"]), (23, True, ["g6"])]
    lines = find_lines(text="23.Nf1?? Qh1#\n23.Nxg4?? Qh1#")
    assert [l.sans for l in lines] == [["Nf1", "Qh1"], ["Nxg4", "Qh1"]]


def test_number_glued_to_a_square_is_not_a_move_number() -> None:
    text = "the king stays stuck around e1/e2.\nBe6 pressures the queenside."
    assert find_lines(text=text) == []
    assert [m.san for m in find_standalone_moves(text=text)] == ["Be6"]


def test_ambiguous_san_counts_as_legal() -> None:
    fen = "r1q1k2r/4ppb1/p3nn2/1p1pNR2/2pP3p/Q1P1P3/PP1NB1PP/R5K1 w kq - 0 18"
    result = check_legality(task=_task(fen), text="18.Rf1 is quieter but also consolidates.")
    assert result.lines[0].legal_prefix_len == 1 and not result.hallucination


GAME = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5"
GAME_END = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_line_continues_the_hypothetical_line_before_it() -> None:
    text = "Better was 3.Bb5, and after 3...a6 4.Bxc6 dxc6 White is comfortable."
    result = check_legality(task=_task(GAME_END, GAME), text=text)
    assert [l.legal_prefix_len for l in result.lines] == [1, 3] and not result.hallucination


def test_hypothetical_line_without_a_known_anchor_is_skipped() -> None:
    text = "Instead of 3.Bc4, 3.Bb5 was better, threatening 4.Bxc6 at once."
    result = check_legality(task=GR_9001.model_copy(update={"pgn": GAME}), text=text)
    assert not result.hallucination and len(result.skipped) == 1 and result.skipped[0].sans == ["Bxc6"]


def test_line_quoting_the_game_then_going_wrong_is_illegal() -> None:
    result = check_legality(task=_task(GAME_END, GAME), text="After 3.Bc4 Bc5 4.Bxc6 White wins a piece.")
    assert result.hallucination and result.lines[0].illegal_san == "Bxc6" and result.lines[0].legal_prefix_len == 2


def test_line_after_the_task_move_is_anchored_there() -> None:
    fen = "rn2k2r/ppq1bpp1/2p1pn1p/5b2/2B5/2NP1N1P/PPPB1PP1/R2QR1K1 w kq - 2 11"
    task = _task(fen).model_copy(update={"family": "move_choice", "move": "Nd4"})
    result = check_legality(task=task, text="After 11...Bh7 12.Nxe6 fxe6 13.Rxe6 Kf7 Black consolidates.")
    assert result.lines[0].legal_prefix_len == 5 and not result.hallucination


def test_sequence_of_one_sides_moves_is_ambiguous_not_illegal() -> None:
    fen = "r4rk1/1pq2ppp/p1nbpn2/3p4/3P4/4BB1P/PPP1NPP1/R2QR1K1 w - - 8 13"
    result = check_legality(task=_task(fen), text="The plan is 13.c3 Ng3 Qd2 and then f4.")
    assert result.lines == [] and len(result.skipped) == 1 and not result.hallucination
    wrong = check_legality(task=_task(fen), text="After 13.c3 Bxe3 14.fxe3 the structure changes.")
    assert wrong.hallucination and wrong.lines[0].illegal_san == "Bxe3"


def test_threat_after_an_implied_pass_is_anchored_at_the_task_position() -> None:
    fen = "4n3/p4pk1/1b4p1/5b2/5p1p/2PB1P1P/PP1R2P1/4K3 b - - 1 34"
    task = _task(fen).model_copy(update={"family": "move_judgement", "move": "Bxd3"})
    result = check_legality(task=task, text="White threatened 35.Bxf5 gxf5 with doubled f-pawns.")
    assert result.lines[0].legal_prefix_len == 2 and not result.hallucination


def test_listed_alternative_at_the_task_move_anchors_its_continuation() -> None:
    fen = "r1q1k2r/4ppb1/p3nn2/1p1pNR2/2pP3p/Q1P1P3/PP1NB1PP/R5K1 w kq - 0 18"
    task = _task(fen).model_copy(update={"family": "move_judgement", "move": "Ndf3"})
    text = "Better:\n18. Raf1! protects the rook (if 18... Nxd4 19. exd4 Qxf5 20. Rxf5 White wins the queen).\n18. Rf2 is also safe."
    result = check_legality(task=task, text=text)
    assert [l.legal_prefix_len for l in result.lines] == [1, 4, 1] and not result.hallucination


def test_hyphenated_long_algebraic_is_one_move() -> None:
    assert [l.sans for l in find_lines(text="1.e2-e4 e7-e5 2.Ng1-f3")] == [["e2-e4", "e7-e5", "Ng1-f3"]]
    fen = "rn2k2r/ppq1bpp1/2p1pn1p/5b2/2B5/2NP1N1P/PPPB1PP1/R2QR1K1 w kq - 2 11"
    result = check_legality(task=_task(fen), text="11.Nf3-d4 Bg6 is natural.")
    assert result.lines[0].legal_prefix_len == 2 and not result.hallucination


def test_illegal_moves_counts_each_illegal_move_in_sequences_with_a_certain_start() -> None:
    assert check_legality(task=_task(FEN12), text="After 12...Nf6 13.Bg5 h6 White is fine.").illegal_moves == 0
    assert check_legality(task=_task(FEN12), text="After 12...Nf6 13.Bg5 Qh9 White is fine.").illegal_moves == 1
    two = check_legality(task=_task(FEN12), text="After 12...Nf6 13.Bg5 Qh9 White is fine, but 12...Nf6 13.Bg5 Bxh9 is no better.")
    assert two.illegal_moves == 2 and two.hallucination


def test_impossible_squares_in_loose_text_count_as_illegal_moves() -> None:
    assert check_legality(task=_task(FEN_CASTLE), text="Nh9 would be a dream.").illegal_moves == 1


def test_skipped_ambiguous_lines_and_ambiguous_standalone_moves_are_not_counted() -> None:
    skipped = check_legality(task=_task(FEN12), text="Later 20.Qh9 Qh8 21.Rz1 decides.")
    assert skipped.illegal_moves == 0 and len(skipped.skipped) == 1
    fen = "rn2k2r/ppq1bpp1/2p1pn1p/5b2/2B5/2NP1N1P/PPPB1PP1/R2QR1K1 w kq - 2 11"
    loose = check_legality(task=_task(fen), text="With the Bc4 and Re1 lined up on e6, ideas like Nxf5 and cxd3 hang in the air.")
    assert loose.illegal_moves == 0


def test_squares_off_the_board_count_in_sequences_bare_mentions_and_game_reviews() -> None:
    fen = "6k1/5pp1/6qp/8/8/1PKP1Q2/r5P1/2R5 w - - 0 28"
    assert check_legality(task=_task(fen), text="28.Qh0 wins").illegal_moves == 1
    assert check_legality(task=_task(fen), text="the idea Qh9 and then Rc0").illegal_moves == 2
    review = _task(fen).model_copy(update={"family": "game_review", "fen": None, "side_to_move": None, "pgn": "1. e4 e5 2. Nf3 Nc6"})
    result = check_legality(task=review, text="Black should have played Qh9 here")
    assert result.illegal_moves == 1 and result.hallucination and [s.san for s in result.standalone] == ["Qh9"]
    assert check_legality(task=review, text="Black should have played Nf6 here").standalone == []
