import chess

from review_app.board_line import position_label
from review_app.claim_lines import ILLEGAL_FROM_HERE, NOT_FROM_TASK, UNREAD, claim_lines

TASK_FEN = "r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"


def _lines(text: str):
    return claim_lines(text=text, positions=[chess.Board(TASK_FEN)], starts=(0,))


def test_a_numbered_legal_line_with_annotations_is_clickable() -> None:
    [line] = _lines("Black should play 5...Na5! 6. Bb5+?! c6, keeping the pawn.")
    assert line.clickable and line.note == ""
    assert line.playable == "5...Na5! 6. Bb5+?! c6" and line.rest == ""
    assert line.uci == ("c6a5", "c4b5", "c7c6") and line.start_ply == 0


def test_a_line_without_numbers_starts_from_the_side_to_move() -> None:
    [line] = _lines("Na5 Bb5+ c6 holds")
    assert line.clickable and line.uci == ("c6a5", "c4b5", "c7c6")


def test_a_line_numbered_for_the_other_side_does_not_start_from_the_task_position() -> None:
    [line] = _lines("5. Na5 Bb5+")
    assert not line.clickable and line.note == NOT_FROM_TASK and line.playable == "" and line.rest == "5. Na5 Bb5+"
    [later] = _lines("7...Na5")
    assert not later.clickable and later.note == NOT_FROM_TASK


def test_an_illegal_move_stops_the_line_and_mutes_the_rest() -> None:
    [line] = _lines("5...Na5 6.Bb5+ c6 7.Qxh8 Kd7")
    assert line.clickable and line.note == ILLEGAL_FROM_HERE
    assert line.playable == "5...Na5 6.Bb5+ c6" and line.rest == "7.Qxh8 Kd7" and len(line.uci) == 3
    [first] = _lines("5...Qh4 6.Nf3")
    assert not first.clickable and first.note == ILLEGAL_FROM_HERE and first.rest == "5...Qh4 6.Nf3"


def test_a_dash_between_squares_is_one_move_and_lone_squares_are_prose() -> None:
    first, second, hyphen = _lines("Nxd5 to hit f4, then Qd8–e7 or Qd8-e7")
    assert first.playable == "Nxd5" and first.uci == ("f6d5",)
    assert second.playable == "Qd8–e7" and second.uci == hyphen.uci == ("d8e7",) and second.note == ""
    [white] = claim_lines(text="1. Nb1–c3 e5 2. Ng1-f3", positions=[chess.Board()], starts=(0,))
    assert white.uci == ("b1c3", "e7e5", "g1f3")
    [plan] = claim_lines(text="Rf1 to protect f5, then Rf4–h4", positions=[chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")], starts=(0,))[1:]
    assert not plan.clickable and plan.note == ILLEGAL_FROM_HERE and plan.rest == "Rf4–h4", "a written move whose piece is elsewhere is illegal, as one move"


def test_a_fragment_that_does_not_read_as_one_move_is_muted_without_calling_it_illegal() -> None:
    [line] = claim_lines(text="1. Nd2 Kd7", positions=[chess.Board("4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1")], starts=(0,))
    assert not line.clickable and line.note == UNREAD and line.rest == "1. Nd2 Kd7"


def test_a_game_review_line_starts_at_the_game_position_of_its_number() -> None:
    board = chess.Board()
    positions = [board.copy()]
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(san)
        positions.append(board.copy())
    [line] = claim_lines(text="2...Nf6 3.Nxe5", positions=positions, starts=None)
    assert line.clickable and line.start_ply == 3 and line.uci == ("g8f6", "f3e5")
    [unnumbered] = claim_lines(text="Bc4 Nf6", positions=positions, starts=None)
    assert not unnumbered.clickable and unnumbered.note == NOT_FROM_TASK


def test_a_move_judgement_line_may_start_after_the_judged_move() -> None:
    task = chess.Board(TASK_FEN)
    after = task.copy()
    after.push_san("Na5")
    positions = [task, after]
    [numbered] = claim_lines(text="6.Bb5+ c6 7.dxc6", positions=positions, starts=(0, 1))
    assert numbered.clickable and numbered.start_ply == 1 and numbered.uci == ("c4b5", "c7c6", "d5c6")
    [from_task] = claim_lines(text="5...Na5 6.Bb5+", positions=positions, starts=(0, 1))
    assert from_task.start_ply == 0
    [unnumbered] = claim_lines(text="Bb5+ c6", positions=positions, starts=(0, 1))
    assert unnumbered.clickable and unnumbered.start_ply == 1, "an unnumbered line starts where its first move is legal"
    [elsewhere] = claim_lines(text="8.Qh5", positions=positions, starts=(0, 1))
    assert elsewhere.note == NOT_FROM_TASK


def test_the_position_label_names_the_move_number_and_side_to_move() -> None:
    task = chess.Board(TASK_FEN)
    assert position_label(board=task, at_task=True) == "Task position · move 5, Black to move"
    after = task.copy()
    after.push_san("Na5")
    assert position_label(board=after, at_task=False) == "after 5...Na5"
    after.push_san("Bb5+")
    assert position_label(board=after, at_task=False) == "after 6.Bb5+"
    assert position_label(board=chess.Board(TASK_FEN), at_task=False) == "move 5, Black to move"
