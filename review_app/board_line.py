"""The moves the review board steps through for a task: the game that led to the task position, then the task's own move."""

import io
from dataclasses import dataclass

import chess
import chess.pgn

from bench.models import Task


@dataclass(frozen=True)
class TaskLine:
    start_fen: str
    moves: list[str]
    task_ply: int


def _game_moves(*, pgn: str, task_id: str) -> tuple[chess.Board, list[str]]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"{task_id} has a pgn that does not parse")
    board = game.board()
    start = board.copy()
    moves: list[str] = []
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return start, moves


def task_line(*, task: Task) -> TaskLine:
    if task.family == "game_review":
        if task.pgn is None:
            raise ValueError(f"{task.task_id} is a game review without a pgn")
        start, moves = _game_moves(pgn=task.pgn, task_id=task.task_id)
        return TaskLine(start_fen=start.fen(), moves=moves, task_ply=0)
    if task.fen is None:
        raise ValueError(f"{task.task_id} has no fen")
    if task.pgn is None:
        start, moves = chess.Board(task.fen), []
    else:
        start, moves = _game_moves(pgn=task.pgn, task_id=task.task_id)
        reached = start.copy()
        for san in moves:
            reached.push_san(san)
        if reached.epd() != chess.Board(task.fen).epd():
            raise ValueError(f"{task.task_id}: the pgn does not reach the task position")
    task_board = start.copy()
    for san in moves:
        task_board.push_san(san)
    task_ply = len(moves)
    if task.family == "move_judgement":
        if task.move is None:
            raise ValueError(f"{task.task_id} needs a move")
        moves = [*moves, task_board.san(task_board.parse_san(task.move))]
    return TaskLine(start_fen=start.fen(), moves=moves, task_ply=task_ply)


def position_label(*, board: chess.Board, at_task: bool) -> str:
    """Which move the board shows: "Task position · move 20, White to move", "after 20...f5", or "move 21, Black to move"
    when no move led here. The page's board script words its labels the same way."""
    to_move = f"move {board.fullmove_number}, {'White' if board.turn == chess.WHITE else 'Black'} to move"
    if at_task:
        return f"Task position · {to_move}"
    if not board.move_stack:
        return to_move
    before = board.copy()
    last = before.pop()
    return f"after {before.fullmove_number}{'.' if before.turn == chess.WHITE else '...'}{before.san(last)}"


def line_positions(*, line: TaskLine) -> list[chess.Board]:
    """The board at each ply of the line, from its start; each carries the moves that led to it."""
    board = chess.Board(line.start_fen)
    positions = [board.copy()]
    for san in line.moves:
        board.push_san(san)
        positions.append(board.copy())
    return positions
