"""Draw the chess boards in the docs as SVG (python-chess), into docs/img/board_<name>.svg.

Run from the repository root: uv run python docs/img/make_boards.py. tests/test_doc_images.py checks that the committed
files match what this script draws. Each board is seen from the side to move; a green arrow is the best move, a red one
the move under discussion when it is not the best."""

from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.svg

OUT = Path(__file__).resolve().parent
SIZE = 360
COLORS = {"square light": "#f0d9b5", "square dark": "#b58863", "margin": "#2b2f36", "coord": "#e6edf3"}
GREEN, RED = "#15781bb3", "#b42318b3"


@dataclass(frozen=True)
class Board:
    name: str
    fen: str
    arrows: list[tuple[str, str]] = field(default_factory=list)  # (uci move, colour)


BOARDS = [
    # docs/worked_example.md: the Opera Game before 10.Nxb5
    Board("worked_example", "rn2kb1r/p3qppp/2p2n2/1p2p1B1/2B1P3/1QN5/PPP2PPP/R3K2R w KQkq - 0 10"),
    # docs/annotator_guide.md: Réti's study, 1.Kg7
    Board("reti_study", "7K/8/k1P5/7p/8/8/8/8 w - - 0 1", [("h8g7", GREEN)]),
    # docs/annotator_guide.md: the pawn ending and the judged move, 1.e5
    Board("pawn_ending", "8/4k3/8/5K2/4P3/8/8/8 w - - 0 1", [("e4e5", RED)]),
    # docs/annotator_guide.md: Queen's Gambit Declined, Exchange Variation, move 11
    Board("qgd_exchange", "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBP3/PPQ1NPPP/R4RK1 w - - 0 11"),
]


def draw(*, spec: Board) -> str:
    board = chess.Board(spec.fen)
    arrows = []
    for uci, colour in spec.arrows:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"{spec.name}: {uci} is not legal in {spec.fen}")
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=colour))
    return chess.svg.board(board, orientation=board.turn, arrows=arrows, size=SIZE, coordinates=True, colors=COLORS) + "\n"


def main(*, out: Path = OUT) -> None:
    for spec in BOARDS:
        (out / f"board_{spec.name}.svg").write_text(draw(spec=spec), encoding="utf-8")


if __name__ == "__main__":
    main()
