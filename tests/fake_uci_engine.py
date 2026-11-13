"""A tiny UCI engine for the review app's live-engine tests: it names itself like the pinned engine and answers `go` with a few
scripted info lines whose principal variations are legal in the position it was given (the k-th line starts with the k-th legal move)."""

import sys

import chess

NAME = sys.argv[1] if len(sys.argv) > 1 else "Stockfish 19"
DEPTHS = (1, 2, 3)


def main() -> None:
    board = chess.Board()
    multipv = 1
    for raw in sys.stdin:
        parts = raw.split()
        if not parts:
            continue
        command = parts[0]
        if command == "uci":
            print(f"id name {NAME}")
            print("option name Threads type spin default 1 min 1 max 1024")
            print("option name Hash type spin default 16 min 1 max 33554432")
            print("option name MultiPV type spin default 1 min 1 max 256")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "setoption" and parts[2] == "MultiPV":
            multipv = int(parts[4])
        elif command == "position":
            board = _board(parts=parts)
        elif command == "go":
            moves = list(board.legal_moves)
            for depth in DEPTHS:
                for rank, move in enumerate(moves[:multipv], start=1):
                    after = board.copy()
                    after.push(move)
                    reply = next(iter(after.legal_moves), None)
                    pv = move.uci() + (f" {reply.uci()}" if reply else "")
                    print(f"info depth {depth} seldepth {depth + 2} multipv {rank} score cp {50 - 10 * rank} nodes {depth * 1000} nps 2500000 time {depth * 10} pv {pv}")
            print(f"bestmove {moves[0].uci() if moves else '(none)'}")
        elif command == "quit":
            return
        sys.stdout.flush()


def _board(*, parts: list[str]) -> chess.Board:
    if parts[1] == "startpos":
        board, rest = chess.Board(), parts[2:]
    else:
        end = parts.index("moves") if "moves" in parts else len(parts)
        board, rest = chess.Board(" ".join(parts[2:end])), parts[end:]
    for uci in rest[1:]:
        board.push_uci(uci)
    return board


if __name__ == "__main__":
    main()
