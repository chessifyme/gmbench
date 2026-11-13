"""Lines of moves written in a claim's text ("20...f5 21.bxc5 Rxc5"), checked on the board so the page can play them."""

import re
from dataclasses import dataclass

import chess

NOT_FROM_TASK = "this line doesn't start from the task position"
ILLEGAL_FROM_HERE = "illegal from here"
UNREAD = "not read as one move"

_SAN = r"(?:O-O-O|O-O|0-0-0|0-0|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=?[QRBN])?)[+#]?"
_LONG = r"(?P<piece>[KQRBN])?(?P<from>[a-h][1-8])[–—-](?P<to>[a-h][1-8])(?:=?(?P<promotion>[QRBN]))?[+#]?"
_TOKEN = re.compile(r"(?<![\w.…])(?:(?P<number>\d+)\s*(?P<dots>\.\.\.|…|\.)\s*)?(?:" + _LONG + r"|(?P<san>" + _SAN + r"))(?P<note>[!?]{1,2})?(?![\w])")
_GAP = re.compile(r"[\s,;…]*|\s*\.\.\.\s*")
_LONE_SQUARE = re.compile(r"[a-h][1-8]")
_PIECES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


@dataclass(frozen=True)
class _Token:
    """One written move: SAN (san), or long algebraic with a dash (from_square, to_square, the piece and promotion if named)."""

    start: int
    end: int
    number: int | None
    black: bool | None
    san: str | None = None
    piece: chess.PieceType | None = None
    from_square: chess.Square | None = None
    to_square: chess.Square | None = None
    promotion: chess.PieceType | None = None


@dataclass(frozen=True)
class ClaimLine:
    """playable: the text of the moves that can be played in turn from where the line starts; rest: the text after them,
    shown muted with the note. uci: the playable moves; start_ply: the board's ply the line starts from."""

    playable: str
    rest: str
    uci: tuple[str, ...]
    start_ply: int | None
    note: str

    @property
    def clickable(self) -> bool:
        return bool(self.uci)


def _tokens(*, text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _TOKEN.finditer(text):
        number = match.group("number")
        where = {"start": match.start(), "end": match.end(), "number": int(number) if number else None, "black": None if number is None else match.group("dots") != "."}
        if match.group("san"):
            tokens.append(_Token(**where, san=match.group("san").replace("0", "O")))
            continue
        piece, promotion = match.group("piece"), match.group("promotion")
        tokens.append(_Token(**where, piece=_PIECES[piece] if piece else None, from_square=chess.parse_square(match.group("from")), to_square=chess.parse_square(match.group("to")), promotion=_PIECES[promotion] if promotion else None))
    return tokens


def _move(*, board: chess.Board, token: _Token) -> chess.Move | str:
    """The legal move the token names, or why not: ILLEGAL_FROM_HERE for a clear move that is not legal here, UNREAD for one
    that names no single move (an ambiguous SAN, a long move that fits several)."""
    if token.san is not None:
        try:
            return board.parse_san(token.san)
        except chess.IllegalMoveError:
            return ILLEGAL_FROM_HERE
        except (chess.AmbiguousMoveError, chess.InvalidMoveError):
            return UNREAD
    fits = [
        move
        for move in board.legal_moves
        if move.from_square == token.from_square
        and move.to_square == token.to_square
        and (token.piece is None or board.piece_type_at(move.from_square) == token.piece)
        and (token.promotion is None or move.promotion == token.promotion)
    ]
    if len(fits) == 1:
        return fits[0]
    return UNREAD if fits else ILLEGAL_FROM_HERE


def _runs(*, text: str, tokens: list[_Token]) -> list[list[_Token]]:
    """Tokens joined by nothing but spaces and punctuation; a token numbered out of turn starts a new run."""
    runs: list[list[_Token]] = []
    for token in tokens:
        run = runs[-1] if runs else None
        joined = run is not None and _GAP.fullmatch(text[run[-1].end : token.start]) is not None
        if joined and run is not None and token.number is not None:
            first = next((t for t in run if t.number is not None), None)
            joined = first is None or _expected(first=first, steps=len(run) - run.index(first)) == (token.number, token.black)
        if joined and run is not None:
            run.append(token)
        else:
            runs.append([token])
    return [run for run in runs if len(run) > 1 or run[0].number is not None or run[0].san is None or not _LONE_SQUARE.fullmatch(run[0].san)]


def _expected(*, first: _Token, steps: int) -> tuple[int, bool]:
    if first.number is None or first.black is None:
        raise ValueError("a run's first numbered token has a number")
    half = first.number * 2 + (1 if first.black else 0) + steps
    return half // 2, bool(half % 2)


def _start(*, run: list[_Token], positions: list[chess.Board], starts: tuple[int, ...] | None) -> int | None:
    """The ply the run starts from: the start its first move's number and side name, or for an unnumbered run the first start
    where its first move is legal (the first start when none); None when nothing fits. starts None (a game review): a numbered
    run starts at the game's position at that move, an unnumbered one nowhere."""
    first = run[0]
    if first.number is None:
        if not starts:
            return None
        return next((ply for ply in starts if isinstance(_move(board=positions[ply], token=first), chess.Move)), starts[0])
    wanted = (first.number, chess.BLACK if first.black else chess.WHITE)
    candidates = range(len(positions)) if starts is None else starts
    return next((ply for ply in candidates if (positions[ply].fullmove_number, positions[ply].turn) == wanted), None)


def claim_lines(*, text: str, positions: list[chess.Board], starts: tuple[int, ...] | None) -> list[ClaimLine]:
    """positions: the board at each ply the page steps through. starts: the plies a line may start from (the task position,
    and for a move judgement also the position after the judged move), or None for a game review."""
    lines: list[ClaimLine] = []
    for run in _runs(text=text, tokens=_tokens(text=text)):
        whole = text[run[0].start : run[-1].end]
        start = _start(run=run, positions=positions, starts=starts)
        if start is None:
            lines.append(ClaimLine(playable="", rest=whole, uci=(), start_ply=None, note=NOT_FROM_TASK))
            continue
        board = positions[start].copy(stack=False)
        played: list[str] = []
        note = ""
        for token in run:
            move = _move(board=board, token=token)
            if not isinstance(move, chess.Move):
                note = move
                break
            played.append(move.uci())
            board.push(move)
        if not note:
            lines.append(ClaimLine(playable=whole, rest="", uci=tuple(played), start_ply=start, note=""))
            continue
        playable = text[run[0].start : run[len(played) - 1].end] if played else ""
        lines.append(ClaimLine(playable=playable, rest=text[run[len(played)].start : run[-1].end], uci=tuple(played), start_ply=start, note=note))
    return lines
