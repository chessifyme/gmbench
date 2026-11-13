"""Live engine analysis for the review app: one engine process per analysis, streamed to the browser as server-sent events."""

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import chess
import chess.engine

EMIT_EVERY_SECONDS = 0.25
PV_PLIES_SHOWN = 16
LINES_SHOWN = 3

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveEngineSettings:
    command: list[str]
    version_expected: str
    hash_mb: int
    threads_max: int
    sessions_max: int
    seconds_max: int


@dataclass(frozen=True)
class AnalysisRequest:
    board: chess.Board
    lines: int
    threads: int
    seconds: int


def board_from_fen(*, fen: str) -> chess.Board:
    """The position the annotator has on the board; a FEN that does not parse or is not a legal position is refused."""
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"not a FEN: {error}") from None
    if not board.is_valid():
        raise ValueError(f"not a legal position: {board.status()!r}")
    return board


class EngineSlots:
    """Counts running analyses. A slot is taken before the stream starts and given back exactly once, even when the stream never runs."""

    def __init__(self, *, limit: int) -> None:
        self.limit = limit
        self.taken = 0

    def take(self) -> Callable[[], None]:
        if self.taken >= self.limit:
            raise RuntimeError(f"the server is busy: {self.limit} analyses are already running, try again shortly")
        self.taken += 1
        given_back = False

        def give_back() -> None:
            nonlocal given_back
            if not given_back:
                given_back = True
                self.taken -= 1

        return give_back


def format_score(*, score: chess.engine.PovScore) -> str:
    white = score.white()
    mate = white.mate()
    if mate is not None:
        return f"#{mate}"
    cp = white.score()
    if cp is None:
        raise ValueError(f"score {score!r} has neither mate nor centipawns")
    return "0.00" if cp == 0 else f"{cp / 100:+.2f}"


def pv_moves(*, board: chess.Board, pv: list[chess.Move]) -> list[dict[str, str]]:
    """The line as separate moves, each with the move number a reader expects before it: "35." before White's moves, "34..." before a first Black move."""
    probe = board.copy()
    moves: list[dict[str, str]] = []
    for index, move in enumerate(pv):
        if probe.turn == chess.WHITE:
            number = f"{probe.fullmove_number}."
        else:
            number = f"{probe.fullmove_number}..." if index == 0 else ""
        moves.append({"number": number, "san": probe.san(move)})
        probe.push(move)
    return moves


def _line(*, board: chess.Board, info: chess.engine.InfoDict, rank: int) -> dict[str, object]:
    pv = info.get("pv", [])[:PV_PLIES_SHOWN]
    score = info.get("score")
    return {"rank": rank, "score": format_score(score=score) if score is not None else "", "moves": pv_moves(board=board, pv=pv)}


def snapshot(*, board: chess.Board, infos: dict[int, chess.engine.InfoDict], started: float) -> dict[str, object]:
    head = infos.get(1, {})
    nps = head.get("nps")
    return {
        "depth": head.get("depth"),
        "seldepth": head.get("seldepth"),
        "nodes": head.get("nodes"),
        "mnps": round(nps / 1_000_000, 2) if nps is not None else None,
        "seconds": round(time.monotonic() - started, 1),
        "lines": [_line(board=board, info=infos[rank], rank=rank) for rank in sorted(infos)],
    }


def event(*, name: str, payload: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


async def analyse(*, settings: LiveEngineSettings, request: AnalysisRequest, disconnected: Callable[[], Awaitable[bool]]) -> AsyncIterator[str]:
    board = request.board
    if board.is_game_over():
        yield event(name="done", payload={"reason": f"no analysis: the game is over ({board.result()})"})
        return
    try:
        _, engine = await chess.engine.popen_uci(settings.command)
    except (OSError, chess.engine.EngineError) as error:
        log.error("live engine failed to start: %s", error)
        yield event(name="error", payload={"message": f"the analysis could not start ({error})"})
        return
    try:
        name = str(engine.id.get("name", ""))
        if settings.version_expected not in name:
            yield event(name="error", payload={"message": f"the analysis program reports {name!r}, expected {settings.version_expected}"})
            return
        await engine.configure({"Threads": request.threads, "Hash": settings.hash_mb})
        yield event(name="start", payload={"engine": name, "fen": board.fen(), "lines": request.lines, "threads": request.threads, "seconds": request.seconds})
        started = time.monotonic()
        last_emit = 0.0
        infos: dict[int, chess.engine.InfoDict] = {}
        with await engine.analysis(board, chess.engine.Limit(time=request.seconds), multipv=request.lines) as analysis:
            async for info in analysis:
                if "pv" not in info:
                    continue
                infos[int(info.get("multipv", 1))] = info
                now = time.monotonic()
                if now - last_emit >= EMIT_EVERY_SECONDS:
                    last_emit = now
                    yield event(name="info", payload=snapshot(board=board, infos=infos, started=started))
                    if await disconnected():
                        analysis.stop()
                        return
        if infos:
            yield event(name="info", payload=snapshot(board=board, infos=infos, started=started))
        yield event(name="done", payload={"reason": "time limit reached"})
    except chess.engine.EngineError as error:
        log.error("live engine failed during analysis: %s", error)
        yield event(name="error", payload={"message": f"the analysis failed ({error})"})
    finally:
        await engine.quit()
