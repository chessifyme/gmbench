from collections.abc import Callable
from typing import Self

import chess
import chess.engine

from bench.config import LabelThresholds, load_bench
from bench.engine import Engine

Scripted = Callable[[chess.Board, int], list[tuple[str, int]] | None]
DEFAULT_CPS = (30, 20, 10)


class ScriptedUci:
    def __init__(self, *, override: Scripted | None = None) -> None:
        self.override = override
        self.calls: list[tuple[str, int, int | None]] = []
        self.id = {"name": "Stockfish 19 scripted"}
        self.quit_called = False

    def _lines(self, *, board: chess.Board, nodes: int) -> list[tuple[str, int]]:
        scripted = self.override(board, nodes) if self.override else None
        if scripted is not None:
            return scripted
        return [(move.uci(), cp) for move, cp in zip(board.legal_moves, DEFAULT_CPS, strict=False)]

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, multipv: int | None = None, game: object = None, root_moves: list[chess.Move] | None = None) -> object:
        nodes = limit.nodes or 0
        self.calls.append((board.fen(), nodes, multipv))
        entries = self._lines(board=board, nodes=nodes)
        if root_moves is not None:
            entries = [entry for entry in entries if entry[0] in {move.uci() for move in root_moves}]
        entries = entries[: multipv or 1]
        infos = [{"pv": [chess.Move.from_uci(uci)], "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)} for uci, cp in entries]
        return infos if multipv is not None else infos[0]

    def analysis(self, board: chess.Board, limit: chess.engine.Limit, multipv: int | None = None, game: object = None) -> "ScriptedAnalysis":
        """The stream of a settled search: an unfinished first report (a bound), then every line exact at one depth."""
        infos = self.analyse(board, limit, multipv=multipv or 1, game=game)
        assert isinstance(infos, list)
        first = [{**infos[0], "multipv": 1, "depth": 1, "upperbound": True}]
        settled = [{**info, "multipv": rank, "depth": 2} for rank, info in enumerate(infos, start=1)]
        return ScriptedAnalysis(infos=first + settled)

    def quit(self) -> None:
        self.quit_called = True


def engine_over(*, uci: ScriptedUci, thresholds: LabelThresholds | None = None) -> Engine:
    return Engine(engine=uci, version="Stockfish 19 scripted", thresholds=thresholds or load_bench().labels)  # type: ignore[arg-type]


class ScriptedAnalysis:
    def __init__(self, *, infos: list[dict[str, object]]) -> None:
        self.infos = infos

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.infos)
