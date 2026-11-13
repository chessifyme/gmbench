from bench.engine import Line, PositionAnalysis
from bench.io_yaml import read_task
from bench.llm import ChatModel, Completion
from bench.reference import Material, Reference
from tests.test_drafting import FIXTURES, _line, _position, mv_reference


class FakeAnalyser:
    def __init__(self, *, by_fen: dict[str, int], by_move: dict[tuple[str, str], int] | None = None, pv_by_fen: dict[str, list[str]] | None = None) -> None:
        self.by_fen = by_fen
        self.by_move = by_move or {}
        self.pv_by_fen = pv_by_fen or {}
        self.calls: list[str] = []
        self.move_calls: list[tuple[str, str, int]] = []

    def analyse_position(self, *, fen: str, nodes: int, multipv: int) -> PositionAnalysis:
        self.calls.append(fen)
        cp = self.by_fen[fen]
        pv = self.pv_by_fen.get(fen, [])
        return PositionAnalysis(fen=fen, version="Fake 1", nodes=nodes, multipv=multipv, lines=[Line(rank=1, move_san=pv[0] if pv else "Kf1", pv_san=pv, cp_white=cp, mate=None)])

    def analyse_move(self, *, fen: str, move_uci: str, nodes: int) -> PositionAnalysis:
        self.move_calls.append((fen, move_uci, nodes))
        cp = self.by_move[(fen, move_uci)]
        return PositionAnalysis(fen=fen, version="Fake 1", nodes=nodes, multipv=1, lines=[Line(rank=1, move_san=move_uci, pv_san=[move_uci], cp_white=cp, mate=None)])


class FakeJson(ChatModel):
    def __init__(self, *, payload: str | None, raise_if_called: bool = False) -> None:
        super().__init__(model_id="fake", vendor="anthropic", transport="claude_cli")
        self.payload = payload
        self.raise_if_called = raise_if_called
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
        if self.raise_if_called:
            raise AssertionError("the model must not be called")
        self.calls.append((system, user))
        return Completion(text=self.payload or "", model_id="fake", transport="claude_cli", latency_s=0.0, input_tokens=None, output_tokens=None, first_token_s=None)


MV_TASK = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
GR_TASK = read_task(path=FIXTURES / "tasks" / "gr_9001.yaml")
MV_REF = mv_reference()
__all__ = ["FakeAnalyser", "FakeJson", "MV_TASK", "GR_TASK", "MV_REF", "Material", "Reference", "_line", "_position"]
