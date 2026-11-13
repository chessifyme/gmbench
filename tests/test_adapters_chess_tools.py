import builtins
import json
from pathlib import Path

import chess
import pytest

from bench.adapters.base import AnswerRecord
from bench.adapters.chess_tools import ENTRANT_TOOL_SPECS, ChessToolsAdapter, chess_tools_system_prompt, entrant_tool_specs
from bench.adapters.prompts import NEUTRAL_SYSTEM_PROMPT, render_user_message
from bench.adapters.registry import get_adapter
from bench.config import load_models
from bench.grading.grade import grade_answer
from bench.grading.tool_loop import cap_message
from bench.judge_tools import BASIC_TOOLS, GAME_POSITION_TEXT, POSITION_TEXT, RESULT_REFERENCE_NOTE, TaskPositions
from bench.llm import LLMTransportError
from bench.paths import DATA_PUBLIC_REFERENCE, DATA_PUBLIC_SHEETS
from bench.aggregate import aggregate
from bench.publish_check import GateFailure, tool_trace_access
from tests.grading_fakes import GR_TASK, MV_REF, MV_TASK
from tests.judge_tool_fakes import BUDGET, CFG, ScriptedAnalyser, line
from tests.test_grade import _extractor, _extractors, _judges, _sheet
from tests.tool_loop_fakes import SERVED_MODEL, ScriptedToolModel, answers, asks, raises
from tests.run_fakes import FIXTURE_TASK_FAMILY, fake_sheets, make_run, tasks_for
from tests.verify_fakes import ScriptedChecker

MODELS = load_models()
SPEC = MODELS.entrants["claude_openrouter_tools"]
FEN = MV_TASK.fen or ""
SETTINGS = CFG.chess_tools.model_copy(update={"engine": BUDGET})
ANSWER = "5...Nxd5 is a mistake: after 6.Nxf7 Black's king is dragged into the open; 5...Na5 was better."


class ClosingModel(ScriptedToolModel):
    def __init__(self, *steps: object) -> None:
        super().__init__(*steps)  # type: ignore[arg-type]
        self.timeout_s = 600.0
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ClosingAnalyser(ScriptedAnalyser):
    closed = False

    def close(self) -> None:
        self.closed = True


def _adapter(*, model: ClosingModel, analyser: ClosingAnalyser, max_tool_calls: int = 25) -> ChessToolsAdapter:
    return ChessToolsAdapter(
        entrant_key="claude_openrouter_tools", spec=SPEC, model_factory=lambda timeout_s: model, analyser_factory=lambda: analyser,
        settings=SETTINGS.model_copy(update={"max_tool_calls": max_tool_calls}), bands=CFG.eval_bands, thresholds=CFG.labels,
    )


def _lines() -> ClosingAnalyser:
    return ClosingAnalyser(lines={FEN: [line(rank=1, san="Na5", cp=20, pv=["Na5", "Bb5+", "c6"]), line(rank=2, san="Nd4", cp=60), line(rank=3, san="Nxd5", cp=250)]})


def test_the_tool_calls_are_run_on_the_task_and_traced_with_the_answer() -> None:
    model, analyser = ClosingModel(asks(("board", {"position": "task"}), ("play", {"position": "task", "moves": "Nxd5 Nxf7"})), answers(ANSWER)), _lines()
    record = _adapter(model=model, analyser=analyser).answer(task=MV_TASK, run_index=0, timeout_s=900)
    assert isinstance(record, AnswerRecord) and record.status == "ok" and record.text == ANSWER and record.error is None
    trace = record.raw["tool_trace"]
    assert isinstance(trace, list) and [call["tool"] for call in trace] == ["board", "play"] and record.raw["tool_calls"] == 2
    assert json.loads(trace[1]["input"]) == {"moves": "Nxd5 Nxf7", "position": "task"} and trace[1]["result"]["moves"][0]["san"] == "Nxd5"
    assert trace[0]["result"]["fen"] == FEN and "squares" not in trace[0]["result"]
    assert record.raw["transport"] == "openrouter_api" and model.closed and analyser.closed
    assert model.offered == [list(BASIC_TOOLS), list(BASIC_TOOLS)]
    system, user = model.requests[0][0]["content"], model.requests[0][1]["content"]
    assert system == chess_tools_system_prompt() and system.startswith(NEUTRAL_SYSTEM_PROMPT) and user == render_user_message(task=MV_TASK)
    assert all(f"- {name}: " in system for name in BASIC_TOOLS) and "gm_" not in system and "reference analysis" not in system.lower() and "published" not in system.lower()


def test_the_entrant_is_offered_its_tasks_position_text_on_every_position_parameter_and_told_results_carry_a_reference() -> None:
    model = ClosingModel(asks(("play", {"position": "task", "moves": "Nxd5 Nxf7"})), answers(ANSWER))
    record = _adapter(model=model, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900)
    assert model.specs[0] == entrant_tool_specs(positions=TaskPositions.for_task(task=MV_TASK)) == ENTRANT_TOOL_SPECS
    for task, own in ((MV_TASK, POSITION_TEXT), (GR_TASK, GAME_POSITION_TEXT)):
        specs = entrant_tool_specs(positions=TaskPositions.for_task(task=task))
        texts = {str(spec["function"]["parameters"]["properties"]["position"]["description"]) for spec in specs}  # type: ignore[index]
        assert len(specs) == len(BASIC_TOOLS) and texts == {TaskPositions.for_task(task=task).position_text} and next(iter(texts)).startswith(own)
        assert all(str(spec["function"]["description"]).endswith(RESULT_REFERENCE_NOTE) for spec in specs)  # type: ignore[index]
    assert all(f"- {name}: " in chess_tools_system_prompt() for name in BASIC_TOOLS) and chess_tools_system_prompt().count(RESULT_REFERENCE_NOTE) == len(BASIC_TOOLS)
    end = record.raw["tool_trace"][0]["result"]["end"]  # type: ignore[index]
    assert end["reference"] == "after: 5... Nxd5 6. Nxf7"


def test_the_tools_name_no_grading_rule_and_return_no_band_or_verdict() -> None:
    described = chess_tools_system_prompt().lower() + json.dumps(ENTRANT_TOOL_SPECS).lower()
    assert not any(word in described for word in ("band", "verdict", "inaccuracy", "mistake", "blunder", "label", "grandmaster", "answer key"))
    assert [spec["function"]["name"] for spec in ENTRANT_TOOL_SPECS] == list(BASIC_TOOLS)  # type: ignore[index]
    model = ClosingModel(asks(("evaluate", {"position": "task"}), ("judge_move", {"position": "task", "move": "Nxd5"})), answers(ANSWER))
    evaluated, judged = (call["result"] for call in _adapter(model=model, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900).raw["tool_trace"])  # type: ignore[union-attr]
    assert "band" not in evaluated and "verdict" not in judged and judged["win_pct_drop"] > 0 and judged["rank"] == 3
    told = [json.loads(str(message["content"])) for message in model.requests[1] if message["role"] == "tool"]
    assert told == [evaluated, judged]


def test_every_answer_opens_its_own_engine_and_closes_it() -> None:
    opened: list[ClosingAnalyser] = []

    def fresh() -> ClosingAnalyser:
        opened.append(_lines())
        return opened[-1]

    for run_index in (0, 1):
        model = ClosingModel(asks(("evaluate", {"position": "task"})), answers(ANSWER))
        adapter = ChessToolsAdapter(
            entrant_key="claude_openrouter_tools", spec=SPEC, model_factory=lambda timeout_s, model=model: model, analyser_factory=fresh,
            settings=SETTINGS, bands=CFG.eval_bands, thresholds=CFG.labels,
        )
        assert adapter.answer(task=MV_TASK, run_index=run_index, timeout_s=900).status == "ok"
    assert len(opened) == 2 and all(analyser.closed and analyser.position_calls == [FEN] for analyser in opened)


def test_the_served_model_and_provider_of_every_turn_are_recorded() -> None:
    model = ClosingModel(asks(("board", {"position": "task"})), answers(ANSWER))
    raw = _adapter(model=model, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900).raw
    assert (raw["served_model"], raw["provider"]) == (SERVED_MODEL, "Anthropic")
    assert raw["served"] == [{"generation_id": "gen-2", "provider": "Anthropic", "served_model": SERVED_MODEL}, {"generation_id": "gen-4", "provider": "Anthropic", "served_model": SERVED_MODEL}]
    mixed = ClosingModel(asks(("board", {"position": "task"}), served_model="anthropic/claude-opus-5.5-20260901"), answers(ANSWER))
    raw = _adapter(model=mixed, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900).raw
    assert raw["served_model"] == f"anthropic/claude-opus-5.5-20260901, {SERVED_MODEL}" and raw["provider"] == "Anthropic"
    assert [turn["served_model"] for turn in raw["served"]] == ["anthropic/claude-opus-5.5-20260901", SERVED_MODEL]  # type: ignore[index,union-attr]


def test_a_transport_failure_keeps_the_calls_made_and_their_cost() -> None:
    model = ClosingModel(asks(("board", {"position": "task"})), raises(error=LLMTransportError("OpenRouter fake: 502 twice")))
    analyser = _lines()
    record = _adapter(model=model, analyser=analyser).answer(task=MV_TASK, run_index=0, timeout_s=900)
    assert record.status == "error" and str(record.error).startswith("transport:") and model.closed and analyser.closed
    assert [call["tool"] for call in record.raw["tool_trace"]] == ["board"] and record.raw["tool_calls"] == 1  # type: ignore[union-attr]
    assert record.raw["usage"] == {"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.001} and record.raw["tool_loop"]["stopped"] == "failed"  # type: ignore[index]
    assert record.raw["served_model"] == SERVED_MODEL and len(record.raw["served"]) == 1  # type: ignore[arg-type]


def test_evaluate_searches_fresh_at_the_configured_budget_and_never_reads_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    guarded = (str(DATA_PUBLIC_REFERENCE), str(DATA_PUBLIC_SHEETS))
    real_open, real_read = builtins.open, Path.read_text

    def refuse(path: object) -> None:
        if any(str(path).startswith(root) or "data/public/reference" in str(path) for root in guarded):
            raise AssertionError(f"the entrant's tools read {path}")

    monkeypatch.setattr(builtins, "open", lambda file, *a, **k: (refuse(file), real_open(file, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (refuse(self), real_read(self, *a, **k))[1])
    monkeypatch.setattr("bench.reference.read_reference", lambda **kwargs: (_ for _ in ()).throw(AssertionError("read_reference called")))
    model, analyser = ClosingModel(asks(("evaluate", {"position": "task"}), ("judge_move", {"position": "task", "move": "Nxd5"})), answers(ANSWER)), _lines()
    record = _adapter(model=model, analyser=analyser).answer(task=MV_TASK, run_index=0, timeout_s=900)
    evaluated, judged = (call["result"] for call in record.raw["tool_trace"])  # type: ignore[union-attr]
    assert MV_REF.position is not None and MV_REF.position.fen == FEN
    assert analyser.position_calls == [FEN] and evaluated["engine"] == {"source": "search", "nodes": BUDGET.nodes, "multipv": BUDGET.multipv}
    assert judged["engine"]["before"]["source"] == "search" and judged["best_move"] == "Na5"


def test_the_grandmaster_tools_and_game_position_are_not_available() -> None:
    calls = (("gm_claims", {}), ("gm_label", {"claim_id": "mj_9001_c001"}), ("game_position", {"move": "5."}))
    model = ClosingModel(asks(*calls), answers(ANSWER))
    record = _adapter(model=model, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900)
    results = [call["result"] for call in record.raw["tool_trace"]]  # type: ignore[union-attr]
    assert record.status == "ok" and all("unknown tool" in result["error"] for result in results)


def test_at_the_call_cap_the_model_is_asked_for_its_answer_in_a_turn_without_tools() -> None:
    model = ClosingModel(asks(("board", {"position": "task"}), ("play", {"position": "task", "moves": "Nxd5"}), ("board", {"position": "after: Na5"})), answers(ANSWER))
    record = _adapter(model=model, analyser=_lines(), max_tool_calls=2).answer(task=MV_TASK, run_index=0, timeout_s=900)
    assert record.status == "ok" and record.text == ANSWER and record.raw["tool_calls"] == 2
    assert model.offered == [list(BASIC_TOOLS), []]
    system, user = model.requests[1]
    assert system == {"role": "system", "content": chess_tools_system_prompt()} and user["role"] == "user" and len(model.requests[1]) == 2
    told = str(user["content"])
    assert told.startswith(render_user_message(task=MV_TASK)) and told.endswith(cap_message(max_calls=2))
    assert '1. board {"position":"task"}' in told and '2. play {"moves":"Nxd5","position":"task"}' in told and '"san":"Nxd5"' in told and "after: Na5" not in told
    assert record.raw["tool_loop"] == {"stopped": "answered", "turns": 2, "refused_calls": 1, "max_tool_calls": 2, "timeout_s": SETTINGS.timeout_s, "engine": BUDGET.model_dump()}
    stubborn = ClosingModel(asks(("board", {"position": "task"})))
    failed = _adapter(model=stubborn, analyser=_lines(), max_tool_calls=1).answer(task=MV_TASK, run_index=0, timeout_s=900)
    assert failed.status == "error" and "the entrant kept calling tools" in str(failed.error) and stubborn.offered == [list(BASIC_TOOLS), []]
    assert failed.raw["tool_loop"]["stopped"] == "call_cap" and failed.raw["usage"]["cost_usd"] == 0.002  # type: ignore[index]


def test_the_publish_gate_audits_and_accepts_the_chess_tools_traces(tmp_path: Path) -> None:
    model = ClosingModel(asks(("board", {"position": "task"}), ("evaluate", {"position": "after: Nxd5"})), answers(ANSWER))
    after = chess.Board(FEN)
    after.push_san("Nxd5")
    analyser = ClosingAnalyser(lines={FEN: [line(rank=1, san="Na5", cp=20)], after.fen(): [line(rank=1, san="Nxf7", cp=250)]})
    trace = _adapter(model=model, analyser=analyser).answer(task=MV_TASK, run_index=0, timeout_s=900).raw["tool_trace"]
    root = tmp_path / "repo"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY))
    run = make_run(root / "runs", entrant="claude_openrouter_tools", families=set(FIXTURE_TASK_FAMILY.values()), scores=dict.fromkeys(FIXTURE_TASK_FAMILY, 50.0), task_family=FIXTURE_TASK_FAMILY, sheets=sheets, tools=frozenset({"chess_tools"}), access_method="api")

    def gate(*, raw: dict[str, object]) -> list[GateFailure]:
        rows = [{**json.loads(row), "raw": raw} for row in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
        (run / "answers.jsonl").write_text("".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8")
        board = aggregate(run_dirs=[run], cfg=CFG, sheets=sheets, tasks=tasks_for(sheets=sheets), resamples=20, root=root)
        return tool_trace_access(leaderboard=board, root=root, private_dir=tmp_path / "private", home=tmp_path / "home")

    assert gate(raw={"tool_trace": trace}) == []
    assert {failure.gate for failure in gate(raw={})} == {"tool_trace_access"}


def test_an_answer_with_chess_tools_is_graded_like_any_other() -> None:
    model = ClosingModel(asks(("judge_move", {"position": "task", "move": "Nxd5"})), answers(ANSWER))
    record = _adapter(model=model, analyser=_lines()).answer(task=MV_TASK, run_index=0, timeout_s=900)
    grade = grade_answer(
        record=record, task=MV_TASK, sheet=_sheet(), reference=MV_REF, judges=dict(_judges("present")), extractors=_extractors(anthropic=_extractor()),
        claim_checker=ScriptedChecker(), cfg=CFG.model_copy(update={"jury_size": 2}), declared_family="anthropic", grader_version="test",
    )
    assert grade.status == "ok" and grade.entrant == "claude_openrouter_tools" and grade.score.coverage > 0


def test_the_registry_builds_the_chess_tools_entrants_on_the_api_with_their_card() -> None:
    for name, plain in (("claude_openrouter_tools", "claude_openrouter"), ("gemini_openrouter_tools", "gemini_openrouter")):
        adapter = get_adapter(name=name, models=MODELS, cfg=CFG)
        spec, base = MODELS.entrants[name], MODELS.entrants[plain]
        assert isinstance(adapter, ChessToolsAdapter) and (spec.model_id, spec.provider, spec.effort, spec.transport) == (base.model_id, base.provider, base.effort, "openrouter_api")
        card = adapter.card()
        assert card.tools == frozenset({"chess_tools"}) and not card.client_tools and card.access_method == "api" and card.prompt is None
