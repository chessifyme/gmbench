from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench.adapters.base import AnswerRecord
from bench.adapters.prompts import NEUTRAL_SYSTEM_PROMPT, render_user_message
from bench.adapters.registry import OTHER_ENTRANTS, get_adapter, known_entrants
from bench.adapters.stock import StockAdapter
from bench.config import EngineContextRules, load_bench, load_models
from bench.engine import EngineContext
from bench.io_yaml import read_task
from bench.llm import ChatModel, Chunk, Completion, LLMAnswerRefused, LLMTimeoutError, LLMTransportError, ToolCall
from bench.models import Task
from tests.engine_fakes import ScriptedUci, engine_over

FIXTURES = Path(__file__).parent / "fixtures" / "tasks"
POS_TASK = read_task(path=FIXTURES / "ap_9001.yaml")


class FakeStream(ChatModel):
    def __init__(self, *, deltas: list[str], failure: Exception | None = None) -> None:
        super().__init__(model_id="fake-model", vendor="google", transport="cursor_cli")
        self.deltas = deltas
        self.failure = failure
        self.prompts: list[tuple[str, str]] = []
        self.closed = False
        self.unstamped = False

    def close(self) -> None:
        self.closed = True

    def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
        raise AssertionError("stream_text is the path under test")

    def stream_text(self, *, system: str, user: str) -> Iterator[Chunk]:
        self.prompts.append((system, user))
        if self.failure is not None:
            raise self.failure
        for delta in self.deltas:
            yield Chunk(delta=delta, at=None if self.unstamped else datetime.now(UTC))


def _adapter(model: ChatModel) -> StockAdapter:
    return StockAdapter(entrant_key="gemini_default", spec=load_models().entrants["gemini_default"], model_factory=lambda timeout_s: model)


def test_stock_adapter_records_first_token() -> None:
    model = FakeStream(deltas=["", "White ", "should ", "castle."])
    rec = _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert isinstance(rec, AnswerRecord)
    assert rec.status == "ok" and rec.error is None
    assert rec.text == "White should castle."
    assert rec.first_token_at is not None and rec.started_at <= rec.first_token_at <= rec.finished_at
    assert rec.task_id == "ap_9001" and rec.entrant == "gemini_default" and rec.run_index == 0
    assert rec.raw["model_id"] == "fake-model" and rec.raw["transport"] == "cursor_cli"
    system, user = model.prompts[0]
    assert system == "You are a chess coach. Answer the user's question about the given position or game."
    assert user.endswith(f"Question: {POS_TASK.question}")


def test_stock_adapter_timeout_yields_timeout_status() -> None:
    rec = _adapter(FakeStream(deltas=[], failure=LLMTimeoutError("cursor-agent timed out after 10s"))).answer(task=POS_TASK, run_index=1, timeout_s=10)
    assert rec.status == "timeout" and rec.text == "" and rec.first_token_at is None
    assert rec.error is not None and "timed out" in rec.error


def test_stock_adapter_timeout_keeps_the_partial_stream() -> None:
    failure = LLMTimeoutError("claude timed out after 10s", partial_stdout='{"type": "system", "subtype": "init"}\n')
    rec = _adapter(FakeStream(deltas=[], failure=failure)).answer(task=POS_TASK, run_index=1, timeout_s=10)
    assert rec.status == "timeout" and rec.raw["partial_stream"] == '{"type": "system", "subtype": "init"}\n'


def test_stock_adapter_transport_error_is_marked_for_retry() -> None:
    rec = _adapter(FakeStream(deltas=[], failure=LLMTransportError("exited with 1: not logged in"))).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "error" and rec.error is not None and rec.error.startswith("transport: ")


def test_stock_adapter_empty_answer_is_an_error() -> None:
    rec = _adapter(FakeStream(deltas=["", "  "])).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "error" and rec.error == "empty_answer"


def test_stock_adapter_card() -> None:
    card = _adapter(FakeStream(deltas=["x"])).card()
    assert card.name == "gemini_default" and card.declared_family == "google"
    assert card.access_method == "cli" and card.version == load_models().entrants["gemini_default"].model_id
    assert card.attempted_families == frozenset({"assessment_plan", "move_choice", "move_judgement", "line_explanation", "game_review"})
    assert card.tools == frozenset() and card.affiliated_with_maintainer is False


class FakeContextSource:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self._engine = engine_over(uci=ScriptedUci())

    def context_for(self, *, task: Task) -> EngineContext:
        self.tasks.append(task.task_id)
        return self._engine.engine_context_for(task=task, rules=EngineContextRules(nodes=1_000_000, multipv=3, nodes_per_ply=200_000))


def _engine_adapter(model: ChatModel, source: FakeContextSource) -> StockAdapter:
    return StockAdapter(entrant_key="gemini_engine", spec=load_models().entrants["gemini_engine"], model_factory=lambda timeout_s: model, engine_context=source)


def test_engine_entrant_gets_the_default_prompt_plus_fresh_engine_analysis() -> None:
    model = FakeStream(deltas=["Play Be3."])
    source = FakeContextSource()
    rec = _engine_adapter(model, source).answer(task=POS_TASK, run_index=0, timeout_s=10)
    system, user = model.prompts[0]
    assert system == NEUTRAL_SYSTEM_PROMPT
    assert user.startswith(render_user_message(task=POS_TASK) + "\n\n") and "Stockfish 19 scripted analysis" in user
    assert source.tasks == ["ap_9001"] and rec.status == "ok"
    assert rec.raw["engine_context"] == {"engine": "Stockfish 19 scripted", "nodes": 1_000_000, "multipv": 3}


def test_engine_entrant_card_marks_the_engine_tool() -> None:
    card = _engine_adapter(FakeStream(deltas=["x"]), FakeContextSource()).card()
    assert card.name == "gemini_engine" and card.tools == frozenset({"engine"}) and card.declared_family == "google"
    assert card.affiliated_with_maintainer is False


def test_default_entrant_prompt_has_no_engine_block_and_no_budget_in_raw() -> None:
    model = FakeStream(deltas=["x"])
    rec = _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert "analysis computed" not in model.prompts[0][1] and "engine_context" not in rec.raw


def test_engine_flag_and_context_source_must_agree() -> None:
    models = load_models()
    with pytest.raises(ValueError, match="gemini_engine"):
        StockAdapter(entrant_key="gemini_engine", spec=models.entrants["gemini_engine"], model_factory=lambda timeout_s: FakeStream(deltas=[]))
    with pytest.raises(ValueError, match="gemini_default"):
        StockAdapter(entrant_key="gemini_default", spec=models.entrants["gemini_default"], model_factory=lambda timeout_s: FakeStream(deltas=[]), engine_context=FakeContextSource())


def test_engine_entrants_pair_with_default_entrants_on_model_and_transport() -> None:
    entrants = load_models().entrants
    for vendor in ("gemini", "openai", "claude"):
        default, engine = entrants[f"{vendor}_default"], entrants[f"{vendor}_engine"]
        assert (default.family, default.model_id, default.transport) == (engine.family, engine.model_id, engine.transport)
        assert default.engine_context is False and engine.engine_context is True


def test_registry_builds_stock_engine_and_calibration_entrants_and_rejects_unknown() -> None:
    models = load_models()
    cfg = load_bench()
    for name, spec in {**models.entrants, **models.calibration_entrants}.items():
        adapter = get_adapter(name=name, models=models, cfg=cfg)
        assert adapter.name == name and adapter.declared_family == spec.family
        assert adapter.card().name == name and adapter.card().tools == frozenset({*(["engine"] if spec.engine_context else []), *(["code"] if spec.tools else []), *(["chess_tools"] if spec.chess_tools else [])})
    with pytest.raises(KeyError, match="gemini_engine"):
        get_adapter(name="nobody", models=models, cfg=cfg)
    assert {"gemini_default", "gemini_engine", "claude_engine", "openai_engine", "calibration_haiku", "calibration_grok"} <= set(known_entrants(models=models))


def test_with_tools_card_and_tool_call_count() -> None:
    models = load_models()
    adapter = get_adapter(name="claude_tools", models=models, cfg=load_bench())
    card = adapter.card()
    assert card.client_tools is True and card.tools == frozenset({"code"}) and card.label == models.entrants["claude_tools"].label and card.version == "claude-sonnet-5-5"
    assert card.effort == "medium"
    plain = _adapter(FakeStream(deltas=["x"])).card()
    assert plain.client_tools is False and plain.tools == frozenset() and plain.label == models.entrants["gemini_default"].label
    model = FakeStream(deltas=["a4"])
    model.last_completion = Completion(text="a4", model_id="fake-model", transport="cursor_cli", latency_s=1.0, input_tokens=None, output_tokens=None, first_token_s=None, tool_calls=3)
    rec = _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.raw["tool_calls"] == 3
    assert {"claude_tools", "openai_tools", "gemini_tools"} <= set(known_entrants(models=models))


def test_gemini_tools_runs_on_antigravity_with_code_tools_only() -> None:
    models = load_models()
    spec = models.entrants["gemini_tools"]
    assert (spec.family, spec.model_id, spec.transport, spec.tools, spec.engine_context) == ("google", "gemini-3.8-flash-high", "agy_cli", True, False)
    card = get_adapter(name="gemini_tools", models=models, cfg=load_bench()).card()
    assert card.client_tools is True and card.tools == frozenset({"code"}) and card.provider == "google"


def test_quota_exhaustion_is_not_recorded_as_an_answer() -> None:
    from bench.llm import LLMQuotaExhausted

    with pytest.raises(LLMQuotaExhausted):
        _adapter(FakeStream(deltas=[], failure=LLMQuotaExhausted(message="agy quota reached", resets_at=None))).answer(task=POS_TASK, run_index=0, timeout_s=10)


class _TracedStream(FakeStream):
    def tool_trace(self, *, stdout: str) -> list[ToolCall]:
        return [ToolCall(tool="Bash", input=line) for line in stdout.splitlines() if line]


def _with_tools(model: ChatModel) -> StockAdapter:
    return StockAdapter(entrant_key="claude_tools", spec=load_models().entrants["claude_tools"], model_factory=lambda timeout_s: model)


def test_with_tools_answer_records_the_tool_trace_in_raw() -> None:
    model = _TracedStream(deltas=["a4"])
    model.last_completion = Completion(text="a4", model_id="fake-model", transport="claude_cli", latency_s=1.0, input_tokens=None, output_tokens=None, first_token_s=None, tool_calls=1, tool_trace=[ToolCall(tool="Bash", input='{"command": "ls"}')])
    rec = _with_tools(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "ok" and rec.raw["tool_trace"] == [{"tool": "Bash", "input": '{"command": "ls"}'}]
    plain = FakeStream(deltas=["a4"])
    plain.last_completion = model.last_completion
    assert "tool_trace" not in _adapter(plain).answer(task=POS_TASK, run_index=0, timeout_s=10).raw


def test_a_timed_out_with_tools_answer_keeps_its_partial_trace() -> None:
    failure = LLMTimeoutError("claude timed out after 10s", partial_stdout="cat /tmp/a\npython3 /tmp/b.py\n")
    rec = _with_tools(_TracedStream(deltas=[], failure=failure)).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "timeout" and rec.raw["tool_trace"] == [{"tool": "Bash", "input": "cat /tmp/a"}, {"tool": "Bash", "input": "python3 /tmp/b.py"}]


def test_a_failed_with_tools_answer_keeps_the_trace_of_what_ran() -> None:
    model = _TracedStream(deltas=[], failure=LLMTransportError("codex exited with 1"))
    model.last_stdout = "cat /tmp/a\n"
    rec = _with_tools(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "error" and rec.raw["tool_trace"] == [{"tool": "Bash", "input": "cat /tmp/a"}]


def test_a_coach_tools_entrant_names_its_tools_and_the_players_level_without_benchmark_words() -> None:
    tools = load_bench().coach_tools
    assert tools is not None
    spec = load_models().entrants["claude_coach_tools"]
    model = FakeStream(deltas=["Play Be3."])
    adapter = StockAdapter(entrant_key="claude_coach_tools", spec=spec, model_factory=lambda timeout_s: model, coach_tools=tools)
    rec = adapter.answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "ok" and adapter.card().prompt == "coach_tools" and adapter.card().client_tools and adapter.card().drafting_only
    assert "drafting only" in (adapter.card().label or "") and "drafting_only" not in _adapter(FakeStream(deltas=["x"])).card().model_dump()
    system, user = model.prompts[0]
    assert f"`{tools.python}`" in system and f'popen_uci("{tools.stockfish}")' in system and "{" not in system.replace("{tools", "")
    assert f"The player is {'an' if POS_TASK.audience == 'advanced' else 'a'} {POS_TASK.audience} player;" in system
    assert user.startswith(f"Position (FEN): {POS_TASK.fen}\n") and user.endswith(f"Question: {POS_TASK.question}")
    assert "Audience:" not in user and "Task type:" not in user
    for word in ("benchmark", "grad", "claim extraction", "answer key"):
        assert word not in (system + user).lower()


def test_the_coach_tools_prompt_needs_tools_and_its_settings() -> None:
    spec = load_models().entrants["openai_coach_tools"]
    with pytest.raises(ValueError, match="needs coach_tools"):
        StockAdapter(entrant_key="openai_coach_tools", spec=spec, model_factory=lambda timeout_s: FakeStream(deltas=[]))
    with pytest.raises(ValueError, match="needs tools: true"):
        spec.model_validate({**spec.model_dump(), "tools": False})


def _openrouter(model: ChatModel) -> StockAdapter:
    return StockAdapter(entrant_key="claude_openrouter", spec=load_models().entrants["claude_openrouter"], model_factory=lambda timeout_s: model)


def test_an_openrouter_entrant_discloses_api_access_through_openrouter_and_its_pinned_provider() -> None:
    adapter = _openrouter(FakeStream(deltas=["x"]))
    card = adapter.card()
    assert card.access_method == "api" and card.provider == "OpenRouter, routed to Anthropic" and card.version == "anthropic/claude-opus-5.5"
    assert card.effort == "medium" and card.label == "Claude Opus 5.5 (API)" and card.declared_family == "anthropic" and card.tools == frozenset()
    assert adapter.run_details() == {"route": "openrouter", "provider": "Anthropic", "max_tokens": "16000"}
    assert _adapter(FakeStream(deltas=["x"])).run_details() is None


def test_an_answer_records_usage_cost_provider_generation_and_latency_in_raw() -> None:
    model = FakeStream(deltas=["a4"])
    model.last_completion = Completion(
        text="a4", model_id="anthropic/claude-opus-5.5", transport="openrouter_api", latency_s=12.5, input_tokens=120, output_tokens=300, first_token_s=4.0,
        tool_calls=0, reasoning_tokens=250, cost_usd=0.0123, provider="Anthropic", generation_id="gen-abc123", finish_reason="stop",
        served_model="anthropic/claude-opus-5.5-20261001",
    )
    raw = _openrouter(model).answer(task=POS_TASK, run_index=0, timeout_s=10).raw
    assert raw["usage"] == {"input_tokens": 120, "output_tokens": 300, "reasoning_tokens": 250, "cost_usd": 0.0123}
    assert raw["served_model"] == "anthropic/claude-opus-5.5-20261001"
    assert (raw["provider"], raw["generation_id"], raw["latency_s"], raw["first_token_s"], raw["finish_reason"]) == ("Anthropic", "gen-abc123", 12.5, 4.0, "stop")


def test_a_cli_answer_records_the_usage_its_client_reports() -> None:
    model = FakeStream(deltas=["a4"])
    model.last_completion = Completion(text="a4", model_id="fake-model", transport="cursor_cli", latency_s=3.0, input_tokens=10, output_tokens=5, first_token_s=None)
    raw = _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10).raw
    assert raw["usage"] == {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": None, "cost_usd": None} and raw["latency_s"] == 3.0 and raw["provider"] is None


def test_a_refused_answer_is_recorded_as_refused_so_the_runner_never_retries_it() -> None:
    rec = _adapter(FakeStream(deltas=[], failure=LLMAnswerRefused("agy used search_web outside the local command and file tools"))).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "error" and rec.error == "refused: agy used search_web outside the local command and file tools"


def test_a_client_that_does_not_say_when_its_first_token_came_leaves_it_unset() -> None:
    model = FakeStream(deltas=["White ", "castles."])
    model.unstamped = True
    rec = _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert rec.status == "ok" and rec.first_token_at is None


@pytest.mark.parametrize("failure", [None, LLMTransportError("exited with 1"), LLMTimeoutError("timed out after 10s")])
def test_every_answer_closes_its_model(failure: Exception | None) -> None:
    model = FakeStream(deltas=["a4"], failure=failure)
    _adapter(model).answer(task=POS_TASK, run_index=0, timeout_s=10)
    assert model.closed
