import json

import httpx
import pytest

from bench.grading.tool_loop import PromptedToolCalls, cap_message, run_tool_loop, tool_caller
from bench.judge_tools import TOOL_NAMES, TOOL_SPECS, GmClaim, TaskPositions, tool_specs
from pathlib import Path

from bench.llm import ClaudeCli, CodexCli, CommandResult, LLMAnswerRefused, LLMFormatError, LLMQuotaExhausted, LLMRequestRejected, LLMTransportError
from tests.llm_fakes import claude_result
from tests.grading_fakes import GR_TASK, MV_TASK
from tests.judge_tool_fakes import ScriptedAnalyser, line, toolbox
from tests.test_llm_openrouter import FakeOpenRouter
from tests.tool_loop_fakes import Clock, ScriptedToolModel, answers, asks, raises, times_out

MV_FEN = MV_TASK.fen or ""
BOARD = ("board", {"position": "task"})
EVAL = ("evaluate", {"position": "task"})


def _box() -> object:
    return toolbox(task=MV_TASK, analyser=ScriptedAnalyser(lines={MV_FEN: [line(rank=1, san="Na5", cp=-20)]}))


def test_the_judge_is_offered_every_tool_with_its_tasks_own_position_text() -> None:
    for task in (MV_TASK, GR_TASK):
        model = ScriptedToolModel(answers('{"claims": []}'))
        run_tool_loop(model=model, system="s", user="u", toolbox=toolbox(task=task), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
        assert model.specs == [tool_specs(positions=TaskPositions.for_task(task=task))] and model.offered == [list(TOOL_NAMES)]


def test_the_loop_answers_each_call_with_the_tools_result_and_records_it() -> None:
    model = ScriptedToolModel(asks(BOARD, EVAL), answers('{"claims": []}'))
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
    assert outcome.stopped == "answered" and outcome.text == '{"claims": []}' and outcome.turns == 2
    assert [(c.number, c.tool, c.arguments) for c in outcome.calls] == [(1, "board", {"position": "task"}), (2, "evaluate", {"position": "task"})]
    assert outcome.calls[1].result["eval_cp_white"] == -20
    second = model.requests[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert json.loads(str(second[4]["content"])) == outcome.calls[1].result and second[3]["tool_call_id"] == "id_2_0"
    assert (outcome.input_tokens, outcome.output_tokens, outcome.cost_usd) == (200, 60, 0.003)


def test_calls_past_the_cap_are_refused_and_a_judge_that_keeps_calling_is_stopped() -> None:
    model = ScriptedToolModel(asks(BOARD, EVAL))
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=3, timeout_s=60)  # type: ignore[arg-type]
    assert outcome.stopped == "call_cap" and outcome.text is None and len(outcome.calls) == 3 and outcome.refused_calls == 1
    refused = model.requests[2][-1]
    assert json.loads(str(refused["content"])) == {"error": cap_message(max_calls=3)}


def test_a_judge_that_answers_after_the_cap_message_is_heard() -> None:
    model = ScriptedToolModel(asks(BOARD, EVAL), asks(BOARD, EVAL), answers("done"))
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=3, timeout_s=60)  # type: ignore[arg-type]
    assert (outcome.stopped, outcome.text, len(outcome.calls), outcome.refused_calls) == ("answered", "done", 3, 1)


def test_the_loop_stops_at_its_time_limit_and_gives_each_turn_the_time_left() -> None:
    clock = Clock()
    model = ScriptedToolModel(asks(BOARD), on_turn=clock.advance(seconds=40))
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=100, clock=clock)  # type: ignore[arg-type]
    assert outcome.stopped == "timeout" and outcome.text is None and outcome.turns == 3 and "within 100s" in str(outcome.problem)
    assert model.timeouts == [100, 60, 20]


def test_a_transport_timeout_inside_a_turn_stops_the_loop_as_a_timeout() -> None:
    outcome = run_tool_loop(model=ScriptedToolModel(asks(BOARD), times_out()), system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
    assert outcome.stopped == "timeout" and len(outcome.calls) == 1


@pytest.mark.parametrize("error", [LLMAnswerRefused("tool turn truncated at max_tokens")])
def test_an_answer_level_failure_ends_the_loop_with_the_reason_recorded(error: Exception) -> None:
    outcome = run_tool_loop(model=ScriptedToolModel(asks(BOARD), raises(error=error)), system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
    assert outcome.stopped == "failed" and outcome.text is None and len(outcome.calls) == 1 and str(error) in str(outcome.problem)


@pytest.mark.parametrize("error", [LLMQuotaExhausted(message="credit exhausted", resets_at=None), LLMRequestRejected("unknown model"), LLMTransportError("gave up after 3 attempts"), LLMFormatError("a tool turn that is not a ProtocolTurn")])
def test_a_run_level_failure_stops_the_whole_run(error: Exception) -> None:
    with pytest.raises(type(error)):
        run_tool_loop(model=ScriptedToolModel(raises(error=error)), system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]


def test_arguments_that_are_not_a_json_object_of_strings_get_an_error_result() -> None:
    model = ScriptedToolModel(asks(("board", {"position": 3})), answers("x"))  # type: ignore[dict-item]
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
    assert "JSON object of strings" in str(outcome.calls[0].result["error"])


# transports


def _tool_reply(*, message: dict[str, object]) -> object:
    body = {"id": "gen-1", "provider": "Anthropic", "model": "anthropic/claude-opus-5.5-20261001", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 900, "completion_tokens": 40, "cost": 0.004}}
    return lambda request: httpx.Response(200, json=body)


def test_an_openrouter_tool_turn_offers_the_tools_and_reads_the_calls_back() -> None:
    call = {"id": "toolu_1", "type": "function", "function": {"name": "evaluate", "arguments": '{"position":"task"}'}}
    api = FakeOpenRouter(_tool_reply(message={"role": "assistant", "content": None, "tool_calls": [call], "reasoning_details": [{"type": "reasoning.text", "text": "hm"}]}))  # type: ignore[arg-type]
    messages: list[dict[str, object]] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    turn = api.model().tool_turn(messages=messages, tools=TOOL_SPECS, timeout_s=30)
    body = api.body()
    assert body["messages"] == messages and body["tools"] == TOOL_SPECS and body["tool_choice"] == "auto" and body["stream"] is False
    assert body["provider"] == {"order": ["Anthropic"], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}
    assert [(c.call_id, c.name, c.arguments) for c in turn.tool_calls] == [("toolu_1", "evaluate", '{"position":"task"}')]
    assert turn.assistant_message == {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_details": [{"type": "reasoning.text", "text": "hm"}]}
    assert (turn.input_tokens, turn.output_tokens, turn.cost_usd) == (900, 40, 0.004)
    assert (turn.provider, turn.served_model, turn.generation_id) == ("Anthropic", "anthropic/claude-opus-5.5-20261001", "gen-1")


def test_an_openrouter_tool_turn_offered_no_tools_declares_none() -> None:
    api = FakeOpenRouter(_tool_reply(message={"role": "assistant", "content": "the answer"}))  # type: ignore[arg-type]
    turn = api.model().tool_turn(messages=[{"role": "user", "content": "u"}], tools=[], timeout_s=30)
    body = api.body()
    assert "tools" not in body and "tool_choice" not in body and turn.content == "the answer" and turn.tool_calls == []


def test_an_openrouter_tool_turn_without_calls_is_the_answer() -> None:
    api = FakeOpenRouter(_tool_reply(message={"role": "assistant", "content": '{"claims": []}'}))  # type: ignore[arg-type]
    turn = api.model().tool_turn(messages=[{"role": "user", "content": "u"}], tools=TOOL_SPECS, timeout_s=30)
    assert turn.tool_calls == [] and turn.content == '{"claims": []}'
    assert tool_caller(model=api.model()) is not None


class TimedRunner:
    """Answers claude -p calls with scripted structured outputs and keeps each call's time limit."""

    def __init__(self, *, replies: list[dict[str, object] | None]) -> None:
        self.replies = replies
        self.users: list[str] = []
        self.timeouts: list[float] = []

    def __call__(self, *, argv: list[str], cwd: Path, timeout_s: float, extra_env: dict[str, str] | None = None, run_as: str | None = None) -> CommandResult:
        self.users.append(argv[argv.index("-p") + 1])
        self.timeouts.append(timeout_s)
        reply = self.replies[len(self.users) - 1]
        return CommandResult(stdout=claude_result(text="not json" if reply is None else json.dumps(reply), structured=reply), stderr="", returncode=0)


def _claude(*, runner: TimedRunner, tmp_path: Path) -> ClaudeCli:
    return ClaudeCli(model_id="claude-haiku-5-5", runner=runner, work_dir=tmp_path, timeout_s=300, effort="low")


def test_prompted_tool_calls_run_the_same_loop_over_claude_cli_within_each_turns_time(tmp_path: Path) -> None:
    runner = TimedRunner(replies=[{"tool_calls": [{"name": "evaluate", "position": "task", "moves": None, "move": None, "claim_id": None}], "final": None}, {"tool_calls": [], "final": '{"claims": []}'}])
    outcome = run_tool_loop(model=tool_caller(model=_claude(runner=runner, tmp_path=tmp_path)), system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60)  # type: ignore[arg-type]
    assert outcome.stopped == "answered" and outcome.text == '{"claims": []}' and [c.tool for c in outcome.calls] == ["evaluate"]
    assert '"eval_cp_white":-20' in runner.users[1]
    assert all(limit <= 60 for limit in runner.timeouts)


def test_prompted_tool_calls_pass_a_grandmaster_claim_id_to_gm_label(tmp_path: Path) -> None:
    box = toolbox(task=MV_TASK)
    box.gm = (GmClaim(claim_id="mj_9001_c001", text="Nxd5 is a mistake", label="true", comment=""),)
    call = {"name": "gm_label", "position": "task", "moves": None, "move": None, "claim_id": "mj_9001_c001"}
    runner = TimedRunner(replies=[{"tool_calls": [call], "final": None}, {"tool_calls": [], "final": "{}"}])
    outcome = run_tool_loop(model=tool_caller(model=_claude(runner=runner, tmp_path=tmp_path)), system="s", user="u", toolbox=box, max_calls=15, timeout_s=60)
    assert [(c.tool, c.arguments, c.result["label"]) for c in outcome.calls] == [("gm_label", {"claim_id": "mj_9001_c001"}, "true")]


def test_prompted_tool_calls_refuse_other_clients_and_malformed_turns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only on claude_cli"):
        PromptedToolCalls(chat=CodexCli(model_id="gpt-6.1-sol", runner=TimedRunner(replies=[]), work_dir=tmp_path, timeout_s=10))
    with pytest.raises(LLMFormatError):
        PromptedToolCalls(chat=_claude(runner=TimedRunner(replies=[None]), tmp_path=tmp_path)).tool_turn(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tools=TOOL_SPECS, timeout_s=10)


def test_a_wrong_final_answer_is_sent_back_once_with_the_reason() -> None:
    check = lambda text: None if text == "good" else "not the JSON asked for"  # noqa: E731
    model = ScriptedToolModel(asks(BOARD), answers("bad"), answers("good"))
    outcome = run_tool_loop(model=model, system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60, check_answer=check)  # type: ignore[arg-type]
    assert (outcome.stopped, outcome.text, outcome.turns) == ("answered", "good", 3)
    assert "not the JSON asked for" in str(model.requests[2][-1]["content"]) and model.requests[2][-2] == {"role": "assistant", "content": "bad"}
    twice = run_tool_loop(model=ScriptedToolModel(answers("bad")), system="s", user="u", toolbox=_box(), max_calls=15, timeout_s=60, check_answer=check)  # type: ignore[arg-type]
    assert (twice.text, twice.turns) == ("bad", 2)
