import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from bench.llm import CommandResult


def claude_result(*, text: str, structured: dict | None, is_error: bool = False) -> str:
    result = {
        "type": "result",
        "subtype": "error_during_execution" if is_error else "success",
        "is_error": is_error,
        "result": text,
        "duration_ms": 1200,
        "first_content_frame_ms": 400,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if structured is not None:
        result["structured_output"] = structured
    return json.dumps([{"type": "system", "subtype": "init"}, result])


@dataclass
class FakeRunner:
    responses: list[Callable[[list[str], Path], CommandResult]]
    calls: list[list[str]] = field(default_factory=list)
    run_as: list[str | None] = field(default_factory=list)

    def __call__(self, *, argv: list[str], cwd: Path, timeout_s: float, extra_env: dict[str, str] | None = None, run_as: str | None = None) -> CommandResult:
        self.calls.append(argv)
        self.run_as.append(run_as)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index](argv, cwd)


def stdout(text: str) -> Callable[[list[str], Path], CommandResult]:
    return lambda argv, cwd: CommandResult(stdout=text, stderr="", returncode=0)


def agy_tool_step(*, index: int, tool: str, parameters: dict[str, object]) -> list[dict[str, object]]:
    info = {"name": tool, "parameters": parameters}
    return [
        {"event": "step_update", "step_update": {"step_index": index, "state": "ACTIVE", "step_type": "tool", "tool_name": tool, "tool_info": info}},
        {"event": "step_update", "step_update": {"step_index": index, "state": "DONE", "step_type": "tool", "tool_name": tool, "tool_info": info}},
    ]


def agy_stream(*, response: str, steps: list[dict[str, object]] | None = None, status: str = "SUCCESS", denied: list[str] | None = None) -> str:
    result: dict[str, object] = {"status": status, "response": response, "num_turns": 1, "usage": {"input_tokens": 12672, "output_tokens": 124, "thinking_tokens": 0}}
    if denied:
        result["denied_actions"] = [{"action": action, "display_name": action} for action in denied]
    events = [
        {"event": "init", "init": {"model": "gemini-3.8-flash-high", "permission_mode": "request-review", "tools": ["run_command", "search_web", "view_file"]}},
        {"event": "step_update", "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"}},
        {"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "agent_response"}},
        *(steps or []),
        {"event": "result", "result": result},
    ]
    return "\n".join(json.dumps(event) for event in events)
