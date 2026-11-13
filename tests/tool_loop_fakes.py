import json
from collections.abc import Callable

from bench.llm import LLMTimeoutError, RequestedToolCall, ToolTurn

Step = Callable[[list[dict[str, object]]], ToolTurn]


SERVED_MODEL = "anthropic/claude-opus-5.5-20261001"


def asks(*calls: tuple[str, dict[str, str]], served_model: str = SERVED_MODEL, provider: str = "Anthropic") -> Step:
    def step(messages: list[dict[str, object]]) -> ToolTurn:
        requested = [RequestedToolCall(call_id=f"id_{len(messages)}_{n}", name=name, arguments=json.dumps(arguments)) for n, (name, arguments) in enumerate(calls)]
        wire = [{"id": c.call_id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in requested]
        return ToolTurn(
            content="", tool_calls=requested, assistant_message={"role": "assistant", "content": None, "tool_calls": wire}, input_tokens=100, output_tokens=10, cost_usd=0.001,
            provider=provider, served_model=served_model, generation_id=f"gen-{len(messages)}",
        )

    return step


def answers(text: str, *, served_model: str = SERVED_MODEL, provider: str = "Anthropic") -> Step:
    return lambda messages: ToolTurn(
        content=text, tool_calls=[], assistant_message={"role": "assistant", "content": text}, input_tokens=100, output_tokens=50, cost_usd=0.002,
        provider=provider, served_model=served_model, generation_id=f"gen-{len(messages)}",
    )


def times_out() -> Step:
    def step(messages: list[dict[str, object]]) -> ToolTurn:
        raise LLMTimeoutError("OpenRouter fake timed out in a tool turn")

    return step


class ScriptedToolModel:
    """A tool-calling model that plays its steps in order (the last one repeats) and keeps every request it was sent."""

    def __init__(self, *steps: Step, model_id: str = "scripted/tool-judge", on_turn: Callable[[], None] | None = None) -> None:
        self.steps = list(steps)
        self.model_id = model_id
        self.transport = "openrouter_api"
        self.requests: list[list[dict[str, object]]] = []
        self.timeouts: list[float] = []
        self.offered: list[list[str]] = []
        self.on_turn = on_turn

    def tool_turn(self, *, messages: list[dict[str, object]], tools: list[dict[str, object]], timeout_s: float) -> ToolTurn:
        self.requests.append(list(messages))
        self.offered.append([str(spec["function"]["name"]) for spec in tools])  # type: ignore[index]
        self.timeouts.append(timeout_s)
        if self.on_turn is not None:
            self.on_turn()
        step = self.steps[min(len(self.requests), len(self.steps)) - 1]
        return step(messages)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, *, seconds: float) -> Callable[[], None]:
        def tick() -> None:
            self.now += seconds

        return tick


def raises(*, error: Exception) -> Step:
    def step(messages: list[dict[str, object]]) -> ToolTurn:
        raise error

    return step
