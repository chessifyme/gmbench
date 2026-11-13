"""One short call per model before grading or drafting: a client that cannot answer stops the command before any grade
is written, naming the model and the switch to the Claude-only profile."""

from bench.config import PROFILE_ENV
from bench.llm import ChatModel, LLMTimeoutError, LLMTransportError


class ModelsUnavailable(RuntimeError):
    pass


def assert_models_available(*, models: dict[str, ChatModel]) -> float:
    """Probe each model client once and print what the probes cost, as the clients reported it; return that cost in USD."""
    failures: list[str] = []
    probed: list[ChatModel] = []
    cost = 0.0
    for key, model in models.items():
        if any(model is other for other in probed):
            continue
        before = model.usage.snapshot()
        try:
            model.complete_text(system="You answer exactly as asked.", user="Reply with the single word OK.")
        except (LLMTransportError, LLMTimeoutError) as error:
            failures.append(f"{key} ({model.model_id} through {model.transport}): {error}")
        probed.append(model)
        cost += float(model.usage.since(before=before)["cost_usd"])
    cost = round(cost, 6)
    if not failures:
        print(f"availability check: {len(probed)} model(s) answered, {cost} USD as the clients reported it", flush=True)
    if failures:
        raise ModelsUnavailable("not available right now: " + "; ".join(failures) + f". Run with {PROFILE_ENV}=claude_only to use Claude Code for every role, or retry later.")
    return cost
