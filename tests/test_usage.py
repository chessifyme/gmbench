from pydantic import BaseModel

from bench.cli.grade import usage_report
from bench.llm import ChatModel, Completion, UsageTotals


class Answer(BaseModel):
    ok: bool


class Metered(ChatModel):
    def __init__(self, *, cost: float | None) -> None:
        super().__init__(model_id="metered", vendor="openai", transport="openrouter_api")
        self.cost = cost

    def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
        return Completion(text='{"ok": true}', model_id=self.model_id, transport=self.transport, latency_s=0.0, input_tokens=1000, output_tokens=200, first_token_s=None, cost_usd=self.cost)


class Served(Metered):
    def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
        return Completion(
            text='{"ok": true}', model_id=self.model_id, transport=self.transport, latency_s=0.0, input_tokens=1, output_tokens=1, first_token_s=None, cost_usd=self.cost,
            provider="OpenAI", served_model="openai/gpt-6-20261001",
        )


def test_every_completion_a_model_makes_is_added_to_its_usage() -> None:
    model = Metered(cost=0.01)
    model.complete_text(system="s", user="u")
    model.complete_json(system="s", user="u", schema=Answer)
    assert model.usage.snapshot() == {"calls": 2, "input_tokens": 2000, "output_tokens": 400, "cost_usd": 0.02, "calls_without_cost": 0}


def test_a_call_without_a_reported_cost_is_counted_apart() -> None:
    model = Metered(cost=None)
    model.complete_text(system="s", user="u")
    assert model.usage.snapshot() == {"calls": 1, "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.0, "calls_without_cost": 1}


def test_the_usage_since_a_snapshot_leaves_out_what_came_before() -> None:
    totals = UsageTotals()
    totals.add(calls=1, input_tokens=10, output_tokens=1, cost_usd=0.5)
    before = totals.snapshot()
    totals.add(calls=3, input_tokens=300, output_tokens=30, cost_usd=0.25)
    assert totals.since(before=before) == {"calls": 3, "input_tokens": 300, "output_tokens": 30, "cost_usd": 0.25, "calls_without_cost": 0}


def test_the_grading_usage_report_names_each_role_and_the_cost_per_answer() -> None:
    judge, extractor = Metered(cost=0.03), Metered(cost=0.01)
    before = {"judge openai": judge.usage.snapshot(), "extractor openai": extractor.usage.snapshot()}
    judge.complete_text(system="s", user="u")
    extractor.complete_text(system="s", user="u")
    report = usage_report(models={"judge openai": judge, "extractor openai": extractor}, before=before, answers=2)
    assert report.answers == 2 and report.cost_usd == 0.04 and report.cost_usd_per_answer == 0.02
    assert report.roles["judge openai"].model_id == "metered" and report.roles["judge openai"].cost_usd == 0.03


def test_the_grading_usage_report_names_the_served_model_and_provider_of_each_role() -> None:
    judge = Served(cost=0.01)
    before = {"judge openai": judge.usage.snapshot()}
    judge.complete_text(system="s", user="u")
    report = usage_report(models={"judge openai": judge}, before=before, answers=1)
    assert report.roles["judge openai"].served == ["openai/gpt-6-20261001 via OpenAI"]
    assert Metered(cost=None).usage.served_sorted() == []
