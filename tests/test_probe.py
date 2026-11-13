import pytest

from bench.grading.probe import ModelsUnavailable, assert_models_available
from bench.llm import Completion, LLMTransportError
from tests.grading_fakes import FakeJson


class Down(FakeJson):
    def complete(self, *, system: str, user: str, schema: dict[str, object] | None) -> Completion:
        raise LLMTransportError("agent (cursor) run failed: usage limit")


def test_every_model_answering_passes() -> None:
    assert_models_available(models={"judge anthropic_opus": FakeJson(payload="OK"), "extractor anthropic_haiku": FakeJson(payload="OK")})


def test_a_model_that_does_not_answer_names_itself_and_the_switch() -> None:
    with pytest.raises(ModelsUnavailable, match="extractor cursor") as caught:
        assert_models_available(models={"judge anthropic": FakeJson(payload="OK"), "extractor cursor": Down(payload=None)})
    assert "BENCH_PROFILE=claude_only" in str(caught.value) and "usage limit" in str(caught.value)


def test_the_check_reports_what_it_cost_and_probes_a_shared_client_once(capsys: pytest.CaptureFixture[str]) -> None:
    shared = FakeJson(payload="OK")
    assert assert_models_available(models={"judge a": shared, "claim judge a": shared}) == 0.0
    assert "availability check: 1 model(s) answered, 0.0 USD" in capsys.readouterr().out


def test_make_grade_probes_only_the_judge_panel_and_the_extractors_grading_uses() -> None:
    from bench.cli.grade import probed_roles

    judges = {name: FakeJson(payload="OK") for name in ("openai", "anthropic", "google")}
    extractors = {name: FakeJson(payload="OK") for name in ("openai", "anthropic", "google")}
    roles = probed_roles(judges=judges, extractors=extractors, claim_judges={"openai": judges["openai"]}, jury_size=1)  # type: ignore[arg-type]
    assert sorted(roles) == ["claim judge openai", "extractor anthropic", "extractor openai", "judge openai"]
