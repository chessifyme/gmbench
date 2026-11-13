from pathlib import Path

import pytest

from bench.cli.rescore import BEFORE_RESCORE, build_parser, rescore_grades, rescore_run
from bench.cli.rejudge import BEFORE_FILE, rejudge_run, replace_grades_file
from bench.grading.score import verdict_credit
from bench.models import Sheet, sheet_hash
from bench.runs import append_jsonl, read_jsonl
from tests.grading_fakes import MV_REF, MV_TASK
from tests.test_grade import ANSWER, CFG, Router, _grade, _judges, _sheet
from tests.test_rejudge import _first


def _no_judges() -> dict[str, Router]:
    return {key: Router(by_schema={}, raise_if_called=True) for key in _judges("absent")}


def test_rescoring_needs_no_model_and_applies_the_quote_check_to_stored_votes() -> None:
    stored = _grade(ANSWER, judges=_judges("present"), declared_family="undisclosed")
    forged = stored.model_copy(update={"item_verdicts": [v.model_copy(update={"votes": [vote.model_copy(update={"quote": "never said"}) for vote in v.votes]}) for v in stored.item_verdicts]})
    [again] = rescore_grades(grades=[forged], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": _sheet()}, judges=_no_judges(), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks=set(), drop_tasks=set(), workers=1, references={'mj_9001': MV_REF})
    assert again.grader_version == "g3" and {v.decision for v in again.item_verdicts} == {"uncertain"} and again.score.score < stored.score.score


def test_listed_tasks_are_rejudged_and_dropped_tasks_are_left_for_make_grade() -> None:
    first, second = _first(run_index=0), _first(run_index=1)
    rejudged = rescore_grades(grades=[first], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": _sheet()}, judges=_judges("present"), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks={"mj_9001"}, drop_tasks=set(), workers=1, references={'mj_9001': MV_REF})
    assert rejudged[0].score.coverage > first.score.coverage and rejudged[0].grader_version == "g3"
    assert rescore_grades(grades=[first, second], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": _sheet()}, judges=_no_judges(), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks=set(), drop_tasks={"mj_9001"}, workers=2, references={'mj_9001': MV_REF}) == []


def _rewritten(*, index: int, text: str) -> Sheet:
    sheet = _sheet()
    return sheet.model_copy(update={"claims": [point.model_copy(update={"text": text}) if i == index else point for i, point in enumerate(sheet.claims)]})


def test_a_sheet_change_outside_the_judged_texts_rescores_without_a_model() -> None:
    first = _first()
    edited = _sheet().model_copy(update={"verdict": "inaccuracy", "notes": "relabelled"})
    assert first.stated_verdict == "mistake" and first.score.verdict_credit == 1.0
    [again] = rescore_grades(grades=[first], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": edited}, judges=_no_judges(), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks=set(), drop_tasks=set(), workers=1, references={"mj_9001": MV_REF})
    assert again.sheet_hash == sheet_hash(sheet=edited) and again.item_verdicts == first.item_verdicts and again.audience_verdict == first.audience_verdict
    assert again.score.verdict_credit == verdict_credit(sheet_verdict="inaccuracy", stated="mistake") and again.score.verdict_credit < 1.0


def test_a_rewritten_text_is_still_refused_without_rejudge() -> None:
    rewritten = _rewritten(index=3, text="Black's king is stuck in the centre")
    with pytest.raises(SystemExit, match="mj_9001"):
        rescore_grades(grades=[_first()], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": rewritten}, judges=_no_judges(), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks=set(), drop_tasks=set(), workers=1, references={"mj_9001": MV_REF})


def test_the_rescore_keeps_its_own_copy_of_the_previous_grades(tmp_path: Path) -> None:
    append_jsonl(path=tmp_path / "grades.jsonl", record=_first())
    replace_grades_file(run_dir=tmp_path, grades=[_first().model_copy(update={"grader_version": "g3"})], before_name=BEFORE_RESCORE)
    assert [row["grader_version"] for row in read_jsonl(path=tmp_path / BEFORE_RESCORE)] == ["test"]


def test_a_task_listed_for_rejudging_is_judged_against_its_new_sheet_and_records_the_new_hash() -> None:
    edited = _rewritten(index=3, text="Black's king is stuck in the centre")
    first = _first()
    assert first.sheet_hash != sheet_hash(sheet=edited)
    [again] = rescore_grades(grades=[first], tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": edited}, judges=_judges("present"), cfg=CFG, declared_family="undisclosed", grader_version="g3", rejudge_tasks={"mj_9001"}, drop_tasks=set(), workers=1, references={"mj_9001": MV_REF})
    assert again.sheet_hash == sheet_hash(sheet=edited) and again.grader_version == "g3" and again.score.coverage > first.score.coverage
    kept = [index for index in range(len(first.item_verdicts)) if index != 3]
    assert [again.item_verdicts[index].model_dump_json() for index in kept] == [first.item_verdicts[index].model_dump_json() for index in kept]
    assert again.item_verdicts[3].item_text == "Black's king is stuck in the centre" and again.item_verdicts[3].decision == "present"


def test_a_rescore_takes_no_engine_flag_because_it_rechecks_only_the_stored_tool_calls(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    assert not hasattr(build_parser().parse_args(["--run", str(tmp_path)]), "engine")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--run", str(tmp_path), "--engine"])


def test_a_run_option_that_names_no_run_folder_is_refused_with_what_to_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--run", str(tmp_path / "nope")])
    assert "is not a run folder (it has no meta.json); pass the folder make run printed" in capsys.readouterr().err


def test_rescore_and_rejudge_stop_with_a_message_on_grades_of_another_grader(tmp_path: Path) -> None:
    from tests.run_fakes import FIXTURE_TASK_FAMILY, make_run

    run_dir = make_run(tmp_path / "runs", entrant="sample_coach", families=set(FIXTURE_TASK_FAMILY.values()), scores={}, task_family=FIXTURE_TASK_FAMILY, grader_version="pilot_3")
    with pytest.raises(SystemExit, match="grader pilot_3; only .*grade the run again"):
        rescore_run(run_dir=run_dir, rejudge_tasks=set(), drop_tasks=set())
    with pytest.raises(SystemExit, match="grader pilot_3; only .*grade the run again"):
        rejudge_run(run_dir=run_dir)


def test_a_rescore_or_rejudge_refuses_an_existing_backup_before_any_work(tmp_path: Path) -> None:
    (tmp_path / BEFORE_RESCORE).write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match=BEFORE_RESCORE):
        rescore_run(run_dir=tmp_path, rejudge_tasks={"mj_9001"}, drop_tasks=set())
    (tmp_path / BEFORE_FILE).write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match=BEFORE_FILE):
        rejudge_run(run_dir=tmp_path)


def test_a_card_lists_its_families_and_tools_sorted_whatever_the_set_order() -> None:
    from bench.adapters.base import AgentCard

    card = AgentCard(name="x", version="v", provider="p", declared_family="anthropic", access_method="cli", attempted_families=frozenset({"move_judgement", "move_choice", "assessment_plan", "game_review"}), tools=frozenset({"web", "analysis"}), affiliated_with_maintainer=False)
    dumped = card.model_dump(mode="json")
    assert dumped["attempted_families"] == ["assessment_plan", "game_review", "move_choice", "move_judgement"] and dumped["tools"] == ["analysis", "web"]
