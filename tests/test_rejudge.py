import json
from pathlib import Path

import pytest

from bench import GRADER_VERSION
from bench.cli.grade import grade_line
from bench.cli.rejudge import load_grade_rows, rejudge_grades, replace_grades_file
from bench.grading.grade import GradeRecord
from bench.models import Sheet, Task
from bench.runs import append_jsonl, read_jsonl
from tests.grading_fakes import MV_REF, MV_TASK
from tests.test_grade import ANSWER, AUDIENCE_CFG, CFG, _grade, _judges, _sheet


def _first(*, run_index: int = 0, verdict: str = "absent") -> GradeRecord:
    return _grade(ANSWER, judges=_judges(verdict), declared_family="undisclosed", cfg=AUDIENCE_CFG).model_copy(update={"run_index": run_index})




def test_a_missing_engine_reference_stops_rejudge_and_rescore_naming_the_task(tmp_path: Path) -> None:
    from bench.cli.rejudge import references_for

    with pytest.raises(SystemExit, match="mj_9001.*make reference"):
        references_for(grades=[_first()], reference_dir=tmp_path)




def test_current_rows_load_as_they_are_and_rows_of_another_grader_are_refused() -> None:
    current = _first().model_copy(update={"grader_version": GRADER_VERSION})
    row = current.model_dump(mode="json")
    assert load_grade_rows(rows=[row], declared_family="anthropic", grader_version=GRADER_VERSION, path=Path("grades.jsonl")) == [current]
    with pytest.raises(ValueError, match=r"grader pilot_3; only .*grades can be rescored or rejudged: grade the run again \(make grade\)"):
        load_grade_rows(rows=[row, {**row, "grader_version": "pilot_3"}], declared_family="anthropic", grader_version=GRADER_VERSION, path=Path("grades.jsonl"))




def test_rejudge_grades_keeps_the_order_and_reports_each_answer() -> None:
    grades = [_first(run_index=i) for i in range(4)]
    seen: list[GradeRecord] = []
    result = rejudge_grades(grades=grades, tasks={"mj_9001": MV_TASK}, sheets={"mj_9001": _sheet()}, judges=_judges("present"), cfg=AUDIENCE_CFG, declared_family="undisclosed", grader_version="g2", workers=3, report=seen.append, references={"mj_9001": MV_REF})
    assert [g.run_index for g in result] == [0, 1, 2, 3] and {g.grader_version for g in result} == {"g2"}
    assert sorted(g.run_index for g in seen) == [0, 1, 2, 3]
    assert all(g.score.audience_credit == 1.0 and g.score.coverage > 0 for g in result)


def test_rejudge_grades_names_a_missing_task_or_sheet() -> None:
    with pytest.raises(SystemExit, match="no sheet"):
        rejudge_grades(grades=[_first()], tasks={"mj_9001": MV_TASK}, sheets={}, judges=_judges("present"), cfg=CFG, declared_family="undisclosed", grader_version="g2", workers=1, report=lambda grade: None, references={"mj_9001": MV_REF})
    with pytest.raises(SystemExit, match="no task"):
        rejudge_grades(grades=[_first()], tasks={}, sheets={"mj_9001": _sheet()}, judges=_judges("present"), cfg=CFG, declared_family="undisclosed", grader_version="g2", workers=1, report=lambda grade: None, references={"mj_9001": MV_REF})


def test_the_old_file_is_kept_and_the_new_one_replaces_it_whole(tmp_path: Path) -> None:
    old = _first()
    append_jsonl(path=tmp_path / "grades.jsonl", record=old)
    new = [_first(verdict="present").model_copy(update={"grader_version": "g2"})]
    replace_grades_file(run_dir=tmp_path, grades=new)
    assert [row["grader_version"] for row in read_jsonl(path=tmp_path / "grades.jsonl")] == ["g2"]
    assert [row["grader_version"] for row in read_jsonl(path=tmp_path / "grades_before_rejudge.jsonl")] == ["test"]
    assert not (tmp_path / "grades.jsonl.tmp").exists()
    with pytest.raises(FileExistsError, match="grades_before_rejudge.jsonl"):
        replace_grades_file(run_dir=tmp_path, grades=new)
    assert [row["grader_version"] for row in read_jsonl(path=tmp_path / "grades.jsonl")] == ["g2"]


def test_the_progress_line_shows_the_audience_credit() -> None:
    line = grade_line(grade=_first())
    assert line.startswith("mj_9001 run 0: ") and "A 1.00" in line and "panel " in line
    uncertain = _first().model_copy(update={"score": _first().score.model_copy(update={"audience_credit": None})})
    assert "A n/a" in grade_line(grade=uncertain)


def test_the_progress_line_names_a_claim_judge_that_did_not_answer_and_why_as_a_grading_failure() -> None:
    first = _first()
    assert "claim judge" not in grade_line(grade=first)
    from tests.verify_fakes import claim_result

    omitted = claim_result(kind="plan", text="Black should castle", verdict="unverifiable", detail="the judge gave no final answer (failed): tool turn truncated").model_copy(update={"evidence_status": "omitted"})
    failed = first.model_copy(update={"claim_judge_stopped": "failed", "claims": [omitted]})
    assert grade_line(grade=failed) == "mj_9001 run 0: GRADING FAILURE, not a score: 0 of 1 claim judges answered, a verdict needs 1: claim_judge failed; the judge gave no final answer (failed): tool turn truncated"
