import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.annotations import annotator_view
from bench.io_yaml import read_sheet, read_task
from review_app.app import create_app
from review_app.edit_locks import EditLocks
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_auth import sign_in_adjudicator
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


@pytest.fixture
def app(data: Path):
    return create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})


def _client(*, app, name: str | None) -> TestClient:
    client = TestClient(app)
    if name is not None:
        login(client, name)
    return client


def _form(*, data: Path, task_id: str, reviewer: str) -> dict[str, list[str] | str]:
    sheet = read_sheet(path=data / "sheets" / f"{task_id}.yaml")
    return {
        "claim_id": [c.claim_id for c in sheet.claims],
        "claim_text": [c.text for c in sheet.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims],
        "claim_label": [c.latest_labels[reviewer].value if reviewer in c.latest_labels else "" for c in sheet.claims],
        "notes": sheet.notes,
        "verdict": sheet.verdict or "",
        "reviewer": reviewer,
    }


def _view(*, data: Path, task_id: str, name: str):
    return annotator_view(sheet=read_sheet(path=data / "sheets" / f"{task_id}.yaml"), annotator=name)


def _sheet_text(*, data: Path, task_id: str) -> str:
    return (data / "sheets" / f"{task_id}.yaml").read_text(encoding="utf-8")


def test_a_save_without_a_name_is_refused_on_the_server(app, data: Path) -> None:
    before = _sheet_text(data=data, task_id="ap_9001")
    response = _client(app=app, name=None).post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Aram") | {"notes": "no name"})
    assert response.status_code == 422 and "access link" in response.text
    assert _sheet_text(data=data, task_id="ap_9001") == before


def test_every_save_records_who_saved_it_and_keeps_the_task_open(app, data: Path) -> None:
    response = _client(app=app, name="Aram").post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Aram") | {"notes": "first pass"}, follow_redirects=False)
    assert response.status_code == 303
    record = read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"]
    assert record.updated_at is not None and record.completed_at is None and record.notes == "first pass"
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status != "signed"


def test_mark_as_complete_signs_and_locks_the_task(app, data: Path) -> None:
    client = _client(app=app, name="Aram")
    response = client.post("/sheets/ap_9001/complete", follow_redirects=False)
    assert response.status_code == 303
    view = _view(data=data, task_id="ap_9001", name="Aram")
    assert view.author == "Aram" and view.signed_at is not None
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"
    before = _sheet_text(data=data, task_id="ap_9001")
    refused = client.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Aram") | {"notes": "late change"})
    assert refused.status_code == 409 and "Reopen" in refused.text
    assert client.post("/sheets/ap_9001/flag", data={"reason": "late"}).status_code == 409
    assert _sheet_text(data=data, task_id="ap_9001") == before
    html = client.get("/sheets/ap_9001").text
    assert 'id="reopen_button"' in html and 'id="save_button"' not in html and 'id="complete_button"' not in html
    assert "<fieldset disabled" in html


def test_save_then_complete_in_one_step(app, data: Path) -> None:
    form = _form(data=data, task_id="ap_9001", reviewer="Aram") | {"notes": "final words", "then": "complete"}
    response = _client(app=app, name="Aram").post("/sheets/ap_9001/save", data=form, follow_redirects=False)
    assert response.status_code == 303
    view = _view(data=data, task_id="ap_9001", name="Aram")
    assert view.notes == "final words" and view.author == "Aram"
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"


def test_complete_needs_a_name_and_a_sheet(app) -> None:
    assert _client(app=app, name=None).post("/sheets/ap_9001/complete").status_code == 422
    assert _client(app=app, name="Aram").post("/sheets/mj_9002/complete").status_code == 422


def test_reopen_is_per_grandmaster_and_opens_the_task_again(app, data: Path) -> None:
    aram = _client(app=app, name="Aram")
    aram.post("/sheets/ap_9001/complete")
    assert _client(app=app, name=None).post("/sheets/ap_9001/reopen").status_code == 422
    assert _client(app=app, name="Bella").post("/sheets/ap_9001/reopen").status_code == 409, "Bella has not completed it"
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"
    response = aram.post("/sheets/ap_9001/reopen", follow_redirects=False)
    assert response.status_code == 303
    record = read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"]
    assert record.completed_at is None and record.updated_at is not None
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "draft", "reopened is in progress again: its checks for finished work are warnings until it is complete"
    assert aram.post("/sheets/ap_9001/reopen").status_code == 409


def test_completion_counts_per_grandmaster_against_the_double_annotation_set(data: Path, tmp_path: Path) -> None:
    double = tmp_path / "double_annotation.yaml"
    double.write_text("required_annotators: 2\ntask_ids: [ap_9001]\n", encoding="utf-8")
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, double_annotation_file=double)
    aram, bella, lilit = (_client(app=app, name=name) for name in ("Aram", "Bella", "Lilit"))
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "draft", "one of two required reviews"
    assert "ap_9001" in _groups(bella.get("/sheets").text)["Not started"]
    assert _next(bella, "gr_9001") == "ap_9001", "the task still needs a second review"
    assert bella.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"
    assert "ap_9001" in _groups(lilit.get("/sheets").text)["Complete"]
    assert _next(lilit, "mj_9002") != "ap_9001", "a task with its required reviews is not offered again"
    assert bella.post("/sheets/ap_9001/reopen", follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "draft"


def test_a_grandmaster_cannot_reject_but_flags_with_a_reason(app, data: Path) -> None:
    aram = _client(app=app, name="Aram")
    assert aram.post("/sheets/ap_9001/reject", data={"reason": "book line"}).status_code == 404
    assert "rejected" not in read_task(path=data / "tasks" / "ap_9001.yaml").tags
    assert aram.post("/sheets/ap_9001/flag", data={"reason": "  "}).status_code == 422
    assert _client(app=app, name=None).post("/sheets/ap_9001/flag", data={"reason": "book line"}).status_code == 422
    response = aram.post("/sheets/ap_9001/flag", data={"reason": "book line"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] != "/sheets/ap_9001"
    flag = read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].flag
    assert flag is not None and flag.reason == "book line"
    task = read_task(path=data / "tasks" / "ap_9001.yaml")
    assert "rejected" not in task.tags, "a flag does not hide the task"
    assert "ap_9001" in _groups(aram.get("/sheets").text)["Flagged by you"]
    assert "ap_9001" in _groups(_client(app=app, name="Bella").get("/sheets").text)["Not started"], "another grandmaster still gets it"
    assert _next(aram, "mj_9002") != "ap_9001"


def test_only_an_adjudicator_rejects_clears_a_flag_or_takes_a_rejection_back(app, data: Path) -> None:
    aram, judge = _client(app=app, name="Aram"), _client(app=app, name="GM Judge")
    aram.post("/sheets/ap_9001/flag", data={"reason": "book line"})
    assert aram.post("/sheets/ap_9001/clear_flag", data={"annotator": "Aram"}).status_code == 404
    assert aram.post("/sheets/ap_9001/unreject").status_code == 404
    assert judge.post("/sheets/ap_9001/clear_flag", data={"annotator": "Aram"}, follow_redirects=False).status_code == 303
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].flag is None
    assert judge.post("/sheets/ap_9001/reject", data={"reason": "book line"}, follow_redirects=False).status_code == 303
    assert "rejected" in read_task(path=data / "tasks" / "ap_9001.yaml").tags
    assert "Rejected by GM Judge: book line" in read_sheet(path=data / "sheets" / "ap_9001.yaml").notes
    assert judge.post("/sheets/ap_9001/unreject", follow_redirects=False).status_code == 303
    assert "rejected" not in read_task(path=data / "tasks" / "ap_9001.yaml").tags


def test_two_grandmasters_edit_the_same_task_at_once(app, data: Path) -> None:
    aram, bella = _client(app=app, name="Aram"), _client(app=app, name="Bella")
    aram.get("/sheets/ap_9001")
    html = bella.get("/sheets/ap_9001").text
    assert "is editing this task" not in html and "<fieldset disabled" not in html and 'id="save_button"' in html
    assert bella.post("/sheets/ap_9001/lock").status_code == 204
    assert aram.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Aram") | {"notes": "Aram's"}, follow_redirects=False).status_code == 303
    assert bella.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Bella") | {"notes": "Bella's"}, follow_redirects=False).status_code == 303
    sheet = read_sheet(path=data / "sheets" / "ap_9001.yaml")
    assert sheet.annotations["Aram"].notes == "Aram's" and sheet.annotations["Bella"].notes == "Bella's"


def test_leaving_the_page_frees_the_lock(app, data: Path) -> None:
    aram = _client(app=app, name="Aram")
    aram.get("/sheets/ap_9001")
    assert aram.post("/sheets/ap_9001/unlock").status_code == 204
    assert aram.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", reviewer="Aram"), follow_redirects=False).status_code == 303


def test_a_lock_runs_out_when_it_is_not_renewed() -> None:
    locks = EditLocks(seconds=120)
    assert locks.hold(task_id="t", name="Aram", now=0) is None
    assert locks.hold(task_id="t", name="Bella", now=60) == "Aram"
    assert locks.hold(task_id="t", name="Aram", now=100) is None
    assert locks.hold(task_id="t", name="Bella", now=200) == "Aram"
    assert locks.hold(task_id="t", name="Bella", now=221) is None and locks.holder(task_id="t", now=221) == "Bella"
    locks.release(task_id="t", name="Aram")
    assert locks.holder(task_id="t", now=221) == "Bella"


def test_the_list_shows_plain_status_names(app) -> None:
    html = _client(app=app, name=None).get("/sheets").text
    assert "Not started" in html and "In progress" in html and "Complete" in html
    assert "0 / 5 complete" in html


def test_the_list_warns_about_broken_data_files(app, data: Path) -> None:
    assert 'id="data_check"' not in _client(app=app, name=None).get("/sheets").text
    (data / "sheets" / "mj_9001.yaml").write_text("task_id: mj_9001\nverdict: maybe\n", encoding="utf-8")
    html = _client(app=app, name=None).get("/sheets").text
    assert 'id="data_check"' in html and "mj_9001.yaml" in html


def _groups(html: str) -> dict[str, list[str]]:
    import re

    groups = {}
    for section in html.split("<h2>")[1:]:
        title = section.split(" <span", 1)[0]
        groups[title] = re.findall(r'<a href="/sheets/(\w+)">', section)
    return groups


def _next(client: TestClient, task_id: str) -> str | None:
    import re

    found = re.search(r'<a href="/sheets/(\w+)" id="next_link">', client.get(f"/sheets/{task_id}").text)
    return found.group(1) if found else None


def test_in_progress_is_per_reviewer(app, data: Path) -> None:
    me = _client(app=app, name="GM Test")
    for task_id in ("ap_9001", "gr_9001"):
        assert me.post(f"/sheets/{task_id}/save", data=_form(data=data, task_id=task_id, reviewer="GM Test"), follow_redirects=False).status_code == 303
    mine = _groups(me.get("/sheets").text)
    assert mine["In progress (yours)"] == ["ap_9001", "gr_9001"], "GM Test saved these two"
    assert mine["Not started"] == ["mc_9001", "mj_9001", "mj_9002"]
    other = _groups(_client(app=app, name="Aram").get("/sheets").text)
    assert other["In progress (yours)"] == [] and len(other["Not started"]) == 5


def test_next_prefers_own_work_skips_completed_flagged_and_rejected_tasks(app, data: Path) -> None:
    me, judge = _client(app=app, name="GM Test"), _client(app=app, name="GM Judge")
    for task_id in ("ap_9001", "gr_9001"):
        me.post(f"/sheets/{task_id}/save", data=_form(data=data, task_id=task_id, reviewer="GM Test"))
    assert _next(me, "gr_9001") == "ap_9001", "own tasks in progress come first"
    assert _next(me, "ap_9001") == "gr_9001"
    me.post("/sheets/gr_9001/flag", data={"reason": "odd game"})
    assert _next(me, "ap_9001") == "mc_9001", "a flagged task is skipped; then the tasks not started"
    assert judge.post("/sheets/mc_9001/reject", data={"reason": "book line"}, follow_redirects=False).status_code == 303
    assert _next(me, "ap_9001") == "mj_9001"
    assert "mc_9001" not in _groups(me.get("/sheets").text)["Not started"]
    assert "Rejected" not in _groups(me.get("/sheets?rejected=1").text), "the flag is ignored for a reviewer"
    lead = _client(app=app, name=None)
    sign_in_adjudicator(client=lead, name="Lead")
    lead.cookies.set("reviewer", "Lead")
    assert _groups(lead.get("/sheets?rejected=1").text)["Rejected"] == ["mc_9001"]
