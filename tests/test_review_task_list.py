from datetime import UTC, datetime
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.io_yaml import read_sheet, read_task, write_task
from review_app.app import create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login
from tests.private_origin import PRIVATE_ORIGIN

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


def _client(*, data: Path, name: str | None = "Aram", show_internal: bool = False, adjudicator: bool = False) -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, show_internal=show_internal)
    client = TestClient(app)
    if adjudicator:
        login(client, "Lead")
    elif name is not None:
        login(client, name)
    return client


def _edit(data: Path, *, task_id: str, origin: str | None = None, reject: bool = False) -> None:
    path = data / "tasks" / f"{task_id}.yaml"
    task = read_task(path=path)
    if origin is not None:
        task = task.model_copy(update={"source": task.source.model_copy(update={"origin": origin})})
    if reject:
        task = task.model_copy(update={"tags": [*task.tags, "rejected"]})
    write_task(task=task, path=path)


def _listed(html: str) -> list[str]:
    return re.findall(r'<a href="/sheets/(\w+)">', html)


def _next(client: TestClient, task_id: str) -> str | None:
    found = re.search(r'<a href="/sheets/(\w+)" id="next_link">', client.get(f"/sheets/{task_id}").text)
    return found.group(1) if found else None


def test_rejected_and_internal_tasks_are_hidden_and_only_the_unlisted_flags_show_them(data: Path) -> None:
    _edit(data, task_id="mc_9001", reject=True)
    _edit(data, task_id="mj_9002", origin=PRIVATE_ORIGIN)
    lead = _client(data=data, name="Lead", adjudicator=True)
    html = lead.get("/sheets").text
    assert "mc_9001" not in _listed(html) and "mj_9002" not in _listed(html)
    for words in ("?rejected=1", "?internal=1", "Show rejected", "Show internal", "Hide rejected", 'id="hidden_links"'):
        assert words not in html, "the flags are for the maintainers only; the list never offers them"
    assert "Hide rejected" not in lead.get("/sheets?rejected=1&internal=1").text
    assert "mc_9001" in _listed(lead.get("/sheets?rejected=1").text)
    assert "mj_9002" not in _listed(lead.get("/sheets?rejected=1").text)
    assert "mj_9002" in _listed(lead.get("/sheets?internal=1").text)
    assert lead.get("/sheets/mc_9001").status_code == 200 and lead.get("/sheets/mj_9002").status_code == 200


def test_a_reviewer_never_reaches_a_hidden_task_and_a_typed_adjudicator_name_is_not_enough(data: Path) -> None:
    _edit(data, task_id="mc_9001", reject=True)
    _edit(data, task_id="mj_9002", origin=PRIVATE_ORIGIN)
    forged = _client(data=data, name=None)
    forged.cookies.set("reviewer", "Lead")
    for client in (_client(data=data), _client(data=data, name="Mallory"), _client(data=data, name=None), forged):
        assert _listed(client.get("/sheets?rejected=1&internal=1").text) == _listed(client.get("/sheets").text)
        for task_id in ("mc_9001", "mj_9002"):
            assert client.get(f"/sheets/{task_id}").status_code == 404
            for action in ("save", "lock", "complete", "reopen", "reject"):
                assert client.post(f"/sheets/{task_id}/{action}", data={"reviewer": "Aram", "reason": "x"}).status_code == 404, action
        assert client.get(f"/sheets/{task_id}?adjudicate=1").status_code == 404
    assert _client(data=data, adjudicator=True).get("/sheets/mc_9001").status_code == 200, "an adjudicator signed in by their own link opens it"


def test_the_data_check_never_names_a_hidden_task_to_a_reviewer(data: Path) -> None:
    _edit(data, task_id="mc_9001", reject=True)
    (data / "sheets" / "mc_9001.yaml").write_text("task_id: mc_9001\n", encoding="utf-8")
    assert "mc_9001" not in _client(data=data).get("/sheets").text
    assert "mc_9001.yaml" in _client(data=data, name="Lead", adjudicator=True).get("/sheets").text


def test_the_api_docs_are_not_served(data: Path) -> None:
    client = _client(data=data)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_next_never_offers_a_hidden_task(data: Path) -> None:
    _edit(data, task_id="mj_9001", origin=PRIVATE_ORIGIN)
    _edit(data, task_id="mj_9002", reject=True)
    client = _client(data=data)
    seen = {_next(client, task_id) for task_id in ("ap_9001", "gr_9001", "mc_9001")}
    assert seen.isdisjoint({"mj_9001", "mj_9002"}) and seen == {"ap_9001", "gr_9001", "mc_9001"}
    lead = _client(data=data, name="Lead", adjudicator=True)
    assert {_next(lead, task_id) for task_id in ("mj_9001", "mj_9002")} <= {"ap_9001", "gr_9001", "mc_9001"}


def test_the_setting_shows_internal_tasks_to_everyone(data: Path) -> None:
    _edit(data, task_id="mj_9002", origin=PRIVATE_ORIGIN)
    client = _client(data=data, show_internal=True)
    html = client.get("/sheets").text
    assert "mj_9002" in _listed(html)
    assert _next(client, "mj_9001") == "mj_9002"


def test_the_list_orders_by_family_assessment_first_game_reviews_last_then_by_id(data: Path) -> None:
    assert _listed(_client(data=data).get("/sheets").text) == ["ap_9001", "mc_9001", "mj_9001", "mj_9002", "gr_9001"]


def _start(client: TestClient, data: Path, *task_ids: str) -> None:
    """Save each task once as it is shown: the reviewer's own record of it, so it is their work in progress."""
    for task_id in task_ids:
        sheet = read_sheet(path=data / "sheets" / f"{task_id}.yaml")
        form = {"claim_id": [c.claim_id for c in sheet.claims], "claim_text": [c.text for c in sheet.claims], "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims], "claim_label": [""] * len(sheet.claims), "notes": ""}
        assert client.post(f"/sheets/{task_id}/save", data=form, follow_redirects=False).status_code == 303


def test_next_follows_the_list_order_own_work_first(data: Path) -> None:
    fresh = _client(data=data)
    assert [_next(fresh, task_id) for task_id in ("ap_9001", "mc_9001", "mj_9001", "mj_9002", "gr_9001")] == ["mc_9001", "mj_9001", "mj_9002", "gr_9001", "ap_9001"]
    mine = _client(data=data, name="GM Test")
    _start(mine, data, "ap_9001", "gr_9001")
    assert _next(mine, "gr_9001") == "ap_9001" and _next(mine, "ap_9001") == "gr_9001", "own tasks in progress before those not started"


def _sections(html: str) -> dict[str, dict[str, list[str]]]:
    sections = {}
    for section in html.split("<h2")[1:]:
        title = re.sub(r"<[^>]+>", "", section.split(">", 1)[1].split("</h2>", 1)[0]).strip()
        sections[title] = {family.split("</h3>", 1)[0].split(">", 1)[1]: _listed(family) for family in section.split("<h3")[1:]}
    return sections


def test_the_list_has_your_work_then_not_started_then_complete_with_family_subheadings(data: Path) -> None:
    path = data / "tasks" / "mj_9002.yaml"
    write_task(task=read_task(path=path).model_copy(update={"status": "signed"}), path=path)
    mine = _client(data=data, name="GM Test")
    _start(mine, data, "ap_9001", "gr_9001")
    html = mine.get("/sheets").text
    assert list(_sections(html)) == ["In progress (yours) 2", "Not started 2", "Flagged by you 0", "Complete 1"]
    assert _sections(html)["In progress (yours) 2"] == {"Assessment &amp; plan": ["ap_9001"], "Game review": ["gr_9001"]}
    assert _sections(html)["Not started 2"] == {"Move choice": ["mc_9001"], "Move judgement": ["mj_9001"]}


def test_the_progress_line_counts_visible_tasks_only(data: Path) -> None:
    _edit(data, task_id="mc_9001", reject=True)
    _edit(data, task_id="mj_9002", origin=PRIVATE_ORIGIN)
    from bench.annotations import set_completed
    from bench.io_yaml import read_sheet, write_sheet

    for task_id in ("ap_9001", "mj_9002"):
        path = data / "tasks" / f"{task_id}.yaml"
        write_task(task=read_task(path=path).model_copy(update={"status": "signed"}), path=path)
    sheet_path = data / "sheets" / "ap_9001.yaml"
    write_sheet(sheet=set_completed(sheet=read_sheet(path=sheet_path), annotator="Aram", at=datetime.now(UTC)), path=sheet_path)
    html = _client(data=data).get("/sheets").text
    assert re.search(r'id="progress_line"[^>]*>1 of 3 done<', html)
    assert "1 / 3 complete" in html
    assert "0 / 3 complete" in _client(data=data, name="Bella").get("/sheets").text, "the line counts the viewer's own completions, never the task's state"
