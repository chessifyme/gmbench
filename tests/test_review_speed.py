"""The review app reads each task and sheet once per change, not on every page: parsed files are cached by their stat, and the
task list's data errors are checked again only when a file they come from changed."""

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from bench.io_yaml import read_sheet, write_sheet
from review_app.app import create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"


def _data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


def test_a_reader_changing_the_sheet_it_got_never_changes_what_the_next_reader_gets(tmp_path: Path) -> None:
    path = _data(tmp_path) / "sheets" / "ap_9001.yaml"
    first = read_sheet(path=path)
    text = first.claims[0].text
    first.claims[0].text = "changed in memory only"
    first.claims.clear()
    again = read_sheet(path=path)
    assert again.claims and again.claims[0].text == text


def test_a_saved_sheet_is_read_back_as_saved(tmp_path: Path) -> None:
    path = _data(tmp_path) / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    write_sheet(sheet=sheet.model_copy(update={"notes": "first save"}), path=path)
    assert read_sheet(path=path).notes == "first save"
    write_sheet(sheet=sheet.model_copy(update={"notes": "second save"}), path=path)
    assert read_sheet(path=path).notes == "second save"


def test_the_task_list_reports_a_file_broken_after_it_was_last_checked(tmp_path: Path) -> None:
    data = _data(tmp_path)
    app = create_app(token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, reviewers=TEST_REVIEWERS)
    client = TestClient(app)
    login(client, "GM Test")
    assert "does not load" not in client.get("/sheets").text
    (data / "sheets" / "ap_9001.yaml").write_text("not: [a sheet", encoding="utf-8")
    page = client.get("/sheets").text
    assert "ap_9001.yaml does not load as a task" in page, "a broken sheet is listed (as a task: grandmasters never see the word sheet), not a 500"
    assert client.get("/healthz").json()["tasks"] == len(list((data / "tasks").glob("*.yaml")))
