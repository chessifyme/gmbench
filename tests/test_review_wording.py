import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from review_app.app import create_app
from review_app.messages import card_notes
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
SHEET_WORD = re.compile(r"\bsheets?\b", re.IGNORECASE)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    client = TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))
    login(client, "Aram")
    return client


def _visible(html: str) -> str:
    """The words a reader sees: text between tags and the title, placeholder and aria-label attributes, without scripts or styles."""
    bare = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.DOTALL)
    shown = re.findall(r'(?:title|placeholder|aria-label)="([^"]*)"', bare)
    return " ".join([re.sub(r"<[^>]+>", " ", bare), *shown])


@pytest.mark.parametrize("path", ["/sheets", "/sheets?rejected=1&internal=1", "/sheets/ap_9001", "/sheets/mj_9002", "/sheets/gr_9001"])
def test_no_page_a_grandmaster_sees_says_sheet(client: TestClient, path: str) -> None:
    assert SHEET_WORD.findall(_visible(client.get(path).text)) == []


def test_the_nav_and_list_say_tasks_and_a_task_without_saved_work_says_so(client: TestClient) -> None:
    listed = client.get("/sheets").text
    assert '<a href="/sheets">Tasks</a>' in listed and "<title>Tasks" in listed
    assert "Nothing saved yet" in client.get("/sheets/mj_9002").text


def test_refusals_speak_of_the_task(client: TestClient) -> None:
    response = client.post("/sheets/mj_9002/complete", data={"reviewer": "Aram"})
    assert response.status_code == 422 and SHEET_WORD.findall(_visible(response.text)) == []


def test_check_messages_never_say_sheet() -> None:
    messages = [
        "question 'Why is Na5 a mistake?' presupposes 'mistake' but the sheet verdict is best; ask it neutrally",
        "move_judgement sheet needs a verdict",
        "task is signed but the sheet lacks author or signed_at",
        "answer a1 in run.json: no claim on the sheet points to it",
        "sheet mj_0007: answer-key claims ['x'] have the same text; weight one of them and mark the others as duplicates",
        "claim mj_0007_c001 calls Bxe6 a mistake but the sheet's turning point labels it a blunder",
    ]
    notes = card_notes(messages=messages, numbers={"mj_0007_c001": 1})
    assert [m for m in notes.summary if SHEET_WORD.search(m)] == []
    assert "Pick a verdict." in notes.summary
