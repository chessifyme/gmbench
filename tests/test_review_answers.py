import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.answers import answers_path, answers_text
from bench.io_yaml import read_sheet, write_sheet
from review_app.app import create_app
from tests.answer_fakes import stored
from tests.claim_fakes import claim, model_source, page_token
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS

FIXTURES = Path(__file__).parent / "fixtures"
REF = "sample_coach/20261105_160110/ap_9001/0"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    sheet = read_sheet(path=tmp_path / "sheets" / "ap_9001.yaml")
    pooled = claim(claim_id="ap_9001_c004", text="White should push g4 soon", weight=None, source=model_source(name="sample_coach").model_copy(update={"answer_ref": REF}), kind="plan")
    write_sheet(sheet=sheet.model_copy(update={"claims": [*sheet.claims, pooled], "last_claim_number": 4}), path=tmp_path / "sheets" / "ap_9001.yaml")
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    return TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))


def _write(data: Path, *answers: object) -> None:
    path = answers_path(folder=data, task_id="ap_9001")
    path.parent.mkdir(exist_ok=True)
    path.write_text(answers_text(answers=list(answers)), encoding="utf-8")  # type: ignore[arg-type]


def test_no_answer_text_is_shown_even_when_answers_are_stored(client: TestClient, data: Path) -> None:
    """The answers/ store stays for the verifier and the bundles; a reviewer judges each claim on its own, never the answer."""
    _write(data, stored(answer_ref=REF, text="A distinctive answer sentence about the g4 push"), stored(answer_ref="sample_coach/20261105_160110/ap_9001/1", text="An answer no claim points to"))
    page = client.get("/sheets/ap_9001").text
    assert 'id="answers"' not in page and "answer-text" not in page
    assert "A distinctive answer sentence" not in page and "An answer no claim points to" not in page and "staging run" not in page
    row = page.split(f'data-claim-id="{page_token("ap_9001_c004")}"', 1)[1].split('class="claim-card', 1)[0]
    assert "answer 1" not in row and "from answer" not in row


def test_a_task_without_answers_shows_no_panel(client: TestClient) -> None:
    assert 'id="answers"' not in client.get("/sheets/ap_9001").text
    assert 'id="answers"' not in client.get("/sheets/mj_9001").text
