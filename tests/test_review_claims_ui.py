import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.annotations import annotator_view
from bench.io_yaml import read_sheet
from bench.models import Sheet
from bench.reference import write_reference
from review_app.app import create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.grading_fakes import MV_REF
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
DETAIL_KEYS = ("claim_kind", "claim_move", "claim_band", "claim_line", "claim_outcome", "claim_mate_in", "claim_side", "claim_about")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    write_reference(reference=MV_REF, path=tmp_path / "reference" / "mj_9001.json")
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    return login(TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})), "GM Test")


def _mine(path: Path) -> Sheet:
    """The sheet as GM Test sees it: their own record (bench.annotations.annotator_view)."""
    return annotator_view(sheet=read_sheet(path=path), annotator="GM Test")


def _form(path: Path) -> dict[str, list[str] | str]:
    sheet = _mine(path)
    rows = len(sheet.claims)
    return {
        "claim_id": [c.claim_id for c in sheet.claims],
        "claim_text": [c.text for c in sheet.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims],
        "claim_label": [c.latest_labels["GM Test"].value if "GM Test" in c.latest_labels else "" for c in sheet.claims],
        **{key: [""] * rows for key in DETAIL_KEYS},
        "notes": sheet.notes,
        "verdict": sheet.verdict or "",
    }


def _add(form: dict[str, list[str] | str], *, text: str, weight: str, **details: str) -> None:
    row = {"claim_id": "", "claim_text": text, "claim_weight": weight, "claim_label": "", **{key: details.get(key.removeprefix("claim_"), "") for key in DETAIL_KEYS}}
    for key, value in row.items():
        form[key].append(value)  # type: ignore[union-attr]


def test_the_actions_sit_in_the_title_row_and_the_sheet_scrolls_beside_the_board(client: TestClient) -> None:
    html = client.get("/sheets/mj_9001").text
    title, save, complete, board, pane, claims, add, verdict = (
        html.index(marker) for marker in ('class="title-row"', 'id="save_button"', 'id="complete_button"', 'class="board-pane"', 'class="sheet-pane"', "<h3>Claims</h3>", 'id="add_claim"', 'name="verdict"')
    )
    assert title < save < complete < board < pane < claims < add < verdict




def test_add_claim_keeps_its_structured_fields_behind_a_details_toggle(client: TestClient) -> None:
    html = client.get("/sheets/ap_9001").text
    assert 'id="claim_template"' in html and "add details" in html
    for key in DETAIL_KEYS:
        assert f'name="{key}"' in html
    for kind in ("best_move", "eval_band", "line_outcome", "mate", "material", "threat", "plan", "other"):
        assert f'value="{kind}"' in html


def test_a_move_judgement_task_shows_the_move_under_discussion(client: TestClient) -> None:
    html = client.get("/sheets/mj_9001").text
    assert "Move under discussion" in html and "5...Nxd5" in html
    assert "Move under discussion" not in client.get("/sheets/ap_9001").text


def test_a_task_without_moves_numbers_its_move_from_the_fen(client: TestClient, data: Path) -> None:
    from bench.io_yaml import read_task, write_task

    path = data / "tasks" / "mj_9001.yaml"
    task = read_task(path=path)
    write_task(task=task.model_copy(update={"pgn": None, "fen": "8/8/4k3/8/8/4K3/7p/8 b - - 0 93", "move": "Kd5", "question": "Was 93...Kd5 a good move?"}), path=path)
    html = client.get("/sheets/mj_9001").text
    assert "Move under discussion:</strong> 93...Kd5" in html
    write_task(task=task.model_copy(update={"pgn": None, "fen": "8/8/4k3/8/8/4K3/7P/8 w - - 0 57", "move": "h4", "side_to_move": "white", "question": "Was 57.h4 a good move?"}), path=path)
    assert "Move under discussion:</strong> 57.h4" in client.get("/sheets/mj_9001").text


def test_analysis_and_grandmaster_claims_offer_labels_with_true_chosen_by_default(client: TestClient, data: Path) -> None:
    html = client.get("/sheets/ap_9001").text
    listed = html[html.index('id="claim_list"') : html.index('id="claim_template"')]
    sheet = read_sheet(path=data / "sheets" / "ap_9001.yaml")
    unlabelled_default_true = sum(1 for c in sheet.claims if c.source.type in ("engine", "gm") and not c.labels)
    assert unlabelled_default_true > 0
    assert listed.count('data-label="true"') == len(sheet.claims)
    assert listed.count('data-label="true" aria-pressed="true"') == unlabelled_default_true
    assert listed.count("true by default") == unlabelled_default_true
    assert "true (auto)" not in listed


def test_a_new_claim_saves_its_structured_fields(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    _add(form, text="Be3 is the best move", weight="3", kind="best_move", move="Be3", about="task_position")
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    added = _mine(path).claims[-1]
    assert (added.text, added.weight, added.kind, added.move, added.about, added.source.type) == ("Be3 is the best move", 3, "best_move", "Be3", "task_position", "gm")


def test_a_new_plan_claim_needs_no_structured_fields(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    _add(form, text="White attacks on the kingside", weight="2", kind="plan")
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    added = _mine(path).claims[-1]
    assert (added.kind, added.move, added.about) == ("plan", None, None)


def test_a_new_checkable_claim_without_its_fields_is_refused_in_plain_words(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    before = path.read_text(encoding="utf-8")
    form = _form(path)
    _add(form, text="There is a forced mate", weight="3", kind="mate", about="task_position")
    response = client.post("/sheets/ap_9001/save", data=form)
    assert response.status_code == 422 and "mate needs mate_in" in response.text
    assert path.read_text(encoding="utf-8") == before


def test_details_sent_for_an_existing_claim_never_change_it(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    before = _mine(path).claims
    form = _form(path)
    form["claim_kind"][0] = "threat"  # type: ignore[index]
    form["claim_move"][0] = "Qh5"  # type: ignore[index]
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    after = _mine(path).claims
    assert [(c.kind, c.move) for c in after] == [(c.kind, c.move) for c in before]


def test_detail_lists_of_the_wrong_length_are_refused(client: TestClient, data: Path) -> None:
    form = _form(data / "sheets" / "ap_9001.yaml")
    form["claim_kind"] = [""]
    response = client.post("/sheets/ap_9001/save", data=form)
    assert response.status_code == 422 and "claim rows are inconsistent" in response.text


def test_the_key_hint_sits_at_the_top_of_the_claim_list_with_no_button_that_labels_the_rest(client: TestClient) -> None:
    html = client.get("/sheets/ap_9001").text
    assert html.index("<h3>Claims</h3>") < html.index('id="key_hint"') < html.index('id="claim_list"')
    assert 'id="mark_rest_true"' not in html
    assert 'data-held-true=' in html
