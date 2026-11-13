import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape

from bench.io_yaml import read_sheet, write_sheet
from bench.annotations import annotator_view
from bench.models import TEXT_EDITED, Sheet
from bench.reference import write_reference
from review_app.app import create_app
from tests.claim_fakes import claim, model_source, page_token
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.grading_fakes import MV_REF
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
POOLED_TEXT = "White should push g4 soon, because Black's king has few defenders and the g-file opens after gxf5"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    write_reference(reference=MV_REF, path=tmp_path / "reference" / "mj_9001.json")
    path = tmp_path / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    pooled = claim(claim_id="ap_9001_c004", text=POOLED_TEXT, weight=None, source=model_source(name="claude-opus-5-5"), kind="plan")
    write_sheet(sheet=sheet.model_copy(update={"claims": [*sheet.claims, pooled], "last_claim_number": 4}), path=path)
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    client = TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))
    login(client, "GM Test")  # the cards show the viewer's own label and comment
    return client


def _listed(html: str) -> str:
    return html[html.index('id="claim_list"') : html.index('id="claim_template"')]


def _mine(path: Path) -> Sheet:
    """The sheet as GM Test sees it: their own record (bench.annotations.annotator_view)."""
    return annotator_view(sheet=read_sheet(path=path), annotator="GM Test")


def _form(path: Path) -> dict[str, list[str] | str]:
    sheet = _mine(path)
    return {
        "claim_id": [c.claim_id for c in sheet.claims],
        "claim_text": [c.text for c in sheet.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims],
        "claim_label": [c.latest_labels["GM Test"].value if "GM Test" in c.latest_labels else "" for c in sheet.claims],
        "claim_comment": ["" for _ in sheet.claims],
        "notes": sheet.notes,
        "verdict": sheet.verdict or "",
    }


def _save(client: TestClient, form: dict[str, list[str] | str]) -> int:
    return client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code


def test_claim_text_is_shown_in_full_in_a_text_area(client: TestClient) -> None:
    listed = _listed(client.get("/sheets/ap_9001").text)
    assert f">{escape(POOLED_TEXT)}</textarea>" in listed
    assert '<input type="text" name="claim_text"' not in listed


def test_labels_and_weights_are_one_click_buttons_not_dropdowns(client: TestClient, data: Path) -> None:
    listed = _listed(client.get("/sheets/ap_9001").text)
    assert '<select name="claim_label"' not in listed and '<select name="claim_weight"' not in listed
    sheet = read_sheet(path=data / "sheets" / "ap_9001.yaml")
    assert listed.count('class="claim-card') == len(sheet.claims)
    for value in ("true", "partly_true", "false", "unclear"):
        assert f'data-label="{value}"' in listed
    for value in ("", "1", "2", "3"):
        assert f'data-weight="{value}"' in listed


def test_weight_buttons_read_as_words_from_least_to_most_with_a_tooltip(client: TestClient) -> None:
    listed = _listed(client.get("/sheets/ap_9001").text)
    group = listed[listed.index('class="seg seg-weight"') :].split("</div>", 1)[0]
    buttons = re.findall(r'data-weight="(\d?)"[^>]*title="([^"]+)">([^<]+)</button>', group)
    assert [(value, text) for value, _, text in buttons] == [("", "Not key"), ("1", "Nice to have"), ("2", "Important"), ("3", "Essential")]
    assert all(title for _, title, _ in buttons) and "can&#39;t do without" in buttons[3][1]
    assert not re.search(r">\s*[123]\s*</button>", group)
    assert '<span class="seg-caption">Importance:</span>' in group and buttons[0][1] == "true but not something a good answer must say"


def test_the_key_hint_names_the_weight_keys(client: TestClient) -> None:
    hint = client.get("/sheets/ap_9001").text.split('id="key_hint">', 1)[1].split("</span>", 1)[0]
    assert all(part in hint for part in ("e Essential", "i Important", "h Nice to have", "0 Not key"))


@pytest.mark.parametrize(("posted", "stored"), [("Essential", 3), ("Important", 2), ("nice to have", 1), ("3", 3), ("1", 1), ("Not key", None)])
def test_a_posted_weight_word_or_number_stores_its_number(client: TestClient, data: Path, posted: str, stored: int | None) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    form["claim_label"][3] = "true"  # type: ignore[index]
    form["claim_weight"][3] = posted  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    assert _mine(path).claims[3].weight == stored


def test_the_chosen_label_and_weight_are_marked_pressed(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    form["claim_label"][3] = "true"  # type: ignore[index]
    form["claim_weight"][3] = "3"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    listed = _listed(client.get("/sheets/ap_9001").text)
    card = listed[listed.index(f'data-claim-id="{page_token("ap_9001_c004")}"') :]
    assert 'data-label="true" aria-pressed="true"' in card and 'data-weight="3" aria-pressed="true"' in card
    assert 'name="claim_label" value="true"' in card and 'name="claim_weight" value="3"' in card


def test_a_comment_is_saved_with_the_label(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    form["claim_label"][3] = "partly_true"  # type: ignore[index]
    form["claim_comment"][3] = "g4 is right, but gxf5 is not forced"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    latest = _mine(path).claims[3].latest_labels.get("GM Test")
    assert latest is not None and (latest.value, latest.comment, latest.annotator) == ("partly_true", "g4 is right, but gxf5 is not forced", "GM Test")
    assert "g4 is right, but gxf5 is not forced</textarea>" in _listed(client.get("/sheets/ap_9001").text)


def test_changing_only_the_comment_keeps_the_label_and_adds_one_entry(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    form["claim_label"][3] = "false"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    form = _form(path)
    form["claim_comment"][3] = "the g-file stays closed"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    labels = _mine(path).claims[3].labels
    assert [(label.value, label.comment) for label in labels] == [("false", ""), ("false", "the g-file stays closed")]
    assert _save(client, _form_with_comment(path=path, index=3, comment="the g-file stays closed")) in (302, 303)
    assert len(_mine(path).claims[3].labels) == 2


def _form_with_comment(*, path: Path, index: int, comment: str) -> dict[str, list[str] | str]:
    form = _form(path)
    form["claim_comment"][index] = comment  # type: ignore[index]
    return form


def test_a_comment_on_an_unlabelled_model_claim_asks_for_a_label_first(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    before = path.read_text(encoding="utf-8")
    response = client.post("/sheets/ap_9001/save", data=_form_with_comment(path=path, index=3, comment="looks doubtful"))
    assert response.status_code == 422 and "A comment goes with a label: pick a label for this claim." in response.text
    assert path.read_text(encoding="utf-8") == before


def test_a_comment_on_an_analysis_or_grandmaster_claim_goes_on_the_true_label_the_card_shows(client: TestClient, data: Path) -> None:
    """The card shows true chosen by default, and the page posts that label with the comment; the server infers no label from the
    source, so a comment posted without one is refused like any other."""
    path = data / "sheets" / "ap_9001.yaml"
    before = path.read_text(encoding="utf-8")
    response = client.post("/sheets/ap_9001/save", data=_form_with_comment(path=path, index=1, comment="also Nd5 is good"))
    assert response.status_code == 422 and "pick a label for this claim" in response.text and path.read_text(encoding="utf-8") == before
    form = _form_with_comment(path=path, index=1, comment="also Nd5 is good")
    form["claim_label"][1] = "true"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    latest = _mine(path).claims[1].latest_labels.get("GM Test")
    assert latest is not None and (latest.value, latest.comment) == ("true", "also Nd5 is good")


def test_a_new_claim_keeps_its_comment(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    for key, value in (("claim_id", ""), ("claim_text", "Black's knight belongs on d4"), ("claim_weight", "1"), ("claim_label", "true"), ("claim_comment", "typical in this structure")):
        form[key].append(value)  # type: ignore[union-attr]
    assert _save(client, form) in (302, 303)
    added = _mine(path).claims[-1]
    assert [(label.value, label.comment) for label in added.labels] == [("true", "typical in this structure")]


def test_a_text_edit_with_a_comment_keeps_both_the_trace_and_the_comment_on_one_label(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    form = _form(path)
    form["claim_text"][3] = "White should push g4 soon"  # type: ignore[index]
    form["claim_label"][3] = "true"  # type: ignore[index]
    form["claim_comment"][3] = "shortened"  # type: ignore[index]
    assert _save(client, form) in (302, 303)
    labels = _mine(path).claims[3].labels
    assert [(label.value, label.comment) for label in labels] == [("true", f"{TEXT_EDITED}; shortened")]
    assert "shortened</textarea>" in _listed(client.get("/sheets/ap_9001").text)


def test_comment_lists_of_the_wrong_length_are_refused(client: TestClient, data: Path) -> None:
    form = _form(data / "sheets" / "ap_9001.yaml")
    form["claim_comment"] = [""]
    response = client.post("/sheets/ap_9001/save", data=form)
    assert response.status_code == 422 and "claim rows are inconsistent" in response.text


def test_weight_keys_set_the_stored_numbers_and_clash_with_no_other_key() -> None:
    script = (Path(__file__).parent.parent / "review_app" / "templates" / "sheet_edit.html").read_text(encoding="utf-8")
    assert 'const WEIGHT_KEYS = { "e": "3", "i": "2", "h": "1", "0": "" };' in script
    taken = {"1", "2", "3", "4", "j", "k", "s", "n", "t"}
    assert not taken & {"e", "i", "h", "0"}
