import shutil
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from bench.io_yaml import read_sheet
from bench.models import ClaimSource, review_claim
from bench.reference import write_reference
from review_app.app import create_app
from tests.claim_fakes import LABELLED_AT, claim, label
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.grading_fakes import MV_REF
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
GM = ClaimSource(type="gm", name="GM Test", answer_ref=None)


def test_an_unlabelled_engine_or_gm_claim_is_true_by_default() -> None:
    assert claim(claim_id="ap_9001_c001", text="Be3 is best", weight=3, kind="best_move", move="Be3", about="task_position").held_true
    assert claim(claim_id="ap_9001_c002", text="White attacks", weight=2, source=GM, kind="plan").held_true


def test_a_grandmaster_label_overrides_the_default_for_engine_and_gm_claims() -> None:
    engine = claim(claim_id="ap_9001_c001", text="Be3 is best", weight=None, kind="best_move", move="Be3", about="task_position", labels=[label(value="false")])
    gm = claim(claim_id="ap_9001_c002", text="White attacks", weight=None, source=GM, kind="plan", labels=[label(value="partly_true")])
    assert not engine.held_true and not gm.held_true
    relabelled = claim(claim_id="ap_9001_c003", text="White attacks", weight=2, source=GM, kind="plan", labels=[label(value="false"), label(value="true", at=LABELLED_AT + timedelta(minutes=1))])
    assert relabelled.held_true


def test_an_engine_claim_labelled_false_cannot_keep_its_weight() -> None:
    with pytest.raises(ValidationError, match="latest label is not true"):
        claim(claim_id="ap_9001_c001", text="Be3 is best", weight=3, kind="best_move", move="Be3", about="task_position", labels=[label(value="false")])


def test_a_text_edit_takes_the_label_given_with_it_and_never_infers_one_from_the_source() -> None:
    refuted = claim(claim_id="ap_9001_c003", text="White attacks", weight=None, source=GM, kind="plan", labels=[label(value="false")])
    reworded = review_claim(claim=refuted, annotator="GM Test", value="false", comment="", text="White attacks the king", weight=None, at=LABELLED_AT + timedelta(minutes=1), how_to_name="name")
    assert reworded.latest_labels["GM Test"].value == "false" and not reworded.held_true
    with pytest.raises(ValueError, match="a new wording needs a label"):
        review_claim(claim=refuted, annotator="GM Test", value=None, comment="", text="White attacks the king", weight=None, at=LABELLED_AT, how_to_name="name")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    write_reference(reference=MV_REF, path=tmp_path / "reference" / "mj_9001.json")
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    client = TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))
    login(client, "GM Test")  # the cards show the viewer's own label
    return client


def _form(path: Path) -> dict[str, list[str] | str]:
    sheet = read_sheet(path=path)
    return {
        "claim_id": [c.claim_id for c in sheet.claims],
        "claim_text": [c.text for c in sheet.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims],
        "claim_label": [c.latest_labels["GM Test"].value if "GM Test" in c.latest_labels else "" for c in sheet.claims],
        "claim_comment": [""] * len(sheet.claims),
        "reviewer": "GM Test",
        "notes": sheet.notes,
        "verdict": sheet.verdict or "",
    }


def _add_weighted_gm_claim(form: dict[str, list[str] | str]) -> None:
    """Keeps the answer key at its minimum size when a test takes an analysis claim out of it."""
    for key, value in (("claim_id", ""), ("claim_text", "Black's king is safe on g8"), ("claim_weight", "1"), ("claim_label", ""), ("claim_comment", "")):
        form[key].append(value)  # type: ignore[union-attr]



def _engine_index(path: Path) -> int:
    return next(i for i, c in enumerate(read_sheet(path=path).claims) if c.source.type == "engine")


def test_the_app_saves_a_false_label_on_an_analysis_claim_and_drops_it_from_the_answer_key(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    index = _engine_index(path)
    target = read_sheet(path=path).claims[index].claim_id
    form = _form(path)
    form["claim_label"][index] = "false"  # type: ignore[index]
    form["claim_weight"][index] = ""  # type: ignore[index]
    _add_weighted_gm_claim(form)
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    sheet = read_sheet(path=path)
    changed = next(c for c in sheet.claims if c.claim_id == target)
    assert changed.latest_labels["GM Test"].value == "false" and not changed.held_true
    assert target not in [c.claim_id for c in sheet.answer_key]
    listed = client.get("/sheets/ap_9001").text
    card = listed[listed.index(f'data-claim-id="{target}"') :]
    assert 'data-label="false" aria-pressed="true"' in card[: card.index("</textarea>", card.index("claim_comment"))]


def test_the_app_refuses_a_false_analysis_claim_that_keeps_its_weight(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    before = path.read_text(encoding="utf-8")
    index = _engine_index(path)
    form = _form(path)
    form["claim_label"][index] = "false"  # type: ignore[index]
    response = client.post("/sheets/ap_9001/save", data=form)
    assert response.status_code == 422 and "isn&#39;t labelled True" in response.text
    assert path.read_text(encoding="utf-8") == before


def test_a_comment_on_a_refuted_analysis_claim_keeps_the_false_label(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    index = _engine_index(path)
    form = _form(path)
    form["claim_label"][index] = "false"  # type: ignore[index]
    form["claim_weight"][index] = ""  # type: ignore[index]
    _add_weighted_gm_claim(form)
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    form = _form(path)
    form["claim_comment"][index] = "Nd5 is just as good"  # type: ignore[index]
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code in (302, 303)
    latest = read_sheet(path=path).claims[index].latest_labels.get("GM Test")
    assert latest is not None and (latest.value, latest.comment) == ("false", "Nd5 is just as good")
