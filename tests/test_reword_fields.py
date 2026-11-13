"""A reworded claim's structured fields (kind, band, line, ...) were drafted from its old words: a rewording clears them to a
neutral kind unless the reviewer sets them again, so a save is never refused, nor a key graded, on a tag the text no longer says."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.annotations import annotator_view, completed_views, fields_for_wording, resolve
from bench.config import load_bench
from bench.io_yaml import read_sheet, read_task, write_sheet
from bench.models import AnnotatorRecord, ClaimFields, RecordWeight, Sheet, review_claim
from bench.reference import write_reference
from bench.validate import sheet_problems
from review_app.app import create_app
from tests.claim_fakes import claim, label, model_source
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.grading_fakes import MV_REF
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
OLD = "White is better here"
NEW = "Every other move leaves Black clearly worse or losing"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    write_reference(reference=MV_REF, path=tmp_path / "reference" / "mj_9001.json")  # the task position is -20 cp: equal
    stale = claim(claim_id="mj_9001_c001", text=OLD, weight=None, source=model_source(), kind="eval_band", band="white_better", about="task_position")
    sheet = Sheet(task_id="mj_9001", verdict=None, turning_points=[], claims=[stale], last_claim_number=1, notes="", author=None, reviewed_by=None, signed_at=None)
    write_sheet(sheet=sheet, path=tmp_path / "sheets" / "mj_9001.yaml")
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    return login(TestClient(app), "GM Test")


def _form(*, text: str, weight: str = "2", label: str = "true") -> dict[str, object]:
    return {"claim_id": ["mj_9001_c001"], "claim_text": [text], "claim_weight": [weight], "claim_weight_was": [""], "claim_label": [label], "claim_comment": [""], "notes": "", "verdict": "good"}


def _sheet(data: Path) -> Sheet:
    return read_sheet(path=data / "sheets" / "mj_9001.yaml")


def test_a_reworded_claim_with_a_stale_band_saves_and_loses_the_tag(client: TestClient, data: Path) -> None:
    response = client.post("/sheets/mj_9001/save", data=_form(text=NEW), follow_redirects=False)
    assert response.status_code == 303, response.text
    stored = _sheet(data)
    shared = stored.claims[0]
    assert (shared.text, shared.kind, shared.band) == (OLD, "eval_band", "white_better")  # the extraction stays as drafted
    mine = annotator_view(sheet=stored, annotator="GM Test").claims[0]
    assert (mine.text, mine.kind, mine.band, mine.about, mine.weight) == (NEW, "other", None, None, 2)
    client.post("/sheets/mj_9001/complete", follow_redirects=False)
    key = resolve(sheet=_sheet(data), required=1).sheet.answer_key
    assert [(c.text, c.kind, c.band) for c in key] == [(NEW, "other", None)]


def test_a_wording_saved_before_fields_were_recorded_still_saves_with_its_stale_band(client: TestClient, data: Path) -> None:
    """A record from before this change keeps the extraction's band with its wording: the reviewer cannot see that tag, so the
    contradiction with the reference is a warning for make validate, not a refusal of their save."""
    stored = _sheet(data)
    record = AnnotatorRecord(weights={"mj_9001_c001": RecordWeight(weight=2, suggested=None)}, rewords={"mj_9001_c001": NEW}, verdict="good", updated_at=datetime(2026, 11, 1, tzinfo=UTC))
    labelled = stored.claims[0].model_copy(update={"labels": [label(value="true", annotator="GM Test")]})
    write_sheet(sheet=stored.model_copy(update={"claims": [labelled], "annotations": {"GM Test": record}}), path=data / "sheets" / "mj_9001.yaml")
    view = annotator_view(sheet=_sheet(data), annotator="GM Test")
    assert (view.claims[0].kind, view.claims[0].band) == ("eval_band", "white_better")
    response = client.post("/sheets/mj_9001/save", data=_form(text=NEW) | {"notes": "saved again"}, follow_redirects=False)
    assert response.status_code == 303, response.text
    problems = sheet_problems(sheet=view, task=read_task(path=data / "tasks" / "mj_9001.yaml"), path=data / "sheets" / "mj_9001.yaml", reference=MV_REF, bands=load_bench().eval_bands)
    assert [p.severity for p in problems if "credits white_better" in p.message] == ["warning"]


def test_review_claim_clears_the_fields_of_a_new_wording_unless_they_are_given() -> None:
    at = datetime(2026, 11, 2, tzinfo=UTC)
    stale = claim(claim_id="mj_9001_c001", text=OLD, weight=None, source=model_source(), kind="eval_band", band="white_better", about="task_position")
    cleared = review_claim(claim=stale, annotator="GM Test", value="true", comment="", text=NEW, weight=None, at=at, how_to_name="name")
    assert (cleared.kind, cleared.band, cleared.about, cleared.original_text) == ("other", None, None, OLD)
    kept = review_claim(claim=stale, annotator="GM Test", value="true", comment="", text="White is better", weight=None, at=at, how_to_name="name", fields=ClaimFields(kind="eval_band", band="white_better", about="task_position"))
    assert (kept.kind, kept.band) == ("eval_band", "white_better")
    plan = claim(claim_id="mj_9001_c002", text="Black plays on the queenside", weight=None, source=model_source(), kind="plan")
    assert review_claim(claim=plan, annotator="GM Test", value="true", comment="", text="Black attacks on the queenside", weight=None, at=at, how_to_name="name").kind == "plan"
    untouched = review_claim(claim=stale, annotator="GM Test", value="true", comment="", text=OLD, weight=None, at=at, how_to_name="name")
    assert (untouched.kind, untouched.band) == ("eval_band", "white_better")
    engine = claim(claim_id="mj_9001_c003", text="The position is equal", weight=None, kind="eval_band", band="equal", about="task_position")
    with pytest.raises(ValueError, match="engine claim text is generated from its fields"):
        review_claim(claim=engine, annotator="GM Test", value=None, comment="", text=engine.text, weight=None, at=at, how_to_name="name", fields=ClaimFields(kind="other"))


def _two_reviews(*, data: Path, fields_b: ClaimFields | None) -> Sheet:
    done = datetime(2026, 11, 3, tzinfo=UTC)
    stored = _sheet(data)
    labelled = stored.claims[0].model_copy(update={"labels": [label(value="true", annotator="GM Test"), label(value="true", annotator="GM Other")]})

    def record(fields: ClaimFields | None) -> AnnotatorRecord:
        return AnnotatorRecord(weights={"mj_9001_c001": RecordWeight(weight=2, suggested=None)}, rewords={"mj_9001_c001": NEW}, fields={"mj_9001_c001": fields} if fields is not None else {}, verdict="good", updated_at=done, completed_at=done)

    return stored.model_copy(update={"claims": [labelled], "annotations": {"GM Test": record(ClaimFields(kind="other")), "GM Other": record(fields_b)}})


def test_two_reviews_with_one_wording_resolve_to_the_fields_they_agree_on(data: Path) -> None:
    agreed = resolve(sheet=_two_reviews(data=data, fields_b=ClaimFields(kind="other")), required=2)
    assert agreed.unresolved == () and [(c.text, c.kind, c.band) for c in agreed.sheet.answer_key] == [(NEW, "other", None)]
    split = resolve(sheet=_two_reviews(data=data, fields_b=None), required=2)  # the second keeps the stale band (an old record)
    assert split.unresolved == ("claim mj_9001_c001: structured fields differ",)
    assert [(c.text, c.kind, c.band) for c in split.sheet.answer_key] == [(NEW, "other", None)]


def test_an_adjudicated_wording_takes_the_fields_of_the_reviews_that_chose_it(data: Path) -> None:
    sheet = _two_reviews(data=data, fields_b=ClaimFields(kind="other"))
    views = completed_views(sheet=sheet)
    shared = sheet.claims[0]
    assert fields_for_wording(views=views, claim=shared, text=NEW) == ClaimFields(kind="other")
    assert fields_for_wording(views=views, claim=shared, text="The adjudicator's own wording") == ClaimFields(kind="other")
    assert fields_for_wording(views=views, claim=shared, text=OLD) is None  # the extraction's wording keeps its own fields


def test_a_refused_save_shows_its_reason_at_the_top_and_brings_it_into_view(client: TestClient) -> None:
    response = client.post("/sheets/mj_9001/save", data=_form(text=NEW, label=""), follow_redirects=False)
    assert response.status_code == 422
    assert 'id="error_summary"' in response.text and "pick a label for the new wording" in response.text
    assert 'getElementById("error_summary")' in response.text and "scrollIntoView" in response.text


DETAILS = ("kind", "move", "band", "line", "outcome", "mate_in", "side", "about")


def _with_details(form: dict[str, object], **chosen: str) -> dict[str, object]:
    return form | {f"claim_{name}": [chosen.get(name, "")] for name in DETAILS}


def _legacy_reword(data: Path) -> None:
    """Saved before rewording cleared the fields: the record has the wording, and the claim the extraction's band."""
    stored = _sheet(data)
    record = AnnotatorRecord(weights={"mj_9001_c001": RecordWeight(weight=2, suggested=None)}, rewords={"mj_9001_c001": NEW}, verdict="good", updated_at=datetime(2026, 11, 1, tzinfo=UTC))
    labelled = stored.claims[0].model_copy(update={"labels": [label(value="true", annotator="GM Test")]})
    write_sheet(sheet=stored.model_copy(update={"claims": [labelled], "annotations": {"GM Test": record}}), path=data / "sheets" / "mj_9001.yaml")


def test_a_reviewer_may_set_the_details_of_their_wording_like_on_an_added_claim(client: TestClient, data: Path) -> None:
    form = _with_details(_form(text="White is slightly better"), kind="eval_band", band="equal", about="task_position")
    assert client.post("/sheets/mj_9001/save", data=form, follow_redirects=False).status_code == 303
    stored = _sheet(data)
    assert stored.annotations["GM Test"].fields["mj_9001_c001"] == ClaimFields(kind="eval_band", band="equal", about="task_position")
    mine = annotator_view(sheet=stored, annotator="GM Test").claims[0]
    assert (mine.kind, mine.band, mine.about) == ("eval_band", "equal", "task_position")
    page = client.get("/sheets/mj_9001").text
    assert "details of your wording" in page and 'value="equal" selected' in page
    blank = _with_details(_form(text="White is slightly better"))  # the reviewer empties the details: the tag goes
    assert client.post("/sheets/mj_9001/save", data=blank, follow_redirects=False).status_code == 303
    assert annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0].kind == "other"
    refused = client.post("/sheets/mj_9001/save", data=_with_details(_form(text="Black is fine"), kind="eval_band"), follow_redirects=False)
    assert refused.status_code == 422 and "a eval_band claim needs about" in refused.text


def test_an_unreworded_card_ignores_the_empty_details_it_posts(client: TestClient, data: Path) -> None:
    assert client.post("/sheets/mj_9001/save", data=_with_details(_form(text=OLD, weight="")), follow_redirects=False).status_code == 303
    mine = annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0]
    assert (mine.kind, mine.band) == ("eval_band", "white_better") and "mj_9001_c001" not in _sheet(data).annotations["GM Test"].fields


def test_a_stale_tag_is_named_on_its_card_with_a_one_click_clear(client: TestClient, data: Path) -> None:
    _legacy_reword(data)
    page = client.get("/sheets/mj_9001").text
    assert "This claim is tagged as an evaluation of the task position (White better), but the engine gives -0.20 for White." in page
    assert 'class="link-button clear-tag"' in page and "(tagged as an evaluation of the task position (White better))" in page
    token = page.split('class="link-button clear-tag" data-token="', 1)[1].split('"', 1)[0]
    form = _with_details(_form(text=NEW), kind="eval_band", band="white_better", about="task_position") | {"clear_tag": token}
    assert client.post("/sheets/mj_9001/save", data=form, follow_redirects=False).status_code == 303
    mine = annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0]
    assert (mine.text, mine.kind, mine.band, mine.weight) == (NEW, "other", None, 2)
    assert "clear-tag" not in client.get("/sheets/mj_9001").text.split('id="claim_list"', 1)[1].split("<template", 1)[0]


def _complete(data: Path) -> None:
    stored = _sheet(data)
    record = stored.annotations["GM Test"].model_copy(update={"completed_at": datetime(2026, 11, 2, tzinfo=UTC)})
    write_sheet(sheet=stored.model_copy(update={"annotations": {"GM Test": record}}), path=data / "sheets" / "mj_9001.yaml")


def test_a_completed_task_names_its_stale_tag_without_a_control_that_posts_a_save(client: TestClient, data: Path) -> None:
    _legacy_reword(data)
    _complete(data)
    cards = client.get("/sheets/mj_9001").text.split('id="claim_list"', 1)[1].split("<template", 1)[0]
    assert "This claim is tagged as an evaluation of the task position (White better)" in cards
    assert "clear-tag" not in cards and "Clear the tag" not in cards
    assert "(tagged as an evaluation of the task position (White better): Reopen to change it)" in cards


def test_a_save_refused_on_a_completed_task_shows_why_and_keeps_the_draft(client: TestClient, data: Path) -> None:
    """A second tab, opened before the first marked the task complete, posts a save (or a Clear the tag): the page comes back
    locked with the reason at the top, in view, and the posted changes stay in the browser's draft for Reopen."""
    _legacy_reword(data)
    page = client.get("/sheets/mj_9001").text
    token = page.split('class="link-button clear-tag" data-token="', 1)[1].split('"', 1)[0]
    _complete(data)
    refused = client.post("/sheets/mj_9001/save", data=_form(text=NEW) | {"clear_tag": token}, follow_redirects=False)
    assert refused.status_code == 409
    assert 'id="error_summary"' in refused.text and "you marked this task complete: press Reopen to change it" in refused.text
    assert 'getElementById("error_summary")' in refused.text and "scrollIntoView" in refused.text
    assert "const unsavedPost = true;" in refused.text
    assert annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0].kind == "eval_band"


def test_the_tag_message_says_what_is_checked() -> None:
    from review_app.messages import card_notes, tag_words

    notes = card_notes(messages=["claim mj_9001_c001 credits white_better (after_move) but the reference has +0 cp for White"], numbers={"mj_9001_c001": 2})
    assert notes.cards["mj_9001_c001"] == ["This claim is tagged as an evaluation of the position after the move (White better), but the engine gives +0.00 for White. If the claim does not say that, clear the tag."]
    line = claim(claim_id="mj_9001_c002", text="Nxd5 Nxf7 wins", weight=None, source=model_source(), kind="line_outcome", line="Nxd5 Nxf7", outcome="wins_material", about="task_position")
    assert tag_words(claim=line.model_copy(update={"side": "white"})) == "the line Nxd5 Nxf7 from the task position, in which White wins material"
    assert tag_words(claim=claim(claim_id="mj_9001_c003", text="Black plans b5", weight=None, source=model_source(), kind="plan")) == ""


def test_validate_warns_of_a_reworded_key_claim_whose_kept_band_the_reference_contradicts(data: Path) -> None:
    from bench.annotations import set_completed
    from bench.validate import validate_all

    _legacy_reword(data)
    write_sheet(sheet=set_completed(sheet=_sheet(data), annotator="GM Test", at=datetime(2026, 11, 2, tzinfo=UTC)), path=data / "sheets" / "mj_9001.yaml")
    problems = validate_all(tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", bands=load_bench().eval_bands, labelling=load_bench().reference_labelling())
    reworded = [p for p in problems if "was reworded but keeps the eval_band fields" in p.message]
    assert [p.severity for p in reworded] == ["warning"] and "white_better (task_position) against -20 cp" in reworded[0].message
    cleared = _sheet(data)
    record = cleared.annotations["GM Test"].model_copy(update={"fields": {"mj_9001_c001": ClaimFields(kind="other")}})
    write_sheet(sheet=cleared.model_copy(update={"annotations": {"GM Test": record}}), path=data / "sheets" / "mj_9001.yaml")
    again = validate_all(tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", bands=load_bench().eval_bands, labelling=load_bench().reference_labelling())
    assert not [p for p in again if "was reworded" in p.message or "credits" in p.message]


def test_a_reviewer_rewording_their_own_claim_keeps_the_details_they_set() -> None:
    from bench.models import ClaimSource

    at = datetime(2026, 11, 4, tzinfo=UTC)
    mine = claim(claim_id="mj_9001_c002", text="Nxd5 is bset", weight=None, source=ClaimSource(type="gm", name="GM Test", answer_ref=None), kind="best_move", move="Nxd5", about="task_position")
    fixed = review_claim(claim=mine, annotator="GM Test", value="true", comment="", text="Nxd5 is best", weight=None, at=at, how_to_name="name")
    assert (fixed.text, fixed.kind, fixed.move, fixed.about) == ("Nxd5 is best", "best_move", "Nxd5", "task_position")
    theirs = review_claim(claim=mine, annotator="GM Other", value="true", comment="", text="Nxd5 is clearly best", weight=None, at=at, how_to_name="name")
    assert (theirs.kind, theirs.move) == ("other", None)  # another reviewer's wording of it: the details went with the old words


def test_a_typo_fix_on_an_own_added_claim_keeps_its_details_through_the_app(client: TestClient, data: Path) -> None:
    form = _form(text=OLD, weight="") | {"claim_id": ["mj_9001_c001", ""], "claim_text": [OLD, "Nxd5 is bset"], "claim_weight": ["", "2"], "claim_weight_was": ["", ""], "claim_label": ["", ""], "claim_comment": ["", ""]}
    form |= {f"claim_{name}": ["", value] for name, value in (("kind", "best_move"), ("move", "Nxd5"), ("band", ""), ("line", ""), ("outcome", ""), ("mate_in", ""), ("side", ""), ("about", "task_position"))}
    assert client.post("/sheets/mj_9001/save", data=form, follow_redirects=False).status_code == 303
    added = annotator_view(sheet=_sheet(data), annotator="GM Test").claims[1]
    fix = _form(text=OLD, weight="") | {"claim_id": ["mj_9001_c001", added.claim_id], "claim_text": [OLD, "Nxd5 is best"], "claim_weight": ["", "2"], "claim_weight_was": ["", "2"], "claim_label": ["", "true"], "claim_comment": ["", ""]}
    assert client.post("/sheets/mj_9001/save", data=fix, follow_redirects=False).status_code == 303
    again = annotator_view(sheet=_sheet(data), annotator="GM Test").claims[1]
    assert (again.text, again.kind, again.move, again.about) == ("Nxd5 is best", "best_move", "Nxd5", "task_position")


def test_a_wording_taken_back_to_the_extraction_takes_back_its_fields(client: TestClient, data: Path) -> None:
    assert client.post("/sheets/mj_9001/save", data=_form(text=NEW), follow_redirects=False).status_code == 303
    assert annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0].kind == "other"
    assert client.post("/sheets/mj_9001/save", data=_form(text=OLD, weight=""), follow_redirects=False).status_code == 303
    stored = _sheet(data)
    record = stored.annotations["GM Test"]
    assert "mj_9001_c001" not in record.rewords and "mj_9001_c001" not in record.fields
    back = annotator_view(sheet=stored, annotator="GM Test").claims[0]
    assert (back.text, back.kind, back.band, back.about) == (OLD, "eval_band", "white_better", "task_position")


def test_on_the_page_a_wording_taken_back_with_the_details_it_showed_takes_back_its_fields(client: TestClient, data: Path) -> None:
    """The page posts every card's detail controls: a reworded card shows (and posts) the neutral kind its new wording got."""
    assert client.post("/sheets/mj_9001/save", data=_with_details(_form(text=NEW)), follow_redirects=False).status_code == 303
    assert client.post("/sheets/mj_9001/save", data=_with_details(_form(text=OLD, weight=""), kind="other"), follow_redirects=False).status_code == 303
    back = annotator_view(sheet=_sheet(data), annotator="GM Test").claims[0]
    assert (back.text, back.kind, back.band) == (OLD, "eval_band", "white_better")
