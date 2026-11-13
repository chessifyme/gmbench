import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bench.annotations import (
    DoubleAnnotation,
    annotator_view,
    completed_annotators,
    load_resolved_sheets,
    migrated,
    per_annotator_sheet,
    resolve,
    set_completed,
    store_view,
    unlabelled_claims,
)
from bench.io_yaml import read_sheet, write_sheet
from bench.models import Adjudication, ClaimLabel, RecordWeight, Sheet, TurningPoint, new_gm_claim
from bench.validate import errors_of, validate_all
from tests.claim_fakes import claim, model_source

FIXTURES = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _sheet(task_id: str = "ap_9001") -> Sheet:
    return read_sheet(path=FIXTURES / "sheets" / f"{task_id}.yaml")


def _label(*, view: Sheet, claim_id: str, annotator: str, value: str, at: datetime) -> Sheet:
    claims = [c.model_copy(update={"labels": [*c.labels, ClaimLabel(annotator=annotator, value=value, comment="", at=at)], "weight": None if value != "true" else c.weight}) if c.claim_id == claim_id else c for c in view.claims]
    return view.model_copy(update={"claims": claims})


def _weight(*, view: Sheet, claim_id: str, weight: int | None) -> Sheet:
    return view.model_copy(update={"claims": [c.model_copy(update={"weight": weight}) if c.claim_id == claim_id else c for c in view.claims]})


def _key(sheet: Sheet) -> dict[str, int | None]:
    return {c.claim_id: c.weight for c in sheet.claims}


def test_the_migration_turns_shared_values_into_suggestions_and_keeps_the_key() -> None:
    sheet = _sheet("gr_9001")
    after = per_annotator_sheet(sheet=sheet)
    assert all(c.weight is None for c in after.claims) and after.turning_points == [] and after.verdict is None
    assert after.suggested is not None and after.suggested.turning_points == sheet.turning_points
    assert after.suggested.weights == {c.claim_id: c.weight for c in sheet.claims if c.weight is not None}
    assert migrated(sheet=after) and per_annotator_sheet(sheet=after) == after
    assert resolve(sheet=after, required=1).sheet == resolve(sheet=sheet, required=1).sheet == sheet


def test_a_grandmaster_sees_suggested_weights_but_never_a_preselected_verdict() -> None:
    sheet = per_annotator_sheet(sheet=_sheet().model_copy(update={"verdict": "mistake"}))
    view = annotator_view(sheet=sheet, annotator="Aram")
    assert _key(view) == {"ap_9001_c001": 3, "ap_9001_c002": 2, "ap_9001_c003": 1}
    assert view.verdict is None and sheet.suggested is not None and sheet.suggested.verdict == "mistake"


def test_a_save_records_each_weight_with_the_suggestion_it_confirmed_or_changed() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    view = _weight(view=annotator_view(sheet=sheet, annotator="Aram"), claim_id="ap_9001_c003", weight=3)
    stored = store_view(sheet=sheet, annotator="Aram", view=view, at=T0, confirmed=frozenset({"ap_9001_c002"}))
    weights = stored.annotations["Aram"].weights
    assert weights["ap_9001_c002"] == RecordWeight(weight=2, suggested=2), "accepted unchanged"
    assert weights["ap_9001_c003"] == RecordWeight(weight=3, suggested=1), "changed"
    assert "ap_9001_c001" not in weights, "never touched: still a suggestion, not Aram's"
    assert annotator_view(sheet=stored, annotator="Aram").claims[0].weight == 3
    done = set_completed(sheet=stored, annotator="Aram", at=T0)
    assert done.annotations["Aram"].left_suggested == ["ap_9001_c001"]
    assert stored.suggested == sheet.suggested and all(c.weight is None for c in stored.claims)


def test_one_grandmaster_never_sees_anothers_labels_wordings_weights_verdict_notes_or_claims() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    mine = annotator_view(sheet=sheet, annotator="Aram")
    mine = _label(view=mine, claim_id="ap_9001_c002", annotator="Aram", value="false", at=T0)
    mine = mine.model_copy(update={"claims": [c.model_copy(update={"text": "Be3 is the best move here"}) if c.claim_id == "ap_9001_c001" else c for c in mine.claims], "verdict": "blunder", "notes": "Aram's private note"})
    added = new_gm_claim(task_id="ap_9001", number=4, text="Black's king is safe", annotator="Aram", weight=2, at=T0, how_to_name="log in")
    mine = mine.model_copy(update={"claims": [*mine.claims, added], "last_claim_number": 4})
    stored = store_view(sheet=sheet, annotator="Aram", view=mine, at=T0, confirmed=frozenset())
    theirs = annotator_view(sheet=stored, annotator="Lilit")
    assert theirs == annotator_view(sheet=sheet, annotator="Lilit").model_copy(update={"last_claim_number": 4})
    assert "Aram" not in theirs.model_dump_json() and "private note" not in theirs.model_dump_json() and "best move here" not in theirs.model_dump_json()
    assert [c.claim_id for c in theirs.claims] == ["ap_9001_c001", "ap_9001_c002", "ap_9001_c003"]
    again = annotator_view(sheet=stored, annotator="Aram")
    assert again.verdict == "blunder" and again.notes == "Aram's private note" and again.claims[0].text == "Be3 is the best move here"
    assert again.claims[3].weight == 2 and again.claims[1].weight is None
    assert stored.claims[0].text == _sheet().claims[0].text


def test_two_grandmasters_saving_the_same_task_keep_each_others_work() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    aram_view, lilit_view = annotator_view(sheet=sheet, annotator="Aram"), annotator_view(sheet=sheet, annotator="Lilit")
    after_aram = store_view(sheet=sheet, annotator="Aram", view=_label(view=aram_view, claim_id="ap_9001_c002", annotator="Aram", value="false", at=T0), at=T0, confirmed=frozenset())
    after_both = store_view(sheet=after_aram, annotator="Lilit", view=_weight(view=lilit_view, claim_id="ap_9001_c003", weight=3), at=T0 + timedelta(seconds=1), confirmed=frozenset())
    assert annotator_view(sheet=after_both, annotator="Aram") == annotator_view(sheet=after_aram, annotator="Aram")
    assert [label.annotator for label in after_both.claims[1].labels] == ["Aram"]
    assert after_both.annotations["Lilit"].weights["ap_9001_c003"].weight == 3


def _done(*, sheet: Sheet, annotator: str, view: Sheet, at: datetime) -> Sheet:
    return set_completed(sheet=store_view(sheet=sheet, annotator=annotator, view=view, at=at, confirmed=frozenset()), annotator=annotator, at=at)


def test_resolution_draft_single_agreed_and_adjudicated() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    assert resolve(sheet=sheet, required=1).sheet == _sheet() and not resolve(sheet=sheet, required=1).complete
    base = annotator_view(sheet=sheet, annotator="Aram")
    one = _done(sheet=sheet, annotator="Aram", view=_weight(view=base, claim_id="ap_9001_c003", weight=None), at=T0)
    single = resolve(sheet=one, required=1)
    assert single.complete and single.annotators == ("Aram",) and _key(single.sheet)["ap_9001_c003"] is None and single.sheet.author == "Aram"
    assert not resolve(sheet=one, required=2).complete
    lilit = _label(view=annotator_view(sheet=one, annotator="Lilit"), claim_id="ap_9001_c002", annotator="Lilit", value="false", at=T0)
    two = _done(sheet=one, annotator="Lilit", view=lilit, at=T0 + timedelta(minutes=1))
    agreed = resolve(sheet=two, required=2)
    assert agreed.complete and completed_annotators(sheet=two) == ["Aram", "Lilit"]
    assert _key(agreed.sheet) == {"ap_9001_c001": 3, "ap_9001_c002": None, "ap_9001_c003": None}
    assert set(agreed.unresolved) == {"claim ap_9001_c002: weights differ", "claim ap_9001_c003: weights differ"}
    ruling = ClaimLabel(annotator="Judge", value="true", comment="adjudicated", at=T0 + timedelta(hours=1), role="adjudicator")
    ruled_claims = [c.model_copy(update={"labels": [*c.labels, ruling]}) if c.claim_id == "ap_9001_c002" else c for c in two.claims]
    adjudication = Adjudication(adjudicator="Judge", weights={"ap_9001_c001": RecordWeight(weight=3, suggested=3), "ap_9001_c002": RecordWeight(weight=2, suggested=2)}, completed_at=T0 + timedelta(hours=1))
    final = resolve(sheet=two.model_copy(update={"claims": ruled_claims, "adjudicated": adjudication}), required=2)
    assert final.adjudicated and final.unresolved == () and _key(final.sheet) == {"ap_9001_c001": 3, "ap_9001_c002": 2, "ap_9001_c003": None}
    assert final.sheet.author == "Judge"


def test_a_completed_adjudication_corrects_a_single_review_and_leaves_the_record_as_it_was() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    one = _done(sheet=sheet, annotator="Aram", view=annotator_view(sheet=sheet, annotator="Aram"), at=T0)
    assert _key(resolve(sheet=one, required=1).sheet) == {"ap_9001_c001": 3, "ap_9001_c002": 2, "ap_9001_c003": 1}
    ruling = ClaimLabel(annotator="Judge", value="false", comment="defended three times", at=T0 + timedelta(hours=1), role="adjudicator")
    ruled_claims = [c.model_copy(update={"labels": [*c.labels, ruling]}) if c.claim_id == "ap_9001_c002" else c for c in one.claims]
    adjudication = Adjudication(adjudicator="Judge", weights={"ap_9001_c001": RecordWeight(weight=3, suggested=3), "ap_9001_c002": RecordWeight(weight=None, suggested=2), "ap_9001_c003": RecordWeight(weight=None, suggested=1)}, completed_at=T0 + timedelta(hours=1))
    adjudicated = one.model_copy(update={"claims": ruled_claims, "adjudicated": adjudication})
    final = resolve(sheet=adjudicated, required=1)
    assert final.final and final.adjudicated and final.annotators == ("Aram",) and final.sheet.author == "Judge"
    assert _key(final.sheet) == {"ap_9001_c001": 3, "ap_9001_c002": None, "ap_9001_c003": None}
    assert not next(c for c in final.sheet.claims if c.claim_id == "ap_9001_c002").held_true
    assert annotator_view(sheet=adjudicated, annotator="Aram") == annotator_view(sheet=one, annotator="Aram"), "the reviewer's own view stays theirs"
    unfinished = adjudicated.model_copy(update={"adjudicated": adjudication.model_copy(update={"completed_at": None})})
    assert resolve(sheet=unfinished, required=1).sheet == resolve(sheet=one, required=1).sheet, "an adjudication counts once complete"


def test_an_adjudication_links_claims_of_one_idea_in_the_key_and_leaves_the_record_as_it_was() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    one = _done(sheet=sheet, annotator="Aram", view=annotator_view(sheet=sheet, annotator="Aram"), at=T0)
    weights = {c: RecordWeight(weight=w, suggested=w) for c, w in (("ap_9001_c001", 3), ("ap_9001_c002", 2), ("ap_9001_c003", 1))}
    adjudication = Adjudication(adjudicator="Judge", weights=weights, same_idea={"ap_9001_c003": "ap_9001_c001"}, completed_at=T0 + timedelta(hours=1))
    adjudicated = one.model_copy(update={"adjudicated": adjudication})
    final = resolve(sheet=adjudicated, required=1)
    assert final.final and [c.same_idea_as for c in final.sheet.claims] == [None, None, "ap_9001_c001"]
    assert _key(final.sheet) == _key(resolve(sheet=one, required=1).sheet), "the weights stay; only the link is added"
    assert all(c.same_idea_as is None for c in annotator_view(sheet=adjudicated, annotator="Aram").claims)
    assert annotator_view(sheet=adjudicated, annotator="Aram") == annotator_view(sheet=one, annotator="Aram")
    assert _json_roundtrip(sheet=adjudicated) == adjudicated
    with pytest.raises(ValueError, match="names claims the sheet does not have"):
        Sheet.model_validate({**adjudicated.model_dump(), "adjudicated": {**adjudication.model_dump(), "same_idea": {"ap_9001_c003": "ap_9001_c009"}}})
    with pytest.raises(ValueError, match="same idea as themselves"):
        Adjudication(adjudicator="Judge", same_idea={"ap_9001_c003": "ap_9001_c003"})
    assert "same_idea" not in Adjudication(adjudicator="Judge").model_dump()


def test_an_adjudication_cannot_link_a_duplicate_or_a_claim_of_an_unfinished_review_the_key_leaves_out() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    one = _done(sheet=sheet, annotator="Aram", view=annotator_view(sheet=sheet, annotator="Aram"), at=T0)
    duplicate = claim(claim_id="ap_9001_c004", text="Be3 is the strongest move", weight=None, source=model_source(name="claude-opus-5-5"), kind="plan").model_copy(update={"duplicate_of": "ap_9001_c002"})
    lilit_claim = new_gm_claim(task_id="ap_9001", number=5, text="White castles queenside and storms the king", annotator="Lilit", weight=None, at=T0, how_to_name="test")
    lilit = one.annotations["Aram"].model_copy(update={"added": [lilit_claim.claim_id], "completed_at": None, "weights": {}})
    data = {**one.model_dump(), "claims": [*(c.model_dump() for c in one.claims), duplicate.model_dump(), lilit_claim.model_dump()], "last_claim_number": 5, "annotations": {**one.model_dump()["annotations"], "Lilit": lilit.model_dump()}}
    base = Sheet.model_validate(data)
    assert resolve(sheet=base, required=1).final
    for links in ({"ap_9001_c004": "ap_9001_c001"}, {"ap_9001_c001": "ap_9001_c004"}, {"ap_9001_c005": "ap_9001_c001"}, {"ap_9001_c001": "ap_9001_c005"}):
        ruling = Adjudication(adjudicator="Judge", same_idea=links, completed_at=T0 + timedelta(hours=1))
        with pytest.raises(ValueError, match="duplicates or claims of an unfinished review"):
            Sheet.model_validate({**data, "adjudicated": ruling.model_dump()})
    for target in ("ap_9001_c004", "ap_9001_c005"):
        claims = [{**c, "same_idea_as": target} if c["claim_id"] == "ap_9001_c003" else c for c in data["claims"]]
        with pytest.raises(ValueError, match="same_idea_as must name another claim"):
            Sheet.model_validate({**data, "claims": claims})


def _json_roundtrip(*, sheet: Sheet) -> Sheet:
    return Sheet.model_validate_json(sheet.model_dump_json())


def test_turning_points_and_verdicts_resolve_to_what_every_record_agrees_on() -> None:
    sheet = per_annotator_sheet(sheet=_sheet("gr_9001"))
    first = annotator_view(sheet=sheet, annotator="Aram")
    one = _done(sheet=sheet, annotator="Aram", view=first, at=T0)
    other = annotator_view(sheet=one, annotator="Lilit").model_copy(update={"turning_points": [TurningPoint(ply=6, move="Bg4", label="mistake"), TurningPoint(ply=18, move="b5", label="blunder")]})
    two = _done(sheet=one, annotator="Lilit", view=other, at=T0)
    resolution = resolve(sheet=two, required=2)
    assert [(tp.ply, tp.label) for tp in resolution.sheet.turning_points] == [(18, "blunder")]
    assert "turning points differ" in resolution.unresolved
    assert two.annotations["Lilit"].suggested_turning_points == sheet.suggested.turning_points if sheet.suggested else False


def test_load_resolved_sheets_reads_the_double_annotation_set(tmp_path: Path) -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    done = _done(sheet=sheet, annotator="Aram", view=_weight(view=annotator_view(sheet=sheet, annotator="Aram"), claim_id="ap_9001_c003", weight=None), at=T0)
    write_sheet(sheet=done, path=tmp_path / "ap_9001.yaml")
    resolved = load_resolved_sheets(sheets_dir=tmp_path, double=DoubleAnnotation(required_annotators=2, task_ids=["ap_9001"]))
    assert _key(resolved["ap_9001"])["ap_9001_c003"] is None
    assert DoubleAnnotation(required_annotators=2, task_ids=["ap_9001"]).required(task_id="ap_9001") == 2
    assert DoubleAnnotation(required_annotators=2, task_ids=["ap_9001"]).required(task_id="ap_9002") == 1


def test_validation_checks_the_resolved_key(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    before = errors_of(problems=validate_all(tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets"))
    for path in (tmp_path / "sheets").glob("*.yaml"):
        write_sheet(sheet=per_annotator_sheet(sheet=read_sheet(path=path)), path=path)
    assert errors_of(problems=validate_all(tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets")) == before




def test_a_record_naming_an_unknown_claim_does_not_load() -> None:
    with pytest.raises(ValueError, match="names claims the sheet does not have"):
        Sheet.model_validate({**_sheet().model_dump(), "annotations": {"Aram": {"weights": {"ap_9001_c099": {"weight": 1, "suggested": None}}}}})


def test_a_duplicate_of_another_grandmasters_added_claim_is_hidden_with_it() -> None:
    sheet = per_annotator_sheet(sheet=_sheet())
    added = new_gm_claim(task_id="ap_9001", number=4, text="Black's king is safe", annotator="Aram", weight=None, at=T0, how_to_name="log in")
    view = annotator_view(sheet=sheet, annotator="Aram")
    stored = store_view(sheet=sheet, annotator="Aram", view=view.model_copy(update={"claims": [*view.claims, added], "last_claim_number": 4}), at=T0, confirmed=frozenset())
    duplicate = stored.claims[1].model_copy(update={"claim_id": "ap_9001_c005", "text": "Black's king is safe.", "duplicate_of": "ap_9001_c004", "weight": None, "labels": [], "weighted_by": None})
    with_duplicate = Sheet.model_validate({**stored.model_dump(), "claims": [*[c.model_dump() for c in stored.claims], duplicate.model_dump()], "last_claim_number": 5})
    assert [c.claim_id for c in annotator_view(sheet=with_duplicate, annotator="Lilit").claims] == ["ap_9001_c001", "ap_9001_c002", "ap_9001_c003"]
    assert "ap_9001_c005" in [c.claim_id for c in annotator_view(sheet=with_duplicate, annotator="Aram").claims]
    done = set_completed(sheet=store_view(sheet=with_duplicate, annotator="Lilit", view=annotator_view(sheet=with_duplicate, annotator="Lilit"), at=T0, confirmed=frozenset()), annotator="Lilit", at=T0)
    assert resolve(sheet=done, required=1).annotators == ("Lilit",)


def test_a_superseded_engine_claim_stays_false_in_every_view_and_key() -> None:
    from bench.models import SUPERSEDED_COMMENT

    superseded = ClaimLabel(annotator="reference re-analysis", value="false", comment=f"{SUPERSEDED_COMMENT}: 13.Rc1 is first now", at=T0)
    base = _sheet()
    sheet = per_annotator_sheet(sheet=base.model_copy(update={"claims": [c.model_copy(update={"labels": [superseded], "weight": None}) if c.claim_id == "ap_9001_c002" else c for c in base.claims]}))
    view = annotator_view(sheet=sheet, annotator="Aram")
    claim = next(c for c in view.claims if c.claim_id == "ap_9001_c002")
    assert claim.dropped is not None and not claim.held_true and claim.weight is None, "set aside for the grandmaster, never true"
    done = set_completed(sheet=store_view(sheet=sheet, annotator="Aram", view=view, at=T0, confirmed=frozenset()), annotator="Aram", at=T0)
    for required in (1, 2):
        key = next(c for c in resolve(sheet=done, required=required).sheet.claims if c.claim_id == "ap_9001_c002")
        assert not key.held_true and key.weight is None
    two = set_completed(sheet=store_view(sheet=done, annotator="Lilit", view=annotator_view(sheet=done, annotator="Lilit"), at=T0, confirmed=frozenset()), annotator="Lilit", at=T0)
    assert not next(c for c in resolve(sheet=two, required=2).sheet.claims if c.claim_id == "ap_9001_c002").held_true
    assert [label.annotator for label in two.claims[1].labels] == ["reference re-analysis"]


def test_a_key_is_signed_and_graded_only_once_final() -> None:
    from bench.annotations import has_gm_work, sheet_has_gm_work

    sheet = per_annotator_sheet(sheet=_sheet())
    assert not sheet_has_gm_work(sheet=sheet) and not any(has_gm_work(sheet=sheet, claim=c) for c in sheet.claims if c.source.type == "engine")
    one = _done(sheet=sheet, annotator="Aram", view=_weight(view=annotator_view(sheet=sheet, annotator="Aram"), claim_id="ap_9001_c003", weight=3), at=T0)
    assert has_gm_work(sheet=one, claim=one.claims[2]) and sheet_has_gm_work(sheet=one)
    half = resolve(sheet=one, required=2)
    assert half.held_back == "1 completed review(s), more needed" and half.sheet.author is None and half.sheet.signed_at is None
    final = resolve(sheet=one, required=1)
    assert final.final and final.held_back is None and final.sheet.author == "Aram"
    other = _done(sheet=one, annotator="Lilit", view=annotator_view(sheet=one, annotator="Lilit"), at=T0)
    disputed = resolve(sheet=other, required=2)
    assert disputed.held_back == "claim ap_9001_c003: weights differ" and disputed.sheet.author is None


def _with_model_claims() -> Sheet:
    sheet = per_annotator_sheet(sheet=_sheet())
    added = [claim(claim_id=f"ap_9001_c00{n}", text=f"Model claim {n}", weight=None, source=model_source()) for n in (4, 5)]
    return Sheet.model_validate({**sheet.model_dump(), "claims": [*(c.model_dump() for c in sheet.claims), *(c.model_dump() for c in added)], "last_claim_number": 5})


def test_a_completion_records_the_claims_left_unlabelled_as_skipped_and_recomputes_them_each_time(tmp_path: Path) -> None:
    sheet = _with_model_claims()
    view = _label(view=annotator_view(sheet=sheet, annotator="Aram"), claim_id="ap_9001_c004", annotator="Aram", value="true", at=T0)
    done = _done(sheet=sheet, annotator="Aram", view=view, at=T0)
    assert done.annotations["Aram"].skipped == ["ap_9001_c005"]
    assert resolve(sheet=done, required=1).complete, "a completion with skips is a completed review"
    reopened = set_completed(sheet=done, annotator="Aram", at=None)
    assert reopened.annotations["Aram"].skipped == []
    relabelled = _label(view=annotator_view(sheet=reopened, annotator="Aram"), claim_id="ap_9001_c005", annotator="Aram", value="unclear", at=T0 + timedelta(minutes=1))
    again = _done(sheet=reopened, annotator="Aram", view=relabelled, at=T0 + timedelta(minutes=1))
    assert again.annotations["Aram"].skipped == []
    write_sheet(sheet=again, path=tmp_path / "ap_9001.yaml")
    assert "skipped" not in (tmp_path / "ap_9001.yaml").read_text(encoding="utf-8"), "an empty skipped list is not written"
    write_sheet(sheet=done, path=tmp_path / "ap_9001.yaml")
    assert "skipped:\n    - ap_9001_c005" in (tmp_path / "ap_9001.yaml").read_text(encoding="utf-8")
    assert read_sheet(path=tmp_path / "ap_9001.yaml") == done


def test_a_skipped_claim_is_no_label_in_the_key_and_is_listed_until_a_completed_review_labels_it() -> None:
    sheet = _with_model_claims()
    aram = _weight(view=_label(view=annotator_view(sheet=sheet, annotator="Aram"), claim_id="ap_9001_c004", annotator="Aram", value="true", at=T0), claim_id="ap_9001_c004", weight=2)
    one = _done(sheet=sheet, annotator="Aram", view=aram, at=T0)
    single = {c.claim_id: c for c in resolve(sheet=one, required=1).sheet.claims}
    assert single["ap_9001_c005"].labels == [] and single["ap_9001_c005"].weight is None and not single["ap_9001_c005"].held_true
    assert unlabelled_claims(sheet=one) == ["ap_9001_c005"]
    two = _done(sheet=one, annotator="Lilit", view=annotator_view(sheet=one, annotator="Lilit"), at=T0 + timedelta(minutes=1))
    assert two.annotations["Lilit"].skipped == ["ap_9001_c004", "ap_9001_c005"]
    agreed = resolve(sheet=two, required=2)
    keyed = {c.claim_id: c for c in agreed.sheet.claims}
    assert keyed["ap_9001_c004"].weight == 2 and [label.annotator for label in keyed["ap_9001_c004"].labels] == ["Aram"]
    assert not any("ap_9001_c004" in reason or "ap_9001_c005" in reason for reason in agreed.unresolved), "a skip is no opinion, not a disagreement"
    assert unlabelled_claims(sheet=two) == ["ap_9001_c005"]
