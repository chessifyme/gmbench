import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from bench.annotations import annotator_view, per_annotator_sheet, read_double_annotation, resolve, set_completed, store_view
from bench.io_yaml import read_sheet, yaml_files
from bench.models import Adjudication, ClaimLabel, RecordWeight, Sheet
from bench.paths import DATA_PUBLIC, ROOT
from bench.published_sheets import (
    Pseudonyms,
    PublishError,
    key_content,
    name_problems,
    pseudonymised,
    pseudonyms_in_use,
    published_key_content,
    published_sheet,
    reviewer_names,
    unpublished_claim_ids,
)

FIXTURES = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)
PUBLIC_SHEETS = yaml_files(directory=DATA_PUBLIC / "expert_points")








@pytest.mark.parametrize("path", PUBLIC_SHEETS, ids=lambda path: path.stem)
def test_every_public_sheet_is_in_published_form_with_pseudonyms_only(path: Path, tmp_path: Path) -> None:
    sheet = read_sheet(path=path)
    assert name_problems(data=sheet.model_dump()) == []
    names = Pseudonyms(path=tmp_path / "pseudonyms.yaml")
    assert published_sheet(sheet=sheet, required=read_double_annotation().required(task_id=sheet.task_id), pseudonyms=names) == sheet and names.added == []


def _done(*, sheet: Sheet, annotator: str, view: Sheet, at: datetime) -> Sheet:
    return set_completed(sheet=store_view(sheet=sheet, annotator=annotator, view=view, at=at, confirmed=frozenset()), annotator=annotator, at=at)


def _labelled(*, view: Sheet, claim_id: str, annotator: str, value: str, at: datetime) -> Sheet:
    claims = [c.model_copy(update={"labels": [*c.labels, ClaimLabel(annotator=annotator, value=value, comment="why", at=at)], "weight": None if value != "true" else c.weight}) if c.claim_id == claim_id else c for c in view.claims]
    return view.model_copy(update={"claims": claims})


def _two_reviews() -> Sheet:
    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    first = _labelled(view=annotator_view(sheet=sheet, annotator="Aram Real"), claim_id="ap_9001_c002", annotator="Aram Real", value="true", at=T0)
    one = _done(sheet=sheet, annotator="Aram Real", view=first, at=T0)
    second = _labelled(view=annotator_view(sheet=one, annotator="Lilit Real"), claim_id="ap_9001_c002", annotator="Lilit Real", value="false", at=T0 + timedelta(minutes=5))
    return _done(sheet=one, annotator="Lilit Real", view=second, at=T0 + timedelta(minutes=5))


def test_an_adjudicated_double_review_publishes_its_ruling_under_pseudonyms_and_dates(tmp_path: Path) -> None:
    two = _two_reviews()
    ruling = ClaimLabel(annotator="Judge Real", value="true", comment="adjudicated", at=T0 + timedelta(minutes=10), role="adjudicator")
    claims = [c.model_copy(update={"labels": [*c.labels, ruling]}) if c.claim_id == "ap_9001_c002" else c for c in two.claims]
    weights = {"ap_9001_c001": RecordWeight(weight=3, suggested=3), "ap_9001_c002": RecordWeight(weight=2, suggested=2), "ap_9001_c003": RecordWeight(weight=1, suggested=1)}
    adjudicated = two.model_copy(update={"claims": claims, "adjudicated": Adjudication(adjudicator="Judge Real", weights=weights, notes="internal", completed_at=T0 + timedelta(minutes=10))})
    names = Pseudonyms(path=tmp_path / "pseudonyms.yaml", mapping={"Lilit Real": "Expert A"})
    published = published_sheet(sheet=adjudicated, required=2, pseudonyms=names)
    text = published.model_dump_json()
    assert "Real" not in text and "internal" not in text and name_problems(data=published.model_dump()) == []
    assert names.mapping == {"Lilit Real": "Expert A", "Aram Real": "Expert B", "GM Test": "Expert C", "Judge Real": "Expert D"}
    assert published.author == "Expert D" and published.signed_at == datetime(2026, 11, 1, tzinfo=UTC)
    disputed = next(c for c in published.claims if c.claim_id == "ap_9001_c002")
    assert [(label.annotator, label.value, label.role) for label in disputed.labels] == [("Expert B", "true", "annotator"), ("Expert A", "false", "annotator"), ("Expert D", "true", "adjudicator")]
    assert all(label.at == datetime(2026, 11, 1, tzinfo=UTC) for label in disputed.labels) and disputed.adjudication is not None and disputed.weight == 2
    full = resolve(sheet=pseudonymised(sheet=adjudicated, pseudonyms=names), required=2)
    assert key_content(resolution=resolve(sheet=published, required=2)) == key_content(resolution=full) and full.final


def test_a_held_back_review_is_not_published(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="held back"):
        published_sheet(sheet=_two_reviews(), required=2, pseudonyms=Pseudonyms(path=tmp_path / "pseudonyms.yaml"))


def test_a_real_name_or_a_working_record_in_a_public_sheet_is_found() -> None:
    sheet = _two_reviews()
    problems = name_problems(data=sheet.model_dump())
    assert any("'Aram Real' is not a pseudonym" in problem for problem in problems) and any("working record" in problem for problem in problems)


def test_the_pseudonym_mapping_keeps_its_names_and_refuses_a_pseudonym_given_twice(tmp_path: Path) -> None:
    path = tmp_path / "pseudonyms.yaml"
    path.write_text("", encoding="utf-8")
    names = Pseudonyms.load(path=path)
    assert [names.of(name=name) for name in ("B Person", "A Person", "B Person", "Expert A")] == ["Expert A", "Expert B", "Expert A", "Expert A"]
    names.save()
    assert Pseudonyms.load(path=path).mapping == {"A Person": "Expert B", "B Person": "Expert A"}
    path.write_text(yaml.safe_dump({"A Person": "Expert A", "B Person": "Expert A"}), encoding="utf-8")
    with pytest.raises(PublishError, match="two reviewers"):
        Pseudonyms.load(path=path)


def test_a_missing_pseudonym_mapping_is_refused_rather_than_started_afresh(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="does not exist"):
        Pseudonyms.load(path=tmp_path / "pseudonyms.yaml")


def test_a_new_reviewer_never_takes_a_pseudonym_the_public_sheets_already_use(tmp_path: Path) -> None:
    names = Pseudonyms(path=tmp_path / "pseudonyms.yaml", reserved=frozenset({"Expert A", "Expert C"}))
    assert [names.of(name=name) for name in ("First Person", "Second Person", "Expert A")] == ["Expert B", "Expert D", "Expert A"]


def test_the_old_gm_pseudonym_form_is_not_a_pseudonym(tmp_path: Path) -> None:
    data = {"task_id": "ap_0001", "author": "GM-A", "reviewed_by": "Expert C", "claims": [{"labels": [{"annotator": "GM-C"}]}]}
    problems = name_problems(data=data)
    assert len(problems) == 2 and any("'GM-A' is not a pseudonym (Expert A, Expert B, ...)" in problem for problem in problems) and any("'GM-C'" in problem for problem in problems)
    path = tmp_path / "pseudonyms.yaml"
    path.write_text(yaml.safe_dump({"A Person": "GM-A"}), encoding="utf-8")
    with pytest.raises(PublishError, match="of the form Expert A"):
        Pseudonyms.load(path=path)


def test_pseudonyms_past_z_take_two_letters(tmp_path: Path) -> None:
    names = Pseudonyms(path=tmp_path / "pseudonyms.yaml", reserved=frozenset(f"Expert {chr(code)}" for code in range(ord("A"), ord("Z") + 1)))
    assert [names.of(name=name) for name in ("First Person", "Second Person")] == ["Expert AA", "Expert AB"]


def test_the_pseudonyms_in_use_are_read_from_the_public_sheets() -> None:
    assert pseudonyms_in_use(sheets_dir=DATA_PUBLIC / "expert_points") == {"Expert A", "Expert B", "Expert C"}


def test_a_reviewer_whose_unfinished_record_the_published_form_leaves_out_gets_no_pseudonym_and_no_letter(tmp_path: Path) -> None:
    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    started = _labelled(view=annotator_view(sheet=sheet, annotator="Aaa Tester"), claim_id="ap_9001_c002", annotator="Aaa Tester", value="false", at=T0)
    trying = store_view(sheet=sheet, annotator="Aaa Tester", view=started, at=T0, confirmed=frozenset())
    finished = _labelled(view=annotator_view(sheet=trying, annotator="Zed Real"), claim_id="ap_9001_c002", annotator="Zed Real", value="true", at=T0 + timedelta(minutes=5))
    signed = _done(sheet=trying, annotator="Zed Real", view=finished, at=T0 + timedelta(minutes=5))
    names = Pseudonyms(path=tmp_path / "pseudonyms.yaml", mapping={"GM Test": "Expert A"})
    published = published_sheet(sheet=signed, required=1, pseudonyms=names)
    assert "Aaa Tester" not in names.mapping and names.mapping["Zed Real"] == "Expert B" and names.added == ["Zed Real"]
    assert "Tester" not in published.model_dump_json() and reviewer_names(sheet=published) <= {"Expert A", "Expert B"}


def test_the_published_form_names_a_model_source_only_as_an_ai_model_without_its_run_and_gives_a_cleanup_drop_a_neutral_comment(tmp_path: Path) -> None:
    from bench.models import CAP_ANNOTATOR, ClaimDrop, ClaimSource

    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    model = ClaimSource(type="model", name="some_model", answer_ref="some_model/20261105_125422/ap_9001/0")
    drops = {
        "ap_9001_c002": ClaimDrop(annotator=CAP_ANNOTATOR, comment="over the review cap (20 positions / 30 games); kept for later", at=T0),
        "ap_9001_c003": ClaimDrop(annotator="cleanup", comment="not a chess claim: a meta statement (an entrant's non-answer)", at=T0),
    }
    claims = [c.model_copy(update={"source": model, "dropped": drops[c.claim_id], "weight": None}) if c.claim_id in drops else c for c in sheet.claims]
    published = published_sheet(sheet=sheet.model_copy(update={"claims": claims}), required=1, pseudonyms=Pseudonyms(path=tmp_path / "pseudonyms.yaml"))
    by_id = {c.claim_id: c for c in published.claims}
    assert by_id["ap_9001_c003"].source == ClaimSource(type="model", name="ai_model", answer_ref=None)
    others = [c for c in sheet.claims if c.source.type != "model" and c.claim_id not in drops]
    assert others and all(by_id[c.claim_id].source.type == c.source.type and by_id[c.claim_id].source.name != "ai_model" for c in others)
    assert all(by_id[c.claim_id].source == c.source.model_copy(update={"answer_ref": None}) for c in others if c.source.type == "engine")
    assert "ap_9001_c002" not in by_id, "a model claim set aside unreviewed over the review limit is not published"
    assert by_id["ap_9001_c003"].dropped is not None and by_id["ap_9001_c003"].dropped.comment == "not a chess claim"


def test_the_published_form_keeps_a_same_idea_link_and_its_key_names_it(tmp_path: Path) -> None:
    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    linked = sheet.model_copy(update={"claims": [c.model_copy(update={"same_idea_as": "ap_9001_c001"}) if c.claim_id == "ap_9001_c003" else c for c in sheet.claims]})
    published = published_sheet(sheet=linked, required=1, pseudonyms=Pseudonyms(path=tmp_path / "pseudonyms.yaml"))
    assert next(c for c in published.claims if c.claim_id == "ap_9001_c003").same_idea_as == "ap_9001_c001"
    content = key_content(resolution=resolve(sheet=published, required=1))
    assert [c["same_idea_as"] for c in content["claims"]] == [None, None, "ap_9001_c001"]  # type: ignore[index, union-attr]


def _set_aside(*, sheet: Sheet, claim_id: str) -> Sheet:
    from bench.models import CAP_ANNOTATOR, ClaimDrop

    drop = ClaimDrop(annotator=CAP_ANNOTATOR, comment="over the review cap", at=T0)
    return sheet.model_copy(update={"claims": [c.model_copy(update={"dropped": drop, "weight": None, "labels": []}) if c.claim_id == claim_id else c for c in sheet.claims]})


def _duplicate(*, sheet: Sheet, claim_id: str, of: str, number: int) -> Sheet:
    """`sheet` with an unlabelled, unweighted model claim, a copy of `claim_id`'s text, marked a duplicate of `of`."""
    from bench.models import ClaimSource

    source = next(c for c in sheet.claims if c.claim_id == claim_id)
    copy = source.model_copy(update={"claim_id": f"ap_9001_c{number:03d}", "source": ClaimSource(type="model", name="some_model", answer_ref=None), "labels": [], "weight": None, "dropped": None, "duplicate_of": of})
    return Sheet.model_validate({**sheet.model_dump(), "claims": [*(c.model_dump() for c in sheet.claims), copy.model_dump()], "last_claim_number": max(sheet.last_claim_number, number)})


def test_a_claim_set_aside_unreviewed_and_its_unreviewed_duplicates_are_not_published_and_the_rest_keep_their_ids(tmp_path: Path) -> None:
    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    aside = _set_aside(sheet=sheet, claim_id="ap_9001_c002")
    duplicated = _duplicate(sheet=_duplicate(sheet=aside, claim_id="ap_9001_c002", of="ap_9001_c002", number=50), claim_id="ap_9001_c002", of="ap_9001_c002", number=51)
    assert unpublished_claim_ids(sheet=duplicated) == {"ap_9001_c002", "ap_9001_c050", "ap_9001_c051"}
    published = published_sheet(sheet=duplicated, required=1, pseudonyms=Pseudonyms(path=tmp_path / "pseudonyms.yaml"))
    kept = [c.claim_id for c in duplicated.claims if c.claim_id not in {"ap_9001_c002", "ap_9001_c050", "ap_9001_c051"}]
    assert [c.claim_id for c in published.claims] == kept and published.last_claim_number == duplicated.last_claim_number


def test_a_reviewed_claim_that_points_at_a_claim_set_aside_unreviewed_stops_the_publication(tmp_path: Path) -> None:
    sheet = per_annotator_sheet(sheet=read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml"))
    aside = _set_aside(sheet=sheet, claim_id="ap_9001_c002")
    duplicate = _duplicate(sheet=aside, claim_id="ap_9001_c002", of="ap_9001_c002", number=50)
    weighted = duplicate.model_copy(update={"claims": [c.model_copy(update={"weight": 1}) if c.claim_id == "ap_9001_c050" else c for c in duplicate.claims]})
    labelled = duplicate.model_copy(update={"claims": [c.model_copy(update={"labels": [ClaimLabel(annotator="GM Test", value="false", comment="", at=T0)]}) if c.claim_id == "ap_9001_c050" else c for c in duplicate.claims]})
    idea = aside.model_copy(update={"claims": [c.model_copy(update={"same_idea_as": "ap_9001_c002"}) if c.claim_id == "ap_9001_c003" else c for c in aside.claims]})
    for bad in (weighted, labelled, idea):
        with pytest.raises(PublishError, match="set aside unreviewed"):
            unpublished_claim_ids(sheet=bad)
