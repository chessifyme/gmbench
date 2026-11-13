"""Per-annotator records in the review app: blindness through every route, two grandmasters at once, completion counts,
flag and reject permissions, and the adjudicator's side-by-side view."""

import re
import shutil
import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from bench.annotations import annotator_view, resolve
from bench.io_yaml import read_sheet, read_task, write_sheet
from bench.models import Sheet
from review_app.app import create_app
from review_app.identity import ACCESS_COOKIE
from tests.claim_fakes import claim, model_source
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
MARKERS = ("ANOTE7", "AREWORD7", "ACLAIM7", "AFLAG7", "ACOMMENT7")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    mj = read_sheet(path=FIXTURES / "sheets" / "ap_9001.yaml")
    plan = mj.claims[0].model_copy(update={"claim_id": "mj_9001_c001", "text": "Black recaptures in the centre and keeps the extra pawn", "source": mj.claims[0].source.model_copy(update={"type": "engine", "name": "stockfish_19"})})
    write_sheet(sheet=mj.model_copy(update={"task_id": "mj_9001", "verdict": "mistake", "claims": [plan], "last_claim_number": 1}), path=tmp_path / "sheets" / "mj_9001.yaml")
    (tmp_path / "double.yaml").write_text(yaml.safe_dump({"required_annotators": 2, "task_ids": ["ap_9001"]}), encoding="utf-8")
    return tmp_path


@pytest.fixture
def app(data: Path):
    return create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, double_annotation_file=data / "double.yaml")


def _as(app, name: str) -> TestClient:
    return login(TestClient(app), name)


def _sheet(data: Path, task_id: str) -> Sheet:
    return read_sheet(path=data / "sheets" / f"{task_id}.yaml")


def _form(*, data: Path, task_id: str, name: str, **changes: object) -> dict[str, object]:
    view = annotator_view(sheet=_sheet(data, task_id), annotator=name)
    form: dict[str, object] = {
        "claim_id": [c.claim_id for c in view.claims],
        "claim_text": [c.text for c in view.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in view.claims],
        "claim_weight_was": ["" if c.weight is None else str(c.weight) for c in view.claims],
        "claim_label": [c.latest_labels[name].value if name in c.latest_labels else "" for c in view.claims],
        "claim_comment": ["" for _ in view.claims],
        "notes": view.notes,
        "verdict": view.verdict or "",
    }
    return form | changes


def _aram_works(*, app, data: Path) -> None:
    """Aram leaves a mark of every kind: a label with a comment, a changed weight, a wording, a new claim, notes, a verdict,
    turning points and a flag."""
    aram = _as(app, "Aram")
    form = _form(data=data, task_id="ap_9001", name="Aram")
    form["claim_id"] = [*form["claim_id"], ""]  # type: ignore[list-item]
    form["claim_text"] = ["AREWORD7 is White's plan", *form["claim_text"][1:], "ACLAIM7 White's king is safe"]  # type: ignore[index]
    form["claim_weight"] = ["3", "", "1", "2"]
    form["claim_weight_was"] = [*form["claim_weight_was"], ""]  # type: ignore[list-item]
    form["claim_label"] = ["true", "false", "", ""]
    form["claim_comment"] = ["", "ACOMMENT7 Nf3 is better", "", ""]
    assert aram.post("/sheets/ap_9001/save", data=form | {"notes": "ANOTE7"}, follow_redirects=False).status_code == 303
    assert aram.post("/sheets/mj_9001/save", data=_form(data=data, task_id="mj_9001", name="Aram", verdict="blunder"), follow_redirects=False).status_code == 303
    tps = {"tp_6": "1", "tp_label_6": "blunder"}
    assert aram.post("/sheets/gr_9001/save", data=_form(data=data, task_id="gr_9001", name="Aram") | tps, follow_redirects=False).status_code == 303
    assert aram.post("/sheets/gr_9001/flag", data={"reason": "AFLAG7 the game is a duplicate"}, follow_redirects=False).status_code == 303


def test_one_grandmaster_sees_nothing_of_anothers_record_on_any_route(app, data: Path) -> None:
    _aram_works(app=app, data=data)
    stored = _sheet(data, "ap_9001").model_dump_json()
    assert all(marker in stored for marker in ("ANOTE7", "AREWORD7", "ACLAIM7", "ACOMMENT7"))
    lilit = _as(app, "Lilit")
    pages = {path: lilit.get(path) for path in ("/sheets", "/sheets?rejected=1&internal=1", "/sheets/ap_9001", "/sheets/ap_9001?adjudicate=1", "/sheets/mj_9001", "/sheets/gr_9001", "/healthz")}
    pages["refused save"] = lilit.post("/sheets/ap_9001/save", data={"claim_id": ["nope"], "claim_text": ["x"], "claim_weight": [""], "claim_label": [""], "claim_comment": [""]})
    pages["adjudicate"] = lilit.get("/sheets/ap_9001/adjudicate")
    for name, response in pages.items():
        for marker in MARKERS:
            assert marker not in response.text, f"{marker} on {name}"
        assert "Aram" not in response.text, f"Aram named on {name}"
    assert pages["adjudicate"].status_code == 404
    mine = annotator_view(sheet=_sheet(data, "ap_9001"), annotator="Lilit")
    assert [c.weight for c in mine.claims] == [3, 2, 1] and not any(c.labels for c in mine.claims)
    assert annotator_view(sheet=_sheet(data, "mj_9001"), annotator="Lilit").verdict is None
    gr_page = pages["/sheets/gr_9001"].text
    assert 'name="tp_6" value="1" checked' in gr_page and 'name="tp_18" value="1" checked' in gr_page


def test_the_verdict_is_never_preselected_but_the_analysis_is_a_hint(app, data: Path) -> None:
    page = _as(app, "Lilit").get("/sheets/mj_9001").text
    assert '<option value="" selected>choose your verdict</option>' in page
    assert "The analysis says mistake" in page and 'value="mistake" selected' not in page


def test_an_adjudicator_sees_every_record_side_by_side(app, data: Path) -> None:
    _aram_works(app=app, data=data)
    lilit = _as(app, "Lilit")
    assert lilit.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Lilit") | {"notes": "LNOTE"}, follow_redirects=False).status_code == 303
    page = _as(app, "GM Judge").get("/sheets/ap_9001/adjudicate")
    assert page.status_code == 200
    assert "Aram" in page.text and "Lilit" in page.text and "AREWORD7" in page.text and 'class="disagree"' in page.text
    flagged = _as(app, "GM Judge").get("/sheets/gr_9001/adjudicate").text
    assert "AFLAG7" in flagged


def test_two_grandmasters_saving_at_the_same_time_keep_both_records(app, data: Path) -> None:
    names = ["Aram", "Lilit", "Bella", "GM A", "GM B"]
    forms = {name: _form(data=data, task_id="ap_9001", name=name, notes=f"note of {name}") for name in names}
    results: dict[str, int] = {}

    def save(name: str) -> None:
        results[name] = _as(app, name).post("/sheets/ap_9001/save", data=forms[name], follow_redirects=False).status_code

    threads = [threading.Thread(target=save, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == dict.fromkeys(names, 303)
    stored = _sheet(data, "ap_9001")
    assert {name: stored.annotations[name].notes for name in names} == {name: f"note of {name}" for name in names}


def test_completion_counts_and_next_follow_the_double_annotation_set(app, data: Path) -> None:
    aram, lilit, bella = _as(app, "Aram"), _as(app, "Lilit"), _as(app, "Bella")
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").annotations["Aram"].completed_at is not None
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "draft"
    assert 'href="/sheets/ap_9001" id="next_link"' in lilit.get("/sheets/mj_9002").text or "ap_9001" not in aram.get("/sheets/mj_9002").text.split('id="next_link"')[0][-40:]
    assert lilit.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Lilit", then="complete"), follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"
    resolution = resolve(sheet=_sheet(data, "ap_9001"), required=2)
    assert resolution.complete and resolution.annotators == ("Aram", "Lilit")
    assert "status-complete" in bella.get("/sheets/ap_9001").text
    for client in (aram, lilit, bella):
        assert '/sheets/ap_9001" id="next_link"' not in client.get("/sheets/mc_9001").text
    assert aram.post("/sheets/ap_9001/reopen", follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "draft"


def test_flags_are_for_grandmasters_and_only_an_adjudicator_rejects_or_clears(app, data: Path) -> None:
    aram, judge = _as(app, "Aram"), _as(app, "GM Judge")
    assert aram.post("/sheets/ap_9001/flag", data={"reason": ""}).status_code == 422
    assert aram.post("/sheets/ap_9001/flag", data={"reason": "the question is ambiguous"}, follow_redirects=False).status_code == 303
    assert "rejected" not in read_task(path=data / "tasks" / "ap_9001.yaml").tags
    assert "ap_9001" in TestClient(app).get("/sheets").text
    for route, form in (("reject", {"reason": "bad"}), ("unreject", {}), ("clear_flag", {"annotator": "Aram"})):
        assert aram.post(f"/sheets/ap_9001/{route}", data=form).status_code == 404
        assert TestClient(app).post(f"/sheets/ap_9001/{route}", data=form).status_code == 404
    assert judge.post("/sheets/ap_9001/clear_flag", data={"annotator": "Aram"}, follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").annotations["Aram"].flag is None
    assert judge.post("/sheets/ap_9001/reject", data={"reason": "duplicate position"}, follow_redirects=False).status_code == 303
    assert "rejected" in read_task(path=data / "tasks" / "ap_9001.yaml").tags
    assert judge.post("/sheets/ap_9001/unreject", follow_redirects=False).status_code == 303
    assert "rejected" not in read_task(path=data / "tasks" / "ap_9001.yaml").tags


def test_a_forged_or_unknown_access_cookie_signs_nobody_in(app, data: Path) -> None:
    client = TestClient(app)
    client.cookies.set(ACCESS_COOKIE, "0" * 32)
    assert client.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Aram")).status_code == 422
    client.cookies.set("reviewer", "Aram")
    assert client.post("/sheets/ap_9001/flag", data={"reason": "x", "reviewer": "Aram"}).status_code == 422
    assert "Aram" not in _sheet(data, "ap_9001").annotations


def test_disagreeing_double_reviews_stay_draft_until_adjudicated_and_a_game_review_adjudicates(app, data: Path) -> None:
    for name, label in (("Aram", "blunder"), ("Lilit", "mistake")):
        form = _form(data=data, task_id="gr_9001", name=name) | {"tp_6": "1", "tp_label_6": label, "tp_18": "1", "tp_label_18": "blunder", "then": "complete"}
        assert _as(app, name).post("/sheets/gr_9001/save", data=form, follow_redirects=False).status_code == 303
    assert read_task(path=data / "tasks" / "gr_9001.yaml").status == "draft"
    judge = _as(app, "GM Judge")
    page = judge.get("/sheets/gr_9001/adjudicate").text
    import re
    fields = {m[1]: m[2] for m in re.finditer(r'name="(text_[^"]+)" value="([^"]*)"', page)}
    for m in re.finditer(r'<select name="(weight_[^"]+)">(.*?)</select>', page, re.S):
        chosen = re.search(r'<option value="([^"]*)" selected', m[2])
        fields[m[1]] = chosen[1] if chosen else ""
    fields |= {"tp_6": "1", "tp_label_6": "mistake", "tp_move_6": "Bg4", "tp_18": "1", "tp_label_18": "blunder", "tp_move_18": "b5", "then": "complete"}
    response = judge.post("/sheets/gr_9001/adjudicate", data=fields, follow_redirects=False)
    assert response.status_code == 303, re.findall(r"<li>(.*?)</li>", response.text)
    resolution = resolve(sheet=_sheet(data, "gr_9001"), required=1)
    assert read_task(path=data / "tasks" / "gr_9001.yaml").status == "signed"
    assert resolution.adjudicated and [(tp.ply, tp.label) for tp in resolution.sheet.turning_points] == [(6, "mistake"), (18, "blunder")]
    # idea links recorded on the adjudication outside the app survive the adjudicator saving again in the app
    stored = _sheet(data, "gr_9001")
    assert stored.adjudicated is not None
    linked = {"gr_9001_c003": "gr_9001_c001"}
    write_sheet(sheet=stored.model_copy(update={"adjudicated": stored.adjudicated.model_copy(update={"same_idea": linked})}), path=data / "sheets" / "gr_9001.yaml")
    assert judge.post("/sheets/gr_9001/adjudicate", data=fields, follow_redirects=False).status_code == 303
    again = _sheet(data, "gr_9001")
    assert again.adjudicated is not None and again.adjudicated.same_idea == linked
    assert next(c for c in resolve(sheet=again, required=1).sheet.claims if c.claim_id == "gr_9001_c003").same_idea_as == "gr_9001_c001"


def test_a_claim_a_grandmaster_added_is_named_by_a_token_so_numbers_tell_nothing(app, data: Path) -> None:
    _aram_works(app=app, data=data)
    added = _sheet(data, "ap_9001").annotations["Aram"].added
    assert added == ["ap_9001_c004"]
    aram = _as(app, "Aram")
    page = aram.get("/sheets/ap_9001").text
    assert "ACLAIM7" in page and "ap_9001_c004" not in page
    form = _form(data=data, task_id="ap_9001", name="Aram")
    token = re.search(r'data-claim-id="(ap_9001_m[0-9a-f]{16})"', page)
    assert token is not None
    form["claim_id"] = [token[1] if claim_id == "ap_9001_c004" else claim_id for claim_id in form["claim_id"]]  # type: ignore[union-attr]
    form["claim_comment"] = ["", "ACOMMENT7 Nf3 is better", "", ""]
    assert aram.post("/sheets/ap_9001/save", data=form | {"notes": "again"}, follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").annotations["Aram"].notes == "again"
    judge = _as(app, "GM Judge").get("/sheets/ap_9001/adjudicate").text
    assert "ACLAIM7" in judge and "ap_9001_c004" not in judge


def _app_with(data: Path, *, annotators: frozenset[str], adjudicators: frozenset[str]):
    from tests.review_login import reviewers_for

    return create_app(reviewers=reviewers_for(annotators=annotators, adjudicators=adjudicators), token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, double_annotation_file=data / "double.yaml")


def test_an_adjudicator_who_is_a_grandmaster_sees_the_records_only_after_their_own_review(data: Path) -> None:
    app = _app_with(data, annotators=frozenset({"Alpha", "Bravo", "Charlie"}), adjudicators=frozenset({"Alpha", "Lead"}))
    alpha, bravo, lead = _as(app, "Alpha"), _as(app, "Bravo"), _as(app, "Lead")
    assert alpha.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Alpha", notes="mine"), follow_redirects=False).status_code == 303
    assert bravo.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    blocked = alpha.get("/sheets/ap_9001/adjudicate")
    assert blocked.status_code == 409 and "Bravo" not in blocked.text
    assert alpha.post("/sheets/ap_9001/adjudicate", data={"then": "complete"}).status_code == 409
    assert alpha.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    page = alpha.get("/sheets/ap_9001/adjudicate")
    assert page.status_code == 200 and "Bravo" in page.text and 'id="own_review_warning"' in page.text
    assert 'id="own_review_warning"' not in lead.get("/sheets/ap_9001/adjudicate").text
    fields = {m[1]: m[2] for m in re.finditer(r'name="(text_[^"]+)" value="([^"]*)"', page.text)}
    for m in re.finditer(r'<select name="(weight_[^"]+)">(.*?)</select>', page.text, re.S):
        chosen = re.search(r'<option value="([^"]*)" selected', m[2])
        fields[m[1]] = chosen[1] if chosen else ""
    assert alpha.post("/sheets/ap_9001/adjudicate", data=fields | {"then": "complete"}, follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").adjudicated.own_review_included  # type: ignore[union-attr]
    assert lead.post("/sheets/mj_9002/save", data={"notes": "x"}).status_code == 403, "an adjudicator alone writes no grandmaster record"
    assert lead.post("/sheets/ap_9001/complete").status_code == 403 and "Lead" not in _sheet(data, "ap_9001").annotations
    assert alpha.get("/sheets/gr_9001/adjudicate").status_code == 409, "a task still under review: Alpha could be one of its reviewers"


def test_suggested_weights_are_marked_and_only_set_or_accepted_ones_are_recorded(app, data: Path) -> None:
    aram = _as(app, "Aram")
    page = aram.get("/sheets/ap_9001").text
    assert page.count('class="suggested-note weight-suggested"') == 3 and 'id="accept_suggestions"' in page
    form = _form(data=data, task_id="ap_9001", name="Aram")
    form["claim_weight_state"] = ["suggested", "set", "suggested"]
    form["claim_weight"] = ["3", "2", "2"]
    assert aram.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code == 303
    record = _sheet(data, "ap_9001").annotations["Aram"]
    assert set(record.weights) == {"ap_9001_c002", "ap_9001_c003"}, "c002 accepted, c003 changed, c001 never touched"
    assert record.weights["ap_9001_c002"].weight == record.weights["ap_9001_c002"].suggested == 2
    assert record.weights["ap_9001_c003"].weight == 2 and record.weights["ap_9001_c003"].suggested == 1
    assert aram.get("/sheets/ap_9001").text.count('class="suggested-note weight-suggested"') == 1
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").annotations["Aram"].left_suggested == ["ap_9001_c001"]


def test_a_flagged_task_cannot_be_marked_complete_until_the_flag_is_cleared(app, data: Path) -> None:
    aram, judge = _as(app, "Aram"), _as(app, "GM Judge")
    assert aram.post("/sheets/ap_9001/flag", data={"reason": "two best moves"}, follow_redirects=False).status_code == 303
    refused = aram.post("/sheets/ap_9001/complete")
    assert refused.status_code == 409 and "flag" in refused.text
    assert judge.post("/sheets/ap_9001/clear_flag", data={"annotator": "Aram"}, follow_redirects=False).status_code == 303
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303


def test_the_progress_line_counts_only_the_viewers_own_completions(app, data: Path) -> None:
    aram, lilit = _as(app, "Aram"), _as(app, "Lilit")
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert lilit.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert "1 / 5 complete" in aram.get("/sheets").text and "1 / 5 complete" in lilit.get("/sheets").text
    assert "0 / 5 complete" in _as(app, "Bella").get("/sheets").text


def test_a_save_records_no_weight_for_a_claim_without_a_suggestion_until_the_grandmaster_sets_one(app, data: Path) -> None:
    sheet = _sheet(data, "ap_9001")
    unsuggested = sheet.claims[0].model_copy(update={"claim_id": "ap_9001_c004", "text": "Black castles queenside", "weight": None})
    write_sheet(sheet=sheet.model_copy(update={"claims": [*sheet.claims, unsuggested], "last_claim_number": 4}), path=data / "sheets" / "ap_9001.yaml")
    aram = _as(app, "Aram")
    page = aram.get("/sheets/ap_9001").text
    assert page.count('name="claim_weight_state" value="untouched"') == 1 and page.count('value="suggested"') >= 3
    form = _form(data=data, task_id="ap_9001", name="Aram") | {"claim_weight_state": ["suggested", "suggested", "suggested", "untouched"]}
    assert aram.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code == 303
    assert _sheet(data, "ap_9001").annotations["Aram"].weights == {}, "nothing set, nothing recorded"
    form["claim_weight_state"] = ["suggested", "suggested", "suggested", "set"]
    assert aram.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code == 303
    assert set(_sheet(data, "ap_9001").annotations["Aram"].weights) == {"ap_9001_c004"}, "Not key pressed on purpose"
    assert 'name="claim_weight_state" value="set"' in aram.get("/sheets/ap_9001").text


def test_an_adjudicator_grandmaster_who_has_seen_the_others_cannot_change_their_own_review(data: Path) -> None:
    app = _app_with(data, annotators=frozenset({"Alpha", "Bravo"}), adjudicators=frozenset({"Alpha"}))
    alpha, bravo = _as(app, "Alpha"), _as(app, "Bravo")
    assert alpha.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert bravo.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert "Alpha" in _sheet(data, "ap_9001").annotations and "Alpha" not in _sheet(data, "ap_9001").seen_by_adjudicators
    listed = alpha.get("/sheets").text.split('id="to_adjudicate"')[1].split("</table>")[0]
    assert "ap_9001" in listed
    assert alpha.get("/sheets/ap_9001/adjudicate").status_code == 200
    assert "Alpha" in _sheet(data, "ap_9001").seen_by_adjudicators
    before = _sheet(data, "ap_9001").annotations["Alpha"]
    assert alpha.post("/sheets/ap_9001/reopen", follow_redirects=False).status_code == 409
    assert alpha.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Alpha")).status_code == 409
    assert alpha.post("/sheets/ap_9001/flag", data={"reason": "x"}).status_code == 409
    assert _sheet(data, "ap_9001").annotations["Alpha"] == before


def test_an_adjudicator_grandmaster_lists_no_task_whose_reviews_they_may_not_see_yet(data: Path) -> None:
    app = _app_with(data, annotators=frozenset({"Alpha", "Bravo", "Charlie"}), adjudicators=frozenset({"Alpha"}))
    for name in ("Bravo", "Charlie"):
        assert _as(app, name).post("/sheets/gr_9001/flag", data={"reason": "AFLAG7"}, follow_redirects=False).status_code == 303
    alpha = _as(app, "Alpha")
    assert alpha.post("/sheets/gr_9001/save", data=_form(data=data, task_id="gr_9001", name="Alpha"), follow_redirects=False).status_code == 303
    page = alpha.get("/sheets").text
    table = page.split('id="to_adjudicate"')[1].split("</table>")[0]
    assert "AFLAG7" not in page and "Bravo" not in page and "Charlie" not in table
    row = table.split('class="flag-pending"')[1].split("</tr>")[0]
    assert 'href="/sheets/gr_9001"' in row and "flagged" in row and "/adjudicate" not in row, "the flag is never lost, but shows only the task"
    assert "<td>0</td>" not in table and "<td>2</td>" not in table, "no completion count"
    assert alpha.post("/sheets/gr_9001/complete", follow_redirects=False).status_code == 303
    seen = alpha.get("/sheets").text.split('id="to_adjudicate"')[1].split("</table>")[0]
    assert 'class="flag-pending"' not in seen and "Bravo, Charlie" in seen and 'href="/sheets/gr_9001/adjudicate"' in seen


def test_double_reviews_that_skip_a_claim_are_complete_and_the_adjudicator_sees_the_skip(app, data: Path) -> None:
    sheet = _sheet(data, "ap_9001")
    added = claim(claim_id="ap_9001_c004", text="White should open the c-file with c4", weight=None, source=model_source())
    write_sheet(sheet=Sheet.model_validate({**sheet.model_dump(), "claims": [*(c.model_dump() for c in sheet.claims), added.model_dump()], "last_claim_number": 4}), path=data / "sheets" / "ap_9001.yaml")
    aram, lilit = _as(app, "Aram"), _as(app, "Lilit")
    assert aram.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Aram"), follow_redirects=False).status_code == 303
    assert aram.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    assert lilit.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001", name="Lilit", then="complete"), follow_redirects=False).status_code == 303
    stored = _sheet(data, "ap_9001")
    assert {name: stored.annotations[name].skipped for name in ("Aram", "Lilit")} == {"Aram": ["ap_9001_c004"], "Lilit": ["ap_9001_c004"]}
    assert resolve(sheet=stored, required=2).final and read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"
    page = _as(app, "GM Judge").get("/sheets/ap_9001/adjudicate")
    assert page.status_code == 200 and "skipped · Not key" in page.text
    assert "1 claim skipped and labelled by no completed review" in page.text
