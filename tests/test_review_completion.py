import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.annotations import resolve
from bench.io_yaml import read_sheet, read_task, write_sheet
from bench.models import ClaimDrop, Sheet
from bench.validate import errors_of, validate_all, warnings_of
from review_app.app import create_app
from review_app.completion import UNCOMMENTED_NOTE, numbered, shown_claims
from tests.claim_fakes import LABELLED_AT, TEST_TOKEN_SECRET, claim, model_source, page_token
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
PLAN = "White should open the c-file with c4"
DROPPED = "The knight belongs on f5"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """ap_9001 with a grandmaster's claim, two engine claims, and three model claims: one shown, one set aside (dropped) and one
    folded into it as a duplicate."""
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    path = tmp_path / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    shown = claim(claim_id="ap_9001_c004", text=PLAN, weight=None, source=model_source())
    dropped = claim(claim_id="ap_9001_c005", text=DROPPED, weight=None, source=model_source()).model_copy(update={"dropped": ClaimDrop(annotator="cap", comment="cap", at=LABELLED_AT)})
    duplicate = claim(claim_id="ap_9001_c006", text=f"{PLAN}.", weight=None, source=model_source(), duplicate_of="ap_9001_c004")
    write_sheet(sheet=Sheet.model_validate({**sheet.model_dump(), "claims": [c.model_dump() for c in [*sheet.claims, shown, dropped, duplicate]], "last_claim_number": 6}), path=path)
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=TEST_TOKEN_SECRET, tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    return login(TestClient(app), "Aram")


def _card_number(data: Path, claim_id: str) -> int:
    shown = shown_claims(sheet=read_sheet(path=data / "sheets" / "ap_9001.yaml"))
    return [c.claim_id for c in shown].index(claim_id) + 1


def _label(client: TestClient, *, claim_id: str, text: str, value: str, comment: str = "", then: str = "", weight: str = "") -> int:
    """A save that labels one claim, as the page posts it (a model claim by its token); the other claims are left as they are."""
    name = page_token(claim_id) if claim_id.endswith(("c004", "c005", "c006")) else claim_id
    form = {"claim_id": name, "claim_text": text, "claim_weight": weight, "claim_weight_was": "", "claim_label": value, "claim_comment": comment, "notes": "", "reviewer": "Aram", "then": then}
    return client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code


def _card(html: str, claim_id: str) -> str:
    return html.split(f'data-claim-id="{page_token(claim_id)}"', 1)[1].split('class="claim-card', 1)[0]


def test_complete_records_unlabelled_model_claims_as_skipped_without_a_word_to_the_grandmaster(client: TestClient, data: Path) -> None:
    done = client.post("/sheets/ap_9001/complete", data={"reviewer": "Aram"}, follow_redirects=False)
    assert done.status_code == 303
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].skipped == ["ap_9001_c004"], "the set-aside and the duplicate claim are not skipped: they have no card"
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed", "a completion with skips is a completed review"
    assert "skipped" not in client.get("/sheets/ap_9001").text


def test_completing_again_after_labelling_a_skipped_claim_takes_it_off_the_list(client: TestClient, data: Path) -> None:
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="", then="complete") == 303
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].skipped == ["ap_9001_c004"]
    assert client.post("/sheets/ap_9001/reopen", data={"reviewer": "Aram"}, follow_redirects=False).status_code == 303
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].completed_at is None
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="unclear", then="complete") == 303
    record = read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"]
    assert record.completed_at is not None and record.skipped == []
    assert "skipped" not in (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8")


def test_skipping_never_lets_a_partly_or_false_label_go_without_a_comment(client: TestClient, data: Path) -> None:
    engine = read_sheet(path=data / "sheets" / "ap_9001.yaml").claims[2]
    assert _label(client, claim_id=engine.claim_id, text=engine.text, value="false") == 303
    refused = client.post("/sheets/ap_9001/complete", data={"reviewer": "Aram"})
    assert refused.status_code == 422
    assert f"Partly and False need a comment: claim {_card_number(data, engine.claim_id)} has none yet." in refused.text
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].completed_at is None


def test_another_reviewers_labels_do_not_count(client: TestClient, data: Path) -> None:
    login(client, "Bella")
    form = {"claim_id": page_token("ap_9001_c004"), "claim_text": PLAN, "claim_weight": "", "claim_weight_was": "", "claim_label": "true", "claim_comment": "", "notes": "", "reviewer": "Bella"}
    assert client.post("/sheets/ap_9001/save", data=form, follow_redirects=False).status_code == 303
    client.post("/sheets/ap_9001/unlock", data={"reviewer": "Bella"})
    login(client, "Aram")
    client.post("/sheets/ap_9001/complete", data={"reviewer": "Aram"})
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].skipped == ["ap_9001_c004"]


def test_partly_and_false_save_without_a_comment_but_complete_needs_one(client: TestClient, data: Path) -> None:
    engine = read_sheet(path=data / "sheets" / "ap_9001.yaml").claims[2]
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="false") == 303
    assert _label(client, claim_id=engine.claim_id, text=engine.text, value="partly_true") == 303, "a save keeps unfinished work"
    refused = client.post("/sheets/ap_9001/save", data={"claim_id": [], "notes": "", "reviewer": "Aram", "then": "complete"})
    numbers = sorted([_card_number(data, "ap_9001_c004"), _card_number(data, engine.claim_id)])
    assert refused.status_code == 422
    assert f"Partly and False need a comment: {numbered(numbers=numbers)} have none yet." in refused.text
    assert UNCOMMENTED_NOTE in _card(refused.text, "ap_9001_c004")
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="false", comment="c4 is not playable yet") == 303
    assert _label(client, claim_id=engine.claim_id, text=engine.text, value="partly_true", comment="only slightly") == 303
    assert client.post("/sheets/ap_9001/complete", data={"reviewer": "Aram"}, follow_redirects=False).status_code == 303, "a key of two claims is no concern of the grandmaster's"
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"


def test_the_card_shows_the_missing_comment_and_the_page_script_keeps_it_in_step(client: TestClient) -> None:
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="false") == 303
    html = client.get("/sheets/ap_9001").text
    assert 'class="card-note card-warning comment-needed" hidden' in _card(html, "ap_9001_c004")
    assert "function syncComment(card)" in html and 'label === "partly_true" || label === "false"' in html


def test_the_list_shows_each_reviewers_own_progress(client: TestClient, data: Path) -> None:
    def progress() -> str:
        row = client.get("/sheets").text.split('href="/sheets/ap_9001"', 1)[1].split("</tr>", 1)[0]
        return re.sub(r"<[^>]+>", "", row.rsplit("<td>", 1)[1]).strip()

    assert progress() == "not started"
    engine = read_sheet(path=data / "sheets" / "ap_9001.yaml").claims[2]
    assert _label(client, claim_id=engine.claim_id, text=engine.text, value="true") == 303
    assert progress() == "0/1 labelled"
    assert _label(client, claim_id="ap_9001_c004", text=PLAN, value="true", then="complete") == 303
    assert progress() == "complete"
    login(client, "Bella")
    assert progress() == "complete"


def test_numbered_names_the_claims_in_plain_english() -> None:
    assert [numbered(numbers=n) for n in ([3], [3, 7], [3, 7, 12])] == ["claim 3", "claims 3 and 7", "claims 3, 7 and 12"]


def test_the_grandmaster_sees_no_check_but_the_comment_rule_and_completes_with_only_warnings_left(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    compound = "White breaks with 15.b4 cxb4 16.bxc5. White is clearly better."
    extra = [claim(claim_id=f"ap_9001_c{n:03d}", text=compound if n == 7 else f"Engine point {n}", weight=None) for n in range(7, 15)]
    write_sheet(sheet=Sheet.model_validate({**sheet.model_dump(), "claims": [*(c.model_dump() for c in sheet.claims), *(c.model_dump() for c in extra)], "last_claim_number": 14}), path=path)
    view = client.get("/sheets/ap_9001").text
    assert "claims in the key" not in view and "more than one thing" not in view and 'class="warning-compact"' not in view
    for c in extra:
        assert _label(client, claim_id=c.claim_id, text=c.text, value="true", weight="2") == 303
    saved = client.get("/sheets/ap_9001").text
    assert "claims in the key" not in saved and "more than one thing" not in saved and "card-warning\">" not in saved
    assert client.post("/sheets/ap_9001/complete", data={"reviewer": "Aram"}, follow_redirects=False).status_code == 303
    resolved = resolve(sheet=read_sheet(path=path), required=1)
    assert len(resolved.sheet.expert_points) > 8
    problems = validate_all(tasks_dir=data / "tasks", sheets_dir=data / "sheets")
    assert errors_of(problems=problems) == [], "a grandmaster's completion never fails validation"
    warned = [p.message for p in warnings_of(problems=problems)]
    assert any("claim ap_9001_c007 is compound" in m for m in warned) and not any("weighted claim" in m for m in warned)
