import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.annotations import annotator_view
from bench.io_yaml import read_sheet, read_task
from review_app.app import create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from review_app.identity import ACCESS_COOKIE, access_token
from tests.review_login import TEST_ACCESS_SECRET, TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


@pytest.fixture
def client(data: Path) -> TestClient:
    return TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))


def _form(*, data: Path, task_id: str) -> dict[str, list[str] | str]:
    sheet = read_sheet(path=data / "sheets" / f"{task_id}.yaml")
    return {
        "claim_id": [c.claim_id for c in sheet.claims],
        "claim_text": [c.text for c in sheet.claims],
        "claim_weight": ["" if c.weight is None else str(c.weight) for c in sheet.claims],
        "claim_label": [c.latest_labels["GM Test"].value if "GM Test" in c.latest_labels else "" for c in sheet.claims],
        "notes": sheet.notes,
        "verdict": sheet.verdict or "",
    }


def test_an_access_link_signs_the_browser_in_and_goes_to_the_task_list(client: TestClient) -> None:
    token = access_token(secret=TEST_ACCESS_SECRET, name="Aram")
    page = client.get("/login").text
    assert "location.hash" in page and token not in page, "the token travels in the link's fragment, never in a request line"
    response = client.post("/login", data={"token": token}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/sheets"
    cookie = response.headers["set-cookie"]
    assert f"{ACCESS_COOKIE}={token}" in cookie and "httponly" in cookie.lower() and "Aram" not in cookie
    header = client.get("/sheets").text.split("<header>", 1)[1].split("</header>", 1)[0]
    assert "Signed in as" in header and "Aram" in header


def test_an_invalid_access_link_is_refused(client: TestClient) -> None:
    response = client.post("/login", data={"token": "not-a-token"}, follow_redirects=False)
    assert response.status_code == 403 and "set-cookie" not in response.headers


def test_a_name_outside_the_list_cannot_sign_in(client: TestClient, data: Path) -> None:
    stranger = access_token(secret=TEST_ACCESS_SECRET, name="Mallory")
    assert client.post("/login", data={"token": stranger}, follow_redirects=False).status_code == 403
    login(client, "Mallory")
    assert "Not signed in" in client.get("/sheets").text
    before = (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8")
    assert client.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001") | {"notes": "spoofed"}).status_code == 422
    assert (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8") == before


def test_a_typed_reviewer_name_is_ignored(client: TestClient, data: Path) -> None:
    assert client.post("/reviewer", data={"name": "Aram", "next": "/sheets"}).status_code in (404, 405)
    before = (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8")
    response = client.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001") | {"notes": "typed name", "reviewer": "Aram"})
    assert response.status_code == 422 and "access link" in response.text
    assert (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8") == before
    login(client, "Bella")
    assert client.post("/sheets/ap_9001/save", data=_form(data=data, task_id="ap_9001") | {"notes": "Bella's", "reviewer": "Aram"}, follow_redirects=False).status_code == 303
    assert set(read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations) == {"Bella"}


def test_logout_clears_the_sign_in(client: TestClient) -> None:
    login(client, "Aram")
    response = client.post("/logout", follow_redirects=False)
    assert response.status_code == 303 and ACCESS_COOKIE in response.headers["set-cookie"]
    client.cookies.clear()
    client.cookies.update(response.cookies)
    assert "Not signed in" in client.get("/sheets").text


def test_header_names_the_reviewer_not_the_task(client: TestClient) -> None:
    login(client, "Aram")
    html = client.get("/sheets/ap_9001").text
    header = html.split("<header>", 1)[1].split("</header>", 1)[0]
    assert "Aram" in header and "Sign ap_9001" not in html
    assert 'id="unsaved_dialog"' in html and 'id="reviewer_dialog"' in html


def test_save_goes_to_the_page_the_reviewer_was_leaving_for(client: TestClient, data: Path) -> None:
    login(client, "Aram")
    form = _form(data=data, task_id="ap_9001") | {"notes": "kept before leaving", "next": "/sheets/gr_9001"}
    response = client.post("/sheets/ap_9001/save", data=form, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/sheets/gr_9001"
    assert read_sheet(path=data / "sheets" / "ap_9001.yaml").annotations["Aram"].notes == "kept before leaving"


def test_save_and_complete_stores_the_edits_and_signs_them(client: TestClient, data: Path) -> None:
    login(client, "Aram")
    form = _form(data=data, task_id="ap_9001") | {"notes": "final words", "then": "complete"}
    response = client.post("/sheets/ap_9001/save", data=form, follow_redirects=False)
    assert response.status_code == 303
    view = annotator_view(sheet=read_sheet(path=data / "sheets" / "ap_9001.yaml"), annotator="Aram")
    assert view.notes == "final words" and view.author == "Aram" and view.signed_at is not None
    assert read_task(path=data / "tasks" / "ap_9001.yaml").status == "signed"


def test_save_and_complete_without_a_name_writes_nothing(client: TestClient, data: Path) -> None:
    before = (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8")
    form = _form(data=data, task_id="ap_9001") | {"notes": "should not land", "then": "complete"}
    assert client.post("/sheets/ap_9001/save", data=form).status_code == 422
    assert (data / "sheets" / "ap_9001.yaml").read_text(encoding="utf-8") == before


def test_a_cookie_behind_a_tls_proxy_is_secure_and_a_revoked_link_stops_working(data: Path, tmp_path: Path) -> None:
    from review_app.identity import Reviewers, revoke

    generations = tmp_path / "generations.json"
    reviewers = Reviewers(annotators=frozenset({"Aram", "Bella"}), adjudicators=frozenset(), secret=TEST_ACCESS_SECRET, generations_file=generations)
    client = TestClient(create_app(reviewers=reviewers, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}))
    old = reviewers.token(name="Aram")
    response = client.post("/login", data={"token": old}, headers={"x-forwarded-proto": "https"}, follow_redirects=False)
    assert "secure" in response.headers["set-cookie"].lower()
    assert reviewers.links(base_url="https://r.example/")["Aram"] == f"https://r.example/login#{old}"
    bella = reviewers.token(name="Bella")
    revoke(path=generations, name="Aram")
    assert reviewers.name_for(token=old) is None and reviewers.name_for(token=bella) == "Bella"
    assert reviewers.name_for(token=reviewers.token(name="Aram")) == "Aram"
    assert client.post("/login", data={"token": old}, follow_redirects=False).status_code == 403
