"""Similar claims sit together: the app and the workbook show claims by topic (evaluation, best move, lines and tactics, plans,
other), engine claims first in each topic, model claims in a hash order that never groups them by model, grandmasters' claims
last; each topic under a heading that names no source."""

import hashlib
import re
import shutil
from html import unescape
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from bench.claim_display import TOPICS, claim_topic, shown_numbers, shown_order
from bench.config import load_bench
from bench.io_yaml import read_sheet, write_sheet
from bench.models import ClaimSource, ReferenceClaim
from review_app.app import create_app
from tests.claim_fakes import ENGINE, TEST_TOKEN_SECRET, claim, page_token
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ("sample_coach", "claude-opus-5-5", "openai/gpt-5", "google/gemini-3-pro")
FORBIDDEN = ("sample_coach", "claude", "gpt", "gemini", "openai", "google")
GM = ClaimSource(type="gm", name="GM Test", answer_ref=None)
# (text, kind, topic, structured fields): the fixture's own claims are c001 (gm), c002 (engine best_move), c003 (engine eval_band).
POOLED: tuple[tuple[str, str | None, str, dict[str, str]], ...] = (
    ("Black is slightly better here", None, "evaluation", {}),
    ("White keeps an edge after the exchange", "other", "other", {}),
    ("12.Nf3 Bg4 13.h3 keeps the balance of tension", None, "lines", {}),
    ("Be3 is the strongest move", "best_move", "best_move", {"move": "Be3", "about": "task_position"}),
    ("The knight belongs on c4", None, "other", {}),
    ("Nf5 threatens the queen", "threat", "lines", {"move": "Nf5", "about": "task_position"}),
    ("White should castle long and storm", "plan", "plans", {}),
    ("Black is clearly better", "eval_band", "evaluation", {"band": "black_better", "about": "task_position"}),
    ("The only move is Be3", None, "best_move", {}),
    ("Black's plan is ...b5 and ...c4", "plan", "plans", {}),
    ("Be3 Qd7 Qd2 keeps the balance", "line_outcome", "lines", {"line": "Be3 Qd7 Qd2", "outcome": "equalises", "about": "task_position"}),
    ("The position is equal", None, "evaluation", {}),
)

def _model(n: int) -> ClaimSource:
    model = MODELS[(n - 4) // 3]
    return ClaimSource(type="model", name=model, answer_ref=f"{model}/20261105_160110/ap_9001/0")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """ap_9001 with its three claims and twelve model claims, three per model, stored grouped by answer."""
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    path = tmp_path / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    pooled = [claim(claim_id=f"ap_9001_c{n:03d}", text=text, weight=None, source=_model(n), kind=kind, **fields) for n, (text, kind, _, fields) in enumerate(POOLED, start=4)]  # type: ignore[arg-type]
    write_sheet(sheet=sheet.model_copy(update={"claims": [*sheet.claims, *pooled], "last_claim_number": 3 + len(pooled)}), path=path)
    return tmp_path


def _client(data: Path) -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=TEST_TOKEN_SECRET, tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    client = TestClient(app)
    login(client, "GM Test")
    return client


def _real(name: str) -> str:
    known = {page_token(f"ap_9001_c{n:03d}"): f"ap_9001_c{n:03d}" for n in range(4, 4 + len(POOLED))}
    return known.get(name, name)


def _list(page: str) -> str:
    return page.split('id="claim_list"', 1)[1].split('<template id="claim_template">', 1)[0]


def _sections(page: str) -> list[tuple[str, list[str]]]:
    """Each heading of the claim list with the claim ids of the cards under it, in page order."""
    sections: list[tuple[str, list[str]]] = []
    for heading, card in re.findall(r'<h4 class="claim-group">([^<]*)</h4>|data-claim-id="([^"]+)"', _list(page)):
        if heading:
            sections.append((unescape(heading), []))
        else:
            sections[-1][1].append(_real(card))
    return sections


def _topic(claim_id: str) -> str:
    n = int(claim_id[-3:])
    return {1: "added", 2: "best_move", 3: "evaluation"}.get(n) or POOLED[n - 4][2]


def test_a_claim_takes_its_topic_from_its_kind_its_fields_or_its_words() -> None:
    def topic(text: str, kind: str | None = None, source: ClaimSource = ENGINE, **fields: str) -> str:
        return claim_topic(claim=claim(claim_id="ap_9001_c001", text=text, weight=None, kind=kind, source=source, **fields))  # type: ignore[arg-type]

    assert [topic("x", "plan"), topic("x", "other")] == ["plans", "other"]
    assert [topic("x", "eval_band", band="equal", about="task_position"), topic("x", "best_move", move="Be3", about="task_position"), topic("x", "line_outcome", line="Be3 Qd7", outcome="equalises", about="task_position")] == ["evaluation", "best_move", "lines"]
    assert [topic(t) for t in ("White is winning", "Black is better", "It is equal", "Kg7 is a blunder", "White is +1.5")] == ["evaluation"] * 5
    assert [topic(t) for t in ("12.Nf3 Bg4 follows", "After Be3 Qd7 Qd2 White castles", "the only move is Kh7", "The knight belongs on c4")] == ["lines", "lines", "best_move", "other"]
    assert topic("White is winning", "plan", source=GM) == "added", "a grandmaster's claim stays last"


def test_the_page_shows_each_topic_under_its_heading_in_order_with_grandmasters_claims_last(data: Path) -> None:
    sections = _sections(_client(data).get("/sheets/ap_9001").text)
    assert [heading for heading, _ in sections] == list(TOPICS.values())
    for heading, ids in sections:
        assert ids and all(TOPICS[_topic(i)] == heading for i in ids), heading
    best_move = dict(sections)["Best move"]
    assert best_move[0] == "ap_9001_c002", "the engine's claim heads its topic"
    assert dict(sections)["Added by reviewers"] == ["ap_9001_c001"]


def test_the_order_is_stable_and_within_a_topic_ignores_the_source(data: Path) -> None:
    client = _client(data)
    first = _sections(client.get("/sheets/ap_9001").text)
    assert all(_sections(client.get("/sheets/ap_9001").text) == first for _ in range(3)), "the same order on every load"
    claims = read_sheet(path=data / "sheets" / "ap_9001.yaml").claims

    def renamed(claim: ReferenceClaim, name: str) -> ReferenceClaim:
        return claim.model_copy(update={"source": ClaimSource(type="model", name=name, answer_ref=None)}) if claim.source.type == "model" else claim

    order = [c.claim_id for c in shown_order(claims=claims)]
    assert [c.claim_id for c in shown_order(claims=[renamed(c, "one model") for c in claims])] == order, "which model a claim came from never moves it"
    stored = [c.claim_id for c in claims if c.source.type == "model"]
    for _, ids in first:
        models = [i for i in ids if i in stored]
        assert models == sorted(models, key=lambda i: hashlib.sha256(i.encode()).hexdigest()), "model claims in hash order, not by answer"


def test_the_headings_name_no_source_and_card_numbers_and_errors_follow_the_shown_order(data: Path) -> None:
    client = _client(data)
    page = client.get("/sheets/ap_9001").text
    headings = " ".join(re.findall(r'<h4 class="claim-group">([^<]*)</h4>', page)).lower()
    assert not [word for word in FORBIDDEN if word in headings] and "analysis" not in headings
    assert 'class="claim-group claim-card' not in page and "<h4 class=\"claim-card" not in page, "a heading is not a card: the keys skip it"
    shown = [i for _, ids in _sections(page) for i in ids]
    numbers = [int(n) for n in re.findall(r'<span class="claim-number">(\d+)</span>', _list(page))]
    assert numbers == list(range(1, len(shown) + 1))
    assert shown_numbers(claims=read_sheet(path=data / "sheets" / "ap_9001.yaml").claims) == {claim_id: n for n, claim_id in enumerate(shown, start=1)}
    cards = _list(page)
    form: dict[str, list[str]] = {name: [unescape(v) for v in re.findall(rf'name="{name}" value="([^"]*)"', cards)] for name in ("claim_id", "claim_label", "claim_weight", "claim_weight_was")}
    form["claim_text"] = [unescape(v) for v in re.findall(r'<textarea name="claim_text"[^>]*>(.*?)</textarea>', cards, re.DOTALL)]
    form["claim_comment"] = [unescape(v) for v in re.findall(r'<textarea name="claim_comment"[^>]*>(.*?)</textarea>', cards, re.DOTALL)]
    target = "ap_9001_c010"
    form["claim_weight"][form["claim_id"].index(page_token(target))] = "2"
    refused = client.post("/sheets/ap_9001/save", data={**form, "reviewer": ["GM Test"]})
    summary = refused.text.split('id="error_summary"', 1)[1].split("</div>", 1)[0]
    assert refused.status_code == 422 and f"Claim {shown.index(target) + 1}: This claim is in the answer key but isn&#39;t labelled True" in summary, summary




def test_a_grandmasters_card_says_only_analysis_and_the_page_has_the_short_hint(data: Path) -> None:
    sheet = read_sheet(path=data / "sheets" / "ap_9001.yaml")
    duplicate = claim(claim_id="ap_9001_c016", text="Be3 is best", weight=None, source=_model(4), kind="plan", duplicate_of="ap_9001_c010")
    write_sheet(sheet=sheet.model_copy(update={"claims": [*sheet.claims, duplicate], "last_claim_number": 16}), path=data / "sheets" / "ap_9001.yaml")
    page = _client(data).get("/sheets/ap_9001").text
    assert "checked by program" not in page and "+1 similar" not in page, "the count of duplicates is for the adjudicator only"
    sources = re.findall(r'<div class="claim-source">([^<]*)</div>', _list(page))
    assert set(sources) == {"analysis", "", "reviewer (GM Test)"}
    hint = page.split('id="review_hint">', 1)[1].split("</p>", 1)[0]
    assert "Label every claim" in hint and "Save (s)" in hint and 'href="/sheets#how_to_review"' in hint
    assert "The label and comment shown are your own latest ones" not in page


def test_the_task_list_has_a_how_to_review_box_open_on_the_first_visit_and_remembered(data: Path) -> None:
    page = _client(data).get("/sheets").text
    box = page.split('<details class="how-to-review" id="how_to_review" open>', 1)[1].split("</details>", 1)[0]
    assert page.index('id="how_to_review"') < page.index("<h2>"), "near the top, above the task tables"
    for rule in ("A wrong claim is False", "Nothing is ever erased", "count as True unless you change them", "only True claims can be in it", "no other reviewer sees your labels", "The verdict is never chosen for you"):
        assert rule in box, rule
    assert box.count("<li>") <= 10 and "engine" not in box.lower()
    script = page.split('const KEY = "review_how_to_open";', 1)[1].split("</script>", 1)[0]
    assert script.count("try {") == 2 and "localStorage.getItem(KEY)" in script and "localStorage.setItem(KEY" in script


def test_the_new_styles_use_the_theme_variables_only() -> None:
    css = (Path(__file__).parent.parent / "review_app" / "static" / "review.css").read_text(encoding="utf-8")
    rules = [line for line in css.splitlines() if line.startswith((".claim-group", ".how-to-review", ".review-hint"))]
    assert rules and not [line for line in rules if re.search(r"#[0-9a-fA-F]{3,6}\b|rgba?\(", line)]
