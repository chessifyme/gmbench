"""The weight logic of the sheet page, run in node against a stub card: a misclick on a non-true label and back must not
lose the weight the reviewer was shown."""

import json
import shutil
import subprocess
from pathlib import Path
from tests.review_login import TEST_REVIEWERS

import pytest

TEMPLATE = Path(__file__).parent.parent / "review_app" / "templates" / "sheet_edit.html"
NODE = shutil.which("node")

STUB = """
function stubCard(label, weight, heldTrue) {
  const inputs = { claim_label: { value: label }, claim_weight: { value: weight } };
  const group = { classList: { toggle: function () {} }, querySelectorAll: function () { return []; }, remove: function () {} };
  return {
    dataset: { heldTrue: heldTrue },
    querySelector: function (selector) {
      const name = /name="([^"]+)"/.exec(selector);
      return name ? inputs[name[1]] : group;
    },
    inputs: inputs,
  };
}
"""


def _functions() -> str:
    script = TEMPLATE.read_text(encoding="utf-8")
    start = script.index("  function press(")
    end = script.index("  function addClaim(")
    return script[start:end]


def _run(steps: str) -> dict[str, str]:
    program = STUB + _functions() + steps
    out = subprocess.run([str(NODE), "-e", program], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_misclick_on_false_and_back_to_true_keeps_the_shown_weight() -> None:
    result = _run("""
const card = stubCard("true", "2", "1");
syncWeight(card, false);
card.inputs.claim_label.value = "false"; syncWeight(card, true);
const cleared = card.inputs.claim_weight.value;
card.inputs.claim_label.value = "true"; syncWeight(card, true);
console.log(JSON.stringify({ cleared: cleared, restored: card.inputs.claim_weight.value }));
""")
    assert result == {"cleared": "", "restored": "2"}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_loading_the_page_never_changes_a_weight() -> None:
    result = _run("""
const card = stubCard("", "2", "0");
syncWeight(card, false);
console.log(JSON.stringify({ weight: card.inputs.claim_weight.value }));
""")
    assert result == {"weight": "2"}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_rendered_page_script_parses_and_shows_the_key_hint(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from review_app.app import create_app
    from tests.fake_uci_engine_settings import TEST_ENGINE

    fixtures = Path(__file__).parent / "fixtures"
    shutil.copytree(fixtures / "tasks", tmp_path / "tasks")
    shutil.copytree(fixtures / "sheets", tmp_path / "sheets")
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    html = TestClient(app).get("/sheets/ap_9001").text
    assert 'id="key_hint"' in html and "1 True" in html and "4 Unclear" in html and "Off-topic" not in html
    assert "mark_rest_true" not in html and "Mark the rest" not in html and "markRestTrue" not in html and 'event.key === "t"' not in html, "every claim is labelled deliberately"
    script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    (tmp_path / "page.js").write_text(script, encoding="utf-8")
    subprocess.run([str(NODE), "--check", str(tmp_path / "page.js")], check=True, capture_output=True)


def test_a_set_aside_claim_has_no_card_for_the_page_script_to_skip(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from bench.io_yaml import read_sheet, write_sheet
    from bench.models import drop_claim
    from review_app.app import create_app
    from tests.claim_fakes import LABELLED_AT
    from tests.fake_uci_engine_settings import TEST_ENGINE

    fixtures = Path(__file__).parent / "fixtures"
    shutil.copytree(fixtures / "tasks", tmp_path / "tasks")
    shutil.copytree(fixtures / "sheets", tmp_path / "sheets")
    path = tmp_path / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    dropped = drop_claim(claim=sheet.claims[0], annotator="cleanup", comment="", at=LABELLED_AT, how_to_name="n")
    write_sheet(sheet=sheet.model_copy(update={"claims": [dropped, *sheet.claims[1:]]}), path=path)
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    html = TestClient(app).get("/sheets/ap_9001").text
    assert 'data-claim-id="ap_9001_c001"' not in html and dropped.text not in html
    script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
    assert "dropped" not in script and "set-aside" not in script
    if NODE is not None:
        (tmp_path / "page.js").write_text(script, encoding="utf-8")
        subprocess.run([str(NODE), "--check", str(tmp_path / "page.js")], check=True, capture_output=True)
