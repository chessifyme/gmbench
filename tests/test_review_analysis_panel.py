"""The analysis panel keeps a fixed height while the engine updates it, so the board and the page never move
(the browser check is tests/test_review_browser.py::test_fast_analysis_updates_never_move_the_board_or_the_page)."""

import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from review_app.app import STATIC, create_app
from review_app.live_engine import LINES_SHOWN
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
CSS = re.sub(r"/\*.*?\*/", "", (STATIC / "review.css").read_text(encoding="utf-8"), flags=re.DOTALL)
NARROW = "@media (max-width: 900px)"


def _rules(css: str) -> dict[str, str]:
    """Selector -> declarations of the top-level rules (a later rule for the same selector adds to the earlier one)."""
    rules: dict[str, str] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        for one in selector.split(","):
            key = one.strip()
            rules[key] = rules.get(key, "") + ";" + body
    return rules


def _narrow_css() -> str:
    start = CSS.index(NARROW)
    depth, end = 0, start
    for end in range(CSS.index("{", start), len(CSS)):
        depth += {"{": 1, "}": -1}.get(CSS[end], 0)
        if depth == 0:
            break
    return CSS[CSS.index("{", start) + 1 : end]


WIDE = _rules(CSS.replace(_narrow_css(), ""))


def _has(selector: str, *declarations: str, rules: dict[str, str] = WIDE) -> None:
    body = re.sub(r"\s+", " ", rules.get(selector, ""))
    for declaration in declarations:
        assert declaration in body, f"{selector} has no `{declaration}`: {body!r}"


def test_the_panel_reserves_room_for_every_line_the_engine_sends(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    html = login(TestClient(app), "Aram").get("/sheets/ap_9001").text
    assert f'class="analysis" style="--analysis-lines: {LINES_SHOWN}"' in html
    body = html.split('class="analysis-body"', 1)[1].split("</div>", 1)[0]
    assert 'id="analysis_out"' in body and 'class="analysis-placeholder"' in body, "the idle text sits in the lines' own reserved box"


def test_each_line_is_two_rows_high_and_a_longer_one_is_clipped_with_an_ellipsis() -> None:
    _has(".analysis-line", "height: var(--analysis-line-h)")
    _has(".analysis", "--analysis-line-h: calc(2 * var(--analysis-row)")
    _has(".analysis-moves", "-webkit-line-clamp: 2", "overflow: hidden", "max-height: calc(2 * var(--analysis-row))")
    _has(".analysis-line.is-open .analysis-line-body", "position: absolute", "height: auto")
    _has(".analysis-line.is-open .analysis-moves", "-webkit-line-clamp: unset", "overflow: visible")


def test_the_panel_keeps_its_height_when_idle_running_or_finished() -> None:
    _has(".analysis-body", "height: calc(var(--analysis-lines, 3) * var(--analysis-line-h) + 2px)")
    _has(".analysis-lines", "height: 100%")
    _has(".analysis-stats", "flex-wrap: nowrap", "height: 24px", "overflow: hidden")
    _has(".analysis-stats[hidden]", "visibility: hidden")
    _has(".analysis-message", "height: 38px", "overflow: hidden")
    _has(".analysis-message[hidden]", "visibility: hidden")
    _has(".board-move", "height: 2.9em", "overflow: hidden")


def test_live_parts_are_never_scroll_anchors() -> None:
    for selector in (".board-pane", ".analysis", ".analysis *"):
        _has(selector, "overflow-anchor: none")


def test_the_board_column_is_sticky_on_wide_screens_and_stacked_on_narrow_ones() -> None:
    _has(".board-pane", "position: sticky", "top: 12px")
    narrow = _rules(_narrow_css())
    _has(".board-pane", "position: static", rules=narrow)
    _has(".cols", "grid-template-columns: minmax(0, 1fr)", rules=narrow)


def test_the_start_stop_toggle_is_as_small_as_the_state_pill_and_keeps_its_width() -> None:
    _has(".analysis-state", "height: 24px")
    _has(".analysis-toggle", "height: 24px", "min-width: 78px")
    _has(".analysis-toggle:focus-visible", "outline: 2px solid")
    _has(".analysis-toggle[data-running=\"true\"]", "background: var(--bad)")
    assert ".analysis-actions" not in WIDE, "the old full-width button row is gone"
