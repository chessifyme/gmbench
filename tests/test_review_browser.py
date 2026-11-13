import re
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from bench.io_yaml import read_sheet, write_sheet
from review_app.app import create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from review_app.identity import ACCESS_COOKIE, access_token
from tests.review_login import TEST_ACCESS_SECRET, TEST_REVIEWERS

sync_api = pytest.importorskip("playwright.sync_api")

FIXTURES = Path(__file__).parent / "fixtures"
LINE = "6. Be2 e5 7. Nb3 Be7"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    path = tmp_path / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    write_sheet(sheet=sheet.model_copy(update={"claims": [sheet.claims[0].model_copy(update={"text": f"{LINE} is the main line."}), *sheet.claims[1:]]}), path=path)
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise TimeoutError("the review app did not start within 10 seconds")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_a_claim_line_plays_on_the_board_and_steps_back_to_the_task(base_url: str) -> None:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(f"{base_url}/sheets/ap_9001")
        label, analysis = page.locator("#board_move"), page.locator("#analysis_position")
        expect = sync_api.expect
        expect(label).to_have_text("Task position · move 6, White to move")
        page.locator("a.claim-line", has_text=LINE).click()
        expect(label).to_have_text("after 6.Be2")
        expect(analysis).to_have_text("after 6.Be2")
        expect(page.locator("#board_side_line")).to_be_visible()
        page.click("#board_forward")
        expect(label).to_have_text("after 6...e5")
        page.click("#board_back")
        page.click("#board_back")
        expect(label).to_have_text("Task position · move 6, White to move")
        page.click("#board_end")
        expect(label).to_have_text("after 7...Be7")
        page.click("#board_task")
        expect(label).to_have_text("Task position · move 6, White to move")
        expect(page.locator("#board_side_line")).to_be_hidden()
        browser.close()


def test_the_keys_label_key_save_and_move_on_and_t_labels_nothing(base_url: str) -> None:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        context.add_cookies([{"name": ACCESS_COOKIE, "value": access_token(secret=TEST_ACCESS_SECRET, name="Aram"), "url": base_url}])
        page = context.new_page()
        dialogs: list[str] = []
        page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        page.goto(f"{base_url}/sheets/ap_9001")
        cards = page.locator("#claim_list .claim-card")
        expect = sync_api.expect

        def pressed(index: int, group: str) -> str:
            return cards.nth(index).locator(f".seg-{group} [aria-pressed='true']").inner_text()

        for key in ("j", "3", "j", "1", "i", "j", "k"):
            page.keyboard.press(key)
        expect(cards.nth(1)).to_have_class(re.compile(r"\bcurrent\b"))
        page.keyboard.press("t")
        assert dialogs == [] and cards.nth(2).locator('input[name="claim_label"]').input_value() == "", "t labels nothing"
        page.keyboard.press("4")
        page.keyboard.press("j")
        page.keyboard.press("2")
        with page.expect_navigation():
            page.keyboard.press("s")
        expect(page.locator("#dirty_note")).to_be_hidden()
        assert [pressed(0, "label"), pressed(1, "label"), pressed(2, "label")] == ["✗ False", "? Unclear", "◐ Partly"]
        page.keyboard.press("j")
        page.keyboard.press("1")
        page.keyboard.press("e")
        assert pressed(0, "weight") == "Essential"
        with page.expect_navigation():
            page.keyboard.press("s")
        assert pressed(0, "label") == "✓ True" and pressed(0, "weight") == "Essential"
        next_task = page.locator("#next_link").get_attribute("href")
        with page.expect_navigation(url=f"{base_url}{next_task}"):
            page.keyboard.press("n")
        assert dialogs == []
        browser.close()


def test_a_lost_lock_keeps_the_unsaved_work_on_the_page_and_in_a_draft_offered_on_the_next_open(base_url: str) -> None:
    """Locks are per reviewer, so the server refuses a renewal only in a fault; the page is checked against a refused renewal
    (409) put in by the test: the banner shows, nothing reloads, and the unsaved work stays on the page and in a draft."""
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        aram = browser.new_context()
        aram.add_cookies([{"name": ACCESS_COOKIE, "value": access_token(secret=TEST_ACCESS_SECRET, name="Aram"), "url": base_url}])
        page = aram.new_page()
        page.clock.install()
        page.goto(f"{base_url}/sheets/ap_9001")
        expect = sync_api.expect
        cards = page.locator("#claim_list .claim-card")
        page.keyboard.press("j")
        page.keyboard.press("3")
        expect(cards.nth(0).locator(".comment-needed")).to_be_visible()
        cards.nth(0).locator('textarea[name="claim_comment"]').fill("not the plan here")
        expect(cards.nth(0).locator(".comment-needed")).to_be_hidden()
        refuse = lambda route: route.fulfill(status=409, body="held")  # noqa: E731
        page.route("**/sheets/ap_9001/lock", refuse)
        page.clock.fast_forward(31_000)
        expect(page.locator("#lock_lost_notice")).to_be_visible()
        assert page.url == f"{base_url}/sheets/ap_9001", "the page is never reloaded"
        assert cards.nth(0).locator('input[name="claim_label"]').input_value() == "false"
        assert cards.nth(0).locator('textarea[name="claim_comment"]').input_value() == "not the plan here"
        page.unroute("**/sheets/ap_9001/lock", refuse)
        page.clock.fast_forward(31_000)
        expect(page.locator("#lock_lost_notice")).to_be_hidden()
        page.close()
        again = aram.new_page()
        again.goto(f"{base_url}/sheets/ap_9001")
        expect(again.locator("#draft_notice")).to_be_visible()
        assert again.locator('#claim_list .claim-card input[name="claim_label"]').first.input_value() == ""
        again.locator("#draft_restore").click()
        restored = again.locator("#claim_list .claim-card").nth(0)
        assert restored.locator('input[name="claim_label"]').input_value() == "false"
        assert restored.locator('textarea[name="claim_comment"]').input_value() == "not the plan here"
        expect(again.locator("#dirty_note")).to_be_visible()
        with again.expect_navigation():
            again.keyboard.press("s")
        expect(again.locator("#draft_notice")).to_be_hidden()
        again.reload()
        expect(again.locator("#draft_notice")).to_be_hidden()
        browser.close()


# Stands in for the engine stream: an info event every 80 ms whose lines change length (2 to 16 plies) and count (1 to 3),
# so a panel sized by its content would grow and shrink several times a second.
FAST_ANALYSIS = r"""
(function () {
  const real = window.fetch;
  const SAN = ["Be3", "e5", "Nb3", "Be7", "f3", "O-O", "Qd2", "Nbd7", "g4", "b5", "O-O-O", "Bb7", "h4", "Rc8", "Kb1", "Nb6"];
  let tick = 0;
  function line(rank) {
    const plies = 2 + ((tick * 5 + rank * 7) % 15);
    const moves = SAN.slice(0, plies).map(function (san, i) { return { number: i % 2 === 0 ? (6 + i / 2) + "." : "", san: san }; });
    return { rank: rank, score: "+0." + rank, moves: moves };
  }
  window.fetch = function (url, options) {
    if (typeof url !== "string" || !url.startsWith("/analysis/stream")) return real.apply(this, arguments);
    const encoder = new TextEncoder();
    let timer = null;
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode("event: start\ndata: {}\n\n"));
        timer = setInterval(function () {
          tick += 1;
          const lines = [];
          for (let rank = 1; rank <= 1 + (tick % 3); rank++) lines.push(line(rank));
          const info = { depth: tick, seldepth: tick + 4, nodes: tick * 98765432, mnps: tick % 2 ? 1.5 : 123.25, seconds: tick * 0.08, lines: lines };
          controller.enqueue(encoder.encode("event: info\ndata: " + JSON.stringify(info) + "\n\n"));
        }, 80);
        options.signal.addEventListener("abort", function () { clearInterval(timer); controller.error(new DOMException("stopped", "AbortError")); });
      },
      cancel() { clearInterval(timer); },
    });
    return Promise.resolve(new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }));
  };
})();
"""
WHERE = """() => { const board = document.getElementById("board").getBoundingClientRect();
  return { top: board.top, left: board.left, size: board.width, scrollY: window.scrollY, page: document.documentElement.scrollHeight,
           panel: document.getElementById("analysis_panel").getBoundingClientRect().height }; }"""
ROW_TOPS = "() => Array.from(document.querySelectorAll('.analysis-line')).map((row) => row.getBoundingClientRect().top)"
OPEN_IS_TALLER = """() => { const row = document.querySelector('.analysis-line.is-open');
  return row !== null && row.querySelector('.analysis-line-body').getBoundingClientRect().height > row.getBoundingClientRect().height + 5; }"""


def test_a_save_from_a_second_tab_after_mark_as_complete_shows_why_in_view_and_keeps_the_changes_for_reopen(base_url: str) -> None:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        aram = browser.new_context()
        aram.add_cookies([{"name": ACCESS_COOKIE, "value": access_token(secret=TEST_ACCESS_SECRET, name="Aram"), "url": base_url}])
        first, second = aram.new_page(), aram.new_page()
        first.goto(f"{base_url}/sheets/ap_9001")
        second.goto(f"{base_url}/sheets/ap_9001")
        expect = sync_api.expect
        with first.expect_navigation():
            first.click("#complete_button")
        second.keyboard.press("j")
        second.keyboard.press("1")
        with second.expect_navigation():
            second.click("#save_button")
        refusal = second.locator("#error_summary")
        expect(refusal).to_have_text("you marked this task complete: press Reopen to change it")
        expect(refusal).to_be_in_viewport()
        expect(second.locator("#complete_notice")).to_be_visible()
        expect(second.locator("#draft_text")).to_contain_text("Press Reopen to restore them.")
        with second.expect_navigation():
            second.click("#reopen_button")
        expect(second.locator("#draft_restore")).to_be_visible()
        second.locator("#draft_restore").click()
        assert second.locator('#claim_list .claim-card input[name="claim_label"]').first.input_value() == "true"
        browser.close()


def test_fast_analysis_updates_never_move_the_board_or_the_page(base_url: str) -> None:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        context.add_cookies([{"name": ACCESS_COOKIE, "value": access_token(secret=TEST_ACCESS_SECRET, name="Aram"), "url": base_url}])
        page = context.new_page()
        page.add_init_script(FAST_ANALYSIS)
        page.goto(f"{base_url}/sheets/ap_9001")
        expect = sync_api.expect
        toggle = page.locator("#analysis_toggle")
        expect(toggle).to_be_enabled()
        expect(toggle).to_have_text("Analyse")
        expect(toggle).to_have_attribute("aria-label", "Analyse position")
        box, pill = toggle.bounding_box(), page.locator("#analysis_state").bounding_box()
        assert 22 <= box["height"] <= 28 and abs(box["height"] - pill["height"]) < 1 and abs(box["y"] - pill["y"]) < 1, (box, pill)
        idle = page.evaluate(WHERE)
        toggle.focus()
        page.keyboard.press("Enter")
        expect(toggle).to_have_text("Stop")
        expect(toggle).to_have_attribute("aria-label", "Stop analysis")
        expect(page.locator(".analysis-line").first).to_be_visible()
        page.evaluate("() => { const pane = document.querySelector('.sheet-pane'); pane.scrollTop = pane.scrollHeight; window.scrollTo(0, document.documentElement.scrollHeight); }")
        page.wait_for_timeout(200)
        samples = []
        for _ in range(30):
            samples.append(page.evaluate(WHERE))
            page.wait_for_timeout(50)
        assert all(sample == samples[0] for sample in samples), samples
        assert samples[0]["panel"] == idle["panel"], "the idle panel already has the running panel's height"
        rows = page.locator(".analysis-line")
        assert rows.first.get_attribute("title").startswith("+0.1  6. Be3")
        tops = page.evaluate(ROW_TOPS)
        rows.first.click()
        expect(page.locator(".analysis-line.is-open")).to_have_count(1)
        page.wait_for_function(OPEN_IS_TALLER)
        now = page.evaluate(ROW_TOPS)
        shared = min(len(now), len(tops))
        assert now[:shared] == tops[:shared], "an open line moves no other line"
        assert page.evaluate(WHERE) == samples[0], "an open line moves neither the board nor the page"
        page.keyboard.press("Escape")
        expect(page.locator(".analysis-line.is-open")).to_have_count(0)
        rows.first.press("Enter")
        expect(page.locator(".analysis-line.is-open")).to_have_count(1)
        page.locator("h3", has_text="Claims").click()
        expect(page.locator(".analysis-line.is-open")).to_have_count(0)
        page.keyboard.press("j")
        expect(page.locator("#claim_list .claim-card").first).to_have_class(re.compile(r"\bcurrent\b"))
        toggle.click()
        expect(page.locator("#analysis_state")).to_have_text("stopped")
        expect(toggle).to_have_text("Analyse")
        assert page.evaluate(WHERE)["panel"] == idle["panel"]
        toggle.focus()
        page.keyboard.press(" ")
        expect(page.locator("#analysis_state")).to_have_text("running")
        page.keyboard.press(" ")
        expect(page.locator("#analysis_state")).to_have_text("stopped")
        assert page.evaluate(WHERE)["panel"] == idle["panel"]
        browser.close()


def test_on_a_phone_the_board_and_the_claims_stack_and_nothing_is_sticky(base_url: str) -> None:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(f"{base_url}/sheets/ap_9001")
        board_pane, sheet_pane = page.locator(".board-pane").bounding_box(), page.locator(".sheet-pane").bounding_box()
        assert sheet_pane["y"] >= board_pane["y"] + board_pane["height"] and board_pane["width"] <= 390
        assert page.evaluate("() => getComputedStyle(document.querySelector('.board-pane')).position") == "static"
        browser.close()
