import json
import shutil
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from bench.io_yaml import load_all_tasks, read_sheet, read_task, write_sheet
from bench.paths import DATA_PUBLIC_TASKS
from review_app.app import create_app
from review_app.board_line import task_line
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS

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


def _board_config(*, html: str) -> dict[str, object]:
    raw = html.split('<script type="application/json" id="board_config">', 1)[1].split("</script>", 1)[0]
    return json.loads(raw)


def test_a_move_task_line_runs_the_game_then_the_task_move(data: Path) -> None:
    task = read_task(path=data / "tasks" / "mj_9001.yaml")
    line = task_line(task=task)
    assert line.start_fen == chess.STARTING_FEN
    assert line.moves[-1] == "Nxd5" and line.task_ply == len(line.moves) - 1
    board = chess.Board(line.start_fen)
    for san in line.moves[: line.task_ply]:
        board.push_san(san)
    assert board.epd() == chess.Board(task.fen).epd()


def test_a_position_task_line_stops_at_the_task_position(data: Path) -> None:
    line = task_line(task=read_task(path=data / "tasks" / "ap_9001.yaml"))
    assert line.task_ply == len(line.moves) == 10


def test_a_game_review_line_is_the_whole_game_from_the_start(data: Path) -> None:
    task = read_task(path=data / "tasks" / "gr_9001.yaml")
    line = task_line(task=task)
    assert line.start_fen == chess.STARTING_FEN and line.task_ply == 0 and len(line.moves) > 20


def test_a_pgn_that_does_not_reach_the_task_position_is_refused(data: Path) -> None:
    task = read_task(path=data / "tasks" / "mj_9001.yaml")
    broken = task.model_copy(update={"pgn": "1. d4 d5"})
    with pytest.raises(ValueError, match="does not reach"):
        task_line(task=broken)


def test_a_task_without_a_pgn_starts_at_its_position(data: Path) -> None:
    task = read_task(path=data / "tasks" / "mj_9001.yaml").model_copy(update={"pgn": None})
    line = task_line(task=task)
    assert line.start_fen == task.fen and line.moves == ["Nxd5"] and line.task_ply == 0


def test_every_repository_task_has_a_consistent_line() -> None:
    for task in load_all_tasks(tasks_dir=DATA_PUBLIC_TASKS):
        task_line(task=task)


def test_sheet_page_has_a_playable_board_and_no_position_text(client: TestClient) -> None:
    html = client.get("/sheets/mj_9001").text
    config = _board_config(html=html)
    assert config["moves"][-1] == "Nxd5" and config["ply"] == config["task_ply"] == len(config["moves"]) - 1
    assert config["orientation"] == "black"
    assert 'id="board"' in html and 'src="/static/board.js"' in html
    task = read_task(path=FIXTURES / "tasks" / "mj_9001.yaml")
    visible = html.replace(html.split('<script type="application/json" id="board_config">', 1)[1].split("</script>", 1)[0], "")
    assert (task.fen or "") not in visible and "exd5" not in visible
    assert 'type="range"' not in html


def test_game_review_board_opens_at_the_requested_ply_from_the_user_side(client: TestClient) -> None:
    config = _board_config(html=client.get("/sheets/gr_9001?ply=18").text)
    assert config["ply"] == 18 and config["task_ply"] == 0 and config["orientation"] == "black"
    assert client.get("/sheets/gr_9001?ply=999").status_code == 422


def test_board_libraries_are_served_by_the_app(client: TestClient) -> None:
    for path in ("/static/vendor/chess_js/chess.js", "/static/vendor/cm_chessboard/src/Chessboard.js", "/static/vendor/cm_chessboard/assets/pieces/standard.svg", "/static/board.js"):
        assert client.get(path).status_code == 200, path


def test_the_board_label_names_the_move_number_and_side_to_move(client: TestClient) -> None:
    def label(url: str) -> str:
        return client.get(url).text.split('id="board_move" class="board-move">', 1)[1].split("</span>", 1)[0]

    assert label("/sheets/ap_9001") == "Task position · move 6, White to move"
    assert label("/sheets/mj_9001") == "Task position · move 5, Black to move"
    assert label("/sheets/gr_9001?ply=18").startswith("after 9...")
    assert "Start position" not in (Path(__file__).parents[1] / "review_app" / "static" / "board.js").read_text(encoding="utf-8")


def test_lines_in_a_claim_are_links_with_their_moves_and_the_rest_is_muted(client: TestClient, data: Path) -> None:
    path = data / "sheets" / "ap_9001.yaml"
    sheet = read_sheet(path=path)
    text = "6. Be2 e5 7. Nb3 Be7 is the main line; 6...e5 is not. 6. Bg5 Qxh8 fails."
    write_sheet(sheet=sheet.model_copy(update={"claims": [sheet.claims[0].model_copy(update={"text": text}), *sheet.claims[1:]]}), path=path)
    html = client.get("/sheets/ap_9001").text
    assert '<a href="#board" class="claim-line" data-uci="f1e2 e7e5 d4b3 f8e7" data-start-ply="10" title="Play this line on the board">6. Be2 e5 7. Nb3 Be7</a>' in html
    assert '<span class="claim-line-muted" title="this line doesn&#39;t start from the task position">6...e5</span>' in html
    assert 'data-uci="c1g5" data-start-ply="10"' in html and '<span class="claim-line-muted" title="illegal from here">Qxh8</span>' in html
