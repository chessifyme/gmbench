import json
import shutil
import sys
from pathlib import Path

import chess
import chess.engine
import pytest
from fastapi.testclient import TestClient

from review_app.app import create_app
from review_app.live_engine import LINES_SHOWN, LiveEngineSettings, format_score, pv_moves
from tests.review_login import TEST_REVIEWERS

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_ENGINE = [sys.executable, str(Path(__file__).parent / "fake_uci_engine.py")]
MV_FEN = "r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"
MATED_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def _settings(*, command: list[str] = FAKE_ENGINE, sessions_max: int = 2) -> LiveEngineSettings:
    return LiveEngineSettings(command=command, version_expected="Stockfish 19", hash_mb=16, threads_max=2, sessions_max=sessions_max, seconds_max=5)


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


def _client(*, data: Path, settings: LiveEngineSettings) -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=settings, previous_weights={})
    return TestClient(app)


def _events(*, body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def _stream(*, data: Path, settings: LiveEngineSettings, fen: str) -> list[tuple[str, dict[str, object]]]:
    response = _client(data=data, settings=settings).get("/analysis/stream", params={"fen": fen})
    assert response.status_code == 200
    return _events(body=response.text)


def test_scores_read_from_white_side_like_an_analysis_board() -> None:
    assert format_score(score=chess.engine.PovScore(chess.engine.Cp(62), chess.WHITE)) == "+0.62"
    assert format_score(score=chess.engine.PovScore(chess.engine.Cp(62), chess.BLACK)) == "-0.62"
    assert format_score(score=chess.engine.PovScore(chess.engine.Cp(0), chess.BLACK)) == "0.00"
    assert format_score(score=chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)) == "#-3"
    assert format_score(score=chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)) == "#2"


def test_lines_are_split_into_numbered_moves() -> None:
    black_to_move = chess.Board("4n3/p4pk1/1b4p1/5b2/5p1p/2PB1P1P/PP1R2P1/4K3 b - - 1 34")
    pv = [black_to_move.parse_san(san) for san in ("Be6",)]
    probe = black_to_move.copy()
    probe.push(pv[0])
    pv += [probe.parse_san("Bc2")]
    probe.push(pv[1])
    pv += [probe.parse_san("Kf6")]
    assert pv_moves(board=black_to_move, pv=pv) == [
        {"number": "34...", "san": "Be6"},
        {"number": "35.", "san": "Bc2"},
        {"number": "", "san": "Kf6"},
    ]
    assert pv_moves(board=chess.Board(), pv=[chess.Move.from_uci("e2e4")]) == [{"number": "1.", "san": "e4"}]


def test_stream_analyses_the_board_position_with_fixed_options(data: Path) -> None:
    events = _stream(data=data, settings=_settings(), fen=MV_FEN)
    start = events[0][1]
    assert events[0][0] == "start" and start["fen"] == MV_FEN
    assert start["lines"] == LINES_SHOWN == 3 and start["threads"] == 2 and start["seconds"] == 5
    infos = [payload for name, payload in events if name == "info"]
    assert infos, "no live info was streamed"
    last = infos[-1]
    assert last["depth"] == 3 and last["seldepth"] == 5 and last["mnps"] == 2.5
    lines = last["lines"]
    assert isinstance(lines, list) and [line["rank"] for line in lines] == [1, 2, 3]
    assert all(line["score"] and line["moves"] for line in lines)
    assert events[-1][0] == "done"


def test_stream_has_nothing_to_analyse_after_mate(data: Path) -> None:
    events = _stream(data=data, settings=_settings(), fen=MATED_FEN)
    assert events == [("done", {"reason": "no analysis: the game is over (0-1)"})]


@pytest.mark.parametrize("fen", ["not a fen", "8/8/8/8/8/8/8/8 w - - 0 1"])
def test_stream_refuses_a_broken_position(data: Path, fen: str) -> None:
    assert _client(data=data, settings=_settings()).get("/analysis/stream", params={"fen": fen}).status_code == 422


def test_stream_reports_a_wrong_engine_version(data: Path) -> None:
    events = _stream(data=data, settings=_settings(command=[*FAKE_ENGINE, "Stockfish 17"]), fen=MV_FEN)
    assert events[-1][0] == "error"
    assert "Stockfish 19" in str(events[-1][1]["message"])


def test_stream_reports_a_missing_engine(data: Path, tmp_path: Path) -> None:
    events = _stream(data=data, settings=_settings(command=[str(tmp_path / "no_such_engine")]), fen=MV_FEN)
    assert events[-1][0] == "error"
    assert "could not start" in str(events[-1][1]["message"])


def test_stream_refuses_when_every_engine_slot_is_busy(data: Path) -> None:
    response = _client(data=data, settings=_settings(sessions_max=0)).get("/analysis/stream", params={"fen": MV_FEN})
    assert response.status_code == 503


def test_engine_slots_are_given_back_after_each_analysis(data: Path) -> None:
    client = _client(data=data, settings=_settings(sessions_max=1))
    for _ in range(3):
        assert client.get("/analysis/stream", params={"fen": MV_FEN}).status_code == 200
    assert client.app.state.engine_slots.taken == 0  # type: ignore[attr-defined]


def test_page_stylesheet_keeps_its_quotes(data: Path) -> None:
    html = _client(data=data, settings=_settings()).get("/sheets/mj_9001").text
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "&#34;" not in style
    assert '"Segoe UI"' in style and '[data-side="black"]' in style


def test_analysis_panel_has_one_start_stop_toggle_in_its_header(data: Path) -> None:
    html = _client(data=data, settings=_settings()).get("/sheets/mj_9001").text
    head = html.split('<div class="analysis-head">', 1)[1].split('<p class="analysis-position">', 1)[0]
    assert 'id="analysis_state"' in head and '<button id="analysis_toggle" type="button"' in head and 'aria-label="Analyse position"' in head
    assert html.count("<button id=\"analysis_") == 1 and "analysis-actions" not in html
    for removed in ("analysis_start", "analysis_stop"):
        assert f'id="{removed}"' not in html
    for removed in ("analysis_at", "analysis_lines", "analysis_threads", "analysis_seconds"):
        assert f'id="{removed}"' not in html
