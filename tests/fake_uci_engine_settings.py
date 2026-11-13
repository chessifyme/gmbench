import sys
from pathlib import Path

from review_app.live_engine import LiveEngineSettings

TEST_ENGINE = LiveEngineSettings(
    command=[sys.executable, str(Path(__file__).parent / "fake_uci_engine.py")],
    version_expected="Stockfish 19",
    hash_mb=16,
    threads_max=2,
    sessions_max=2,
    seconds_max=60,
)
