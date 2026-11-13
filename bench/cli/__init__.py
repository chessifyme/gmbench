import argparse
from pathlib import Path


def run_folder(value: str) -> Path:
    """The argparse type of a --run option: a run folder `make run` wrote, which holds its meta.json."""
    path = Path(value)
    if not (path / "meta.json").is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a run folder (it has no meta.json); pass the folder make run printed, runs/<entrant>/<timestamp>")
    return path
