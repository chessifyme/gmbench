import json
import shutil
from dataclasses import dataclass
from pathlib import Path


from bench.io_yaml import yaml_files
from bench.paths import (
    CALIBRATION_DIR,
    DATA_PUBLIC_REFERENCE,
    DATA_PUBLIC_SHEETS,
    DATA_PUBLIC_TASKS,
    PREVIOUS_WEIGHTS_FILE,
)
from bench.reference import reference_path


@dataclass(frozen=True)
class ReviewDirs:
    tasks: Path
    sheets: Path
    reference: Path
    calibration: Path
    previous_weights: Path
    public_reference: Path | None = None


def review_dirs(*, data_dir: Path | None) -> ReviewDirs:
    if data_dir is None:
        return ReviewDirs(tasks=DATA_PUBLIC_TASKS, sheets=DATA_PUBLIC_SHEETS, reference=DATA_PUBLIC_REFERENCE, calibration=CALIBRATION_DIR, previous_weights=PREVIOUS_WEIGHTS_FILE)
    return ReviewDirs(tasks=data_dir / "tasks", sheets=data_dir / "sheets", reference=data_dir / "reference", calibration=data_dir / "calibration", previous_weights=data_dir / PREVIOUS_WEIGHTS_FILE.name, public_reference=DATA_PUBLIC_REFERENCE)


def reference_file(*, task_id: str, reference_dir: Path, public_reference_dir: Path | None) -> Path | None:
    """A task's reference: the data dir's own first; with a data dir, the public reference only for a task of the public set, and
    no reference in either is refused. Without a data dir (public_reference_dir None) a missing reference stays absent, as before."""
    own = reference_path(reference_dir=reference_dir, task_id=task_id)
    if own.exists() or public_reference_dir is None:
        return own if own.exists() else None
    public = reference_path(reference_dir=public_reference_dir, task_id=task_id)
    if not (public_reference_dir.parent / "tasks" / f"{task_id}.yaml").exists():
        raise FileNotFoundError(f"task {task_id} has no reference in {reference_dir}, and it is not a public task, so {public_reference_dir} is not used for it")
    if not public.exists():
        raise FileNotFoundError(f"task {task_id} has no reference in {reference_dir} nor in {public_reference_dir}")
    return public


def _copy_missing(*, source: Path, target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in yaml_files(directory=source):
        destination = target / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
            copied.append(destination)
    return copied


def read_previous_weights(*, path: Path) -> dict[str, int]:
    """The earlier weights the review app shows beside some claims, from an optional file a data folder may hold; none without it."""
    if not path.exists():
        return {}
    return {str(claim_id): int(weight) for claim_id, weight in json.loads(path.read_text(encoding="utf-8")).items()}


def seed_data_dir(*, data_dir: Path, source_tasks: Path, source_sheets: Path, source_previous_weights: Path) -> list[Path]:
    """Copy tasks, sheets and the previous weights (when the source has them) the data dir does not have yet; files already
    there hold annotator work and are never replaced."""
    copied = _copy_missing(source=source_tasks, target=data_dir / "tasks") + _copy_missing(source=source_sheets, target=data_dir / "sheets")
    weights = data_dir / source_previous_weights.name
    if not weights.exists() and source_previous_weights.exists():
        shutil.copy2(source_previous_weights, weights)
        copied.append(weights)
    (data_dir / "calibration").mkdir(parents=True, exist_ok=True)
    return copied
