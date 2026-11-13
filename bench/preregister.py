import hashlib
from datetime import UTC, datetime
from pathlib import Path

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from bench.extensions import private_path
from bench.paths import DATA_PUBLIC, REGISTRATIONS_DIR, ROOT
from bench.provenance import file_sha256, git_head

HASHED_DIRS = ("bench",)
HASHED_FILES = ("config/bench.yaml", "config/models.yaml", "config/third_party.yaml", "pyproject.toml", "uv.lock")
REFERENCE_DIR_NAME = "reference"
IGNORED_PARTS = {"__pycache__"}
HASHED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".html", ".css", ".toml", ".lock"}
PUBLIC_DATA_DIRS = ("tasks", "sheets", REFERENCE_DIR_NAME)


class RegistrationError(ValueError):
    pass


TaskSetName = Literal["public", "private"]
WeightsSource = Literal["calibrated", "hand_set"]


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: str
    task_set: TaskSetName = "public"
    grader_version: str
    weights_source: WeightsSource | None
    git_commit: str
    files: dict[str, str]
    combined_sha256: str
    # the registered paths the public release leaves out (bench.extensions.private_path), and the combined sha256 of the others: the
    # public tree can recompute that one, and checks every other file by its own hash
    private_files: list[str] = []
    public_combined_sha256: str | None = None

    def public_files(self) -> dict[str, str]:
        excluded = set(self.private_files)
        return {path: digest for path, digest in self.files.items() if path not in excluded}


class RegistrationLine(BaseModel):
    """One line of a dataset version's registration history: every manifest written, a replacement included, with its time and reason."""

    model_config = ConfigDict(extra="forbid")
    dataset_version: str
    task_set: TaskSetName
    grader_version: str
    git_commit: str
    combined_sha256: str
    registered_at: datetime
    replaced_sha256: str | None
    reason: str | None


def history_path(*, dataset_version: str, registrations_dir: Path = REGISTRATIONS_DIR) -> Path:
    return registrations_dir / f"{dataset_version}_registrations.jsonl"


def read_history(*, path: Path) -> list[RegistrationLine]:
    if not path.is_file():
        return []
    return [RegistrationLine.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_path(*, dataset_version: str, registrations_dir: Path = REGISTRATIONS_DIR, task_set: TaskSetName = "public") -> Path:
    return registrations_dir / (f"{dataset_version}.json" if task_set == "public" else f"{dataset_version}_{task_set}.json")


def combine(*, files: dict[str, str]) -> str:
    text = "".join(f"{path} {digest}\n" for path, digest in sorted(files.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def weights_path(*, root: Path, dataset_version: str) -> Path:
    return root / "calibration" / f"weights_{dataset_version}.json"


def weights_source(*, root: Path, dataset_version: str) -> WeightsSource | None:
    """Where the registered score weights come from: the calibrated file when it exists, else the hand-set weights of config/bench.yaml
    when its calibration.hand_set_weights lists this dataset version, else nowhere yet (an official registration then fails)."""
    if weights_path(root=root, dataset_version=dataset_version).is_file():
        return "calibrated"
    config = yaml.safe_load((root / "config" / "bench.yaml").read_text(encoding="utf-8"))
    calibration = config.get("calibration") or {}
    return "hand_set" if dataset_version in (calibration.get("hand_set_weights") or []) else None


def hash_files(*, root: Path, tasks_dir: Path, sheets_dir: Path, dataset_version: str) -> dict[str, str]:
    paths: list[Path] = [root / name for name in HASHED_FILES]
    chosen_weights = weights_path(root=root, dataset_version=dataset_version)
    if chosen_weights.is_file():
        paths.append(chosen_weights)
    for directory in (*(root / name for name in HASHED_DIRS), tasks_dir, sheets_dir, tasks_dir.parent / REFERENCE_DIR_NAME):
        if not directory.is_dir():
            raise RegistrationError(f"{directory} is not a directory; nothing to register")
        paths.extend(p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix in HASHED_SUFFIXES and not IGNORED_PARTS & set(p.parts))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RegistrationError(f"cannot register, missing files: {', '.join(missing)}")
    return {p.relative_to(root).as_posix(): file_sha256(path=p) for p in paths}


def build_registration(*, root: Path, tasks_dir: Path, sheets_dir: Path, dataset_version: str, grader_version: str, task_set: TaskSetName = "public") -> Registration:
    files = hash_files(root=root, tasks_dir=tasks_dir, sheets_dir=sheets_dir, dataset_version=dataset_version)
    private = sorted(path for path in files if private_path(path=path))
    public = {path: digest for path, digest in files.items() if path not in private}
    return Registration(
        dataset_version=dataset_version,
        task_set=task_set,
        grader_version=grader_version,
        weights_source=weights_source(root=root, dataset_version=dataset_version),
        git_commit=git_head(root=root),
        files=files,
        combined_sha256=combine(files=files),
        private_files=private,
        public_combined_sha256=combine(files=public),
    )


def write_registration(*, registration: Registration, registrations_dir: Path = REGISTRATIONS_DIR, replace: bool = False, reason: str | None = None) -> Path:
    """Write the manifest and append it to the registration history, which keeps every earlier registration (and the attempts
    made under it stay in the attempt ledger), so a replacement never resets what was run before; a replacement needs a reason."""
    path = manifest_path(dataset_version=registration.dataset_version, registrations_dir=registrations_dir, task_set=registration.task_set)
    replaced = Registration.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
    if replaced is not None and not replace:
        raise RegistrationError(f"{path} already exists; a registration is final, pass --replace with a --reason to overwrite it knowingly")
    if replaced is not None and not (reason or "").strip():
        raise RegistrationError(f"replacing {path} needs a reason; it is kept in {history_path(dataset_version=registration.dataset_version, registrations_dir=registrations_dir).name} and published with the results")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registration.model_dump_json(indent=2) + "\n", encoding="utf-8")
    line = RegistrationLine(
        dataset_version=registration.dataset_version,
        task_set=registration.task_set,
        grader_version=registration.grader_version,
        git_commit=registration.git_commit,
        combined_sha256=registration.combined_sha256,
        registered_at=datetime.now(UTC),
        replaced_sha256=replaced.combined_sha256 if replaced is not None else None,
        reason=reason.strip() if reason is not None and reason.strip() else None,
    )
    with history_path(dataset_version=registration.dataset_version, registrations_dir=registrations_dir).open("a", encoding="utf-8") as handle:
        handle.write(line.model_dump_json() + "\n")
    return path


def registration_problems(
    *, root: Path = ROOT, tasks_dir: Path, sheets_dir: Path, dataset_version: str, grader_version: str, registrations_dir: Path = REGISTRATIONS_DIR, require_weights: bool = False, task_set: TaskSetName = "public"
) -> tuple[Registration | None, list[str]]:
    path = manifest_path(dataset_version=dataset_version, registrations_dir=registrations_dir, task_set=task_set)
    if not path.is_file():
        return None, [f"no pre-registration manifest at {path}; run `make preregister` and commit it before any official run"]
    registration = Registration.model_validate_json(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    source = weights_source(root=root, dataset_version=dataset_version)
    if require_weights and source is None:
        chosen_weights = weights_path(root=root, dataset_version=dataset_version).relative_to(root)
        problems.append(
            f"the calibrated weights {chosen_weights} are missing; calibrate the weights for this dataset version, or list {dataset_version} under calibration.hand_set_weights in config/bench.yaml to register the hand-set weights, before an official run"
        )
    if registration.weights_source != source:
        problems.append(f"manifest records {registration.weights_source or 'no'} weights, the tree now has {source or 'none'}")
    if registration.dataset_version != dataset_version:
        problems.append(f"manifest is for dataset {registration.dataset_version}, the config says {dataset_version}")
    if registration.grader_version != grader_version:
        problems.append(f"manifest is for grader {registration.grader_version}, the code is {grader_version}")
    if registration.task_set != task_set:
        problems.append(f"manifest is for the {registration.task_set} set, this is the {task_set} set")
    current = hash_files(root=root, tasks_dir=tasks_dir, sheets_dir=sheets_dir, dataset_version=dataset_version)
    changed = sorted(p for p in current if p in registration.files and current[p] != registration.files[p])
    added = sorted(set(current) - set(registration.files))
    # a registered private file is absent from the public release by design; every other file is checked on its own hash
    removed = sorted(set(registration.files) - set(current) - set(registration.private_files))
    for label, names in (("changed since registration", changed), ("not registered", added), ("registered but gone", removed)):
        if names:
            problems.append(f"{len(names)} file(s) {label}: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    if combine(files=registration.files) != registration.combined_sha256:
        problems.append("the manifest's combined sha256 does not match its own file list; it was edited by hand")
    if registration.public_combined_sha256 is not None and combine(files=registration.public_files()) != registration.public_combined_sha256:
        problems.append("the manifest's public combined sha256 does not match its own file list; it was edited by hand")
    return registration, problems


def verify_registration(
    *, root: Path = ROOT, tasks_dir: Path, sheets_dir: Path, dataset_version: str, grader_version: str, registrations_dir: Path = REGISTRATIONS_DIR, require_weights: bool = True, task_set: TaskSetName = "public"
) -> str:
    registration, problems = registration_problems(
        root=root,
        task_set=task_set,
        tasks_dir=tasks_dir,
        sheets_dir=sheets_dir,
        dataset_version=dataset_version,
        grader_version=grader_version,
        registrations_dir=registrations_dir,
        require_weights=require_weights,
    )
    if registration is None or problems:
        raise RegistrationError("; ".join(problems))
    return registration.combined_sha256


def governed_paths(*, root: Path, dataset_version: str) -> list[Path]:
    """The paths in the repository a run depends on: what a registration hashes (the bench package, the configs including `config/third_party.yaml`, the lock files,
    the public tasks, sheets and references, the calibrated weights) and the manifests themselves. The attempt ledger is not one
    of them, since every registered run appends to it."""
    public = root / DATA_PUBLIC.relative_to(ROOT)
    registrations = root / REGISTRATIONS_DIR.relative_to(ROOT)
    manifests = [manifest_path(dataset_version=dataset_version, registrations_dir=registrations, task_set=task_set) for task_set in ("public", "private")]
    return [*(root / name for name in HASHED_DIRS), *(root / name for name in HASHED_FILES), *(public / name for name in PUBLIC_DATA_DIRS), weights_path(root=root, dataset_version=dataset_version), *manifests]


def registration_for(
    *, task_set: TaskSetName, tasks_dir: Path, dataset_version: str, grader_version: str, root: Path = ROOT, registrations_dir: Path = REGISTRATIONS_DIR
) -> str | None:
    """The combined hash a new run records. A private-set run needs a matching manifest (RegistrationError otherwise). A public-set
    run records None when there is no manifest yet, and also when one exists but no longer matches, which it says, with every problem."""
    versions = {"root": root, "tasks_dir": tasks_dir, "sheets_dir": tasks_dir.parent / "sheets", "dataset_version": dataset_version, "grader_version": grader_version, "registrations_dir": registrations_dir}
    if task_set == "private":
        return verify_registration(**versions, task_set="private")
    registration, problems = registration_problems(**versions, require_weights=True)
    if registration is None:
        return None
    if problems:
        print(f"the {task_set} pre-registration manifest for {dataset_version} does not match, so this run records no registration hash and stays a development result:\n  " + "\n  ".join(problems))
        return None
    return registration.combined_sha256
