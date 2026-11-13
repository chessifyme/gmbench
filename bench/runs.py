import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bench.adapters.base import AgentCard, AnswerRecord
from bench.config import AccessMethod, DeclaredFamily, Effort, Tool
from bench.extensions import private_config_files
from bench.naming import assert_underscore_name
from bench.paths import REGISTRATIONS_DIR, ROOT, RUNS_DIR
from bench.preregister import combine
from bench.provenance import file_sha256

TaskSet = Literal["public", "private", "folder"]
# who ran it: the maintainers, the entrant's maker (self_reported, an entrant of entrants/), or the maker with a sample re-run by
# the maintainers that agreed (verified, make verify_submission)
Provenance = Literal["self_reported", "verified", "maintainer_run"]
AttemptKind = Literal["attempt", "void"]


class RunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entrant: str
    card: AgentCard
    dataset_version: str
    task_set: TaskSet
    grader_version: str | None
    started_at: datetime
    models_config_hash: str
    bench_config_hash: str
    git_commit: str
    client_versions: dict[str, str]
    engine_id: str | None
    engine_sha256: str | None
    temperature: str
    registration_sha256: str | None
    models_profile: str | None = None
    grading_profile: str | None = None
    unavailable_reason: str | None = None
    entrant_config_hash: str | None = None
    git_dirty: bool | None = None
    source_sha256: str | None = None
    tasks_dir: str | None = Field(default=None, exclude_if=lambda value: value is None)
    entrant_details: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)
    tasks_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    # none on the maintainers' runs, so their meta files stay unchanged; the board reads it with board_provenance
    provenance: Provenance | None = Field(default=None, exclude_if=lambda value: value is None)


def folder_tasks_sha256(*, tasks_dir: Path) -> str:
    """The combined hash of a task folder's task files: what a folder run answered, checked again before its answers are graded or pooled."""
    return combine(files={path.name: file_sha256(path=path) for path in sorted(tasks_dir.glob("*.yaml"))})


def repo_relative(*, path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{path} is outside the repository {root}; results must be published with repo-relative paths") from error


def config_hash(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_config_hash(*, path: Path) -> str:
    """The hash a run records for a configuration file: config_hash of the file alone, or, where the working repository's private
    configuration adds to it (bench.extensions.private_config_files), of the file and those files together, so a change to either
    stops a resume."""
    extra = private_config_files()
    if not extra:
        return config_hash(path=path)
    digest = hashlib.sha256()
    for part in (path, *extra):
        digest.update(part.name.encode("utf-8") + b"\0" + part.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def new_run_dir(*, entrant: str, runs_dir: Path = RUNS_DIR, now: datetime | None = None) -> Path:
    assert_underscore_name(name=entrant)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / entrant / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_run_dir(*, entrant: str, runs_dir: Path = RUNS_DIR) -> Path:
    entrant_dir = runs_dir / entrant
    candidates = sorted(p for p in entrant_dir.iterdir() if p.is_dir()) if entrant_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no run directory for entrant {entrant!r} under {runs_dir}")
    return candidates[-1]


def write_meta(*, meta: RunMeta, run_dir: Path) -> None:
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_meta(*, run_dir: Path) -> RunMeta:
    return RunMeta.model_validate_json((run_dir / "meta.json").read_text(encoding="utf-8"))


def append_jsonl(*, path: Path, record: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_jsonl(*, path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_answers(*, run_dir: Path) -> list[AnswerRecord]:
    return [AnswerRecord.model_validate(row) for row in read_jsonl(path=run_dir / "answers.jsonl")]


class AttemptLine(BaseModel):
    """One line of a dataset version's attempt ledger: a registered run that was started, or a later voiding of one."""

    model_config = ConfigDict(extra="forbid")
    kind: AttemptKind
    entrant: str
    run_dir: str
    started_at: datetime
    recorded_at: datetime
    version: str
    declared_family: DeclaredFamily
    access_method: AccessMethod
    tools: list[Tool]
    client_tools: bool
    affiliated_with_maintainer: bool
    effort: Effort | None
    git_commit: str
    registration_sha256: str
    task_set: TaskSet
    reason: str | None


def ledger_path(*, dataset_version: str, registrations_dir: Path = REGISTRATIONS_DIR) -> Path:
    return registrations_dir / f"{dataset_version}_attempts.jsonl"


def read_ledger(*, ledger: Path) -> list[AttemptLine]:
    return [AttemptLine.model_validate(row) for row in read_jsonl(path=ledger)]


def _ledger_line(*, kind: AttemptKind, meta: RunMeta, run_dir: Path, root: Path, reason: str | None) -> AttemptLine:
    if meta.registration_sha256 is None:
        raise ValueError(f"{run_dir} records no registration; only registered runs enter the attempt ledger")
    card = meta.card
    return AttemptLine(
        kind=kind,
        entrant=meta.entrant,
        run_dir=repo_relative(path=run_dir, root=root),
        started_at=meta.started_at,
        recorded_at=datetime.now(UTC),
        version=card.version,
        declared_family=card.declared_family,
        access_method=card.access_method,
        tools=sorted(card.tools),
        client_tools=card.client_tools,
        affiliated_with_maintainer=card.affiliated_with_maintainer,
        effort=card.effort,
        git_commit=meta.git_commit,
        registration_sha256=meta.registration_sha256,
        task_set=meta.task_set,
        reason=reason,
    )


def append_attempt(*, ledger: Path, meta: RunMeta, run_dir: Path, root: Path) -> None:
    line = _ledger_line(kind="attempt", meta=meta, run_dir=run_dir, root=root, reason=None)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path=ledger, record=line)


def void_attempt(*, ledger: Path, run_dir: Path, root: Path, reason: str) -> AttemptLine:
    """Mark a registered attempt void (a harness fault); refused once the run is graded, so a void can never follow a look at the scores."""
    if not reason.strip():
        raise ValueError("a void needs a reason that names the harness fault")
    graded = sorted(path.name for path in run_dir.glob("grades*.jsonl"))
    if graded:
        raise ValueError(f"{run_dir} is already graded ({', '.join(graded)}); a graded attempt cannot be voided")
    relative = repo_relative(path=run_dir, root=root)
    lines = read_ledger(ledger=ledger)
    if not any(line.kind == "attempt" and line.run_dir == relative for line in lines):
        raise ValueError(f"{relative} is not an attempt in {ledger.name}; only registered attempts can be voided")
    if any(line.kind == "void" and line.run_dir == relative for line in lines):
        raise ValueError(f"{relative} is already void")
    line = _ledger_line(kind="void", meta=read_meta(run_dir=run_dir), run_dir=run_dir, root=root, reason=reason.strip())
    append_jsonl(path=ledger, record=line)
    return line


def start_run(*, meta: RunMeta, runs_dir: Path = RUNS_DIR, root: Path = ROOT, registrations_dir: Path = REGISTRATIONS_DIR) -> Path:
    """Create the run directory, write its metadata and, for a registered run, append an attempt to the dataset version's ledger.
    A run recorded as not available asked no task (its client failed the probe), so it is no attempt and writes no line; nor is a
    self-reported run, which its maker started, not the maintainers.
    A registered run never starts from a tree with uncommitted governed files: it could never be published, yet it would count
    as the entrant's attempt."""
    if meta.registration_sha256 is not None and meta.git_dirty is not False and meta.provenance is None:
        state = "has uncommitted changes" if meta.git_dirty else "was not checked for uncommitted changes"
        raise ValueError(f"the registered {meta.entrant} run cannot start: the tree of governed files {state}; commit them first (git status shows which), so the attempt is tied to a commit")
    run_dir = new_run_dir(entrant=meta.entrant, runs_dir=runs_dir)
    write_meta(meta=meta, run_dir=run_dir)
    if meta.registration_sha256 is not None and meta.unavailable_reason is None and meta.provenance is None:
        append_attempt(ledger=ledger_path(dataset_version=meta.dataset_version, registrations_dir=registrations_dir), meta=meta, run_dir=run_dir, root=root)
    return run_dir


# the maintainers' record of every verification of a submitted run (make verify_submission), kept with the attempt ledger and never
# inside a submitted folder: the first line for an entrant and dataset version decides, whatever folder it came from
VERIFICATIONS_LEDGER = REGISTRATIONS_DIR / "verifications.jsonl"


class VerificationLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entrant: str
    dataset_version: str
    answers_sha256: str
    passed: bool
    needs_full_rerun: str | None = None
    rerun_mean: float | None = None
    submitted_mean: float | None = None
    difference: float | None = None
    standard_error: float | None = None
    tolerance: float | None = None
    sample: list[str] = []
    seed: int | None = None
    rerun_dir: str | None = None
    verified_at: datetime


def read_verifications(*, ledger: Path = VERIFICATIONS_LEDGER) -> list[VerificationLine]:
    return [VerificationLine.model_validate(row) for row in read_jsonl(path=ledger)]


def first_verification(*, lines: list[VerificationLine], entrant: str, dataset_version: str) -> VerificationLine | None:
    return next((line for line in lines if line.entrant == entrant and line.dataset_version == dataset_version), None)


def append_verification(*, line: VerificationLine, ledger: Path = VERIFICATIONS_LEDGER) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path=ledger, record=line)


def board_provenance(*, meta: RunMeta, answers_sha256: str, lines: list[VerificationLine], keeps_entrant_file: bool) -> Provenance:
    """Who ran a run, as the board shows it. A run folder that keeps an entrant file (entrant.yaml or entrant.py) is a self-run
    entrant's, never the maintainers', whatever its meta.json says. A submitted run is verified only when the ledger's first
    verification of its entrant and dataset version passed on these very answers; it is self-reported otherwise."""
    if not keeps_entrant_file and meta.provenance in (None, "maintainer_run"):
        return "maintainer_run"
    first = first_verification(lines=lines, entrant=meta.entrant, dataset_version=meta.dataset_version)
    return "verified" if first is not None and first.passed and first.answers_sha256 == answers_sha256 else "self_reported"
