from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from bench.io_yaml import load_all_tasks
from bench.models import Family


class ReleaseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    block: str
    reason: str


class ReleaseManifest(BaseModel):
    """A dataset release (data/public/v0_tasks.yaml): its tasks by family, each with the curation block it was picked from."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    families: dict[Family, list[ReleaseEntry]]


@dataclass(frozen=True)
class Release:
    """The tasks grandmasters review: the others stay in the data folder, hidden. position: each task's place within its
    family, the curation blocks interleaved (interleave_blocks)."""

    name: str
    path: Path
    families: dict[str, Family]
    position: dict[str, int]

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.families

    def __len__(self) -> int:
        return len(self.families)


def interleave_blocks(*, entries: list[ReleaseEntry]) -> list[str]:
    """The task ids of one family with the curation blocks spread evenly, by smooth weighted round-robin: at each step every
    block with tasks left gains its size in credit, the block with the most credit (ties: the larger block, then by name) gives
    its next task by task id and pays the family's size, except that a block which gave the last `run` tasks waits a turn while
    another block has tasks left. `run` is two, or, when one block is too large for runs of two, the shortest run its size
    allows: the largest block split evenly by the others (12 tasks of which 10 from one block: runs of four). Consecutive tasks
    so come from different sources wherever the sizes allow, and the order depends only on which tasks the manifest lists in
    which block."""
    blocks: dict[str, list[str]] = {}
    for entry in entries:
        blocks.setdefault(entry.block, []).append(entry.task_id)
    queues = {block: sorted(ids) for block, ids in sorted(blocks.items())}
    sizes = {block: len(ids) for block, ids in queues.items()}
    total = sum(sizes.values())
    largest = max(sizes.values(), default=0)
    run = max(2, -(-largest // (total - largest + 1)))
    credit = dict.fromkeys(queues, 0)
    order: list[str] = []
    picked: list[str] = []
    while len(order) < total:
        for block in queues:
            if queues[block]:
                credit[block] += sizes[block]
        left = [block for block in queues if queues[block]]
        tired = picked[-1] if len(picked) >= run and len(set(picked[-run:])) == 1 else None
        candidates = [block for block in left if block != tired] or left
        chosen = min(candidates, key=lambda block: (-credit[block], -sizes[block], block))
        credit[chosen] -= total
        picked.append(chosen)
        order.append(queues[chosen].pop(0))
    return order


def read_release(*, path: Path) -> Release:
    """The release manifest at path; a missing, unreadable or malformed manifest, or one that lists a task twice or no task at
    all, stops the app, so the grandmasters never get a wrong or empty task list."""
    if not path.is_file():
        raise FileNotFoundError(f"release manifest {path} is missing: the review app shows reviewers only the tasks it lists (REVIEW_RELEASE_MANIFEST, docs/operations.md)")
    try:
        manifest = ReleaseManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"release manifest {path} does not read as a release (dataset_version, families of task_id/block/reason): {error}") from error
    families: dict[str, Family] = {}
    position: dict[str, int] = {}
    for family, entries in manifest.families.items():
        for index, task_id in enumerate(interleave_blocks(entries=entries)):
            if task_id in families:
                raise ValueError(f"release manifest {path} lists task {task_id} twice")
            families[task_id] = family
            position[task_id] = index
    if not families:
        raise ValueError(f"release manifest {path} lists no task: the reviewers would see an empty list")
    return Release(name=manifest.dataset_version, path=path, families=families, position=position)


def assert_release_tasks(*, release: Release, tasks_dir: Path) -> None:
    """Every task of the release is in the data folder, under the family the manifest gives it; otherwise the app stops rather
    than show grandmasters a release with tasks missing."""
    tasks = {task.task_id: task for task in load_all_tasks(tasks_dir=tasks_dir)}
    missing = sorted(task_id for task_id in release.families if task_id not in tasks)
    if missing:
        raise ValueError(f"release {release.name} ({release.path}) lists {len(missing)} task(s) not in {tasks_dir}: {', '.join(missing)}")
    moved = sorted(f"{task_id} ({tasks[task_id].family}, listed as {family})" for task_id, family in release.families.items() if tasks[task_id].family != family)
    if moved:
        raise ValueError(f"release {release.name} ({release.path}) lists task(s) under another family than their task file: {', '.join(moved)}")
