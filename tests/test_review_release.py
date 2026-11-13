import hashlib
import re
import shutil
from itertools import groupby
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from bench.config import ReviewEnv
from bench.io_yaml import read_task, write_task
from bench.paths import DATA_PUBLIC, DATA_PUBLIC_TASKS, ROOT
from review_app.app import create_app
from review_app.release import Release, ReleaseEntry, assert_release_tasks, interleave_blocks, read_release
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
RELEASE = {
    "dataset_version": "v_test",
    "families": {
        "assessment_plan": [{"task_id": "ap_9001", "block": "lichess_games", "reason": "x"}],
        "move_choice": [{"task_id": "mc_9001", "block": "lichess_puzzles", "reason": "x"}],
        "move_judgement": [{"task_id": "mj_9001", "block": "lichess_games", "reason": "x"}, {"task_id": "mj_9002", "block": "broadcast", "reason": "x"}],
    },
}


@pytest.fixture
def data(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return tmp_path


def _manifest(path: Path, *, drop: tuple[str, ...] = ()) -> Path:
    families = {family: [row for row in rows if row["task_id"] not in drop] for family, rows in RELEASE["families"].items()}
    path.write_text(yaml.safe_dump({**RELEASE, "families": families}), encoding="utf-8")
    return path


def _client(*, data: Path, release: Release | None, name: str | None = "Aram") -> TestClient:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=data / "tasks", sheets_dir=data / "sheets", reference_dir=data / "reference", calibration_dir=data / "calibration", version="test", engine=TEST_ENGINE, previous_weights={}, release=release)
    client = TestClient(app)
    if name is not None:
        login(client, name)
    return client


def _release(data: Path, *, drop: tuple[str, ...] = ("mj_9002",)) -> Release:
    return read_release(path=_manifest(data / "release.yaml", drop=drop))


def _listed(html: str) -> list[str]:
    return re.findall(r'<a href="/sheets/(\w+)">', html)


def _next(client: TestClient, task_id: str) -> str | None:
    found = re.search(r'<a href="/sheets/(\w+)" id="next_link">', client.get(f"/sheets/{task_id}").text)
    return found.group(1) if found else None


def _digest(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.rglob("*")) if path.is_file()}


def test_a_grandmaster_sees_only_the_release_and_never_reaches_the_other_tasks(data: Path) -> None:
    gm = _client(data=data, release=_release(data))
    assert _listed(gm.get("/sheets").text) == ["ap_9001", "mc_9001", "mj_9001"], "gr_9001 and mj_9002 are not in the release"
    assert _listed(gm.get("/sheets?not_in_release=1").text) == ["ap_9001", "mc_9001", "mj_9001"], "the flag is for an adjudicator only"
    for task_id in ("gr_9001", "mj_9002"):
        assert gm.get(f"/sheets/{task_id}").status_code == 404
        assert gm.post(f"/sheets/{task_id}/save", data={"notes": ""}).status_code == 404


def test_an_adjudicator_lists_and_opens_the_tasks_outside_the_release(data: Path) -> None:
    lead = _client(data=data, release=_release(data), name="Lead")
    html = lead.get("/sheets").text
    assert "gr_9001" not in _listed(html) and "?not_in_release=1" not in html, "hidden by default, the flag offered nowhere"
    assert set(_listed(lead.get("/sheets?not_in_release=1").text)) >= {"gr_9001", "mj_9002"}
    assert lead.get("/sheets/gr_9001").status_code == 200 and lead.get("/sheets/mj_9002").status_code == 200


def test_the_release_filter_keeps_rejected_and_internal_hiding(data: Path) -> None:
    task = read_task(path=data / "tasks" / "mc_9001.yaml")
    write_task(task=task.model_copy(update={"tags": [*task.tags, "rejected"]}), path=data / "tasks" / "mc_9001.yaml")
    release = _release(data)
    assert _listed(_client(data=data, release=release).get("/sheets").text) == ["ap_9001", "mj_9001"]
    lead = _client(data=data, release=release, name="Lead")
    assert "mc_9001" in _listed(lead.get("/sheets?rejected=1").text) and "gr_9001" not in _listed(lead.get("/sheets?rejected=1").text)
    assert "mc_9001" not in _listed(lead.get("/sheets?not_in_release=1").text), "a rejected release task needs the rejected flag"


def test_next_skips_the_tasks_outside_the_release(data: Path) -> None:
    gm = _client(data=data, release=_release(data))
    assert [_next(gm, task_id) for task_id in ("ap_9001", "mc_9001", "mj_9001")] == ["mc_9001", "mj_9001", "ap_9001"]


def test_counts_cover_only_the_release_by_family(data: Path) -> None:
    gm = _client(data=data, release=_release(data))
    html = gm.get("/sheets").text
    assert "0 of 3 done" in html and "0 / 3 complete" in html
    assert gm.post("/sheets/ap_9001/complete", follow_redirects=False).status_code == 303
    html = gm.get("/sheets").text
    assert "1 of 3 done" in html and "1 / 3 complete" in html
    lines = re.findall(r'<li id="family_progress_(\w+)">([^<]+)</li>', html)
    assert lines == [("assessment_plan", "Assessment &amp; plan: 1 of 1 done"), ("move_choice", "Move choice: 0 of 1 done"), ("move_judgement", "Move judgement: 0 of 1 done")]


def test_the_list_and_next_follow_the_release_order(data: Path) -> None:
    gm = _client(data=data, release=_release(data, drop=()))
    assert _listed(gm.get("/sheets").text) == ["ap_9001", "mc_9001", "mj_9002", "mj_9001"], "two blocks of one task: by block name, broadcast first"
    assert _next(gm, "mj_9002") == "mj_9001" and _next(gm, "mj_9001") == "ap_9001"


def test_interleaving_spreads_each_block_evenly_and_is_deterministic() -> None:
    entries = [ReleaseEntry(task_id=f"t_{block}_{index}", block=block, reason="x") for block, size in (("puzzles", 6), ("games", 4), ("endgames", 3), ("session", 2)) for index in range(size)]
    order = interleave_blocks(entries=entries)
    assert order == interleave_blocks(entries=list(reversed(entries))), "the manifest's row order does not matter"
    blocks = [task_id.split("_")[1] for task_id in order]
    assert all(left != right for left, right in zip(blocks, blocks[1:])), blocks
    assert [t for t in order if "puzzles" in t] == sorted(t for t in order if "puzzles" in t), "within a block by task id"
    assert blocks[0] == "puzzles", "the largest block opens"
    lopsided = [ReleaseEntry(task_id=f"t_{block}_{index}", block=block, reason="x") for block, size in (("games", 10), ("session", 3), ("broadcast", 2)) for index in range(size)]
    assert max(len(list(run)) for _, run in groupby(task_id.split("_")[1] for task_id in interleave_blocks(entries=lopsided))) == 2
    thin = [ReleaseEntry(task_id=f"t_{block}_{index}", block=block, reason="x") for block, size in (("games", 10), ("session", 3), ("broadcast", 1)) for index in range(size)]
    assert max(len(list(run)) for _, run in groupby(task_id.split("_")[1] for task_id in interleave_blocks(entries=thin))) == 2, "a block that gave two in a row waits a turn"
    big = [ReleaseEntry(task_id=f"t_{block}_{index:02d}", block=block, reason="x") for block, size in (("games", 10), ("session", 2)) for index in range(size)]
    assert [len(list(run)) for _, run in groupby(task_id.split("_")[1] for task_id in interleave_blocks(entries=big))] == [3, 1, 4, 1, 3], "a block too large for runs of two is split evenly by the others"
    only = [ReleaseEntry(task_id=f"t_{block}_{index}", block=block, reason="x") for block, size in (("games", 5), ("session", 1)) for index in range(size)]
    assert len(interleave_blocks(entries=only)) == 6, "when one block is all that is left it gives the rest in a row"


def test_the_v0_release_is_the_default_and_matches_the_public_tasks() -> None:
    assert ReviewEnv().review_release_manifest == DATA_PUBLIC / "v0_tasks.yaml"
    assert ReviewEnv(review_release_manifest=Path("data/public/v0_tasks.yaml")).review_release_manifest == ROOT / "data" / "public" / "v0_tasks.yaml"
    release = read_release(path=DATA_PUBLIC / "v0_tasks.yaml")
    assert release.name == "v0" and len(release) == 45
    assert_release_tasks(release=release, tasks_dir=DATA_PUBLIC_TASKS)
    manifest = yaml.safe_load((DATA_PUBLIC / "v0_tasks.yaml").read_text(encoding="utf-8"))
    blocks = {row["task_id"]: row["block"] for rows in manifest["families"].values() for row in rows}
    for family in ("assessment_plan", "move_choice", "move_judgement"):
        order = sorted((task_id for task_id, f in release.families.items() if f == family), key=lambda task_id: release.position[task_id])
        longest = max(len(list(run)) for _, run in groupby(blocks[task_id] for task_id in order))
        largest = max(sum(1 for task_id in order if blocks[task_id] == block) for block in set(blocks[task_id] for task_id in order))
        shortest_possible = max(2, -(-largest // (len(order) - largest + 1)))
        assert longest <= shortest_possible, (family, [blocks[task_id] for task_id in order])


def test_a_missing_empty_or_malformed_manifest_stops_the_app(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="release manifest .* is missing"):
        read_release(path=tmp_path / "absent.yaml")
    with pytest.raises(FileNotFoundError, match="is missing"):
        read_release(path=tmp_path)
    (tmp_path / "empty.yaml").write_text("dataset_version: v9\nfamilies: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lists no task"):
        read_release(path=tmp_path / "empty.yaml")
    (tmp_path / "bad.yaml").write_text("dataset_version: v9\nfamilies: {move_choice: [{task_id: mc_1}]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not read as a release"):
        read_release(path=tmp_path / "bad.yaml")
    (tmp_path / "broken.yaml").write_text("families: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not read as a release"):
        read_release(path=tmp_path / "broken.yaml")
    twice = {**RELEASE, "families": {"move_choice": [{"task_id": "mc_9001", "block": "a", "reason": "x"}], "move_judgement": [{"task_id": "mc_9001", "block": "a", "reason": "x"}]}}
    (tmp_path / "twice.yaml").write_text(yaml.safe_dump(twice), encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        read_release(path=tmp_path / "twice.yaml")


def test_a_release_task_missing_from_the_data_folder_or_under_another_family_stops_the_app(data: Path) -> None:
    (data / "tasks" / "mj_9002.yaml").unlink()
    with pytest.raises(ValueError, match="not in .*mj_9002"):
        _client(data=data, release=_release(data, drop=()))
    moved = {**RELEASE, "families": {"move_choice": [{"task_id": "ap_9001", "block": "a", "reason": "x"}]}}
    (data / "moved.yaml").write_text(yaml.safe_dump(moved), encoding="utf-8")
    with pytest.raises(ValueError, match="another family"):
        _client(data=data, release=read_release(path=data / "moved.yaml"))


def test_every_write_to_a_task_outside_the_release_is_refused_and_changes_nothing(data: Path) -> None:
    gm = _client(data=data, release=_release(data))
    before = _digest(data)
    for task_id in ("gr_9001", "mj_9002"):
        for action, form in (("save", {"notes": "x"}), ("lock", {}), ("complete", {}), ("reopen", {}), ("flag", {"reason": "x"})):
            assert gm.post(f"/sheets/{task_id}/{action}", data=form).status_code == 404, (task_id, action)
    assert _digest(data) == before


def test_hiding_changes_no_data_file(data: Path) -> None:
    release = _release(data)
    before = _digest(data)
    gm, lead = _client(data=data, release=release), _client(data=data, release=release, name="Lead")
    for client in (gm, lead):
        for path in ("/sheets", "/sheets?not_in_release=1&rejected=1&internal=1", "/sheets/ap_9001", "/sheets/gr_9001", "/sheets/mj_9002", "/healthz"):
            client.get(path)
    assert _digest(data) == before


def test_healthz_reports_the_release(data: Path) -> None:
    body = _client(data=data, release=_release(data), name=None).get("/healthz").json()
    assert body["status"] == "ok" and body["tasks"] == 5 and body["release"] == "v_test" and body["release_tasks"] == 3
    assert "release" not in _client(data=data, release=None, name=None).get("/healthz").json()
