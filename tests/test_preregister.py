import hashlib
import subprocess
from pathlib import Path

import pytest

from bench.preregister import Registration, RegistrationError, build_registration, combine, history_path, manifest_path, read_history, registration_problems, verify_registration, write_registration

VERSION = "pilot"
GRADER = "g1"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for directory in ("bench/prompts", "bench/grading", "config", "data/public/tasks", "data/public/sheets", "data/public/reference"):
        (root / directory).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'bench'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "data" / "public" / "reference" / "ap_0001.json").write_text("{}\n", encoding="utf-8")
    (root / "bench" / "prompts" / "judge_item.md").write_text("judge\n", encoding="utf-8")
    (root / "bench" / "grading" / "score.py").write_text("X = 1\n", encoding="utf-8")
    (root / "bench" / "grading" / "__pycache__").mkdir()
    (root / "bench" / "grading" / "__pycache__" / "score.cpython-312.pyc").write_bytes(b"\x00")
    (root / "config" / "bench.yaml").write_text("dataset_version: pilot\n", encoding="utf-8")
    (root / "config" / "models.yaml").write_text("judges: {}\n", encoding="utf-8")
    (root / "config" / "third_party.yaml").write_text("entrants: {}\n", encoding="utf-8")
    (root / "config" / "other.yaml").write_text("unregistered: true\n", encoding="utf-8")
    (root / "data" / "public" / "tasks" / "ap_0001.yaml").write_text("task_id: ap_0001\n", encoding="utf-8")
    (root / "data" / "public" / "sheets" / "ap_0001.yaml").write_text("task_id: ap_0001\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "init"], cwd=root, check=True)
    return root


def _build(root: Path) -> Registration:
    return build_registration(root=root, tasks_dir=root / "data" / "public" / "tasks", sheets_dir=root / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER)


def _problems(root: Path, registrations: Path) -> list[str]:
    return registration_problems(root=root, tasks_dir=root / "data" / "public" / "tasks", sheets_dir=root / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER, registrations_dir=registrations)[1]


def test_manifest_hashes_every_governed_file_and_a_combined_digest(tree: Path) -> None:
    registration = _build(tree)
    assert sorted(registration.files) == [
        "bench/grading/score.py",
        "bench/prompts/judge_item.md",
        "config/bench.yaml",
        "config/models.yaml",
        "config/third_party.yaml",
        "data/public/reference/ap_0001.json",
        "data/public/sheets/ap_0001.yaml",
        "data/public/tasks/ap_0001.yaml",
        "pyproject.toml",
        "uv.lock",
    ]
    assert registration.files["config/bench.yaml"] == hashlib.sha256(b"dataset_version: pilot\n").hexdigest()
    assert registration.combined_sha256 == combine(files=registration.files) and len(registration.git_commit) == 40
    assert (registration.dataset_version, registration.grader_version) == (VERSION, GRADER)


def test_manifest_is_written_at_the_dataset_version_path_and_is_final(tree: Path) -> None:
    registrations = tree / "registrations"
    path = write_registration(registration=_build(tree), registrations_dir=registrations)
    assert path == registrations / "pilot.json" == manifest_path(dataset_version=VERSION, registrations_dir=registrations)
    with pytest.raises(RegistrationError, match="already exists"):
        write_registration(registration=_build(tree), registrations_dir=registrations)
    with pytest.raises(RegistrationError, match="needs a reason"):
        write_registration(registration=_build(tree), registrations_dir=registrations, replace=True, reason="  ")
    write_registration(registration=_build(tree), registrations_dir=registrations, replace=True, reason="a grading bug fixed before any attempt")


def test_a_replaced_registration_stays_listed_in_the_registration_history(tree: Path) -> None:
    registrations = tree / "registrations"
    first = _build(tree)
    write_registration(registration=first, registrations_dir=registrations)
    (tree / "config" / "bench.yaml").write_text("dataset_version: pilot\nchanged: true\n", encoding="utf-8")
    second = _build(tree)
    write_registration(registration=second, registrations_dir=registrations, replace=True, reason="the answer timeout was wrong")
    history = read_history(path=history_path(dataset_version=VERSION, registrations_dir=registrations))
    assert [line.combined_sha256 for line in history] == [first.combined_sha256, second.combined_sha256]
    assert (history[0].replaced_sha256, history[0].reason) == (None, None)
    assert (history[1].replaced_sha256, history[1].reason) == (first.combined_sha256, "the answer timeout was wrong")
    assert history[0].registered_at <= history[1].registered_at


def test_missing_manifest_is_reported(tree: Path) -> None:
    registration, problems = registration_problems(root=tree, tasks_dir=tree / "data" / "public" / "tasks", sheets_dir=tree / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER, registrations_dir=tree / "registrations")
    assert registration is None and "no pre-registration manifest" in problems[0]
    with pytest.raises(RegistrationError, match="make preregister"):
        verify_registration(root=tree, tasks_dir=tree / "data" / "public" / "tasks", sheets_dir=tree / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER, registrations_dir=tree / "registrations")


def test_matching_manifest_verifies_and_returns_the_combined_digest(tree: Path) -> None:
    registrations = tree / "registrations"
    (tree / "calibration").mkdir()
    (tree / "calibration" / "weights_pilot.json").write_text('{"verdict": 0.4}\n', encoding="utf-8")
    registration = _build(tree)
    write_registration(registration=registration, registrations_dir=registrations)
    assert _problems(tree, registrations) == []
    digest = verify_registration(root=tree, tasks_dir=tree / "data" / "public" / "tasks", sheets_dir=tree / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER, registrations_dir=registrations)
    assert digest == registration.combined_sha256


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda r: (r / "bench" / "grading" / "score.py").write_text("X = 2\n", encoding="utf-8"), "changed since registration: bench/grading/score.py"),
        (lambda r: (r / "data" / "public" / "sheets" / "ap_0001.yaml").write_text("task_id: ap_0001\nnotes: x\n", encoding="utf-8"), "data/public/sheets/ap_0001.yaml"),
        (lambda r: (r / "data" / "public" / "tasks" / "ap_0002.yaml").write_text("task_id: ap_0002\n", encoding="utf-8"), "not registered: data/public/tasks/ap_0002.yaml"),
        (lambda r: (r / "bench" / "prompts" / "judge_item.md").unlink(), "registered but gone: bench/prompts/judge_item.md"),
    ],
)
def test_any_change_to_a_registered_file_is_a_mismatch(tree: Path, edit, expected: str) -> None:
    registrations = tree / "registrations"
    write_registration(registration=_build(tree), registrations_dir=registrations)
    edit(tree)
    problems = _problems(tree, registrations)
    assert any(expected in problem for problem in problems), problems


def test_version_mismatch_and_hand_edited_manifest_are_reported(tree: Path) -> None:
    registrations = tree / "registrations"
    path = write_registration(registration=_build(tree), registrations_dir=registrations)
    assert any("grader g1" in p for p in registration_problems(root=tree, tasks_dir=tree / "data" / "public" / "tasks", sheets_dir=tree / "data" / "public" / "sheets", dataset_version=VERSION, grader_version="g2", registrations_dir=registrations)[1])
    path.write_text(path.read_text(encoding="utf-8").replace("X", "Y"), encoding="utf-8")
    forged = Registration.model_validate_json(path.read_text(encoding="utf-8"))
    forged = forged.model_copy(update={"combined_sha256": "0" * 64})
    path.write_text(forged.model_dump_json(indent=2), encoding="utf-8")
    assert any("edited by hand" in p for p in _problems(tree, registrations))


def test_a_private_run_is_refused_without_a_matching_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.cli import run as run_cli

    from bench import preregister

    monkeypatch.setattr(run_cli, "tasks_dir_for", lambda **kwargs: tmp_path / "private" / "tasks")
    monkeypatch.setattr(preregister, "verify_registration", lambda **kwargs: (_ for _ in ()).throw(RegistrationError("no pre-registration manifest at x")))
    with pytest.raises(SystemExit, match="private-set run needs a matching pre-registration"):
        run_cli.registration_for(task_set="private", dataset_version=VERSION)


def test_calibrated_weights_are_registered_and_required_for_official_runs(tree: Path) -> None:
    registrations = tree / "registrations"
    write_registration(registration=_build(tree), registrations_dir=registrations)
    kwargs = {"root": tree, "tasks_dir": tree / "data" / "public" / "tasks", "sheets_dir": tree / "data" / "public" / "sheets", "dataset_version": VERSION, "grader_version": GRADER, "registrations_dir": registrations}
    assert registration_problems(**kwargs)[1] == []
    _, problems = registration_problems(**kwargs, require_weights=True)
    assert any("calibration/weights_pilot.json" in p and "missing" in p for p in problems)
    with pytest.raises(RegistrationError, match="weights_pilot.json"):
        verify_registration(**kwargs)
    (tree / "calibration").mkdir()
    (tree / "calibration" / "weights_pilot.json").write_text('{"verdict": 0.4}\n', encoding="utf-8")
    assert any("not registered: calibration/weights_pilot.json" in p for p in _problems(tree, registrations))
    write_registration(registration=_build(tree), registrations_dir=registrations, replace=True, reason="the weights were calibrated")
    assert verify_registration(**kwargs)
    (tree / "calibration" / "weights_pilot.json").write_text('{"verdict": 0.5}\n', encoding="utf-8")
    assert any("changed since registration: calibration/weights_pilot.json" in p for p in _problems(tree, registrations))


def test_a_private_set_has_its_own_manifest_bound_to_its_own_tasks_and_sheets(tree: Path) -> None:
    private = tree / "private"
    for directory in ("tasks", "sheets", "reference"):
        (private / directory).mkdir(parents=True)
    (private / "tasks" / "ap_0101.yaml").write_text("task_id: ap_0101\n", encoding="utf-8")
    (private / "sheets" / "ap_0101.yaml").write_text("task_id: ap_0101\n", encoding="utf-8")
    registrations = tree / "registrations"
    registration = build_registration(root=tree, tasks_dir=private / "tasks", sheets_dir=private / "sheets", dataset_version=VERSION, grader_version=GRADER, task_set="private")
    assert registration.task_set == "private" and "private/tasks/ap_0101.yaml" in registration.files and "data/public/tasks/ap_0001.yaml" not in registration.files
    assert write_registration(registration=registration, registrations_dir=registrations) == registrations / "pilot_private.json"
    kwargs = {"root": tree, "tasks_dir": private / "tasks", "sheets_dir": private / "sheets", "dataset_version": VERSION, "grader_version": GRADER, "registrations_dir": registrations, "task_set": "private"}
    assert registration_problems(**kwargs)[1] == []
    _, wrong_set = registration_problems(**{**kwargs, "task_set": "public", "tasks_dir": tree / "data" / "public" / "tasks", "sheets_dir": tree / "data" / "public" / "sheets"})
    assert any("no pre-registration manifest" in p for p in wrong_set)
    (private / "sheets" / "ap_0101.yaml").write_text("task_id: ap_0101\nnotes: edited\n", encoding="utf-8")
    assert any("changed since registration: private/sheets/ap_0101.yaml" in p for p in registration_problems(**kwargs)[1])


def test_registration_for_names_the_problems_of_a_mismatched_public_manifest(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from bench.preregister import registration_for

    registrations = tree / "registrations"
    common = {"task_set": "public", "tasks_dir": tree / "data" / "public" / "tasks", "dataset_version": VERSION, "grader_version": GRADER, "root": tree, "registrations_dir": registrations}
    assert registration_for(**common) is None
    assert capsys.readouterr().out == ""
    unweighted = _build(tree)
    write_registration(registration=unweighted, registrations_dir=registrations)
    assert registration_for(**common) is None
    assert "calibrated weights" in capsys.readouterr().out
    (tree / "calibration").mkdir()
    (tree / "calibration" / f"weights_{VERSION}.json").write_text('{"weights": {}}\n', encoding="utf-8")
    registration = _build(tree)
    write_registration(registration=registration, registrations_dir=registrations, replace=True, reason="the weights were calibrated")
    assert registration_for(**common) == registration.combined_sha256
    (tree / "bench" / "grading" / "score.py").write_text("X = 2\n", encoding="utf-8")
    assert registration_for(**common) is None
    said = capsys.readouterr().out
    assert "records no registration" in said and "bench/grading/score.py" in said


def test_registration_for_refuses_a_private_run_without_a_matching_manifest(tree: Path) -> None:
    from bench.preregister import registration_for

    with pytest.raises(RegistrationError, match="no pre-registration manifest"):
        registration_for(task_set="private", tasks_dir=tree / "data" / "public" / "tasks", dataset_version=VERSION, grader_version=GRADER, root=tree, registrations_dir=tree / "registrations")


def test_git_dirty_sees_uncommitted_changes_to_governed_paths_only(tree: Path) -> None:
    from bench.preregister import governed_paths
    from bench.provenance import git_dirty

    (tree / "README.md").write_text("readme\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tree, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "all"], cwd=tree, check=True)
    paths = governed_paths(root=tree, dataset_version=VERSION)
    assert git_dirty(root=tree, paths=paths) is False
    (tree / "README.md").write_text("changed\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is False
    (tree / "data" / "public" / "sheets" / "ap_0002.yaml").write_text("task_id: ap_0002\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is True
    (tree / "data" / "public" / "sheets" / "ap_0002.yaml").unlink()
    (tree / "config" / "bench.yaml").write_text("dataset_version: other\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is True
    subprocess.run(["git", "checkout", "--", "config/bench.yaml"], cwd=tree, check=True)
    assert git_dirty(root=tree, paths=paths) is False
    (tree / "calibration").mkdir()
    (tree / "calibration" / f"weights_{VERSION}.json").write_text("{}\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is True
    (tree / "calibration" / f"weights_{VERSION}.json").unlink()
    (tree / "registrations").mkdir()
    (tree / "registrations" / f"{VERSION}.json").write_text("{}\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is True
    (tree / "registrations" / f"{VERSION}.json").unlink()
    (tree / "registrations" / f"{VERSION}_attempts.jsonl").write_text("{}\n", encoding="utf-8")
    assert git_dirty(root=tree, paths=paths) is False


def test_hand_set_weights_register_and_are_recorded_as_hand_set(tree: Path) -> None:
    registrations = tree / "registrations"
    kwargs = {"root": tree, "tasks_dir": tree / "data" / "public" / "tasks", "sheets_dir": tree / "data" / "public" / "sheets", "dataset_version": VERSION, "grader_version": GRADER, "registrations_dir": registrations}
    assert _build(tree).weights_source is None
    (tree / "config" / "bench.yaml").write_text("dataset_version: pilot\ncalibration:\n  hand_set_weights: [pilot]\n", encoding="utf-8")
    registration = _build(tree)
    assert registration.weights_source == "hand_set" and "config/bench.yaml" in registration.files
    write_registration(registration=registration, registrations_dir=registrations)
    assert Registration.model_validate_json(manifest_path(dataset_version=VERSION, registrations_dir=registrations).read_text(encoding="utf-8")).weights_source == "hand_set"
    assert verify_registration(**kwargs) == registration.combined_sha256
    (tree / "calibration").mkdir()
    (tree / "calibration" / "weights_pilot.json").write_text('{"verdict": 0.4}\n', encoding="utf-8")
    problems = registration_problems(**kwargs, require_weights=True)[1]
    assert any("manifest records hand_set weights, the tree now has calibrated" in p for p in problems)
    assert _build(tree).weights_source == "calibrated"


def test_hand_set_weights_for_another_version_do_not_satisfy_an_official_registration(tree: Path) -> None:
    (tree / "config" / "bench.yaml").write_text("dataset_version: pilot\ncalibration:\n  hand_set_weights: [v0]\n", encoding="utf-8")
    registrations = tree / "registrations"
    write_registration(registration=_build(tree), registrations_dir=registrations)
    with pytest.raises(RegistrationError, match="hand_set_weights"):
        verify_registration(root=tree, tasks_dir=tree / "data" / "public" / "tasks", sheets_dir=tree / "data" / "public" / "sheets", dataset_version=VERSION, grader_version=GRADER, registrations_dir=registrations)


def test_a_private_file_is_listed_apart_so_the_public_tree_checks_the_rest_and_recomputes_their_combined_digest(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tree / "bench" / "private_part.py").write_text("SECRET = 1\n", encoding="utf-8")
    monkeypatch.setattr("bench.preregister.private_path", lambda *, path: path == "bench/private_part.py")
    registration = _build(tree)
    assert registration.private_files == ["bench/private_part.py"] and "bench/private_part.py" in registration.files
    public = {path: digest for path, digest in registration.files.items() if path != "bench/private_part.py"}
    assert registration.public_combined_sha256 == combine(files=public) != registration.combined_sha256
    registrations = tree / "registrations"
    write_registration(registration=registration, registrations_dir=registrations)
    (tree / "bench" / "private_part.py").unlink()
    assert _problems(tree, registrations) == []
    (tree / "bench" / "grading" / "score.py").write_text("X = 2\n", encoding="utf-8")
    assert any("changed since registration" in problem and "bench/grading/score.py" in problem for problem in _problems(tree, registrations))

