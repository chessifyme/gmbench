import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bench.paths import CALIBRATION_DIR, DATA_PUBLIC_REFERENCE, DATA_PUBLIC_SHEETS, DATA_PUBLIC_TASKS, PREVIOUS_WEIGHTS_FILE
from review_app.app import create_app
from review_app.data_dirs import read_previous_weights, reference_file, review_dirs, seed_data_dir
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    shutil.copytree(FIXTURES / "tasks", root / "tasks")
    shutil.copytree(FIXTURES / "sheets", root / "sheets")
    (root / "previous_weights_data_model_v2.json").write_text('{"ap_9001_c002": 3}', encoding="utf-8")
    return root


def test_without_data_dir_the_app_uses_the_repository_folders() -> None:
    dirs = review_dirs(data_dir=None)
    assert (dirs.tasks, dirs.sheets, dirs.reference, dirs.calibration) == (DATA_PUBLIC_TASKS, DATA_PUBLIC_SHEETS, DATA_PUBLIC_REFERENCE, CALIBRATION_DIR)


def test_with_data_dir_every_folder_moves_there_and_public_references_are_the_fallback(tmp_path: Path) -> None:
    dirs = review_dirs(data_dir=tmp_path)
    assert (dirs.tasks, dirs.sheets, dirs.calibration, dirs.reference) == (tmp_path / "tasks", tmp_path / "sheets", tmp_path / "calibration", tmp_path / "reference")
    assert dirs.public_reference == DATA_PUBLIC_REFERENCE and review_dirs(data_dir=None).public_reference is None


def _public_tree(root: Path) -> Path:
    (root / "tasks").mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "tasks" / "ap_0001.yaml").write_text("task_id: ap_0001\n", encoding="utf-8")
    (root / "reference" / "ap_0001.json").write_text("{}", encoding="utf-8")
    (root / "reference" / "ap_5901.json").write_text("{}", encoding="utf-8")
    return root / "reference"


def test_a_tasks_reference_comes_from_the_data_dir_first(tmp_path: Path) -> None:
    public = _public_tree(tmp_path / "public")
    (tmp_path / "data" / "reference").mkdir(parents=True)
    (tmp_path / "data" / "reference" / "ap_0001.json").write_text("{}", encoding="utf-8")
    assert reference_file(task_id="ap_0001", reference_dir=tmp_path / "data" / "reference", public_reference_dir=public) == tmp_path / "data" / "reference" / "ap_0001.json"


def test_only_a_public_task_falls_back_to_the_public_reference(tmp_path: Path) -> None:
    public = _public_tree(tmp_path / "public")
    own = tmp_path / "data" / "reference"
    assert reference_file(task_id="ap_0001", reference_dir=own, public_reference_dir=public) == public / "ap_0001.json"
    with pytest.raises(FileNotFoundError, match="ap_5901 has no reference in .*data/reference.*not a public task"):
        reference_file(task_id="ap_5901", reference_dir=own, public_reference_dir=public)
    (tmp_path / "public" / "tasks" / "ap_0002.yaml").write_text("task_id: ap_0002\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ap_0002 has no reference in .*data/reference nor in .*public/reference"):
        reference_file(task_id="ap_0002", reference_dir=own, public_reference_dir=public)


def test_without_a_data_dir_a_missing_reference_stays_absent(tmp_path: Path) -> None:
    assert reference_file(task_id="ap_0001", reference_dir=tmp_path / "reference", public_reference_dir=None) is None


def test_the_app_with_a_data_dir_refuses_loudly_a_task_without_a_reference(source: Path, tmp_path: Path) -> None:
    public = _public_tree(tmp_path / "public")
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=source / "tasks", sheets_dir=source / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="t", engine=TEST_ENGINE, previous_weights={}, public_reference_dir=public)
    response = TestClient(app, raise_server_exceptions=False).get("/sheets/ap_9001")
    assert response.status_code == 500 and "ap_9001 has no reference" in response.text


def test_seed_fills_an_empty_data_dir(source: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    copied = seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    expected = sorted(p.name for p in (source / "tasks").glob("*.yaml")) + sorted(p.name for p in (source / "sheets").glob("*.yaml")) + ["previous_weights_data_model_v2.json"]
    assert sorted(p.name for p in copied) == sorted(expected)
    assert (data_dir / "calibration").is_dir()


def test_seed_never_overwrites_annotated_files(source: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    edited = next((data_dir / "sheets").glob("*.yaml"))
    edited.write_text("annotator work\n", encoding="utf-8")
    copied = seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    assert copied == []
    assert edited.read_text(encoding="utf-8") == "annotator work\n"


def test_seed_adds_tasks_that_are_new_in_the_repository(source: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    first = next((source / "tasks").glob("*.yaml"))
    new_task = source / "tasks" / "zz_9999.yaml"
    shutil.copy(first, new_task)
    copied = seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    assert copied == [data_dir / "tasks" / "zz_9999.yaml"]


def test_healthz_reports_the_running_version(source: Path, tmp_path: Path) -> None:
    app = create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=source / "tasks", sheets_dir=source / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="abc1234", engine=TEST_ENGINE, previous_weights={})
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "abc1234", "tasks": len(list((source / "tasks").glob("*.yaml")))}


def test_seed_copies_the_previous_weights_once_and_never_overwrites_them(source: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    weights = data_dir / "previous_weights_data_model_v2.json"
    assert read_previous_weights(path=weights) == {"ap_9001_c002": 3}
    weights.write_text('{"ap_9001_c002": 1}', encoding="utf-8")
    copied = seed_data_dir(data_dir=data_dir, source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    assert copied == [] and read_previous_weights(path=weights) == {"ap_9001_c002": 1}


def test_the_previous_weights_are_optional(source: Path, tmp_path: Path) -> None:
    assert read_previous_weights(path=tmp_path / "previous_weights_data_model_v2.json") == {}
    (source / "previous_weights_data_model_v2.json").unlink()
    copied = seed_data_dir(data_dir=tmp_path / "data", source_tasks=source / "tasks", source_sheets=source / "sheets", source_previous_weights=source / "previous_weights_data_model_v2.json")
    assert copied and not (tmp_path / "data" / "previous_weights_data_model_v2.json").exists()


def test_seeding_from_the_public_set_never_overwrites_a_server_task_or_sheet(tmp_path: Path) -> None:
    """The server's own copies of the public tasks and sheets hold the real reviews (the public sheets carry pseudonyms): seeding
    adds only the files the data folder lacks and leaves every existing one byte for byte."""
    data_dir = tmp_path / "data"
    weights = PREVIOUS_WEIGHTS_FILE
    seed_data_dir(data_dir=data_dir, source_tasks=DATA_PUBLIC_TASKS, source_sheets=DATA_PUBLIC_SHEETS, source_previous_weights=weights)
    own = {path: path.read_bytes() + b"# the server's own work\n" for folder in ("tasks", "sheets") for path in sorted((data_dir / folder).glob("*.yaml"))[::2]}
    for path, content in own.items():
        path.write_bytes(content)
    removed = sorted((data_dir / "sheets").glob("*.yaml"))[1]
    removed.unlink()
    copied = seed_data_dir(data_dir=data_dir, source_tasks=DATA_PUBLIC_TASKS, source_sheets=DATA_PUBLIC_SHEETS, source_previous_weights=weights)
    assert copied == [removed] and removed.read_bytes() == (DATA_PUBLIC_SHEETS / removed.name).read_bytes()
    assert all(path.read_bytes() == content for path, content in own.items())
    assert not (data_dir / "reference").exists(), "references are never seeded"
