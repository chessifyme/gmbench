"""The images in docs/img are what docs/img/make_boards.py and make_diagrams.py draw, and every image the docs name exists."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, IMG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["make_boards", "make_diagrams"])
def test_committed_images_match_their_script(script: str, tmp_path: Path) -> None:
    load(script).main(out=tmp_path)
    drawn = sorted(tmp_path.glob("*.svg"))
    assert drawn
    for path in drawn:
        committed = IMG / path.name
        assert committed.is_file(), f"{path.name} is drawn but not committed; run uv run python docs/img/{script}.py"
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), f"{path.name} is stale; run uv run python docs/img/{script}.py"


def test_every_image_the_docs_name_exists() -> None:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    named = 0
    for page in pages:
        for ref in re.findall(r'(?:src|srcset)="([^"]+\.svg)"', page.read_text(encoding="utf-8")):
            assert (page.parent / ref).is_file(), f"{page.name} names {ref}, which does not exist"
            named += 1
    assert named
