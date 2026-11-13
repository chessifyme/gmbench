"""Every link from a committed Markdown page to another page's heading (`page.md#anchor`) names a heading that page has."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = [*sorted(ROOT.glob("*.md")), ROOT / "data" / "public" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def _slug(*, heading: str) -> str:
    text = re.sub(r"[`*]", "", heading).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


def _anchors(*, page: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.DOTALL)
    return {_slug(heading=h) for h in re.findall(r"^#{1,6} (.+)$", text, flags=re.MULTILINE)}


def test_every_heading_link_between_pages_resolves() -> None:
    checked = 0
    for page in PAGES:
        for target, anchor in re.findall(r"\]\(([^)#\s]*\.md)?#([^)\s]+)\)", page.read_text(encoding="utf-8")):
            linked = (page.parent / target).resolve() if target else page
            assert linked.is_file(), f"{page.name} links {target}, which does not exist"
            assert anchor in _anchors(page=linked), f"{page.name} links {target}#{anchor}, which {linked.name} has no heading for"
            checked += 1
    assert checked
