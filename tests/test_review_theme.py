import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from review_app.app import STATIC, create_app
from tests.fake_uci_engine_settings import TEST_ENGINE
from tests.review_login import TEST_REVIEWERS, login

FIXTURES = Path(__file__).parent / "fixtures"
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(|\b(?:white|black|red|green|blue|orange|yellow|gray|grey|silver)\b", re.IGNORECASE)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    shutil.copytree(FIXTURES / "tasks", tmp_path / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "sheets")
    (tmp_path / "reference").mkdir()
    return login(TestClient(create_app(reviewers=TEST_REVIEWERS, token_secret=b"test-claim-token-secret", tasks_dir=tmp_path / "tasks", sheets_dir=tmp_path / "sheets", reference_dir=tmp_path / "reference", calibration_dir=tmp_path / "calibration", version="test", engine=TEST_ENGINE, previous_weights={})), "GM Judge")


PAGES = ["/sheets", "/sheets/ap_9001", "/sheets/gr_9001"]


@pytest.mark.parametrize("path", PAGES)
def test_every_page_sets_the_theme_before_paint_and_offers_the_toggle(client: TestClient, path: str) -> None:
    html = client.get(path).text
    head = html.split("</head>", 1)[0]
    assert head.index('id="theme_boot"') < head.index("<style>")
    assert "localStorage" in head and "try" in head
    for choice in ["light", "dark", "auto"]:
        assert f'data-theme-choice="{choice}"' in html
    assert html.index('id="theme_toggle"') < html.index("</header>")


def _declarations(*, css: str) -> list[tuple[str, str]]:
    bodies = re.findall(r"\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL))
    found = []
    for body in bodies:
        for declaration in body.split(";"):
            if ":" in declaration:
                name, value = declaration.split(":", 1)
                found.append((name.strip(), value.strip()))
    return found


def test_review_css_keeps_every_colour_in_a_variable() -> None:
    css = (STATIC / "review.css").read_text(encoding="utf-8")
    stray = [(name, value) for name, value in _declarations(css=css) if not name.startswith("--") and COLOUR.search(re.sub(r"var\(--[\w-]+\)", "", value))]
    assert stray == []


def test_light_and_dark_palettes_define_the_same_variables() -> None:
    css = (STATIC / "review.css").read_text(encoding="utf-8")
    light = re.search(r"^:root \{([^{}]*)\}", css, flags=re.MULTILINE)
    auto_dark = re.search(r'@media \(prefers-color-scheme: dark\) \{ :root:not\(\[data-theme="light"\]\) \{([^{}]*)\}', css)
    forced_dark = re.search(r':root\[data-theme="dark"\] \{([^{}]*)\}', css)
    assert light and auto_dark and forced_dark
    names = [{name for name, _ in _declarations(css="{" + block.group(1) + "}") if name.startswith("--")} for block in (light, auto_dark, forced_dark)]
    assert names[0] == names[1] == names[2] and len(names[0]) > 20
    assert auto_dark.group(1) == forced_dark.group(1)
    assert re.search(r"\bbody \{[^}]*background: var\(--bg\)", css)
