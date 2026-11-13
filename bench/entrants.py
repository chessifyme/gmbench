"""Entrants a participant plugs in and runs themselves (docs/taking_part.md, Way 1): a file in entrants/ named after the entrant,
either entrants/<name>.yaml (an HTTP endpoint described by a request template and the path of the answer in the reply) or
entrants/<name>.py (one Python function, answer(message) -> str, with its card in an ENTRANT dict). Their runs are self-reported."""

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from bench.adapters.base import AnswerStatus
from bench.config import DeclaredFamily, StrictModel, Tool, checked_entrant_url
from bench.errors import UserError
from bench.naming import assert_underscore_name
from bench.paths import ROOT

ENTRANTS_DIR = ROOT / "entrants"
# what a body template may hold: the task message every entrant gets, the neutral system prompt, and a new id per answer (for a
# service that wants a session or request id). The task id is never sent, so an entrant cannot look a task up.
PLACEHOLDERS: frozenset[str] = frozenset({"message", "system", "fresh_id"})
_PLACEHOLDER = re.compile(r"(?<!\$)\{([^{}]*)\}")
_ENV_REFERENCE = re.compile(r"\$\{([^}]*)\}")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
# headers that carry credentials: their values must come from .env (${VARIABLE}), so the entrant file never holds a key
_SECRET_HEADER = re.compile(r"authorization|cookie|key|token|secret|password|auth", re.IGNORECASE)


class EntrantFileError(UserError, ValueError):
    """An entrant file that cannot be used; the message names the file and the problem."""


def env_prefix(*, name: str) -> str:
    """The prefix of every variable an entrant's headers may name: BENCH_ENTRANT_<NAME>_, followed by one word (KEY, TOKEN). Apart
    from the maintainers' own variables (OPENROUTER_API_KEY, BENCH_HTTP_*), and one word only, so no entrant file can name
    another entrant's variable (acme could otherwise read BENCH_ENTRANT_ACME_COACH_KEY)."""
    return f"BENCH_ENTRANT_{name.upper()}_"


def placeholders_in(*, value: object) -> set[str]:
    """Every {placeholder} in the strings of a JSON template, keys included."""
    if isinstance(value, str):
        return set(_PLACEHOLDER.findall(value))
    if isinstance(value, dict):
        return {p for key, item in value.items() for p in placeholders_in(value=key) | placeholders_in(value=item)}
    if isinstance(value, list):
        return {p for item in value for p in placeholders_in(value=item)}
    return set()


def fill(*, template: object, values: dict[str, str]) -> object:
    """The template with every {placeholder} in its strings replaced; numbers, booleans and null are kept as they are."""
    if isinstance(template, str):
        return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
    if isinstance(template, dict):
        return {fill(template=key, values=values): fill(template=item, values=values) for key, item in template.items()}
    if isinstance(template, list):
        return [fill(template=item, values=values) for item in template]
    return template


def extract(*, reply: object, path: str) -> tuple[AnswerStatus, str | None, str]:
    """The answer at a dotted path in a JSON reply (`output.text`, `choices.0.message.content`): a key of an object, or the index
    of a list. Returns status, error and text the way every adapter records them."""
    current = reply
    walked: list[str] = []
    for part in path.split("."):
        walked.append(part)
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return "error", f"malformed_response: the reply has no {'.'.join(walked)}", ""
    if not isinstance(current, str):
        return "error", f"unsupported_response: {path} is not a string", ""
    if not current.strip():
        return "error", "empty_answer", current
    return "ok", None, current


class EntrantCard(StrictModel):
    """What the board says about the entrant; the same fields for an HTTP and a Python entrant."""

    name: str
    label: str
    version: str
    provider: str
    declared_family: DeclaredFamily
    tools: list[Tool]
    affiliated_with_maintainer: bool

    @field_validator("name")
    @classmethod
    def _underscore_name(cls, value: str) -> str:
        assert_underscore_name(name=value)
        return value


class HttpCall(StrictModel):
    url: str
    method: Literal["POST", "PUT"] = "POST"
    headers: dict[str, str] = {}
    body: dict[str, object] | list[object]
    answer: str = Field(min_length=1)
    timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        return checked_entrant_url(value=value)

    @field_validator("body")
    @classmethod
    def _known_placeholders(cls, value: dict[str, object] | list[object]) -> dict[str, object] | list[object]:
        if "${" in json.dumps(value):
            raise ValueError("the body holds a ${VARIABLE}; a key goes in a header, never in the body")
        found = placeholders_in(value=value)
        if "task_id" in found:
            raise ValueError("{task_id} is not allowed: an entrant never sees the task id; send {message}, the text every entrant gets")
        unknown = sorted(found - PLACEHOLDERS)
        if unknown:
            raise ValueError(f"unknown placeholder(s) {', '.join('{' + p + '}' for p in unknown)}; the body may use {{message}}, {{system}} and {{fresh_id}}")
        if "message" not in found:
            raise ValueError("the body never sends {message}, the task text every entrant gets")
        return value

    @field_validator("answer")
    @classmethod
    def _dotted_path(cls, value: str) -> str:
        if any(not part for part in value.split(".")):
            raise ValueError(f"answer {value!r} is not a dotted path such as output.text or choices.0.message.content")
        return value

    @field_validator("headers")
    @classmethod
    def _headers_from_env(cls, value: dict[str, str]) -> dict[str, str]:
        for header, text in value.items():
            if placeholders_in(value=text):
                raise ValueError(f"header {header} holds a {{placeholder}}; headers take only fixed text and ${{VARIABLE}} from .env")
            if _SECRET_HEADER.search(header) and not _ENV_REFERENCE.search(text):
                raise ValueError(f"header {header} must take its secret from .env as ${{VARIABLE}}; a key never goes in the entrant file")
            for variable in _ENV_REFERENCE.findall(text):
                if not _ENV_NAME.fullmatch(variable):
                    raise ValueError(f"header {header}: ${{{variable}}} is not an upper-case variable name")
        return value

    def variables(self) -> list[str]:
        return sorted({v for text in self.headers.values() for v in _ENV_REFERENCE.findall(text)})


class HttpEntrantFile(EntrantCard):
    """entrants/<name>.yaml."""

    http: HttpCall

    @model_validator(mode="after")
    def _own_variables(self) -> "HttpEntrantFile":
        prefix = env_prefix(name=self.name)
        foreign = [v for v in self.http.variables() if not (v.startswith(prefix) and re.fullmatch(r"[A-Z0-9]+", v[len(prefix) :]))]
        if foreign:
            raise ValueError(f"header variable(s) {', '.join(foreign)} must be {prefix}<WORD> (one word, such as {prefix}KEY), so an entrant file reads only its own keys")
        return self


EntrantFile = HttpEntrantFile | EntrantCard


def resolve_headers(*, headers: dict[str, str], lookup: Callable[[str], str | None]) -> dict[str, str]:
    """The headers with every ${VARIABLE} replaced by its value; a missing variable stops with its name, never a value."""
    missing = sorted({v for text in headers.values() for v in _ENV_REFERENCE.findall(text) if lookup(v) is None})
    if missing:
        raise EntrantFileError(f"set {', '.join(missing)} in .env (or the environment); the entrant's headers need {'it' if len(missing) == 1 else 'them'}")
    return {header: _ENV_REFERENCE.sub(lambda match: str(lookup(match.group(1))), text) for header, text in headers.items()}


def _python_card(*, path: Path) -> dict[str, object]:
    """The ENTRANT dict of an entrants/<name>.py, read without running the file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as error:  # noqa: BLE001 - any file that does not parse (syntax, encoding, null bytes) is refused alike
        raise EntrantFileError(f"{path} does not parse as Python ({type(error).__name__})") from error
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENTRANT" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except Exception as error:  # noqa: BLE001 - literal_eval raises several kinds on what is not a plain literal
                raise EntrantFileError(f"{path}: ENTRANT must be a plain dict of strings, lists and booleans") from error
            if not isinstance(value, dict):
                raise EntrantFileError(f"{path}: ENTRANT must be a dict")
            return value
    raise EntrantFileError(f"{path} has no ENTRANT = {{...}} dict with the entrant's card (see entrants/example_python.py)")


def load_entrant_file(*, path: Path, expected_name: str | None = None) -> EntrantFile:
    """An entrant file, checked; its name must be the file name (or expected_name, for the copy a run keeps)."""
    try:
        if path.suffix == ".py":
            entrant: EntrantFile = EntrantCard.model_validate(_python_card(path=path))
        else:
            entrant = HttpEntrantFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (ValueError, yaml.YAMLError) as error:
        if isinstance(error, EntrantFileError):
            raise
        raise EntrantFileError(f"{path} is not a valid entrant file: {error}") from error
    expected = expected_name or path.stem
    if entrant.name != expected:
        raise EntrantFileError(f"{path}: name {entrant.name!r} must be {expected!r}")
    return entrant


def _scan(*, directory: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    found: dict[str, Path] = {}
    both: dict[str, list[str]] = {}
    if not directory.is_dir():
        return found, both
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.py")]):
        if path.name.startswith("_"):
            continue
        if path.stem in found:
            both[path.stem] = [found[path.stem].name, path.name]
            continue
        found[path.stem] = path
    return found, both


def entrant_files(*, directory: Path = ENTRANTS_DIR) -> dict[str, Path]:
    """The entrant files by name: <name>.yaml or <name>.py. A name with both is listed, but refused when used (entrant_file)."""
    return _scan(directory=directory)[0]


def entrant_file(*, name: str, directory: Path = ENTRANTS_DIR) -> Path | None:
    """The file of one entrant, or None; refused when the entrant has both a .yaml and a .py, which leaves every other entrant usable."""
    found, both = _scan(directory=directory)
    if name in both:
        raise EntrantFileError(f"{directory} has both {' and '.join(both[name])}; keep one file per entrant")
    return found.get(name)


# the entrant file a self-run entrant's run keeps next to its answers: entrant.yaml or entrant.py (no secret: keys stay in .env)
RUN_ENTRANT_STEM = "entrant"


def copy_entrant_file(*, source: Path, run_dir: Path) -> Path:
    target = run_dir / f"{RUN_ENTRANT_STEM}{source.suffix}"
    target.write_bytes(source.read_bytes())
    return target


def keeps_entrant_file(*, run_dir: Path) -> bool:
    """Whether a run folder keeps an entrant file, which marks a self-run entrant's run whatever its meta.json says."""
    return any((run_dir / f"{RUN_ENTRANT_STEM}{suffix}").exists() for suffix in (".yaml", ".py"))


def run_entrant_file(*, run_dir: Path) -> Path | None:
    """The entrant file a run folder keeps (entrant.yaml or entrant.py), or None."""
    found = [run_dir / f"{RUN_ENTRANT_STEM}{suffix}" for suffix in (".yaml", ".py") if (run_dir / f"{RUN_ENTRANT_STEM}{suffix}").is_file()]
    if len(found) > 1:
        raise EntrantFileError(f"{run_dir} keeps both entrant.yaml and entrant.py; a run has one entrant file")
    return found[0] if found else None


def load_run_entrant(*, run_dir: Path, name: str) -> EntrantFile:
    """The entrant file a run keeps, checked like entrants/<name>.yaml (the name it gives must be the run's entrant)."""
    path = run_entrant_file(run_dir=run_dir)
    if path is None:
        raise EntrantFileError(f"{run_dir} keeps no entrant.yaml or entrant.py")
    return load_entrant_file(path=path, expected_name=name)


def reply_json(*, content: bytes) -> tuple[object | None, str | None]:
    """The parsed reply, or the malformed_response error that says why it is none."""
    try:
        return json.loads(content), None
    except (ValueError, UnicodeDecodeError):
        return None, "malformed_response: the body is not JSON"
