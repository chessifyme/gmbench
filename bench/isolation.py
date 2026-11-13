"""Keeps with-tools entrants away from the benchmark: they run as a separate local user that must not be able to read the
repository, its secrets and data, other entrants' runs, the private set or the operator's home."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from bench.config import BenchConfig, CoachTools, DataEnv, ModelsConfig
from bench.extensions import private_hook
from bench.llm import CommandResult, SandboxIsolation, EntrantIsolation, EntrantIsolationError

Probe = Callable[..., CommandResult]
PROBE_TIMEOUT_S = 30


def sudo_probe(*, argv: list[str]) -> CommandResult:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=PROBE_TIMEOUT_S)
    return CommandResult(stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode)


def coach_tool_paths(*, models: ModelsConfig, tools: CoachTools | None, entrant: str | None = None) -> list[Path]:
    """What the entrant user must be able to run or read when the entrant (or, with none named, any entrant) uses the coach_tools
    prompt: the interpreter, the python-chess package beside it and the engine; none otherwise."""
    specs = {**models.entrants, **models.calibration_entrants}
    chosen = [specs[entrant]] if entrant is not None and entrant in specs else [] if entrant is not None else list(specs.values())
    if not any(spec.prompt == "coach_tools" for spec in chosen) or tools is None:
        return []
    venv = tools.python.parent.parent
    packages = sorted(venv.glob("lib/python*/site-packages/chess/__init__.py"))
    return [tools.python, tools.stockfish, *(packages[:1] or [venv / "lib" / "site-packages" / "chess" / "__init__.py"])]


def needs_isolation(*, name: str, models: ModelsConfig) -> bool:
    spec = models.entrants.get(name) or models.calibration_entrants.get(name)
    return spec is not None and spec.tools


def protected_paths(*, root: Path, private_dir: Path | None, home: Path) -> list[Path]:
    return [root, root / ".env", root / "data" / "public" / "sheets", root / "data" / "public" / "reference", root / "runs", *([private_dir] if private_dir is not None else []), home]


def _as_entrant(*, user: str, argv: list[str]) -> list[str]:
    return ["sudo", "-n", "-u", user, *argv]


def isolation_problems(*, entrant_user: str, work_root: Path, paths: list[Path], probe: Probe = sudo_probe, tools: list[Path] | None = None) -> list[str]:
    """Every protected path the entrant user can read or search (a searchable folder gives away any file whose name is known),
    whether the sudo rule runs /bin/test and /bin/kill for it, and whether the work root works for both users."""
    control = probe(argv=_as_entrant(user=entrant_user, argv=["/bin/test", "-d", "/"]))
    if control.returncode != 0:
        return [f"sudo -n -u {entrant_user} /bin/test does not run ({control.stderr.strip()[:200] or f'exit {control.returncode}'}); the sudo rule must let the operator run the clients, /bin/test and /bin/kill as that user without a password"]
    problems: list[str] = []
    if probe(argv=_as_entrant(user=entrant_user, argv=["/bin/kill", "-l"])).returncode != 0:
        problems.append(f"sudo -n -u {entrant_user} /bin/kill does not run; without it a timed-out answer's processes cannot be stopped")
    for path in paths:
        if probe(argv=_as_entrant(user=entrant_user, argv=["/bin/test", "-r", str(path)])).returncode == 0:
            problems.append(f"{path} is readable by {entrant_user}")
        elif probe(argv=_as_entrant(user=entrant_user, argv=["/bin/test", "-x", str(path)])).returncode == 0:
            problems.append(f"{path} is searchable by {entrant_user}, so any file inside whose name is known can be read")
    if probe(argv=_as_entrant(user=entrant_user, argv=["/bin/test", "-w", str(work_root)])).returncode != 0:
        problems.append(f"{work_root} is not writable by {entrant_user}; with-tools calls get their work folders there")
    if not (work_root.is_dir() and os.access(work_root, os.W_OK | os.X_OK)):
        problems.append(f"{work_root} is not a folder the operator can create work folders in")
    for tool in tools or []:
        flag = "-r" if tool.suffix == ".py" else "-x"
        if probe(argv=_as_entrant(user=entrant_user, argv=["/bin/test", flag, str(tool)])).returncode != 0:
            problems.append(f"{tool} is not {'readable' if flag == '-r' else 'runnable'} by {entrant_user}, and the coach_tools prompt names it ({coach_tools_hint()})")
    return problems


def coach_tools_hint() -> str:
    """Where the tools the coach_tools prompt names come from: the configuration, or the working repository's own installer."""
    hint = private_hook(name="coach_tools_hint")
    return str(hint()) if hint is not None else "config: coach_tools"


def entrant_isolation(*, entrant_user: str | None, entrant_work_dir: Path | None, root: Path, private_dir: Path | None, home: Path, probe: Probe = sudo_probe, tools: list[Path] | None = None) -> EntrantIsolation:
    """The isolation a with-tools run needs, after checking it; raises EntrantIsolationError with everything that is missing or readable."""
    missing = [name for name, value in (("ENTRANT_USER", entrant_user), ("ENTRANT_WORK_DIR", entrant_work_dir)) if value is None]
    if entrant_user is None or entrant_work_dir is None:
        raise EntrantIsolationError(f"a with-tools entrant runs only as a separate local user that cannot read the benchmark; set {' and '.join(missing)} in .env (see .env.example), then run make entrant_isolation_check")
    problems = isolation_problems(entrant_user=entrant_user, work_root=entrant_work_dir, paths=protected_paths(root=root, private_dir=private_dir, home=home), probe=probe, tools=tools)
    if problems:
        raise EntrantIsolationError("the entrant user is not isolated, so no with-tools run starts:\n  " + "\n  ".join(problems) + "\nfix the permissions, then run make entrant_isolation_check")
    return EntrantIsolation(user=entrant_user, work_root=entrant_work_dir)


def isolation_from_env(*, env: DataEnv, cfg: BenchConfig, models: ModelsConfig, root: Path, home: Path, entrant: str | None = None) -> EntrantIsolation | SandboxIsolation:
    """The isolation .env chooses: the separate local user (ENTRANT_ISOLATION empty or user, the default) or a sandbox a private
    extension provides under another name."""
    if env.entrant_isolation not in (None, "user"):
        sandboxed = private_hook(name="sandbox_isolation")
        if sandboxed is None:
            raise EntrantIsolationError(f"ENTRANT_ISOLATION={env.entrant_isolation} is not available in this checkout; leave it empty to run with-tools entrants as ENTRANT_USER")
        isolation: SandboxIsolation = sandboxed(kind=env.entrant_isolation, env=env, cfg=cfg, root=root, home=home)
        return isolation
    return entrant_isolation(entrant_user=env.entrant_user, entrant_work_dir=env.entrant_work_dir, root=root, private_dir=env.private_data_dir, home=home, tools=coach_tool_paths(models=models, tools=cfg.coach_tools, entrant=entrant))
