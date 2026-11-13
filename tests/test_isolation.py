from pathlib import Path

import pytest

from bench.isolation import entrant_isolation, isolation_problems, protected_paths, coach_tools_hint
from bench.llm import CommandResult, EntrantIsolation, EntrantIsolationError

ENTRANT = "bench_entrant"


class _Probe:
    """Answers `sudo -n -u <user> /bin/test <flag> <path>` from a set of readable paths, as the entrant user would."""

    def __init__(self, *, readable: set[str], sudo_works: bool = True, writable: bool = True, searchable: set[str] | None = None, kill_allowed: bool = True) -> None:
        self.readable, self.sudo_works, self.writable, self.searchable, self.kill_allowed = readable, sudo_works, writable, searchable or set(), kill_allowed
        self.calls: list[list[str]] = []

    def __call__(self, *, argv: list[str]) -> CommandResult:
        self.calls.append(argv)
        assert argv[:4] == ["sudo", "-n", "-u", ENTRANT] and argv[4] in ("/bin/test", "/bin/kill")
        if not self.sudo_works or (argv[4] == "/bin/kill" and not self.kill_allowed):
            return CommandResult(stdout="", stderr="sudo: a password is required", returncode=1)
        if argv[4] == "/bin/kill":
            return CommandResult(stdout="HUP INT KILL", stderr="", returncode=0)
        flag, path = argv[5], argv[6]
        allowed = {"-d": path == "/", "-r": path in self.readable, "-x": path in self.searchable, "-w": self.writable}[flag]
        return CommandResult(stdout="", stderr="", returncode=0 if allowed else 1)


def _paths(tmp_path: Path) -> list[Path]:
    return protected_paths(root=tmp_path / "repo", private_dir=tmp_path / "private", home=tmp_path / "home")


def test_protected_paths_cover_the_repository_its_secrets_and_data_the_private_set_and_the_home() -> None:
    paths = protected_paths(root=Path("/r"), private_dir=Path("/p"), home=Path("/h"))
    assert [p.as_posix() for p in paths] == ["/r", "/r/.env", "/r/data/public/expert_points", "/r/data/public/reference", "/r/runs", "/p", "/h"]
    assert Path("/p") not in protected_paths(root=Path("/r"), private_dir=None, home=Path("/h"))


def _work(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return work


def test_isolation_problems_lists_every_readable_path(tmp_path: Path) -> None:
    _work(tmp_path)
    paths = _paths(tmp_path)
    assert isolation_problems(entrant_user=ENTRANT, work_root=tmp_path / "work", paths=paths, probe=_Probe(readable=set())) == []
    problems = isolation_problems(entrant_user=ENTRANT, work_root=tmp_path / "work", paths=paths, probe=_Probe(readable={str(paths[1]), str(paths[4])}))
    assert len(problems) == 2 and str(paths[1]) in problems[0] and str(paths[4]) in problems[1]


def test_a_sudo_rule_that_does_not_work_or_a_work_root_it_cannot_write_is_a_problem(tmp_path: Path) -> None:
    _work(tmp_path)
    refused = isolation_problems(entrant_user=ENTRANT, work_root=tmp_path / "work", paths=_paths(tmp_path), probe=_Probe(readable=set(), sudo_works=False))
    assert len(refused) == 1 and "password is required" in refused[0]
    unwritable = isolation_problems(entrant_user=ENTRANT, work_root=tmp_path / "work", paths=_paths(tmp_path), probe=_Probe(readable=set(), writable=False))
    assert len(unwritable) == 1 and "not writable" in unwritable[0]


def test_a_with_tools_run_refuses_when_the_isolation_check_finds_a_readable_path(tmp_path: Path) -> None:
    _work(tmp_path)
    common = {"root": tmp_path / "repo", "private_dir": None, "home": tmp_path / "home"}
    with pytest.raises(EntrantIsolationError, match="ENTRANT_USER and ENTRANT_WORK_DIR"):
        entrant_isolation(entrant_user=None, entrant_work_dir=None, probe=_Probe(readable=set()), **common)
    with pytest.raises(EntrantIsolationError, match="ENTRANT_WORK_DIR"):
        entrant_isolation(entrant_user=ENTRANT, entrant_work_dir=None, probe=_Probe(readable=set()), **common)
    with pytest.raises(EntrantIsolationError, match=r"\.env"):
        entrant_isolation(entrant_user=ENTRANT, entrant_work_dir=tmp_path / "work", probe=_Probe(readable={str(tmp_path / "repo" / ".env")}), **common)
    assert entrant_isolation(entrant_user=ENTRANT, entrant_work_dir=tmp_path / "work", probe=_Probe(readable=set()), **common) == EntrantIsolation(user=ENTRANT, work_root=tmp_path / "work")


def test_only_with_tools_entrants_need_isolation() -> None:
    from bench.config import load_models
    from bench.isolation import needs_isolation

    models = load_models()
    assert needs_isolation(name="claude_tools", models=models) and needs_isolation(name="gemini_tools", models=models)
    assert not needs_isolation(name="claude_default", models=models) and not needs_isolation(name="sample_coach", models=models) and not needs_isolation(name="claude_engine", models=models)


def test_a_searchable_folder_a_kill_rule_that_does_not_work_or_a_work_root_the_operator_cannot_use_is_a_problem(tmp_path: Path) -> None:
    work = _work(tmp_path)
    paths = _paths(tmp_path)
    searchable = isolation_problems(entrant_user=ENTRANT, work_root=work, paths=paths, probe=_Probe(readable=set(), searchable={str(paths[2])}))
    assert len(searchable) == 1 and str(paths[2]) in searchable[0] and "search" in searchable[0]
    no_kill = isolation_problems(entrant_user=ENTRANT, work_root=work, paths=paths, probe=_Probe(readable=set(), kill_allowed=False))
    assert len(no_kill) == 1 and "/bin/kill" in no_kill[0]
    missing = isolation_problems(entrant_user=ENTRANT, work_root=tmp_path / "absent", paths=paths, probe=_Probe(readable=set()))
    assert len(missing) == 1 and "operator" in missing[0]


def test_the_coach_tools_must_be_runnable_by_the_entrant_user(tmp_path: Path) -> None:
    python, stockfish, package = tmp_path / "venv" / "bin" / "python", Path("/opt/homebrew/bin/stockfish"), tmp_path / "venv" / "lib" / "python3.13" / "site-packages" / "chess" / "__init__.py"
    common = {"entrant_user": ENTRANT, "work_root": tmp_path, "paths": _paths(tmp_path), "tools": [python, stockfish, package]}
    assert isolation_problems(probe=_Probe(readable={str(package)}, searchable={str(python), str(stockfish)}), **common) == []
    problems = isolation_problems(probe=_Probe(readable=set(), searchable={str(python)}), **common)
    assert problems == [f"{stockfish} is not runnable by {ENTRANT}, and the coach_tools prompt names it ({coach_tools_hint()})", f"{package} is not readable by {ENTRANT}, and the coach_tools prompt names it ({coach_tools_hint()})"]
