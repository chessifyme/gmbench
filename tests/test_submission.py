import json
import math
import os
import socket
from collections.abc import Iterator
import re
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from bench.adapters.http_entrant import HttpEntrantAdapter
from bench.adapters.registry import entrant_file_adapter
from bench.aggregate import aggregate
from bench.config import load_bench
from bench.entrants import ENTRANTS_DIR, copy_entrant_file
from bench.io_yaml import load_all_tasks, read_task
from bench.runner import answer_with_retry, run_entrant
from bench.provenance import file_sha256
from bench.runs import RunMeta, VerificationLine, append_verification, board_provenance, config_hash, ledger_path, read_meta, read_verifications, start_run, write_meta
from bench.site.build import build_site
from bench.entrants import HttpEntrantFile
from bench.submission import compare_sample, run_disagreement, sample_tasks, submission_problems, verification_refusal
from tests.fake_agent_server import fake_agent_server
from tests.run_fakes import COMMIT, FIXTURE_TASK_FAMILY, fake_sheets, make_run

FIXTURES = Path(__file__).parent / "fixtures"
TASKS = [read_task(path=FIXTURES / "tasks" / "ap_9001.yaml"), read_task(path=FIXTURES / "tasks" / "gr_9001.yaml")]
RUNS = 2
CFG = load_bench()
KEY = "fake_entrant_key_0123456789abcdef"


def _meta(*, card: object, entrant_file: Path, provenance: str | None = "self_reported") -> RunMeta:
    return RunMeta.model_validate(
        {
            "entrant": "example_http",
            "card": card,
            "dataset_version": CFG.dataset_version,
            "task_set": "public",
            "grader_version": None,
            "started_at": datetime.now(UTC),
            "models_config_hash": "m",
            "bench_config_hash": "b",
            "git_commit": COMMIT,
            "client_versions": {},
            "engine_id": None,
            "engine_sha256": None,
            "temperature": "service_default",
            "registration_sha256": None,
            "entrant_config_hash": config_hash(path=entrant_file),
            "provenance": provenance,
        }
    )


@pytest.fixture
def submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A self-run entrant's run folder, made the way make run makes it, against the fake agent server."""
    monkeypatch.setenv("BENCH_ENTRANT_EXAMPLE_HTTP_KEY", KEY)
    server = fake_agent_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    example = yaml.safe_load((ENTRANTS_DIR / "example_http.yaml").read_text(encoding="utf-8"))
    example["http"]["url"] = f"http://127.0.0.1:{server.server_address[1]}/answer"
    example["http"]["headers"] = {"Authorization": "Bearer ${BENCH_ENTRANT_EXAMPLE_HTTP_KEY}"}
    entrant_file = tmp_path / "entrants" / "example_http.yaml"
    entrant_file.parent.mkdir()
    entrant_file.write_text(yaml.safe_dump(example), encoding="utf-8")
    adapter = entrant_file_adapter(path=entrant_file, cfg=CFG)
    run_dir = start_run(meta=_meta(card=adapter.card(), entrant_file=entrant_file), runs_dir=tmp_path / "runs", root=tmp_path, registrations_dir=tmp_path / "registrations")
    copy_entrant_file(source=entrant_file, run_dir=run_dir)
    try:
        run_entrant(adapter=adapter, tasks=TASKS, runs_per_task=RUNS, timeout_s=30, concurrency=2, out_dir=run_dir)
    finally:
        server.shutdown()
    return run_dir


def _problems(run_dir: Path, *, graded: bool = False, maintained: tuple[str, ...] = ("claude_default",)) -> list[str]:
    return submission_problems(
        run_dir=run_dir,
        tasks=TASKS,
        runs_per_task=RUNS,
        dataset_version=CFG.dataset_version,
        graded=graded,
        lookup=lambda v: KEY if v == "BENCH_ENTRANT_EXAMPLE_HTTP_KEY" else None,
        maintained=maintained,
        runs_dir=run_dir.parent.parent,
    )


def _rows(run_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()]


def _write_rows(run_dir: Path, rows: list[dict[str, object]]) -> None:
    (run_dir / "answers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_a_complete_self_run_folder_passes(submission: Path) -> None:
    assert sorted(p.name for p in submission.iterdir()) == ["answers.jsonl", "entrant.yaml", "meta.json"]
    assert read_meta(run_dir=submission).provenance == "self_reported"
    assert _problems(submission) == []


def test_a_self_reported_run_never_enters_the_attempt_ledger(submission: Path) -> None:
    meta = read_meta(run_dir=submission).model_copy(update={"registration_sha256": "f" * 64, "git_dirty": False})
    root = submission.parents[2]
    start_run(meta=meta, runs_dir=root / "runs2", root=root, registrations_dir=root / "registrations")
    assert not ledger_path(dataset_version=CFG.dataset_version, registrations_dir=root / "registrations").exists()


def test_a_missing_answer_a_duplicate_and_a_stray_answer_are_named(submission: Path) -> None:
    rows = _rows(submission)
    first = rows[0]
    stray = {**first, "task_id": "mj_9999"}
    _write_rows(submission, [*rows[1:], rows[1], stray])
    problems = "\n".join(_problems(submission))
    assert f"1 of {len(TASKS) * RUNS} task runs have no answer" in problems and f"{first['task_id']} run {first['run_index']}" in problems
    assert "answered more than once" in problems and "1 answer(s) are for no task run of the set: mj_9999" in problems


def test_an_answer_that_is_not_the_one_in_its_raw_reply_is_refused(submission: Path) -> None:
    rows = _rows(submission)
    rows[0]["text"] = "A better answer written afterwards."
    _write_rows(submission, rows)
    assert any("1 answer(s) differ from the answer read from their raw reply" in p for p in _problems(submission))


def test_an_answer_without_its_raw_reply_is_refused(submission: Path) -> None:
    rows = _rows(submission)
    del rows[0]["raw"]["reply"]  # type: ignore[attr-defined]
    _write_rows(submission, rows)
    assert any("keep no raw reply" in p for p in _problems(submission))


def test_a_changed_entrant_file_or_card_is_refused(submission: Path) -> None:
    entrant = submission / "entrant.yaml"
    data = yaml.safe_load(entrant.read_text(encoding="utf-8"))
    data["tools"] = ["engine"]
    entrant.write_text(yaml.safe_dump(data), encoding="utf-8")
    problems = "\n".join(_problems(submission))
    assert "is not the entrant file the run started with" in problems and "card in meta.json differs" in problems


def test_a_missing_entrant_file_or_meta_is_refused(submission: Path) -> None:
    (submission / "entrant.yaml").unlink()
    assert any("keeps no entrant.yaml or entrant.py" in p for p in _problems(submission))
    (submission / "meta.json").unlink()
    assert _problems(submission) == [f"{submission} has no meta.json; submit the whole run folder make run wrote"]


def test_a_maintainer_run_is_not_a_submission(submission: Path) -> None:
    write_meta(meta=read_meta(run_dir=submission).model_copy(update={"provenance": None}), run_dir=submission)
    assert any("not self-reported (provenance maintainer_run)" in p for p in _problems(submission))


def test_grades_belong_to_the_maintainers(submission: Path) -> None:
    (submission / "grades.jsonl").write_text("", encoding="utf-8")
    assert any("remove grades.jsonl" in p for p in _problems(submission))
    assert _problems(submission, graded=True) == []


# the key-shaped string is built at run time, so the repository's own leak scan (make publish_scan) never sees one
@pytest.mark.parametrize("secret", ["sk-" + "proj-abcdefghijklmnopqrstuvwxyz0123", "Bearer abcdefghijklmnopqrstuvwxyz", KEY])
def test_secrets_in_any_file_are_found_and_never_printed(submission: Path, secret: str) -> None:
    (submission / "notes.txt").write_text(f"debug: {secret}\n", encoding="utf-8")
    problems = _problems(submission)
    assert any(p.startswith("notes.txt holds") for p in problems)
    assert all(secret not in p for p in problems)


def test_the_entrant_file_keeps_its_key_as_a_variable_only(submission: Path) -> None:
    text = (submission / "entrant.yaml").read_text(encoding="utf-8")
    assert "${BENCH_ENTRANT_EXAMPLE_HTTP_KEY}" in text and KEY not in text


def test_a_submission_cannot_arrive_verified(submission: Path) -> None:
    write_meta(meta=read_meta(run_dir=submission).model_copy(update={"provenance": "verified"}), run_dir=submission)
    assert any("only the maintainers mark one verified" in p for p in _problems(submission))
    assert any("only the maintainers mark one verified" in p for p in _problems(submission, graded=True))


@pytest.mark.parametrize("made_up", ["verification.json", "verification"])
def test_a_submission_cannot_bring_its_own_verification(submission: Path, made_up: str) -> None:
    target = submission / made_up
    if made_up.endswith(".json"):
        target.write_text('{"attempts": [{"passed": true}]}', encoding="utf-8")
    else:
        target.mkdir()
    assert any(f"remove {made_up}: only the maintainers verify" in p for p in _problems(submission))


def test_a_submission_named_like_a_maintainers_entrant_is_refused(submission: Path) -> None:
    assert any("one of the maintainers' entrants" in p for p in _problems(submission, maintained=("example_http",)))


def test_a_name_already_on_the_board_for_another_entrant_is_refused(submission: Path) -> None:
    other = submission.parent / "20200101_000000"
    other.mkdir()
    meta = read_meta(run_dir=submission)
    write_meta(meta=meta.model_copy(update={"provenance": None}), run_dir=other)
    assert any("already on the board as an entrant the maintainers run" in p for p in _problems(submission))
    write_meta(meta=meta.model_copy(update={"card": meta.card.model_copy(update={"provider": "Someone Else"})}), run_dir=other)
    assert any("already on the board for another entrant" in p and "Someone Else" in p for p in _problems(submission))
    write_meta(meta=meta, run_dir=other)  # an earlier submission of the same entrant
    assert _problems(submission) == []


def test_a_submission_is_never_affiliated_with_the_maintainers(submission: Path) -> None:
    meta = read_meta(run_dir=submission)
    write_meta(meta=meta.model_copy(update={"card": meta.card.model_copy(update={"affiliated_with_maintainer": True})}), run_dir=submission)
    assert any("affiliated_with_maintainer is true" in p for p in _problems(submission))


@pytest.mark.parametrize("link", ["verification.json", "verification", "notes.txt"])
def test_links_in_a_submission_are_refused_before_anything_is_read_or_written(submission: Path, tmp_path: Path, link: str) -> None:
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (submission / link).symlink_to(outside / "x" if "." in link else outside)
    problems = _problems(submission, graded=True)
    assert problems == [f"{link} is a symbolic link; a submission holds plain files only"]


def test_a_special_file_in_a_submission_is_refused(submission: Path) -> None:
    os.mkfifo(submission / "pipe")
    assert _problems(submission) == ["pipe is not a plain file; a submission holds plain files only"]


def test_a_self_reported_run_may_start_from_an_uncommitted_tree(submission: Path) -> None:
    meta = read_meta(run_dir=submission).model_copy(update={"registration_sha256": "f" * 64, "git_dirty": True})
    root = submission.parents[2]
    assert start_run(meta=meta, runs_dir=root / "runs3", root=root, registrations_dir=root / "registrations").is_dir()
    with pytest.raises(ValueError, match="uncommitted changes"):
        start_run(meta=meta.model_copy(update={"provenance": None}), runs_dir=root / "runs3", root=root, registrations_dir=root / "registrations")


def test_a_maintainers_run_always_stands_over_a_submission_on_the_board(tmp_path: Path) -> None:
    from bench.cli.aggregate import graded_latest_runs

    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY), signed=1)
    common = {"task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "families": {"assessment_plan"}, "scores": {"ap_9001": 50}}
    ours = make_run(runs, entrant="acme_coach", **common)
    later = ours.parent / "29990101_000000"
    shutil.copytree(ours, later)
    write_meta(meta=read_meta(run_dir=later).model_copy(update={"provenance": "verified"}), run_dir=later)
    assert graded_latest_runs(runs_dir=runs) == [ours]
    shutil.copytree(ours, runs / "_verifications" / "acme_coach" / "20990101_000000_abc123")  # a verification re-run is never an entrant
    assert graded_latest_runs(runs_dir=runs) == [ours]


# verification


def test_the_sample_is_random_reproducible_and_never_larger_than_the_set() -> None:
    ids = [f"ap_{i:04d}" for i in range(45)]
    first = sample_tasks(task_ids=ids, size=10, seed=7)
    assert first == sample_tasks(task_ids=list(reversed(ids)), size=10, seed=7) and len(set(first)) == 10
    assert first != sample_tasks(task_ids=ids, size=10, seed=8)
    assert sample_tasks(task_ids=ids[:3], size=10, seed=7) == ids[:3]
    with pytest.raises(ValueError, match="at least one"):
        sample_tasks(task_ids=ids, size=0, seed=1)


SAMPLE = ["ap_0001", "ap_0002"]
STEADY = {("ap_0001", r): 60.0 for r in range(3)} | {("ap_0002", r): 40.0 for r in range(3)}
NOISY = {("ap_0001", 0): 40.0, ("ap_0001", 1): 60.0, ("ap_0001", 2): 80.0, ("ap_0002", 0): 20.0, ("ap_0002", 1): 40.0, ("ap_0002", 2): 60.0}


@pytest.mark.parametrize(("rerun", "passed"), [({"ap_0001": 60.0, "ap_0002": 40.0}, True), ({"ap_0001": 65.0, "ap_0002": 45.0}, True), ({"ap_0001": 55.0, "ap_0002": 35.0}, True), ({"ap_0001": 66.0, "ap_0002": 46.0}, False), ({"ap_0001": 40.0, "ap_0002": 40.0}, False)])
def test_steady_runs_get_the_floor_of_five_points(rerun: dict[str, float], passed: bool) -> None:
    result = compare_sample(submitted=STEADY, rerun=rerun, sample=SAMPLE, runs_per_task=3)
    assert (result.submitted_mean, result.standard_error, result.tolerance) == (50.0, 0.0, 5.0)
    assert result.passed is passed and result.difference == result.rerun_mean - 50.0


def test_noisy_runs_widen_the_tolerance_to_two_standard_errors_up_to_ten_points() -> None:
    # each task varies by 20 points between runs (variance 400): SE = sqrt(2 * 400 * (1 + 1/3)) / 2 = 16.33, so 2 SE is capped at 10
    result = compare_sample(submitted=NOISY, rerun={"ap_0001": 70.0, "ap_0002": 50.0}, sample=SAMPLE, runs_per_task=3)
    assert result.submitted_mean == 50.0 and result.standard_error == pytest.approx(16.3299, abs=1e-3) and result.tolerance == 10.0 and result.passed is True
    assert compare_sample(submitted=NOISY, rerun={"ap_0001": 71.0, "ap_0002": 50.0}, sample=SAMPLE, runs_per_task=3).passed is False
    # between the floor and the cap: variance 12.5 per task gives 2 SE = 2 * sqrt(2 * 12.5 * 4/3) / 2 = 5.77
    middling = {(t, r): 50.0 + (5.0 if r == 0 else 0.0) for t in SAMPLE for r in range(2)}
    assert compare_sample(submitted=middling, rerun={t: 52.5 for t in SAMPLE}, sample=SAMPLE, runs_per_task=2).tolerance == pytest.approx(2 * math.sqrt(2 * 12.5 * 1.5) / 2, abs=1e-3)


def test_a_weak_decoy_run_does_not_hide_two_made_up_strong_runs() -> None:
    sample = [f"ap_{i:04d}" for i in range(10)]
    gamed = {(t, r): (0.0 if r == 2 else 90.0) for t in sample for r in range(3)}
    result = compare_sample(submitted=gamed, rerun={t: 45.0 for t in sample}, sample=sample, runs_per_task=3)
    assert result.submitted_mean == 60.0 and result.tolerance == 10.0 and result.passed is False
    assert run_disagreement(submitted=gamed, task_ids=sample, runs_per_task=3) == "run 0 averages 90.0 on the full set, the other runs 45.0: more than 10 points apart"


def test_runs_that_agree_within_ten_points_can_be_verified_on_a_sample() -> None:
    ids = ["ap_0001", "ap_0002"]
    close = {(t, r): 50.0 + 5.0 * r for t in ids for r in range(3)}  # run means 50, 55, 60: run 0 is 7.5 from the others
    assert run_disagreement(submitted=close, task_ids=ids, runs_per_task=3) is None
    apart = {(t, r): 50.0 + 7.0 * r for t in ids for r in range(3)}  # 50, 57, 64: run 0 is 10.5 from the others
    assert "more than 10 points apart" in (run_disagreement(submitted=apart, task_ids=ids, runs_per_task=3) or "")
    with pytest.raises(ValueError, match="no grade for ap_0002 run 2"):
        run_disagreement(submitted={k: v for k, v in close.items() if k != ("ap_0002", 2)}, task_ids=ids, runs_per_task=3)


def test_a_comparison_without_every_grade_stops() -> None:
    with pytest.raises(ValueError, match=r"no grade for ap_0002 \(re-run\), ap_0002 run 2;"):
        compare_sample(submitted={k: v for k, v in STEADY.items() if k != ("ap_0002", 2)}, rerun={"ap_0001": 50.0}, sample=SAMPLE, runs_per_task=3)


# the endpoints a maintainers' re-run posts to

# addresses built at run time, so the release's leak scan (which flags IP addresses in shipped files) sees none
PUBLIC_IP = ".".join(map(str, (93, 184, 216, 34)))
PRIVATE_IP = ".".join(map(str, (10, 0, 0, 5)))
METADATA_IP = ".".join(map(str, (169, 254, 169, 254)))
FAKE_DNS = {
    "api.acme.example": PUBLIC_IP,
    "127.0.0.1.nip.io": "127.0.0.1",
    "rebind.acme.example": ".".join(map(str, (10, 0, 0, 7))),
    "intranet": ".".join(map(str, (192, 168, 1, 4))),
    "printer.local": ".".join(map(str, (169, 254, 3, 3))),
}


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    real = socket.getaddrinfo

    def resolve(host: str, port: object, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        if host in FAKE_DNS:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (FAKE_DNS[host], port))]
        return real(host, port, *args, **kwargs)  # IP literals and short forms such as 127.1, resolved without the network

    monkeypatch.setattr(socket, "getaddrinfo", resolve)


def _http_entrant(url: str, name: str = "acme_coach") -> HttpEntrantFile:
    card = {"name": name, "label": "x", "version": "1", "provider": "x", "declared_family": "undisclosed", "tools": [], "affiliated_with_maintainer": False}
    return HttpEntrantFile.model_validate({**card, "http": {"url": url, "body": {"q": "{message}"}, "answer": "a"}})


@pytest.mark.parametrize(
    ("url", "name", "refused"),
    [
        ("https://api.acme.example/ask", "acme_coach", None),
        (f"https://{PUBLIC_IP}/ask", "acme_coach", None),
        ("http://127.0.0.1:8808/answer", "acme_coach", "only to an https:// endpoint"),
        ("http://localhost/answer", "acme_coach", "only to an https:// endpoint"),
        ("https://127.0.0.1.nip.io/ask", "acme_coach", "not public"),
        ("https://rebind.acme.example/ask", "acme_coach", "not public"),
        ("https://127.1/ask", "acme_coach", "not public"),
        (f"https://{PRIVATE_IP}/ask", "acme_coach", "not public"),
        (f"https://{METADATA_IP}/latest", "acme_coach", "not public"),
        ("https://[::1]/ask", "acme_coach", "not public"),
        ("https://intranet/ask", "acme_coach", "not public"),
        ("https://printer.local/ask", "acme_coach", "not public"),
        ("https://api.acme.example/ask", "claude_default", "an entrant the maintainers configure"),
    ],
)
def test_the_maintainers_rerun_only_public_endpoints_under_names_of_their_own(fake_dns: None, url: str, name: str, refused: str | None) -> None:
    reason = verification_refusal(entrant=_http_entrant(url, name), maintained=("claude_default",))
    assert reason is None if refused is None else refused in (reason or "")


def test_a_rerun_request_goes_to_the_checked_address_with_the_original_host_and_tls_name(fake_dns: None) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"a": "Nf3 is best."})

    adapter = HttpEntrantAdapter(entrant=_http_entrant("https://api.acme.example:8443/ask"), answer_timeout_s=900, transport=httpx.MockTransport(handler), public_only=True)
    record = adapter.answer(task=TASKS[0], run_index=0, timeout_s=30)
    assert record.status == "ok" and str(seen[0].url) == f"https://{PUBLIC_IP}:8443/ask"
    assert seen[0].headers["host"] == "api.acme.example:8443" and seen[0].extensions["sni_hostname"] == "api.acme.example"


def test_a_name_that_turns_private_between_requests_is_refused_and_not_retried(fake_dns: None) -> None:
    calls: list[httpx.Request] = []
    adapter = HttpEntrantAdapter(entrant=_http_entrant("https://rebind.acme.example/ask"), answer_timeout_s=900, transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, json={"a": "x"})), public_only=True)
    record = answer_with_retry(adapter=adapter, task=TASKS[0], run_index=0, timeout_s=30)
    assert record.status == "error" and record.error is not None and record.error.startswith("blocked_address:") and calls == [] and "retried_after" not in record.raw


# the board


def _ledger_line(entrant: str, *, answers_sha256: str, passed: bool, dataset_version: str = CFG.dataset_version) -> VerificationLine:
    return VerificationLine(entrant=entrant, dataset_version=dataset_version, answers_sha256=answers_sha256, passed=passed, verified_at=datetime.now(UTC))


def test_the_board_believes_only_the_ledger_and_only_for_the_answers_it_verified(submission: Path) -> None:
    meta = read_meta(run_dir=submission)
    claimed = meta.model_copy(update={"provenance": "verified"})
    assert board_provenance(meta=claimed, answers_sha256="a" * 64, lines=[], keeps_entrant_file=True) == "self_reported"  # a forged meta.json
    passed = [_ledger_line("example_http", answers_sha256="a" * 64, passed=True)]
    assert board_provenance(meta=meta, answers_sha256="a" * 64, lines=passed, keeps_entrant_file=True) == "verified"
    assert board_provenance(meta=meta, answers_sha256="b" * 64, lines=passed, keeps_entrant_file=True) == "self_reported"  # other answers than the verified ones
    assert board_provenance(meta=meta, answers_sha256="a" * 64, lines=[_ledger_line("example_http", answers_sha256="a" * 64, passed=True, dataset_version="v9")], keeps_entrant_file=True) == "self_reported"
    failed_first = [_ledger_line("example_http", answers_sha256="a" * 64, passed=False), *passed]
    assert board_provenance(meta=meta, answers_sha256="a" * 64, lines=failed_first, keeps_entrant_file=True) == "self_reported"
    unmarked = meta.model_copy(update={"provenance": None})
    assert board_provenance(meta=unmarked, answers_sha256="a" * 64, lines=[], keeps_entrant_file=False) == "maintainer_run"
    # a folder that keeps an entrant file is a self-run entrant's, even with provenance removed from meta.json
    assert board_provenance(meta=unmarked, answers_sha256="a" * 64, lines=[], keeps_entrant_file=True) == "self_reported"
    assert board_provenance(meta=meta.model_copy(update={"provenance": "maintainer_run"}), answers_sha256="a" * 64, lines=[], keeps_entrant_file=True) == "self_reported"


def test_the_board_shows_who_ran_each_entrant(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY), signed=1)
    families = {"assessment_plan", "move_choice", "move_judgement", "game_review"}
    scores = {"ap_9001": 60, "mj_9001": 50, "mc_9001": 70, "gr_9001": 40}
    common = {"task_family": FIXTURE_TASK_FAMILY, "sheets": sheets, "families": families, "access_method": "api"}
    dirs = {name: make_run(runs, entrant=name, scores=scores, **common) for name in ("acme_coach", "beta_coach", "claude_default")}
    for name in ("acme_coach", "beta_coach"):
        write_meta(meta=read_meta(run_dir=dirs[name]).model_copy(update={"provenance": "self_reported"}), run_dir=dirs[name])
    ledger = tmp_path / "registrations" / "verifications.jsonl"
    append_verification(line=_ledger_line("beta_coach", answers_sha256=file_sha256(path=dirs["beta_coach"] / "answers.jsonl"), passed=True), ledger=ledger)
    board = aggregate(run_dirs=list(dirs.values()), cfg=CFG, sheets=sheets, tasks=load_all_tasks(tasks_dir=FIXTURES / "tasks"), resamples=20, seed=1, root=tmp_path, verifications_ledger=ledger)
    assert {e.name: e.provenance for e in board.entrants} == {"acme_coach": "self_reported", "beta_coach": "verified", "claude_default": "maintainer_run"}
    leaderboard = runs / "leaderboard.json"
    leaderboard.write_text(board.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(FIXTURES / "tasks", tmp_path / "data" / "tasks")
    shutil.copytree(FIXTURES / "sheets", tmp_path / "data" / "sheets")
    build_site(leaderboard_path=leaderboard, run_dirs=list(dirs.values()), tasks_dir=tmp_path / "data" / "tasks", sheets_dir=tmp_path / "data" / "sheets", out_dir=tmp_path / "site", cfg=CFG, root=tmp_path, registrations_dir=tmp_path / "registrations")
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "<th>Entrant</th><th>Run by</th>" in index
    for name, label in (("acme_coach", "maker, self-reported"), ("beta_coach", "maker, verified"), ("claude_default", "maintainers")):
        row = re.search(rf'<tr data-entrant="{name}".*?</tr>', index, re.DOTALL)
        assert row is not None and f"<td>{label}</td>" in row.group(0)


def test_a_maintainers_run_meta_is_written_without_a_provenance_field(submission: Path) -> None:
    write_meta(meta=read_meta(run_dir=submission).model_copy(update={"provenance": None}), run_dir=submission)
    assert "provenance" not in json.loads((submission / "meta.json").read_text(encoding="utf-8"))


def _verify(monkeypatch: pytest.MonkeyPatch, run_dir: Path, *, rerun_score: float, submitted: dict[int, float], inside_repository: bool = True) -> None:
    """make verify_submission on the fixture tasks, the fake server standing in for the entrant and fixed grades for the grader."""
    from types import SimpleNamespace

    import bench.cli.verify_submission as verify

    def grades(*, run_dir: Path) -> list[SimpleNamespace]:
        if "_verifications" in run_dir.parts:
            return [SimpleNamespace(task_id=t.task_id, run_index=0, score=SimpleNamespace(score=rerun_score)) for t in TASKS]
        return [SimpleNamespace(task_id=t.task_id, run_index=r, score=SimpleNamespace(score=s)) for t in TASKS for r, s in submitted.items()]

    tasks_dir = run_dir.parents[2] / "public_tasks"
    tasks_dir.mkdir(exist_ok=True)
    for task in TASKS:
        shutil.copy(FIXTURES / "tasks" / f"{task.task_id}.yaml", tasks_dir)
    monkeypatch.setattr(verify, "DATA_PUBLIC_TASKS", tasks_dir)
    monkeypatch.setattr(verify, "load_bench", lambda: CFG.model_copy(update={"runs_per_task": RUNS, "run_status_allowed": sorted({t.status for t in TASKS})}))
    monkeypatch.setattr(verify, "grade_run", lambda *, run_dir: None)
    # the fake server is on this machine, which a real re-run refuses
    monkeypatch.setattr(verify, "verification_refusal", lambda **kwargs: None)
    monkeypatch.setattr(verify, "HttpEntrantAdapter", lambda **kwargs: HttpEntrantAdapter(**{**kwargs, "public_only": False}))
    monkeypatch.setattr(verify, "read_grades", grades)
    monkeypatch.setattr(verify, "submission_problems", lambda **kwargs: _problems(kwargs["run_dir"], graded=True))
    write_meta(meta=read_meta(run_dir=run_dir).model_copy(update={"grader_version": verify.GRADER_VERSION}), run_dir=run_dir)
    ledger = run_dir.parents[2] / "registrations" / "verifications.jsonl"
    if inside_repository:
        monkeypatch.setattr(verify, "ROOT", run_dir.parents[2])
    monkeypatch.setattr(verify, "git_head", lambda *, root: COMMIT)
    monkeypatch.setattr("sys.argv", ["verify_submission", "--run", str(run_dir), "--sample", "5", "--seed", "3", "--ledger", str(ledger)])
    verify.main()


@pytest.fixture
def served_submission(submission: Path) -> Iterator[Path]:
    server = fake_agent_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    path = submission / "entrant.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["http"]["url"] = f"http://127.0.0.1:{server.server_address[1]}/answer"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    write_meta(meta=read_meta(run_dir=submission).model_copy(update={"entrant_config_hash": config_hash(path=path)}), run_dir=submission)
    yield submission
    server.shutdown()


def _ledger(run_dir: Path) -> list[VerificationLine]:
    return read_verifications(ledger=run_dir.parents[2] / "registrations" / "verifications.jsonl")


def test_verify_submission_records_a_close_rerun_as_passed_in_the_maintainers_ledger(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = sorted(p.name for p in served_submission.iterdir())
    _verify(monkeypatch, served_submission, rerun_score=57.0, submitted={0: 50.0, 1: 60.0})
    [line] = _ledger(served_submission)
    assert line.passed is True and line.sample == sorted(t.task_id for t in TASKS) and line.seed == 3
    assert (line.entrant, line.answers_sha256) == ("example_http", file_sha256(path=served_submission / "answers.jsonl"))
    assert (line.submitted_mean, line.difference, line.tolerance) == (55.0, 2.0, 10.0)
    assert line.rerun_dir is not None and line.rerun_dir.startswith("runs/_verifications/example_http/")  # repository-relative, never absolute
    rerun = served_submission.parents[2] / line.rerun_dir
    assert rerun.parent.parent.name == "_verifications" and len(_rows(rerun)) == len(TASKS) and {r["run_index"] for r in _rows(rerun)} == {0}
    assert read_meta(run_dir=rerun).provenance == "maintainer_run"
    assert sorted(p.name for p in served_submission.iterdir()) == before and read_meta(run_dir=served_submission).provenance == "self_reported"


def test_verify_submission_records_a_far_rerun_as_failed(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="FAIL example_http: re-run mean 80.0 .*stays self-reported"):
        _verify(monkeypatch, served_submission, rerun_score=80.0, submitted={0: 50.0, 1: 60.0})
    assert [line.passed for line in _ledger(served_submission)] == [False]


def test_a_copy_resubmitted_after_a_fail_gets_no_new_first_attempt(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="FAIL"):
        _verify(monkeypatch, served_submission, rerun_score=80.0, submitted={0: 50.0, 1: 60.0})
    copy = served_submission.parent / "20990101_000000"
    shutil.copytree(served_submission, copy)
    with pytest.raises(SystemExit, match="already verified for dataset .* \\(failed, verifications.jsonl\\).*only a full re-run by the maintainers"):
        _verify(monkeypatch, copy, rerun_score=55.0, submitted={0: 50.0, 1: 60.0})
    assert [line.passed for line in _ledger(served_submission)] == [False]


def test_verify_submission_refuses_a_forged_verification_or_provenance(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (served_submission / "verification.json").write_text('{"attempts": [{"passed": true}]}', encoding="utf-8")
    with pytest.raises(SystemExit, match="remove verification.json"):
        _verify(monkeypatch, served_submission, rerun_score=55.0, submitted={0: 50.0, 1: 60.0})
    (served_submission / "verification.json").unlink()
    write_meta(meta=read_meta(run_dir=served_submission).model_copy(update={"provenance": "verified"}), run_dir=served_submission)
    with pytest.raises(SystemExit, match="only the maintainers mark one verified"):
        _verify(monkeypatch, served_submission, rerun_score=55.0, submitted={0: 50.0, 1: 60.0})
    assert _ledger(served_submission) == []


def test_runs_that_disagree_need_a_full_rerun_and_no_sample_is_asked(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="FAIL example_http: needs a full re-run.*run 0 averages 30.0 on the full set, the other runs 60.0"):
        _verify(monkeypatch, served_submission, rerun_score=45.0, submitted={0: 30.0, 1: 60.0})
    [line] = _ledger(served_submission)
    assert line.passed is False and line.needs_full_rerun is not None and not (served_submission.parents[1] / "_verifications").exists()


def test_a_submission_outside_the_repository_is_not_verified_so_the_ledger_never_holds_a_local_path(served_submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="keep the submission under this repository's runs/ folder"):
        _verify(monkeypatch, served_submission, rerun_score=55.0, submitted={0: 50.0, 1: 60.0}, inside_repository=False)
    assert _ledger(served_submission) == [] and not (served_submission.parents[1] / "_verifications").exists()


def test_an_absolute_path_in_the_registrations_fails_the_scan(tmp_path: Path) -> None:
    from bench.publish_check import MUST_NOT_SHIP_GATES, absolute_paths, committed_absolute_paths

    (tmp_path / "registrations").mkdir()
    ledger = tmp_path / "registrations" / "verifications.jsonl"
    ledger.write_text('{"rerun_dir": "runs/_verifications/acme_coach/x"}\n', encoding="utf-8")
    assert absolute_paths(root=tmp_path) == []
    ledger.write_text('{"rerun_dir": "/Users/someone/gmbench/runs/_verifications/acme_coach/x"}\n', encoding="utf-8")
    assert [f.gate for f in absolute_paths(root=tmp_path)] == ["absolute_path"]
    # make publish_scan runs the must-not-ship gates, which check the committed registrations at every commit
    assert [f.gate for f in committed_absolute_paths(root=tmp_path)] == ["committed_absolute_path"] and "committed_absolute_path" in MUST_NOT_SHIP_GATES


def test_a_folder_with_an_entrant_file_never_stands_as_a_maintainers_run(tmp_path: Path) -> None:
    from bench.cli.aggregate import graded_latest_runs

    runs = tmp_path / "runs"
    sheets = fake_sheets(task_ids=list(FIXTURE_TASK_FAMILY), signed=1)
    ours = make_run(runs, entrant="acme_coach", task_family=FIXTURE_TASK_FAMILY, sheets=sheets, families={"assessment_plan"}, scores={"ap_9001": 50})
    disguised = ours.parent / "29990101_000000"
    shutil.copytree(ours, disguised)
    (disguised / "entrant.yaml").write_text("name: acme_coach\n", encoding="utf-8")  # meta.json has no provenance at all
    assert graded_latest_runs(runs_dir=runs) == [ours]
