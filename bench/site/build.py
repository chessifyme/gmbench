import html as html_lib
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

from bench.adapters.base import AnswerRecord
from bench.aggregate import HEADLINE_GROUPS, EntrantResult, JuryRatingEntry, Leaderboard, Stat, completeness_warnings, read_grades, unreviewed_answer_keys_note
from bench.config import BenchConfig, load_models_file
from bench.grading.grade import GradeRecord
from bench.annotations import load_resolved_sheets
from bench.io_yaml import load_all_tasks
from bench.models import FAMILIES, Sheet, Task
from bench.paths import DOCS_DIR, ROOT
from bench.publish_check import official_result
from bench.grading.normalise import to_plain_text
from bench.grading.score import ClaimTally, compute_score, guarded, idea_groups_of, saturated, verdict_claim
from bench.runs import read_answers, read_meta
from bench.site.boards import svg_for_fen, svg_for_pgn_at_ply

TEMPLATES = Path(__file__).parent / "templates"
CSS = Path(__file__).parent / "static" / "site.css"
DEVELOPMENT_NOTICE = (
    "Development result: the board was graded under the Claude-only profile. Not publishable."
)
CLAUDE_ONLY_PROFILE = "claude_only"


@dataclass(frozen=True)
class RunView:
    run_index: int
    answer: AnswerRecord
    text_plain: str
    grade: GradeRecord | None
    seconds: float


@dataclass(frozen=True)
class EntrantView:
    name: str
    card: object
    runs: list[RunView]


@dataclass(frozen=True)
class ExampleRow:
    family: str
    task_id: str
    question: str
    scores: list[tuple[str, str]]


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    family: str
    audience: str
    scores: list[str]
    mean: str
    fit: str
    votes: list[str]


@dataclass(frozen=True)
class ScoreDefinitions:
    """The numbers of the dashboard's coverage, precision and score definitions, worked out with the grader's own functions."""

    coverage_weight: float
    precision_weight: float
    saturation_percent: float
    key_weight: int
    earned_weight: int
    coverage_example: float
    checked: int
    true: int
    precision_example: float
    min_checked: int
    few_checked: int
    few_checked_precision: float
    score_example: float


UNSCORED_WEIGHTS = ("verdict", "audience", "illegal_move_per_move", "illegal_move_cap", "moments", "moment_labels")
EXAMPLE_KEY_WEIGHT = 10
EXAMPLE_CHECKED = 10
EXAMPLE_TRUE = 8


def score_definitions(*, cfg: BenchConfig) -> ScoreDefinitions:
    """The definitions say the score is coverage and precision alone, with one pair of weights; a configuration where that is not
    so is refused rather than described wrongly."""
    weights = cfg.weights
    if (weights.coverage, weights.precision) != (weights.coverage_without_family, weights.precision_without_family):
        raise ValueError("the dashboard defines one score formula, but move_judgement and the other position families weigh coverage and precision differently")
    scored = [name for name in UNSCORED_WEIGHTS if getattr(weights, name) != 0.0]
    if scored or weights.game_review_coverage != 1.0:
        raise ValueError(f"the dashboard says the score is coverage and precision alone, but config/bench.yaml also weighs {', '.join(scored) or 'game reviews other than by coverage alone'}")
    earned = math.ceil(cfg.coverage_saturation * EXAMPLE_KEY_WEIGHT - 1e-9)
    coverage_value = saturated(earned=earned, total=EXAMPLE_KEY_WEIGHT, saturation=cfg.coverage_saturation)
    precision_value = ClaimTally(checked=EXAMPLE_CHECKED, true=EXAMPLE_TRUE, min_checked=cfg.precision_min_checked).precision
    few = max(1, cfg.precision_min_checked - 1)
    score = compute_score(family="assessment_plan", coverage=coverage_value, precision=precision_value, illegal_moves=0, verdict_credit=None, turning_point_f1=None, audience_credit=None, weights=weights)
    return ScoreDefinitions(
        coverage_weight=weights.coverage_without_family,
        precision_weight=weights.precision_without_family,
        saturation_percent=round(100 * cfg.coverage_saturation, 1),
        key_weight=EXAMPLE_KEY_WEIGHT,
        earned_weight=earned,
        coverage_example=round(100 * coverage_value, 1),
        checked=EXAMPLE_CHECKED,
        true=EXAMPLE_TRUE,
        precision_example=round(100 * precision_value, 1),
        min_checked=cfg.precision_min_checked,
        few_checked=few,
        few_checked_precision=round(100 * ClaimTally(checked=few, true=few, min_checked=cfg.precision_min_checked).precision, 1),
        score_example=round(score, 1),
    )


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _percent_interval(stat: Stat) -> str:
    return f"[{stat.ci_low * 100:.0f}, {stat.ci_high * 100:.0f}]"


def _number(value: float) -> str:
    return f"{value:g}"


def _medians(values: dict[str, float]) -> str:
    return " / ".join(f"{values[f]:.0f}" for f in FAMILIES if f in values) or "—"


def banner_text(*, board: Leaderboard, official: bool) -> str | None:
    if official:
        return None
    parts = ["public set"] if board.task_set == "public" else []
    parts.append(f"{board.sheets_signed} of {board.sheets_total} answer keys signed")
    return " · ".join(["Preview", *parts, "not an official result"])


def conflict_of_interest(*, board: Leaderboard) -> str:
    """The governance statement, naming the published configuration's extractor and its judge, the first configured (config/models.yaml)."""
    models = load_models_file(profile=None)
    extractor = next(iter(models.extractors.values())).model_id
    judge = next(iter(models.judges.values())).model_id
    return board.governance.conflict_of_interest.format(extractor=extractor, judge=judge)


def judging_sentence(*, board: Leaderboard) -> str:
    """How the board's answers were judged, read from the jury and claim-judge models its grades record."""
    jury = [model for model in board.grading_instruments.get("jury", "").split("+") if model]
    claim_judges = [model for model in board.grading_instruments.get("claim_judge", "").split("+") if model]
    if len(jury) == 1 and claim_judges in ([], jury):
        return (
            f"Every answer is graded by one judge model, {jury[0]}, which checks its claims with chess tools and decides which answer-key claims it makes; "
            "code re-checks every tool call the judge cites. A single judge from one model family grades every entrant."
        )
    return f"Every answer is graded by a panel of {len(jury)} judges ({', '.join(jury)}), and more than half of the panel decides."


def development_notice(*, board: Leaderboard) -> str | None:
    return DEVELOPMENT_NOTICE if CLAUDE_ONLY_PROFILE in board.grading_profiles else None


def sheets_sentence(*, board: Leaderboard) -> str:
    if board.sheets_total and board.sheets_signed == board.sheets_total:
        return f"All {board.sheets_total} answer keys are signed by expert chess players."
    return f"{board.sheets_signed} of {board.sheets_total} answer keys are signed by expert chess players; the rest are drafts."


TIMES = {1: "once", 2: "twice", 3: "three times"}


def answers_sentence(*, board: Leaderboard, grades: Iterable[tuple[str, str, int]]) -> str:
    """The lead's first clause, true of the board's own runs: how many of the set's questions the ranked entrants answered, and how often."""
    runs = set(Counter((entrant, task_id) for entrant, task_id, _ in grades).values())
    times = TIMES.get(min(runs), f"{min(runs)} times") if len(runs) == 1 else "several times"
    ranked = [e for e in board.entrants if e.group in HEADLINE_GROUPS] or board.entrants
    graded = {sum(c.graded for c in e.family_tasks.values()) for e in ranked}
    total = max((sum(c.total for c in e.family_tasks.values()) for e in ranked), default=0)
    if graded == {total}:
        return f"Each entrant answers the same {total} questions {times}"
    if len(graded) == 1:
        return f"Each entrant answered {min(graded)} of the {total} questions, {times} each"
    return f"Entrants answered between {min(graded)} and {max(graded)} of the {total} questions, {times} each"


FAMILY_TASKS = {
    "assessment_plan": "assess a position and give the plan",
    "move_choice": "choose a move",
    "move_judgement": "judge a move",
    "line_explanation": "explain a line",
    "game_review": "review a whole game",
}


def headline_tasks(*, board: Leaderboard) -> str:
    """What the headline asks of every entrant, one phrase per headline family in the order of FAMILIES."""
    return ", ".join(FAMILY_TASKS[f] for f in FAMILIES if f in board.headline_families)


def shown_families(*, board: Leaderboard) -> list[str]:
    """The headline families and any other family an entrant was scored on; a family without tasks gets no column."""
    return [f for f in FAMILIES if f in board.headline_families or any(f in e.family_scores for e in board.entrants)]


def family_labels(*, board: Leaderboard) -> dict[str, str]:
    labels: dict[str, str] = {}
    for family in shown_families(board=board):
        name = family.replace("_", " ")
        if family in board.headline_families:
            labels[family] = name
        else:
            labels[family] = f"{name} (not in headline)"
    return labels


def examples_for(*, board: Leaderboard, tasks: list[Task], grades: dict[tuple[str, str, int], GradeRecord]) -> list[ExampleRow]:
    full = [e.name for e in board.entrants if e.group in HEADLINE_GROUPS]
    rows: list[ExampleRow] = []
    for family in FAMILIES:
        candidates = sorted((t for t in tasks if t.family == family), key=lambda t: t.task_id)
        chosen = next((t for t in candidates if any((name, t.task_id, 0) in grades for name in full)), None)
        if chosen is None:
            continue
        scores = [(name, f"{grades[(name, chosen.task_id, 0)].score.score:.1f}") for name in full if (name, chosen.task_id, 0) in grades]
        rows.append(ExampleRow(family=family, task_id=chosen.task_id, question=chosen.question, scores=scores))
    return rows


def _panels(values: dict[str, int]) -> str:
    return ", ".join(f"{panel.replace('+', ' + ')} ({count})" for panel, count in values.items()) or "none"


TOOL_NAMES = {"analysis": "analysis tools or services", "chess_tools": "basic chess tools (play, board, evaluate, judge_move; Stockfish 19)"}


def _tools(values: Iterable[str]) -> str:
    return ", ".join(TOOL_NAMES.get(value, value) for value in sorted(values)) or "none"


LEVEL_INITIALS = (("beginner", "B"), ("club", "C"), ("advanced", "A"))
VOTE_LABELS = {"fits": "fits", "partly": "partly fits", "does_not_fit": "does not fit", "invalid": "no valid vote"}


def audience_label(credit: float | None) -> str:
    if credit is None:
        return "not settled"
    if credit == 1.0:
        return "fits"
    if credit == 0.0:
        return "does not fit"
    return "partly fits" if credit == 0.5 else f"partly fits ({credit:.2f})"


def audience_class(credit: float | None) -> str:
    if credit is None:
        return "absent"
    return {1.0: "present", 0.0: "contradicted"}.get(credit, "uncertain")


def audience_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def audience_levels_line(by_level: dict[str, float]) -> str:
    return " / ".join(f"{initial} {by_level[level] * 100:.0f}" if level in by_level else f"{initial} —" for level, initial in LEVEL_INITIALS)


def audience_vote_lines(*, grade: GradeRecord) -> list[str]:
    verdict = grade.audience_verdict
    if verdict is None:
        return []
    return [f"{v.judge} ({v.model_id}): {VOTE_LABELS[v.vote]}" + (f" — \"{v.quote}\"" if v.quote else "") + (f" — {v.error}" if v.error else "") for v in verdict.votes]


def _task_fit(*, scores: list[GradeRecord]) -> str:
    credits = [g.audience_verdict.credit for g in scores if g.audience_verdict is not None and g.audience_verdict.credit is not None]
    return audience_percent(sum(credits) / len(credits)) if credits else "—"


def jury_by_name(*, board: Leaderboard) -> dict[str, JuryRatingEntry]:
    return {r.name: r for r in board.jury_rating.entrants} if board.jury_rating is not None else {}


def jury_disagreements(*, board: Leaderboard) -> dict[str, list[str]]:
    """For each entrant, the entrants the head-to-head rating orders the other way from the headline."""
    others: dict[str, list[str]] = {}
    for pair in board.jury_rating.different_order if board.jury_rating is not None else []:
        others.setdefault(pair.headline_ahead, []).append(pair.rating_ahead)
        others.setdefault(pair.rating_ahead, []).append(pair.headline_ahead)
    return others


def signed_points(value: float) -> str:
    rounded = round(value)
    return "0" if rounded == 0 else f"{'+' if rounded > 0 else '−'}{abs(rounded)}"


@dataclass(frozen=True)
class IdeaLine:
    """One idea of several key claims in a grading trace: each claim's text, weight and how it fared, and what the idea counted."""

    members: list[tuple[str, int, str]]
    earned: int
    weight: int


def _idea_lines(grade: GradeRecord) -> list[IdeaLine]:
    texts = {o.claim_id: v.item_text for o, v in zip(grade.claim_outcomes, grade.item_verdicts, strict=False)}
    by_id = {o.claim_id: o for o in grade.claim_outcomes}
    return [
        IdeaLine(members=[(texts.get(claim_id, claim_id), by_id[claim_id].weight, by_id[claim_id].reason) for claim_id in idea.claim_ids], earned=idea.earned, weight=idea.weight)
        for idea in idea_groups_of(outcomes=grade.claim_outcomes)
    ]


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.globals["medians"] = _medians
    env.filters["panels"] = _panels
    env.filters["tools"] = _tools
    env.filters["audience_label"] = audience_label
    env.filters["audience_class"] = audience_class
    env.filters["audience_percent"] = audience_percent
    env.filters["audience_levels"] = audience_levels_line
    env.filters["vote_label"] = VOTE_LABELS.__getitem__
    env.filters["signed_points"] = signed_points
    env.filters["percent"] = _percent
    env.filters["idea_lines"] = _idea_lines
    env.filters["percent_interval"] = _percent_interval
    env.filters["number"] = _number
    env.tests["guarded"] = lambda claim: guarded(claim=claim)
    return env


def _board_svg(*, task: Task, sheet: Sheet | None) -> str:
    if task.family == "game_review":
        ply = sheet.turning_points[0].ply if sheet and sheet.turning_points else 0
        return svg_for_pgn_at_ply(pgn=task.pgn or "", ply=ply)
    return svg_for_fen(fen=task.fen or "", lastmove=task.move)


SITE_DOCS = {"methodology": "Methodology", "grading_rules": "Grading rules"}
LINK = re.compile(r'<a href="([^"]*)"([^>]*)>(.*?)</a>', re.S)
IMAGE_SOURCE = re.compile(r'(src|srcset)="([^"]+)"')
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def doc_page_name(*, path: Path, root: Path = ROOT) -> str:
    """The site page of a Markdown file of the repository: the site docs keep their names, the dataset card (data/public/README.md) is doc_dataset_card, any other is doc_<lowercased stem>."""
    if path.parent == root / "docs" and path.stem in SITE_DOCS:
        return f"{path.stem}.html"
    if path == root / "data" / "public" / "README.md":
        return "doc_dataset_card.html"
    return f"doc_{path.stem.lower()}.html"


def _relative_targets(*, html: str, source: Path, root: Path) -> list[Path]:
    targets: list[Path] = []
    for match in LINK.finditer(html):
        href = html_lib.unescape(match.group(1))
        if EXTERNAL.match(href):
            continue
        target = (source.parent / href.split("#", 1)[0]).resolve()
        if target.suffix == ".md" and target.is_file() and root.resolve() in target.parents:
            targets.append(target)
    return targets


def _markdown(*, path: Path) -> str:
    return markdown.markdown(path.read_text(encoding="utf-8"), extensions=["toc", "tables"])


def doc_pages(*, root: Path = ROOT, docs_dir: Path = DOCS_DIR) -> dict[Path, str]:
    """Every Markdown file the site docs link to, and every file those link to in turn, by its site page name. The site carries
    them all, so no link on a docs page leads out of the site to a file it does not have."""
    pages: dict[Path, str] = {}
    queue = [(docs_dir / f"{name}.md").resolve() for name in SITE_DOCS]
    while queue:
        path = queue.pop()
        if path in pages:
            continue
        name = doc_page_name(path=path, root=root.resolve())
        if name in pages.values():
            raise ValueError(f"two documents would share the site page {name}: rename one of them ({path})")
        pages[path] = name
        queue.extend(_relative_targets(html=_markdown(path=path), source=path, root=root))
    return pages


def site_image(*, path: Path) -> str:
    """Where the site keeps a copy of an image a docs page shows (site file names use underscores)."""
    return f"img/{path.name.lower().replace('-', '_')}"


def render_doc_file(*, path: Path, pages: dict[Path, str], images: dict[Path, str]) -> str:
    """A Markdown file as a site page: links to documents the site carries point at their pages, images at the site's img/ copies
    (collected into IMAGES), and a link to any other repository file keeps its text without the link."""
    html = _markdown(path=path)

    def link(match: re.Match[str]) -> str:
        href = html_lib.unescape(match.group(1))
        if EXTERNAL.match(href):
            return match.group(0)
        file_part, _, anchor = href.partition("#")
        target = (path.parent / file_part).resolve()
        if target in pages:
            return f'<a href="{pages[target]}{"#" + anchor if anchor else ""}"{match.group(2)}>{match.group(3)}</a>'
        if target.suffix.lower() in (".svg", ".png", ".jpg", ".jpeg", ".gif") and target.is_file():
            images[target] = site_image(path=target)
            return f'<a href="{site_image(path=target)}"{match.group(2)}>{match.group(3)}</a>'
        return match.group(3)

    def image(match: re.Match[str]) -> str:
        value = html_lib.unescape(match.group(2))
        if EXTERNAL.match(value):
            return match.group(0)
        target = (path.parent / value).resolve()
        if not target.is_file():
            return match.group(0)
        images[target] = site_image(path=target)
        return f'{match.group(1)}="{site_image(path=target)}"'

    return IMAGE_SOURCE.sub(image, LINK.sub(link, html))


def render_doc(*, name: str) -> str:
    """One of SITE_DOCS as HTML, its links rewritten to the site's own pages (images are not copied; build_site does that)."""
    return render_doc_file(path=(DOCS_DIR / f"{name}.md").resolve(), pages=doc_pages(), images={})


def write_doc_pages(*, out_dir: Path, render: Callable[[str, str], str]) -> list[str]:
    """Write every docs page (doc_pages) and the images they show into OUT_DIR; RENDER(title, content) makes the page. Returns
    the page names."""
    pages = doc_pages()
    images: dict[Path, str] = {}
    titles = {(DOCS_DIR / f"{name}.md").resolve(): title for name, title in SITE_DOCS.items()}
    for path, name in pages.items():
        content = render_doc_file(path=path, pages=pages, images=images)
        heading = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.S)
        title = titles.get(path) or (re.sub(r"<[^>]+>", "", heading.group(1)).strip() if heading else path.stem)
        (out_dir / name).write_text(render(title, content), encoding="utf-8")
    if images:
        (out_dir / "img").mkdir(parents=True, exist_ok=True)
        for source, target in images.items():
            shutil.copyfile(source, out_dir / target)
    return sorted(pages.values())


def build_site(*, leaderboard_path: Path, run_dirs: list[Path], tasks_dir: Path, sheets_dir: Path, out_dir: Path, cfg: BenchConfig, root: Path, registrations_dir: Path, set_aside: str | None = None) -> None:
    board = Leaderboard.model_validate_json(leaderboard_path.read_text(encoding="utf-8"))
    tasks = load_all_tasks(tasks_dir=tasks_dir)
    sheets = load_resolved_sheets(sheets_dir=sheets_dir)
    env = _environment()
    css = CSS.read_text(encoding="utf-8")
    results = {e.name: e for e in board.entrants}
    answers: dict[tuple[str, str, int], AnswerRecord] = {}
    grades: dict[tuple[str, str, int], GradeRecord] = {}
    for run_dir in run_dirs:
        meta = read_meta(run_dir=run_dir)
        for answer in read_answers(run_dir=run_dir):
            answers[(meta.entrant, answer.task_id, answer.run_index)] = answer
        for grade in read_grades(run_dir=run_dir):
            grades[(meta.entrant, grade.task_id, grade.run_index)] = grade
    (out_dir / "agents").mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
    common = {
        "css": css,
        "board": board,
        "families": shown_families(board=board),
        "labels": family_labels(board=board),
        "conflict_of_interest": conflict_of_interest(board=board),
        "banner": banner_text(board=board, official=official_result(leaderboard=board, root=root, registrations_dir=registrations_dir)),
        "development_notice": development_notice(board=board),
        "judging_sentence": judging_sentence(board=board),
        "sheets_sentence": sheets_sentence(board=board),
        "answer_keys_note": unreviewed_answer_keys_note(board=board),
        "incomplete": completeness_warnings(board=board),
        "set_aside": set_aside,
        "show_ttft": any(e.ttft_median_s is not None for e in board.entrants),
        "show_audience": any(g.audience_verdict is not None for g in grades.values()),
        "defs": score_definitions(cfg=cfg),
    }
    (out_dir / "index.html").write_text(
        env.get_template("index.html").render(
            title="Leaderboard",
            root="",
            tasks=tasks,
            answers_sentence=answers_sentence(board=board, grades=grades),
            headline_tasks=headline_tasks(board=board),
            examples=examples_for(board=board, tasks=tasks, grades=grades),
            jury=jury_by_name(board=board),
            jury_disagree=jury_disagreements(board=board),
            **common
        ),
        encoding="utf-8",
    )
    write_doc_pages(out_dir=out_dir, render=lambda title, content: env.get_template("methodology.html").render(title=title, root="", content=content, **common))
    for entrant in board.entrants:
        rows: list[TaskRow] = []
        for task in tasks:
            if task.family not in entrant.card.attempted_families:
                continue
            task_grades = [grades[(entrant.name, task.task_id, i)] for i in range(3) if (entrant.name, task.task_id, i) in grades]
            scores = [g.score.score for g in task_grades]
            votes = [f"run {g.run_index + 1}, {line}" for g in task_grades for line in audience_vote_lines(grade=g)]
            rows.append(
                TaskRow(
                    task_id=task.task_id,
                    family=task.family,
                    audience=task.audience,
                    scores=[f"{s:.1f}" for s in scores] + [""] * (3 - len(scores)),
                    mean=f"{sum(scores) / len(scores):.1f}" if scores else "",
                    fit=_task_fit(scores=task_grades) if task.family != "game_review" else "",
                    votes=votes,
                )
            )
        (out_dir / "agents" / f"{entrant.name}.html").write_text(env.get_template("agent.html").render(title=entrant.name, root="../", e=entrant, rows=rows, **common), encoding="utf-8")
    for task in tasks:
        sheet = sheets.get(task.task_id)
        views: list[EntrantView] = []
        for entrant in board.entrants:
            if task.family not in entrant.card.attempted_families:
                continue
            runs: list[RunView] = []
            for run_index in range(3):
                key = (entrant.name, task.task_id, run_index)
                if key not in answers:
                    continue
                answer = answers[key]
                runs.append(RunView(run_index=run_index, answer=answer, text_plain=to_plain_text(markdown=answer.text), grade=grades.get(key), seconds=round((answer.finished_at - answer.started_at).total_seconds(), 1)))
            if runs:
                views.append(EntrantView(name=entrant.name, card=entrant.card, runs=runs))
        (out_dir / "tasks" / f"{task.task_id}.html").write_text(
            env.get_template("task.html").render(title=task.task_id, root="../", task=task, sheet=sheet, key_verdict=verdict_claim(task=task, sheet=sheet, stated=None) if sheet is not None and sheet.verdict is not None else None, board_svg=_board_svg(task=task, sheet=sheet), entrants=views, **common),
            encoding="utf-8",
        )
