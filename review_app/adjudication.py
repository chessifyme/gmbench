"""The adjudication page (/sheets/<id>/adjudicate), for a signed-in adjudicator only: every grandmaster's record of a task side by
side (labels, weights, wording, verdict, turning points, flags), the disagreements highlighted, and a form for the resolved key,
stored as the sheet's adjudicated record (bench.annotations). Rulings on disputed labels are labels with role adjudicator."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, get_args

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.datastructures import FormData

from bench.annotations import annotator_view, completed_annotators, completed_views, fields_for_wording, resolve, suggested_turning_points, suggested_verdict, unlabelled_claims, with_adjudication
from bench.claim_display import shown_order, source_label
from bench.claim_tokens import claim_ids, public_id, recorded_claims
from bench.decisions import Decision, apply_decision, parse_label_word, WORD_VALUES
from bench.io_yaml import write_sheet
from bench.models import (
    LABEL_VALUES,
    ClaimFields,
    TRUE_BY_SOURCE,
    VERDICTS,
    Adjudication,
    RecordWeight,
    ReferenceClaim,
    Sheet,
    Task,
    TurningLabel,
    TurningPoint,
    parse_optional_weight,
    parse_verdict,
    weight_word,
)
from bench.validate import Problem, errors_of, warnings_of

TURNING_LABELS: tuple[TurningLabel, ...] = get_args(TurningLabel)


class Render(Protocol):
    def __call__(self, request: Request, name: str, status_code: int = 200, **context: object) -> HTMLResponse: ...


@dataclass(frozen=True)
class Cell:
    label: str
    weight: str
    wording: str
    suggested: bool


@dataclass(frozen=True)
class Row:
    token: str
    number: int
    text: str
    source: str
    cells: tuple[Cell, ...]
    differs: bool
    disputed: bool
    ruling: str
    final_weight: str
    final_text: str
    wordings: tuple[str, ...]


@dataclass(frozen=True)
class TurningRow:
    ply: int
    move: str
    chosen: tuple[str, ...]
    differs: bool
    checked: bool
    label: str


def _weight_text(weight: int | None) -> str:
    return "" if weight is None else str(weight)


def _shown_weight(weight: int | None) -> str:
    return "Not key" if weight is None else weight_word(weight=weight)


def _shown_label(*, claim: ReferenceClaim, name: str, skipped: list[str]) -> str:
    latest = claim.latest_labels.get(name)
    if latest is not None:
        return latest.value
    if claim.claim_id in skipped:
        return "skipped"
    return "true (default)" if claim.source.type in TRUE_BY_SOURCE else "not labelled"


def review_first(*, name: str, sheet: Sheet, required: int, annotators: frozenset[str]) -> str | None:
    """Why an adjudicator who is also one of the grandmasters may not see a task's other reviews yet: their own review of it is
    open, or they have none and the task still needs reviews (they could be one of its reviewers). Their own review stays blind
    until they have marked it complete."""
    if name not in annotators:
        return None
    record = sheet.annotations.get(name)
    if record is not None:
        return None if record.completed_at is not None else "you are reviewing this task yourself: mark your own review complete before you see the others"
    if len(completed_annotators(sheet=sheet)) < required:
        return "this task still needs reviews and you are one of the reviewers: review it yourself, or wait until it has the reviews it needs"
    return None


def adjudication_routes(
    *,
    app: FastAPI,
    render: Render,
    adjudicator_or_404: Callable[[Request], str],
    read_task_by_id: Callable[[str], Task],
    load_sheet: Callable[[str], Sheet | None],
    sheet_path: Callable[[str], Path],
    problems_for: Callable[..., list[Problem]],
    required: Callable[[str], int],
    token_secret: bytes,
    writing: threading.Lock,
    sync_status: Callable[..., Task],
    annotators: frozenset[str],
) -> None:
    def own_review_first(*, name: str, sheet: Sheet, task_id: str) -> str | None:
        return review_first(name=name, sheet=sheet, required=required(task_id), annotators=annotators)

    def seen(*, name: str, sheet: Sheet) -> None:
        """An adjudicator who is also a grandmaster has now seen the other reviews of this task: from here on their own review
        of it stays as it is (no save, reopen or completion), so it cannot be copied from the others."""
        if name in annotators and name not in sheet.seen_by_adjudicators:
            with writing:
                fresh = load_sheet(sheet.task_id)
                if fresh is not None and name not in fresh.seen_by_adjudicators:
                    write_sheet(sheet=fresh.model_copy(update={"seen_by_adjudicators": {**fresh.seen_by_adjudicators, name: datetime.now(UTC)}}), path=sheet_path(sheet.task_id))

    def page(*, request: Request, task: Task, sheet: Sheet, error: list[str] | None = None, status_code: int = 200, posted: FormData | None = None) -> HTMLResponse:
        adjudicator_name = adjudicator_or_404(request)
        names = sorted(sheet.annotations)
        done = completed_annotators(sheet=sheet)
        views = {name: annotator_view(sheet=sheet, annotator=name) for name in names}
        resolution = resolve(sheet=sheet, required=required(task.task_id))
        ruling = sheet.adjudicated
        agreed = {c.claim_id: c for c in resolution.sheet.claims} if len(done) >= 2 else {}
        rows: list[Row] = []
        claims = [c for c in shown_order(claims=sheet.claims) if c.duplicate_of is None and c.dropped is None]
        for number, claim in enumerate(claims, start=1):
            token = public_id(claim=claim, task_id=task.task_id, secret=token_secret, recorded=recorded_claims(sheet=sheet))
            cells = []
            for name in names:
                view_claim = next((c for c in views[name].claims if c.claim_id == claim.claim_id), None)
                record = sheet.annotations[name]
                if view_claim is None:
                    cells.append(Cell(label="", weight="", wording="", suggested=False))
                    continue
                cells.append(Cell(label=_shown_label(claim=view_claim, name=name, skipped=record.skipped), weight=_shown_weight(view_claim.weight), wording=view_claim.text if view_claim.text != claim.text else "", suggested=claim.claim_id not in record.weights))
            present = [cell for cell in cells if cell.label]
            differs = len({(cell.label, cell.weight, cell.wording) for cell in present}) > 1
            if ruling is not None:
                final_weight, final_text = _weight_text(ruling.weights[claim.claim_id].weight if claim.claim_id in ruling.weights else None), ruling.rewords.get(claim.claim_id, claim.text)
            else:
                settled = agreed.get(claim.claim_id)
                final_weight, final_text = _weight_text(settled.weight if settled else None), settled.text if settled else claim.text
            if posted is not None:
                final_weight = str(posted.get(f"weight_{token}") or "")
                final_text = str(posted.get(f"text_{token}") or final_text)
            settled_label = claim.adjudication
            wordings = tuple(dict.fromkeys([claim.text, *(cell.wording for cell in cells if cell.wording), final_text]))
            rows.append(Row(token=token, number=number, text=claim.text, source=source_label(source=claim.source), cells=tuple(cells), differs=differs, disputed=claim.needs_adjudication, ruling=settled_label.value if settled_label else "", final_weight=final_weight, final_text=final_text, wordings=wordings))
        verdicts = {name: views[name].verdict for name in names}
        final_verdict = ruling.verdict if ruling is not None else resolution.sheet.verdict if len(done) >= 2 else None
        chosen = {name: {tp.ply: tp for tp in sheet.annotations[name].turning_points} for name in names}
        plies = sorted({tp.ply for tp in suggested_turning_points(sheet=sheet)} | {ply for picked in chosen.values() for ply in picked})
        moves = {tp.ply: tp.move for tp in suggested_turning_points(sheet=sheet)} | {ply: tp.move for picked in chosen.values() for ply, tp in picked.items()}
        final_tps = {tp.ply: tp for tp in (ruling.turning_points if ruling is not None else resolution.sheet.turning_points if len(done) >= 2 else [])}
        turning = []
        for ply in plies:
            marks = tuple(chosen[name][ply].label if ply in chosen[name] else "-" for name in names)
            turning.append(TurningRow(ply=ply, move=moves[ply], chosen=marks, differs=len(set(marks)) > 1, checked=ply in final_tps, label=final_tps[ply].label if ply in final_tps else next((chosen[name][ply].label for name in names if ply in chosen[name]), "mistake")))
        return render(
            request,
            "adjudicate.html",
            status_code=status_code,
            title=f"Adjudicate {task.task_id}",
            task=task,
            names=names,
            done=done,
            records=sheet.annotations,
            rows=rows,
            verdicts=verdicts,
            verdict_hint=suggested_verdict(sheet=sheet),
            final_verdict=str(posted.get("verdict") or "") if posted is not None else final_verdict or "",
            verdict_values=VERDICTS,
            turning=turning,
            turning_labels=TURNING_LABELS,
            unresolved=resolution.unresolved,
            adjudicated=ruling,
            notes=str(posted.get("notes") or "") if posted is not None else ruling.notes if ruling is not None else "",
            shared_notes=sheet.notes,
            label_values=LABEL_VALUES,
            rejected="rejected" in task.tags,
            own_review=adjudicator_name in done,
            errors=error or [],
            unlabelled=len(unlabelled_claims(sheet=sheet)),
        )

    @app.get("/sheets/{task_id}/adjudicate", response_class=HTMLResponse)
    def adjudicate_page(request: Request, task_id: str) -> HTMLResponse:
        name = adjudicator_or_404(request)
        task = read_task_by_id(task_id)
        sheet = load_sheet(task_id)
        blocked = own_review_first(name=name, sheet=sheet, task_id=task_id) if sheet is not None else None
        if blocked is not None:
            return HTMLResponse(blocked, status_code=409)
        if sheet is None:
            return render(request, "adjudicate.html", title=f"Adjudicate {task_id}", task=task, names=[], done=[], records={}, rows=[], verdicts={}, verdict_hint=None, final_verdict="", verdict_values=VERDICTS, turning=[], turning_labels=TURNING_LABELS, unresolved=(), adjudicated=None, notes="", shared_notes="", label_values=LABEL_VALUES, rejected="rejected" in task.tags, own_review=False, errors=["nothing is saved for this task yet"], unlabelled=0)
        seen(name=name, sheet=sheet)
        return page(request=request, task=task, sheet=sheet)

    @app.post("/sheets/{task_id}/adjudicate")
    async def adjudicate_save(request: Request, task_id: str):
        name = adjudicator_or_404(request)
        task = read_task_by_id(task_id)
        form = await request.form()
        now = datetime.now(UTC)
        before = load_sheet(task_id)
        if before is not None and own_review_first(name=name, sheet=before, task_id=task_id) is None:
            seen(name=name, sheet=before)
        complete = form.get("then") == "complete"
        with writing:
            sheet = load_sheet(task_id)
            if sheet is None or len(completed_annotators(sheet=sheet)) < 2:
                return HTMLResponse("adjudication needs at least two completed reviews of the task", status_code=409)
            blocked = own_review_first(name=name, sheet=sheet, task_id=task_id)
            if blocked is not None:
                return HTMLResponse(blocked, status_code=409)
            if complete and any(record.flag is not None for record in sheet.annotations.values()):
                return page(request=request, task=task, sheet=sheet, error=["a reviewer flagged this task: clear the flag (or reject the task) before saving the resolved key"], status_code=409, posted=form)
            try:
                updated, adjudication = _adjudicated(form=form, task=task, sheet=sheet, adjudicator=name, now=now, complete=complete)
            except ValueError as error:
                return page(request=request, task=task, sheet=sheet, error=[str(error)], status_code=422, posted=form)
            candidate = with_adjudication(sheet=updated, adjudication=adjudication.model_copy(update={"completed_at": now}))
            resolution = resolve(sheet=candidate, required=required(task_id))
            problems = problems_for(sheet=resolution.sheet, task=task.model_copy(update={"status": "signed" if complete else "draft"}))
            refused = [*resolution.unresolved, *(p.message for p in errors_of(problems=problems))]
            if refused:
                return page(request=request, task=task, sheet=sheet, error=refused, status_code=422, posted=form)
            write_sheet(sheet=with_adjudication(sheet=updated, adjudication=adjudication), path=sheet_path(task_id))
            sync_status(task=task)
        warnings = [p.message for p in warnings_of(problems=problems)]
        return RedirectResponse(url=f"/sheets/{task_id}/adjudicate{'?warnings=' + str(len(warnings)) if warnings else ''}", status_code=303)

    def _adjudicated(*, form: FormData, task: Task, sheet: Sheet, adjudicator: str, now: datetime, complete: bool) -> tuple[Sheet, Adjudication]:
        """The sheet with the posted rulings, and the adjudicated record of the posted weights, wordings, verdict, turning points
        and notes."""
        ids = claim_ids(sheet=sheet, secret=token_secret)
        resolution = resolve(sheet=sheet, required=required(task.task_id))
        agreed = {c.claim_id: c.weight for c in resolution.sheet.claims}
        claims = {c.claim_id: c for c in sheet.claims}
        weights: dict[str, RecordWeight] = {}
        rewords: dict[str, str] = {}
        fields: dict[str, ClaimFields] = {}
        views = completed_views(sheet=sheet)
        for key in form:
            if not str(key).startswith("weight_"):
                continue
            token = str(key).removeprefix("weight_")
            claim_id = ids.get(token)
            if claim_id is None:
                raise ValueError(f"the form names a claim this task does not have ({token}); reload the page")
            where = f"claim {claim_id}"
            weights[claim_id] = RecordWeight(weight=parse_optional_weight(text=str(form.get(key) or ""), where=where), suggested=agreed.get(claim_id))
            text = str(form.get(f"text_{token}") or "").strip()
            if text and text != claims[claim_id].text:
                rewords[claim_id] = text
            settled = fields_for_wording(views=views, claim=claims[claim_id], text=rewords.get(claim_id, claims[claim_id].text))
            if settled is not None:
                fields[claim_id] = settled
            ruling_text = str(form.get(f"ruling_{token}") or "")
            if ruling_text:
                word = parse_label_word(text=ruling_text, where=where)
                if word is not None:
                    claims[claim_id] = apply_decision(claim=claims[claim_id], annotator=adjudicator, decision=Decision(ruling=WORD_VALUES[word]), at=now, how_to_name="sign in as an adjudicator")
        turning = []
        for key in form:
            if str(key).startswith("tp_") and not str(key).startswith(("tp_label_", "tp_move_")):
                ply = int(str(key).removeprefix("tp_"))
                label = str(form.get(f"tp_label_{ply}") or "")
                move = str(form.get(f"tp_move_{ply}") or "")
                turning.append(TurningPoint.model_validate({"ply": ply, "move": move, "label": label}))
        adjudication = Adjudication(
            adjudicator=adjudicator,
            own_review_included=adjudicator in completed_annotators(sheet=sheet),
            weights=weights,
            rewords=rewords,
            fields=fields,
            verdict=parse_verdict(text=str(form.get("verdict") or "")) if task.family == "move_judgement" else None,
            turning_points=sorted(turning, key=lambda tp: tp.ply),
            notes=str(form.get("notes") or ""),
            updated_at=now,
            completed_at=now if complete else None,
            same_idea=dict(sheet.adjudicated.same_idea) if sheet.adjudicated is not None else {},
        )
        updated = Sheet.model_validate({**sheet.model_dump(), "claims": [c.model_dump() for c in claims.values()]})
        return updated, adjudication
