"""The published form of a sheet (docs/review_app.md, "From working sheet to published key"): what data/public/sheets holds, the resolved answer key
(bench.annotations.resolve) without the review app's working record, every reviewer under a pseudonym.

A published sheet is a plain sheet: each claim with its text, kind, checkable fields, source, weight, duplicate link, idea link and drop, and
each reviewer's final label (with an adjudicator's ruling when it decides the claim); the verdict and turning points; who signed and
when. Times keep their date only (midnight UTC). Gone are the per-annotator records, the adjudication record, the suggestions, label
history, notes, original wordings and who set a weight when. `published_sheet` refuses a result whose key is not the full sheet's.

Real names never reach the repository: `Pseudonyms` maps each reviewer (an expert chess player) to Expert A, Expert B, ... through a
mapping file kept outside version control, which must exist; a reviewer it does not know yet gets the next free pseudonym, written to
the file before any sheet is. The maintainers' publishing step (not in this repository) calls this code; it is here so anyone can
see how a published sheet is made."""

import re
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from bench.annotations import Resolution, resolve
from bench.io_yaml import yaml_files
from bench.models import CAP_ANNOTATOR, SUPERSEDED_COMMENT, ClaimDrop, ClaimLabel, ClaimSource, ReferenceClaim, Sheet

PSEUDONYM = re.compile(r"Expert [A-Z]{1,2}")
# Pipeline steps that set claims aside under their own name: not reviewers, so never pseudonymised.
PIPELINE_NAMES = frozenset({CAP_ANNOTATOR, "cleanup"})
WORKING_RECORD = ("annotations", "adjudicated", "suggested", "seen_by_adjudicators")


class PublishError(Exception):
    pass


def _day(*, at: datetime) -> datetime:
    return datetime.combine(at.astimezone(UTC).date(), time(), tzinfo=UTC)


def _signers(*, value: str | None) -> list[str]:
    return [name.strip() for name in value.split(",")] if value else []


def _names(*, data: dict[str, Any]) -> set[str]:
    """Every reviewer's name in a sheet's data, read without the sheet model so a file that does not load is still scanned:
    records, the adjudicator, labels (not the pipeline's), drops that are not a pipeline step's, who set a weight, the authors of
    grandmasters' claims, who reviewed and signed."""
    names = set(data.get("annotations") or {}) | set(data.get("seen_by_adjudicators") or {}) | set(_signers(value=data.get("author"))) | set(_signers(value=data.get("reviewed_by")))
    if data.get("adjudicated"):
        names.add(data["adjudicated"].get("adjudicator"))
    for claim in data.get("claims") or []:
        names |= {label.get("annotator") for label in claim.get("labels") or [] if not str(label.get("comment", "")).startswith(SUPERSEDED_COMMENT)}
        if claim.get("dropped") and claim["dropped"].get("annotator") not in PIPELINE_NAMES:
            names.add(claim["dropped"].get("annotator"))
        if claim.get("weighted_by"):
            names.add(claim["weighted_by"].get("annotator"))
        if (claim.get("source") or {}).get("type") == "gm":
            names.add(claim["source"].get("name"))
    return {str(name) for name in names}


def reviewer_names(*, sheet: Sheet) -> set[str]:
    return _names(data=sheet.model_dump())


def name_problems(*, data: dict[str, Any]) -> list[str]:
    """Why a sheet's data may not sit in data/public: a reviewer named other than by a pseudonym, or a review-app working record."""
    problems = [f"reviewer {name!r} is not a pseudonym (Expert A, Expert B, ...); data/public names its reviewers by pseudonym only" for name in sorted(_names(data=data)) if not PSEUDONYM.fullmatch(name)]
    if any(data.get(key) for key in WORKING_RECORD):
        problems.append(f"holds the review app's working record ({', '.join(key for key in WORKING_RECORD if data.get(key))}); data/public holds the published form only")
    return problems


def pseudonyms_in_use(*, sheets_dir: Path) -> set[str]:
    """The pseudonyms the published sheets of a folder already name."""
    return {name for path in yaml_files(directory=sheets_dir) for name in _names(data=yaml.safe_load(path.read_text(encoding="utf-8"))) if PSEUDONYM.fullmatch(name)}


def _letters() -> list[str]:
    return list(string.ascii_uppercase) + ["".join(pair) for pair in product(string.ascii_uppercase, repeat=2)]


@dataclass
class Pseudonyms:
    """Real reviewer names to pseudonyms, read from and extended in a mapping file that must stay out of git."""

    path: Path
    mapping: dict[str, str] = field(default_factory=dict)
    added: list[str] = field(default_factory=list)
    reserved: frozenset[str] = frozenset()

    @classmethod
    def load(cls, *, path: Path, reserved: frozenset[str] = frozenset()) -> "Pseudonyms":
        """The mapping file must exist (it may be empty for a first publish): a lost file would hand the published pseudonyms
        to other reviewers. `reserved` are the pseudonyms the published sheets already use, never given to a new reviewer."""
        if not path.exists():
            raise PublishError(f"the pseudonym mapping {path} does not exist; restore it, or create it empty for a first publish")
        mapping = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) and PSEUDONYM.fullmatch(v) for k, v in mapping.items()):
            raise PublishError(f"{path} must map each reviewer's name to a pseudonym of the form Expert A")
        if len(set(mapping.values())) != len(mapping):
            raise PublishError(f"{path} gives one pseudonym to two reviewers")
        return cls(path=path, mapping=dict(mapping), reserved=reserved)

    def of(self, *, name: str) -> str:
        if name in self.mapping:
            return self.mapping[name]
        if PSEUDONYM.fullmatch(name):
            return name
        taken = set(self.mapping.values()) | self.reserved
        pseudonym = next(f"Expert {letters}" for letters in _letters() if f"Expert {letters}" not in taken)
        self.mapping[name] = pseudonym
        self.added.append(name)
        return pseudonym

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(self.mapping, sort_keys=True, allow_unicode=True), encoding="utf-8")
        self.path.chmod(0o600)


# what pseudonymised calls a reviewer left out of the published form, in its working copy only: published_sheet refuses a sheet
# that still names one, so it is never published and never written to the mapping
UNSHOWN_REVIEWER = "reviewer not in the published form"


def pseudonymised(*, sheet: Sheet, pseudonyms: Pseudonyms, unshown: frozenset[str] = frozenset()) -> Sheet:
    """The full sheet with every reviewer under their pseudonym (records, labels, drops, weights, claim authors, signatures);
    the names in `unshown` (reviewers whose work the published form leaves out) are given no pseudonym."""
    names = reviewer_names(sheet=sheet)
    rename = {name: f"{UNSHOWN_REVIEWER} {index}" if name in unshown else pseudonyms.of(name=name) for index, name in enumerate(sorted(names))}

    def signers(*, value: str | None) -> str | None:
        return ", ".join(rename[name] for name in _signers(value=value)) if value else value

    data = sheet.model_dump()
    for claim in data["claims"]:
        for label in claim["labels"]:
            label["annotator"] = rename.get(label["annotator"], label["annotator"])
        for key in ("dropped", "weighted_by"):
            if claim.get(key) is not None:
                claim[key]["annotator"] = rename.get(claim[key]["annotator"], claim[key]["annotator"])
        if claim["source"]["type"] == "gm":
            claim["source"]["name"] = rename[claim["source"]["name"]]
    data["annotations"] = {rename[name]: record for name, record in data.get("annotations", {}).items()}
    data["seen_by_adjudicators"] = {rename[name]: at for name, at in data.get("seen_by_adjudicators", {}).items()}
    if data.get("adjudicated") is not None:
        data["adjudicated"]["adjudicator"] = rename[data["adjudicated"]["adjudicator"]]
    data["author"], data["reviewed_by"] = signers(value=sheet.author), signers(value=sheet.reviewed_by)
    return Sheet.model_validate(data)


def _final_labels(*, claim: ReferenceClaim) -> list[ClaimLabel]:
    """Each reviewer's latest label, in the order they were given, then the adjudicator's ruling when it decides the claim; dated."""
    latest = sorted(claim.latest_labels.values(), key=lambda label: label.at)
    ruling = claim.adjudication
    return [label.model_copy(update={"at": _day(at=label.at)}) for label in [*latest, *([ruling] if ruling is not None else [])]]


# what a published drop by the pipeline says, by pipeline step: its working comments name review-process details and entrants
PUBLISHED_DROP_COMMENTS = {CAP_ANNOTATOR: "set aside: over the review limit", "cleanup": "not a chess claim"}


def published_drop_comment(*, drop: ClaimDrop) -> str:
    """A pipeline drop's neutral comment (PUBLISHED_DROP_COMMENTS); a reviewer's own comment stays."""
    return PUBLISHED_DROP_COMMENTS.get(drop.annotator, drop.comment)


# what a published model claim names as its source: which model or agent pre-annotated it is not published
PUBLISHED_MODEL_SOURCE = "ai_model"


def published_source(*, source: ClaimSource) -> ClaimSource:
    """The source as published: an engine or expert source as it is, a model source as PUBLISHED_MODEL_SOURCE; never the answer a
    claim was drawn from (a run folder that is not published)."""
    name = PUBLISHED_MODEL_SOURCE if source.type == "model" else source.name
    return source.model_copy(update={"name": name, "answer_ref": None})


def _published_claim(*, claim: ReferenceClaim) -> ReferenceClaim:
    dropped = claim.dropped.model_copy(update={"at": _day(at=claim.dropped.at), "comment": published_drop_comment(drop=claim.dropped)}) if claim.dropped is not None else None
    return claim.model_copy(update={"labels": _final_labels(claim=claim), "dropped": dropped, "weighted_by": None, "original_text": None, "source": published_source(source=claim.source)})


def key_content(*, resolution: Resolution) -> dict[str, object]:
    """What grading and every consumer read of a resolved key: per claim its text, kind and fields, source (type and name), weight,
    duplicate link, idea link (same_idea_as), drop (its published comment), whether it holds true or awaits adjudication, each reviewer's final label and the ruling; the verdict, turning points,
    signature (its date) and whether the key is final or held back."""
    sheet = resolution.sheet
    claims = []
    for claim in sheet.claims:
        ruling = claim.adjudication
        claims.append(
            {
                **claim.model_dump(include={"claim_id", "text", "kind", "move", "band", "line", "outcome", "mate_in", "side", "about", "weight", "duplicate_of"}),
                "same_idea_as": claim.same_idea_as,
                "source": published_source(source=claim.source).model_dump(),
                "dropped": published_drop_comment(drop=claim.dropped) if claim.dropped is not None else None,
                "held_true": claim.held_true,
                "needs_adjudication": claim.needs_adjudication,
                "labels": {name: (label.value, label.comment) for name, label in sorted(claim.latest_labels.items())},
                "ruling": (ruling.annotator, ruling.value, ruling.comment) if ruling is not None else None,
            }
        )
    return {
        "task_id": sheet.task_id,
        "verdict": sheet.verdict,
        "turning_points": [tp.model_dump() for tp in sheet.turning_points],
        "answer_key": [c.claim_id for c in sheet.answer_key],
        "author": sorted(_signers(value=sheet.author)),
        "reviewed_by": sorted(_signers(value=sheet.reviewed_by)),
        "signed_on": _day(at=sheet.signed_at).date() if sheet.signed_at is not None else None,
        "final": resolution.final,
        "held_back": resolution.held_back is not None,
        "claims": claims,
    }


def published_sheet(*, sheet: Sheet, required: int, pseudonyms: Pseudonyms) -> Sheet:
    """The published form of a full sheet, refused when its review is held back (some required reviews still missing, or
    disagreements no adjudication settles: a plain sheet cannot say so) or when it would not resolve to the same key. A new
    reviewer gets a pseudonym only when the published form shows their work: one whose review never counts (an unfinished
    record, such as a test account's) gets none, and takes no letter from the reviewers who are shown."""
    trial = Pseudonyms(path=pseudonyms.path, mapping=dict(pseudonyms.mapping), reserved=pseudonyms.reserved)
    shown = reviewer_names(sheet=_published_form(sheet=sheet, required=required, pseudonyms=trial))
    unshown = frozenset(name for name in trial.added if trial.mapping[name] not in shown)
    published = _published_form(sheet=sheet, required=required, pseudonyms=pseudonyms, unshown=unshown)
    if any(name.startswith(UNSHOWN_REVIEWER) for name in reviewer_names(sheet=published)):
        raise PublishError(f"sheet {sheet.task_id}: a reviewer given no pseudonym appears in the published form")
    return published


def _published_form(*, sheet: Sheet, required: int, pseudonyms: Pseudonyms, unshown: frozenset[str] = frozenset()) -> Sheet:
    named = pseudonymised(sheet=sheet, pseudonyms=pseudonyms, unshown=unshown)
    resolution = resolve(sheet=named, required=required)
    if resolution.held_back is not None:
        raise PublishError(f"sheet {sheet.task_id} is held back ({resolution.held_back}); it is published once its key is final")
    key = resolution.sheet
    published = Sheet.model_validate(
        {
            "task_id": key.task_id,
            "verdict": key.verdict,
            "turning_points": [tp.model_dump() for tp in key.turning_points],
            "claims": [_published_claim(claim=c).model_dump() for c in key.claims],
            "last_claim_number": key.last_claim_number,
            "notes": "",
            "author": key.author,
            "reviewed_by": key.reviewed_by,
            "signed_at": _day(at=key.signed_at) if key.signed_at is not None else None,
        }
    )
    if key_content(resolution=resolve(sheet=published, required=required)) != key_content(resolution=resolution):
        raise PublishError(f"sheet {sheet.task_id}: the published form does not resolve to the same answer key as the full sheet")
    return published
