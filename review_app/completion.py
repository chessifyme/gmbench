"""What one grandmaster still owes a task before Mark as complete, and how far they are on it, by the card numbers they see.

Every shown claim (not set aside, not a duplicate) drafted from a model answer needs this reviewer's own label: the analysis
claims count as true until overruled and the grandmasters' claims are true by their author, so neither needs one. A claim left
without one does not hold completion up: bench.annotations.set_completed records it as skipped. Every Partly and False label of
this reviewer needs a comment saying why; that alone refuses Mark as complete. A save keeps unfinished work."""

from dataclasses import dataclass

from bench.annotations import needs_label
from bench.claim_display import shown_order
from bench.models import TEXT_EDITED, ReferenceClaim, Sheet

NEEDS_COMMENT = ("partly_true", "false")
UNCOMMENTED_NOTE = "Partly and False need a comment: say why in one line."


def shown_claims(*, sheet: Sheet | None) -> list[ReferenceClaim]:
    """The claims with a card, in card order (card 1 first): a duplicate and a set-aside claim have none."""
    return [c for c in shown_order(claims=sheet.claims if sheet else []) if c.duplicate_of is None and c.dropped is None]


def label_comment(*, comment: str) -> str:
    """The reviewer's own words on a label, without the note a text edit adds."""
    return comment.removeprefix(TEXT_EDITED).removeprefix("; ") if comment.startswith(TEXT_EDITED) else comment


def missing_comment(*, claim: ReferenceClaim, reviewer: str) -> bool:
    label = claim.latest_labels.get(reviewer)
    return label is not None and label.value in NEEDS_COMMENT and not label_comment(comment=label.comment).strip()


def numbered(*, numbers: list[int]) -> str:
    """"claim 3", "claims 3 and 7", "claims 3, 7 and 12"."""
    words = [str(n) for n in numbers]
    if len(words) == 1:
        return f"claim {words[0]}"
    return f"claims {', '.join(words[:-1])} and {words[-1]}"


@dataclass(frozen=True)
class Gaps:
    """The cards one reviewer still has to finish, by claim id, and the messages for the top of the page."""

    uncommented: dict[str, int]

    def summary(self) -> list[str]:
        if not self.uncommented:
            return []
        return [f"Partly and False need a comment: {numbered(numbers=list(self.uncommented.values()))} {'has' if len(self.uncommented) == 1 else 'have'} none yet."]

    def cards(self) -> dict[str, list[str]]:
        return {claim_id: [UNCOMMENTED_NOTE] for claim_id in self.uncommented}


def completion_gaps(*, sheet: Sheet, reviewer: str) -> Gaps:
    shown = list(enumerate(shown_claims(sheet=sheet), start=1))
    return Gaps(uncommented={c.claim_id: n for n, c in shown if missing_comment(claim=c, reviewer=reviewer)})


def progress_word(*, sheet: Sheet | None, reviewer: str, complete: bool, started: bool) -> str:
    """The task list's "Your progress" cell: complete, not started, or how many of the claims that need this reviewer's label
    they have labelled."""
    if complete:
        return "complete"
    if not started or sheet is None:
        return "not started"
    owed = [c for c in shown_claims(sheet=sheet) if needs_label(claim=c)]
    if not owed:
        return "started"
    done = sum(1 for c in owed if reviewer in c.latest_labels)
    return f"{done}/{len(owed)} labelled"
