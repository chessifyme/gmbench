from datetime import timedelta

from bench.judge_tools import GmClaim
from bench.models import ClaimDrop, ClaimLabel, ClaimSource, ReferenceClaim, Sheet
from bench.verify.gm import gm_claims_for, gm_label
from tests.claim_fakes import LABELLED_AT, claim, model_source

GM_SOURCE = ClaimSource(type="gm", name="GM Hidden Author", answer_ref=None)
LATER = LABELLED_AT + timedelta(hours=1)


def said(*, value: str, annotator: str, comment: str = "", at: object = LABELLED_AT, role: str = "annotator") -> ClaimLabel:
    return ClaimLabel.model_validate({"annotator": annotator, "value": value, "comment": comment, "at": at, "role": role})


def sheet(*claims: ReferenceClaim) -> Sheet:
    return Sheet.model_validate({"task_id": "mj_9001", "verdict": None, "turning_points": [], "claims": list(claims), "last_claim_number": 99, "notes": "", "author": None, "reviewed_by": None, "signed_at": None})


def plan(*, number: int, text: str, labels: list[ClaimLabel] | None = None, source: ClaimSource | None = None, duplicate_of: str | None = None) -> ReferenceClaim:
    return claim(claim_id=f"mj_9001_c{number:03d}", text=text, weight=None, source=source or model_source(), kind="plan", labels=labels, duplicate_of=duplicate_of)


AGREED = plan(number=1, text="Black should castle", labels=[said(value="true", annotator="GM Ann", comment="safer king"), said(value="true", annotator="GM Bob", comment="yes", at=LATER)])
DISPUTED = plan(number=2, text="Black should attack", labels=[said(value="true", annotator="GM Ann"), said(value="false", annotator="GM Bob")])
ADJUDICATED = plan(number=3, text="White is better", labels=[said(value="true", annotator="GM Ann"), said(value="false", annotator="GM Bob"), said(value="partly_true", annotator="GM Cid", role="adjudicator", at=LATER)])
UNLABELLED = plan(number=4, text="Black is fine")
WRITTEN = plan(number=5, text="Na5 keeps the balance", source=GM_SOURCE)
DUPLICATE = plan(number=6, text="Castling is best for Black", duplicate_of="mj_9001_c001")
DROPPED = plan(number=7, text="Chess is hard").model_copy(update={"dropped": ClaimDrop(annotator="GM Ann", comment="useless", at=LABELLED_AT)})
ENGINE_UNLABELLED = claim(claim_id="mj_9001_c008", text="Black is slightly better", weight=None, kind="eval_band", about="task_position", band="black_slightly_better")


def test_the_label_a_review_settled_on_or_why_there_is_none() -> None:
    assert gm_label(claim=AGREED) == ("true", None)
    assert gm_label(claim=DISPUTED) == (None, "disputed")
    assert gm_label(claim=ADJUDICATED) == ("partly_true", None)
    assert gm_label(claim=UNLABELLED) == (None, "unlabelled")
    assert gm_label(claim=DROPPED) == (None, "dropped")


def test_the_judge_sees_settled_claims_and_grandmaster_claims_with_their_labels_and_comments() -> None:
    found = gm_claims_for(sheet=sheet(AGREED, DISPUTED, ADJUDICATED, UNLABELLED, WRITTEN, DUPLICATE, DROPPED, ENGINE_UNLABELLED))
    assert found == (
        GmClaim(claim_id="mj_9001_c001", text="Black should castle", label="true", comment="safer king / yes", kind="plan"),
        GmClaim(claim_id="mj_9001_c003", text="White is better", label="partly_true", comment="", kind="plan"),
        GmClaim(claim_id="mj_9001_c005", text="Na5 keeps the balance", label="true", comment="", kind="plan"),
        GmClaim(claim_id="mj_9001_c006", text="Castling is best for Black", label="true", comment="safer king / yes", kind="plan"),
    )


def test_no_reviewer_or_author_name_reaches_the_judge() -> None:
    found = gm_claims_for(sheet=sheet(AGREED, ADJUDICATED, WRITTEN, DUPLICATE))
    shown = repr(found)
    assert not any(name in shown for name in ("GM Ann", "GM Bob", "GM Cid", "GM Hidden Author"))


def test_a_labelled_grandmaster_claim_takes_its_label_not_its_source() -> None:
    relabelled = plan(number=5, text="Na5 keeps the balance", source=GM_SOURCE, labels=[said(value="false", annotator="GM Ann")])
    assert gm_claims_for(sheet=sheet(relabelled))[0].label == "false"


def test_a_duplicate_of_an_unsettled_claim_is_left_out() -> None:
    copy = plan(number=6, text="Black must attack", duplicate_of="mj_9001_c002")
    assert gm_claims_for(sheet=sheet(DISPUTED, copy)) == ()


def test_the_comment_is_only_that_of_the_labels_that_decided_the_claim() -> None:
    overruled = plan(number=3, text="White is better", labels=[said(value="true", annotator="GM Ann", comment="clearly"), said(value="false", annotator="GM Bob", comment="no way"), said(value="false", annotator="GM Cid", comment="Black holds", role="adjudicator", at=LATER)])
    changed = plan(number=1, text="Black should castle", labels=[said(value="false", annotator="GM Ann", comment="too slow"), said(value="true", annotator="GM Ann", comment="safer king", at=LATER)])
    assert [c.comment for c in gm_claims_for(sheet=sheet(overruled, changed))] == ["Black holds", "safer king"]


def test_a_gm_claim_carries_the_sheets_kind_move_band_and_position() -> None:
    labelled = ENGINE_UNLABELLED.model_copy(update={"labels": [said(value="true", annotator="GM Ann")]})
    [found] = gm_claims_for(sheet=sheet(labelled))
    assert (found.kind, found.move, found.band, found.about) == ("eval_band", None, "black_slightly_better", "task_position")
