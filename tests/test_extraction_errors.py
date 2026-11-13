"""v1_rc39: an extraction error is not an illegal move, and a suspicious extraction is made again once. The cases are from the
development runs: the Ziuk mc_0015 grade whose extractor gave one claim with the move "exf5#INVALID" for a long answer (ruled
illegal, precision 0), the claude_openrouter_tools mc_0013 claim "Be2 МCRayavannan ta debuta shah Rhaedekar ad." and mj_0015 line
"R1g7 and then Ra7", and the false-credit and escape cases of the PR's independent review."""

import json

import pytest

from bench.grading.claims import SPARSE_MIN_CHARS, Claim, ClaimList, RawClaim, extract_claims_checked, extraction_problem, normalised_claim
from bench.grading.grade import GradeRecord, guard_claims, guard_passes, illegal_claim_moves, stored_extraction_problem
from bench.grading.moves import normalise_san, readable_san
from bench.verify.check import recheck, unplaced_verdict
from bench.verify.positions import place
from bench.verify.schema import CheckedClaims, ClaimResult, InputClaim
from bench.verify.subject import subject_for_task
from tests.grading_fakes import FakeJson
from tests.test_claim_check import ANSWER, CFG, GM, ILLEGAL, MV, checker, claim
from tests.test_guard_refusals import BANDS, _key, _line_claim, _task, _top_lines
from tests.tool_loop_fakes import ScriptedToolModel, answers
from tests.verify_fakes import claim_result

MC_0015_ANSWER = (
    "Black should simply take the knight with 23... exf5, winning a piece.\n\n"
    "White's tactical idea in playing the knight to f5 was that the Black knight on d5 is pinned or overworked, hoping to regain the piece "
    "after trades on the c- and d-files: 23... exf5 24. Rxc8 Qxc8 25. Rxd5 Rxd5 26. Qxd5.\n\n"
    "However, White's combination falls apart against the decisive counter-blow 26... Qc2.\n\n"
    "From c2, Black's queen creates an unstoppable double threat: attacking the undefended bishop on b2 and threatening back-rank mate with "
    "...Qb1+. White has no adequate defense:\nMoving the bishop with 27. Bd4 walks into immediate checkmate: 27... Qc1#.\n"
    "Defending against back-rank mate with a luft move like 27. g3 or 27. h3 simply drops the bishop to 27... Qxb2, leaving Black a full minor piece ahead.\n"
    "An intermediate check like 27. Qa8+ is easily parried with 27... Bf8, renewing both the mate threat on the back rank and the capture of the "
    "bishop on b2. After 28. g3 Qxb2, Black remains cleanly up a piece.\n\n"
    "If White instead tries 24. Rxd5, Black trades rooks first with 24... Rxc1+ 25. Bxc1 Qxd5, leaving Black a full rook ahead."
)


def _subject(task_id: str):  # type: ignore[no-untyped-def]
    return subject_for_task(task=_task(task_id))


def _checked(*claims: InputClaim, subject=MV, answer: str = ANSWER) -> list[ClaimResult]:  # type: ignore[no-untyped-def]
    return checker(script=ScriptedToolModel(answers("never"))).check(claims=list(claims), subject=subject, answer_text=answer, reference=None, gm=GM).results


# 1. unreadable is not illegal


@pytest.mark.parametrize(("token", "readable", "normal"), [
    ("exf5#INVALID", False, "exf5#INVALID"),
    ("Be2 МCRayavannan ta debuta shah Rhaedekar ad.", False, "Be2 МCRayavannan ta debuta shah Rhaedekar ad"),
    ("castles", False, "castles"),
    ("Bxh6.", True, "Bxh6"),
    ("Nf3!?", True, "Nf3"),
    ("Qh7+!", True, "Qh7"),
    ("♘f3", True, "Nf3"),
    ("0-0", True, "O-O"),
    ("Ra9", True, "Ra9"),  # reads as notation; a square off the board makes it illegal, not unreadable
    ("23...Qh3", True, "Qh3"),
    ("...Qh3", True, "Qh3"),
    ("23. Qh3", True, "Qh3"),
    ("Qh3,", True, "Qh3"),
    ("(Qh3)", True, "Qh3"),
    ("...", False, ""),
])
def test_a_move_reads_as_chess_notation_with_its_full_stop_and_annotations_removed(token: str, readable: bool, normal: str) -> None:
    assert readable_san(token=token) is readable and normalise_san(token=token) == normal


def test_mc_0015_an_unreadable_best_move_is_an_extraction_error_not_an_illegal_move() -> None:
    subject = _subject("mc_0015")
    invalid = claim(kind="best_move", text="Black should simply take the knight with 23...exf5.", move="exf5#INVALID")
    [result] = _checked(invalid, subject=subject, answer=MC_0015_ANSWER)
    assert (result.verdict, result.decided_by) == ("unverifiable", "code") and result.detail.startswith("extraction error: 'exf5#INVALID'")
    assert illegal_claim_moves(results=[result], task=_task("mc_0015")) == 0


def test_mc_0013_a_move_followed_by_garbage_is_an_extraction_error() -> None:
    garbage = claim(kind="best_move", text="I recommend 20.Be2.", move="Be2 МCRayavannan ta debuta shah Rhaedekar ad.")
    [result] = _checked(garbage, subject=_subject("mc_0013"), answer="I recommend 20.Be2. The engine rates it as clearly the best move.")
    assert result.verdict == "unverifiable" and "extraction error" in result.detail


def test_a_move_with_a_full_stop_or_annotation_is_read_without_it_at_extraction() -> None:
    raw = RawClaim(kind="best_move", text="20.Bxh6. wins", move="Bxh6.", band=None, line=["Nf3!?", "...", "e5?!", " ", "Qh7+!"], outcome=None, mate_in=None, about=None)
    cleaned = normalised_claim(raw=raw)
    assert (cleaned.move, cleaned.line) == ("Bxh6", ["Nf3", "e5", "Qh7"])


def test_a_readable_move_not_legal_in_its_position_is_still_an_illegal_move() -> None:
    assert [r.verdict for r in _checked(ILLEGAL)] == ["illegal_move"]
    off_board = claim(kind="best_move", text="Ra9 is best", move="Ra9")
    assert [r.verdict for r in _checked(off_board)] == ["illegal_move"]
    # an illegal move written with its number, dots or a comma is still read, and still illegal
    for written in ("6...Qh3", "...Qh3", "Qh3,"):
        numbered = ILLEGAL.model_copy(update={"line": ["Nxd5", written]})
        [result] = _checked(numbered)
        assert result.verdict == "illegal_move", written


# 2. a line that leaves out the other side's reply


def test_mj_0015_a_line_that_leaves_out_the_reply_is_unverifiable_not_illegal() -> None:
    subject = _subject("mj_0015")
    skipped = claim(kind="line_outcome", text="After Nf8, White can play R1g7 and then Ra7.", line=["Nf8", "R1g7", "Ra7"], outcome="wins", side="white")
    assert isinstance(place(subject=subject, claim=skipped), str)
    [result] = _checked(skipped, subject=subject, answer="After Nf8, White can play R1g7 and then Ra7.")
    assert result.verdict == "unverifiable" and "leaves out the other side's reply before Ra7" in result.detail


def test_a_move_the_answer_numbers_for_the_side_to_move_is_not_a_left_out_reply() -> None:
    # White's queen cannot reach e8; Black's could, but "7.Qe8" gives the move to White: an illegal move, as before
    mates = claim(kind="line_outcome", text="Nxd5 Nxf7 Kxf7 Qe8 mates", line=["Nxd5", "Nxf7", "Kxf7", "Qe8"], outcome="mates", side="white")
    assert [r.verdict for r in _checked(mates, answer="Nxd5 is a mistake, because after 6.Nxf7 Kxf7 7.Qe8 White mates.")] == ["illegal_move"]
    assert [r.verdict for r in _checked(mates, answer="Nxd5 is a mistake: Nxf7 Kxf7 and then Qe8 mates.")] == ["unverifiable"]
    # the same with figurines
    assert [r.verdict for r in _checked(mates, answer="Nxd5 is a mistake, because after 6.♘xf7 ♚xf7 7.♕e8 White mates.")] == ["illegal_move"]


def test_a_first_move_for_the_wrong_side_is_not_a_left_out_reply() -> None:
    # Qh3 is legal for neither side after Nxd5: the line stays an illegal move; a first move is placed from both starts already
    verdict, _ = unplaced_verdict(claim=ILLEGAL, placement="x", subject=MV, answer_text=ANSWER)
    assert verdict == "illegal_move"


# 4. a move the claim itself calls illegal


@pytest.mark.parametrize("text", ["Qh3 is illegal here.", "Qh3 would be illegal.", "Qh3 is not legal.", "Qh3 cannot be played.", "Nxd5, and the illegal Qh3 is often suggested."])
def test_a_move_the_claim_calls_illegal_is_not_counted_against_it(text: str) -> None:
    [result] = _checked(ILLEGAL.model_copy(update={"text": text}))
    assert result.verdict == "unverifiable" and "calls Qh3 illegal" in result.detail


@pytest.mark.parametrize("text", [
    "Nxd5 and now Qh3 is impossible to stop.",
    "Nxd5? Qh3! is impossible to meet.",
    "Nxd5 Qh3 is not possible to defend against.",
    "Nxd5 Qh3!! (impossible to foresee) wins.",
    "Nxd5 Qh3: illegal moves aside, this is the most forcing.",
    "Qh3 makes Black's defence impossible.",
    "White plays Qh3. Then castling is illegal for Black.",
])
def test_impossible_or_an_illegal_word_not_about_the_move_leaves_it_illegal(text: str) -> None:
    [result] = _checked(ILLEGAL.model_copy(update={"text": text}))
    assert result.verdict == "illegal_move"


def test_only_the_claims_own_text_is_read_for_calling_a_move_illegal() -> None:
    [result] = _checked(ILLEGAL, answer="6.Nxd5 Qh3 wins. Later, Qh3 is illegal after g3.")
    assert result.verdict == "illegal_move"


# rescoring a stored grade


def test_a_recheck_rules_a_stored_illegal_move_again_on_the_answer_text() -> None:
    invalid = claim(kind="best_move", text="23...exf5 wins.", move="exf5#INVALID")
    dotted = claim(claim_id="c2", kind="best_move", text="Na5. was better", move="Na5.")
    stored_illegal = _checked(ILLEGAL)[0]
    stored = [stored_illegal.model_copy(update={"claim": c}) for c in (invalid, dotted)]
    again = recheck(checked=CheckedClaims(positions={}, results=[*stored, stored_illegal], judge_model=None), subject=MV, cfg=CFG, reference=None, gm=GM, answer_text=ANSWER).results
    assert [r.verdict for r in again] == ["unverifiable", "unverifiable", "illegal_move"]
    assert "extraction error" in again[0].detail and "grade the answer again" in again[1].detail


# 3. a suspicious extraction is made again once


def _claims_json(*claims: dict[str, object]) -> str:
    base = {"band": None, "line": None, "outcome": None, "mate_in": None, "side": None, "about": None, "move": None}
    return json.dumps({"claims": [{**base, **c} for c in claims]})


class SequencedJson(FakeJson):
    def __init__(self, *payloads: str) -> None:
        super().__init__(payload=None)
        self.payloads = list(payloads)

    def complete(self, *, system: str, user: str, schema: dict[str, object] | None):  # type: ignore[no-untyped-def]
        self.payload = self.payloads[min(len(self.calls), len(self.payloads) - 1)]
        return super().complete(system=system, user=user, schema=schema)


INVALID = _claims_json({"kind": "best_move", "text": "Black should simply take the knight with 23...exf5.", "move": "exf5#INVALID"})
GOOD = _claims_json(
    {"kind": "best_move", "text": "23...exf5 is best.", "move": "exf5"},
    {"kind": "line_outcome", "text": "23...exf5 wins a piece.", "line": ["exf5"], "outcome": "wins_material", "side": "black"},
    {"kind": "threat", "text": "26...Qc2 attacks the b2-bishop.", "move": "Qc2"},
)
ONE_GOOD = _claims_json({"kind": "best_move", "text": "23...exf5 is best.", "move": "exf5"})


def test_mc_0015_an_unreadable_extraction_is_made_again_and_the_retry_kept() -> None:
    extractor = SequencedJson(INVALID, GOOD)
    claims, retry = extract_claims_checked(answer_text=MC_0015_ANSWER, task=_task("mc_0015"), extractor=extractor)
    assert len(extractor.calls) == 2 and [c.move for c in claims] == ["exf5", None, "Qc2"]
    assert retry is not None and (retry.kept, retry.claim_counts, retry.flagged) == ("retry", (1, 3), False) and "exf5#INVALID" in retry.reason


def test_one_claim_for_a_long_answer_full_of_moves_is_made_again_and_flagged_when_the_retry_is_no_better() -> None:
    assert len(MC_0015_ANSWER) >= SPARSE_MIN_CHARS
    extractor = SequencedJson(ONE_GOOD, ONE_GOOD)
    claims, retry = extract_claims_checked(answer_text=MC_0015_ANSWER, task=_task("mc_0015"), extractor=extractor)
    assert len(extractor.calls) == 2 and len(claims) == 1
    assert retry is not None and retry.kept == "first" and retry.flagged and retry.reason.startswith("1 claim for an answer of")
    # an unreadable first and a sparse but readable retry: the readable one is kept, still flagged
    claims, retry = extract_claims_checked(answer_text=MC_0015_ANSWER, task=_task("mc_0015"), extractor=SequencedJson(INVALID, ONE_GOOD))
    assert [c.move for c in claims] == ["exf5"] and retry is not None and retry.kept == "retry" and retry.flagged


def test_a_long_retry_with_one_garbled_move_beats_a_one_claim_first_extraction() -> None:
    many = _claims_json(
        {"kind": "best_move", "text": "23...exf5 is best.", "move": "exf5"},
        *({"kind": "other", "text": f"claim {i}"} for i in range(18)),
        {"kind": "threat", "text": "Qc2 hits b2", "move": "Qc2 then Qb1+"},
    )
    claims, retry = extract_claims_checked(answer_text=MC_0015_ANSWER, task=_task("mc_0015"), extractor=SequencedJson(ONE_GOOD, many))
    assert len(claims) == 20 and retry is not None and retry.kept == "retry" and retry.flagged


def test_a_sound_extraction_or_a_short_answer_is_extracted_once() -> None:
    extractor = SequencedJson(GOOD)
    assert extract_claims_checked(answer_text=MC_0015_ANSWER, task=_task("mc_0015"), extractor=extractor)[1] is None and len(extractor.calls) == 1
    short = "23...exf5 wins a piece: 24.Rxc8 Qxc8 25.Rxd5 Rxd5 26.Qxd5 Qc2."
    extractor = SequencedJson(ONE_GOOD)
    assert extract_claims_checked(answer_text=short, task=_task("mc_0015"), extractor=extractor)[1] is None and len(extractor.calls) == 1
    # a long answer that writes almost no moves is not chess text enough to call one claim a failed extraction
    words = "Black is better because the knight is strong and the bishop is weak. " * 15
    assert extraction_problem(claims=[Claim(**ClaimList.model_validate_json(ONE_GOOD).claims[0].model_dump(), checkable=True)], answer_text=words) is None


def test_a_stored_grade_with_a_suspicious_extraction_is_named_for_regrading() -> None:
    from tests.test_rejudge import _first

    grade: GradeRecord = _first()
    invalid = claim_result(kind="best_move", text="23...exf5", verdict="illegal_move", decided_by="code", move="exf5#INVALID")
    suspect = grade.model_copy(update={"claims": [invalid]})
    assert stored_extraction_problem(grade=grade) is None
    assert "exf5#INVALID" in str(stored_extraction_problem(grade=suspect))


def test_a_rescore_marks_a_grade_whose_extraction_was_never_retried() -> None:
    from bench import GRADER_VERSION
    from bench.grading.grade import rescore
    from tests.grading_fakes import MV_REF, MV_TASK
    from tests.test_rejudge import _first, _sheet

    grade: GradeRecord = _first()
    sound = rescore(grade=grade, task=MV_TASK, sheet=_sheet(), reference=MV_REF, cfg=CFG, grader_version=GRADER_VERSION)
    assert sound.unretried_extraction is None and "unretried_extraction" not in sound.model_dump(mode="json")
    invalid = claim_result(kind="best_move", text="Na5 was better", verdict="illegal_move", decided_by="code", move="Na5#INVALID", position_ids=["P1"])
    marked = rescore(grade=grade.model_copy(update={"claims": [invalid]}), task=MV_TASK, sheet=_sheet(), reference=MV_REF, cfg=CFG, grader_version=GRADER_VERSION)
    assert "Na5#INVALID" in str(marked.unretried_extraction) and marked.claims[0].verdict == "unverifiable"


# 5. evaluations in words are not read by the guards (dropped from v1_rc39 for false credit; they wait for the grader study)


def test_holding_or_a_piece_up_in_words_earns_no_evaluation_band() -> None:
    task, top = _task("mj_0034"), _top_lines("mj_0034")
    _, _, key = _key("mj_0034_c003")  # After 39...Qf6, the position is roughly equal
    holds = claim_result(kind="other", text="Black holds comfortably after Qf6.", verdict="unverifiable")
    assert not guard_passes(check=key, mentioned=set(), lines=[], claims=guard_claims(extracted=[holds], positions={}, task=task, top_lines=top), task=task, bands=BANDS)
    task, top = _task("mc_0015"), _top_lines("mc_0015")
    _, _, key = _key("mc_0015_c002")  # Black is winning
    line = _line_claim("exf5", outcome="wins_material", side="black")
    piece = line.model_copy(update={"claim": line.claim.model_copy(update={"text": "23...exf5 wins a piece."})})
    assert not guard_passes(check=key, mentioned=set(), lines=[], claims=guard_claims(extracted=[piece], positions={}, task=task, top_lines=top), task=task, bands=BANDS)



