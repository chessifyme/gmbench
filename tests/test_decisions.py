from datetime import UTC, datetime

import pytest

from bench.decisions import KEEP, Decision, apply_decision, new_claim, parse_label_word
from tests.claim_fakes import claim, label, model_source

AT = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
HOW = "write your name"
GM = "GM Reviewer"


def _pooled(*, weight: int | None = None, labels: list | None = None):  # type: ignore[no-untyped-def]
    return claim(claim_id="ap_9001_c004", text="White should push g4 soon", weight=weight, source=model_source(), kind="plan", labels=labels)


def _apply(c, **decision):  # type: ignore[no-untyped-def]
    return apply_decision(claim=c, annotator=GM, decision=Decision(**decision), at=AT, how_to_name=HOW)


@pytest.mark.parametrize(("text", "word"), [("True", "True"), ("partly", "Partly"), ("partly_true", "Partly"), (" FALSE ", "False"), ("unclear", "Unclear"), ("", None)])
def test_label_words_are_read_in_any_case(text: str, word: str | None) -> None:
    assert parse_label_word(text=text, where="row 3") == word


def test_an_unknown_label_is_refused() -> None:
    with pytest.raises(ValueError, match="row 3: label 'ok' is not one of True, Partly, False, Unclear$"):
        parse_label_word(text="ok", where="row 3")


def test_no_decision_changes_nothing_and_needs_no_name() -> None:
    c = _pooled()
    assert apply_decision(claim=c, annotator="", decision=Decision(), at=AT, how_to_name=HOW) == c


def test_labels_map_to_the_stored_values_with_the_comment() -> None:
    for word, value in (("True", "true"), ("Partly", "partly_true"), ("False", "false"), ("Unclear", "unclear")):
        labelled = _apply(_pooled(), label=word, comment="why")
        assert [(l.annotator, l.value, l.comment) for l in labelled.labels] == [(GM, value, "why")]


def test_false_takes_the_weight_away_when_the_weight_is_kept_and_true_keeps_it() -> None:
    weighted = claim(claim_id="ap_9001_c002", text="Be3 is best", weight=2, kind="best_move", about="task_position", move="Be3")
    assert _apply(weighted, label="False").weight is None
    assert _apply(weighted, label="True").weight == 2


def test_off_topic_is_not_a_label_and_a_label_brings_back_a_claim_a_script_set_aside() -> None:
    from bench.models import drop_claim

    with pytest.raises(ValueError, match="is not one of"):
        parse_label_word(text="Off-topic", where="row 3")
    aside = drop_claim(claim=_pooled(), annotator="script", comment="", at=AT, how_to_name=HOW)
    with pytest.raises(ValueError, match="was set aside by script; only a True claim carries a weight"):
        _apply(aside, weight=2)
    back = _apply(aside, label="True", weight=2)
    assert back.dropped is None and back.weight == 2


def test_a_rewording_needs_a_label_is_one_claim_and_is_recorded() -> None:
    with pytest.raises(ValueError, match="has a new wording but no label"):
        _apply(_pooled(), new_text="White pushes g4")
    with pytest.raises(ValueError, match="the new wording is not one claim"):
        _apply(_pooled(), label="True", new_text="White pushes g4. Then h4.")
    reworded = _apply(_pooled(), label="True", new_text="White pushes g4", comment="clearer")
    assert reworded.text == "White pushes g4" and reworded.labels[-1].comment == "text edited; clearer"
    assert _apply(reworded, label="True", new_text="White pushes g4", comment="clearer") == reworded


def test_a_comment_needs_a_label() -> None:
    with pytest.raises(ValueError, match="has a comment but no label"):
        _apply(_pooled(), comment="hm")


def test_a_weight_only_on_a_true_claim() -> None:
    with pytest.raises(ValueError, match="claim ap_9001_c004 is not labelled yet; only a True claim carries a weight"):
        _apply(_pooled(), weight=2)
    with pytest.raises(ValueError, match="is not labelled True by every reviewer"):
        _apply(_pooled(), label="Partly", weight=2)
    with pytest.raises(ValueError, match="needs adjudication first"):
        _apply(_pooled(labels=[label(value="true", annotator="A"), label(value="false", annotator="B")]), weight=2)
    weighted = _apply(_pooled(), label="True", weight=3)
    assert weighted.weight == 3 and weighted.weighted_by is not None and weighted.weighted_by.annotator == GM
    assert _apply(weighted, weight=None).weight is None
    assert _apply(weighted, weight=KEEP) == weighted


def test_a_ruling_settles_a_dispute_once() -> None:
    disputed = _pooled(labels=[label(value="true", annotator="A"), label(value="false", annotator="B")])
    settled = _apply(disputed, ruling="true", weight=2)
    assert settled.weight == 2 and settled.labels[-1].role == "adjudicator"
    assert _apply(settled, ruling="true", weight=2) == settled
    with pytest.raises(ValueError, match="does not need adjudication"):
        _apply(_pooled(), ruling="true")


def _new(text: str, *, claims: list | None = None, **decision):  # type: ignore[no-untyped-def]
    return new_claim(task_id="ap_9001", number=5, claims=claims or [], annotator=GM, text=text, decision=Decision(**decision), at=AT, how_to_name=HOW)


def test_a_new_claim_takes_its_weight_label_and_comment() -> None:
    added = _new("White's king is safer on b1", weight=2)
    assert added is not None and (added.claim_id, added.weight, added.source.type, added.labels) == ("ap_9001_c005", 2, "gm", [])
    commented = _new("White's king is safer on b1", comment="key idea")
    assert commented is not None and [(l.value, l.comment) for l in commented.labels] == [("true", "key idea")]
    partly = _new("White's king is safer on b1", label="Partly")
    assert partly is not None and partly.labels[-1].value == "partly_true"


def test_a_new_claim_repeating_yours_is_skipped_or_refused_with_another_weight() -> None:
    first = _new("White's king is safer on b1", weight=2)
    assert first is not None
    assert _new("White's king is  SAFER on b1", claims=[first]) is None
    assert _new("White's king is safer on b1", claims=[first], weight=2) is None
    with pytest.raises(ValueError, match="repeats your claim ap_9001_c005"):
        _new("White's king is safer on b1", claims=[first], weight=3)


def test_a_new_claim_refuses_a_weight_on_a_false_label_and_two_claims() -> None:
    with pytest.raises(ValueError, match="labelled False, so it carries no weight"):
        _new("x is y", label="False", weight=2)
    with pytest.raises(ValueError, match="the new claim is not one claim"):
        _new("White is better, or Black is worse")
