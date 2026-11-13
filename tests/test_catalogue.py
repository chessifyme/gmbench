from collections import Counter

import pytest
from pydantic import ValidationError

from bench.catalogue import Catalogue, Slot, load_catalogue

PILOT = {"assessment_plan": 10, "move_judgement": 8, "move_choice": 6, "game_review": 6}


def test_plan_counts_matches_pilot_total() -> None:
    slots = load_catalogue().plan_counts(family_counts=PILOT)
    assert len(slots) == 30
    assert Counter(s.family for s in slots)["assessment_plan"] == 10
    assert all(isinstance(s, Slot) for s in slots)


def test_plan_counts_allocates_question_types_by_largest_remainder() -> None:
    catalogue = load_catalogue()
    slots = catalogue.plan_counts(family_counts={"assessment_plan": 10})
    by_type = Counter(s.question_type for s in slots)
    shares = {q.id: q.share for q in catalogue.families["assessment_plan"].question_types}
    assert sum(by_type.values()) == 10
    for type_id, count in by_type.items():
        assert abs(count - 10 * shares[type_id]) < 1


def test_plan_counts_spreads_audience_by_mix_within_family() -> None:
    slots = load_catalogue().plan_counts(family_counts={"assessment_plan": 10})
    assert Counter(s.audience for s in slots) == {"beginner": 2, "club": 6, "advanced": 2}


def test_slot_focus_comes_from_question_type() -> None:
    catalogue = load_catalogue()
    slots = catalogue.plan_counts(family_counts=PILOT)
    focus_of = {q.id: q.focus for f in catalogue.families.values() for q in f.question_types}
    assert all(s.focus == focus_of[s.question_type] for s in slots)


def test_templates_use_known_placeholders_only() -> None:
    catalogue = load_catalogue()
    for family in catalogue.families.values():
        for question_type in family.question_types:
            assert 2 <= len(question_type.templates) <= 4
            for template in question_type.templates:
                leftovers = template.format(side="White", move="Nf3", color="white", level="a beginner")
                assert "{" not in leftovers and "}" not in leftovers


def test_shares_sum_to_one_per_family() -> None:
    catalogue = load_catalogue()
    for family in catalogue.families.values():
        assert abs(sum(q.share for q in family.question_types) - 1.0) < 1e-6
    assert abs(sum(catalogue.audience_mix.values()) - 1.0) < 1e-6


def test_move_judgement_asks_only_about_the_move_played() -> None:
    types = {q.id: q.share for q in load_catalogue().families["move_judgement"].question_types}
    assert types == {"rate_move": 0.6, "was_it_good": 0.4}


def test_the_former_position_questions_split_into_move_choice_and_assessment_plan() -> None:
    families = load_catalogue().families
    assert [q.id for q in families["move_choice"].question_types] == ["best_move", "practical_choice"]
    assert [q.id for q in families["assessment_plan"].question_types] == ["best_plan", "evaluate_both_sides", "explain_for_level"]


def test_a_leading_move_judgement_template_is_rejected() -> None:
    data = load_catalogue().model_dump()
    data["families"]["move_judgement"]["question_types"][0]["templates"][0] = "Why was {move} a mistake?"
    with pytest.raises(ValidationError, match="presupposes a verdict"):
        Catalogue.model_validate(data)
