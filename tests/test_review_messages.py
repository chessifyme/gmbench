from review_app.messages import card_notes

NUMBERS = {"mj_0007_c001": 1, "mj_0007_c004": 4, "new1": 5}


def test_a_message_goes_on_its_card_by_the_number_the_reviewer_sees_in_chess_words() -> None:
    notes = card_notes(messages=["claim mj_0007_c004 calls Bxe6 the best move but the sheet verdict is mistake"], numbers=NUMBERS)
    assert notes.cards == {"mj_0007_c004": ["This claim says Bxe6 is best, but your verdict says it's a mistake."]}
    assert notes.summary == ["Claim 4: This claim says Bxe6 is best, but your verdict says it's a mistake."]


def test_an_unrecognised_message_keeps_its_words_with_card_numbers_for_ids() -> None:
    notes = card_notes(messages=["the form lists claim mj_0007_c001 twice; reload the page", "new claim row 1 is marked Important but has no text"], numbers=NUMBERS)
    assert notes.cards == {"mj_0007_c001": ["the form lists claim 1 twice; reload the page"], "new1": ["claim 5 is marked Important but has no text"]}
    assert "mj_0007" not in " ".join(notes.summary)


def test_a_message_about_no_claim_stays_at_the_top_only() -> None:
    notes = card_notes(messages=["the answer key is empty: every family scores coverage on it, so weight at least one true claim", "enter your name to save"], numbers=NUMBERS)
    assert notes.cards == {} and notes.summary == ["No claim is in the answer key yet: mark at least one True claim Essential, Important or Nice to have.", "enter your name to save"]


def test_answer_key_messages_speak_of_the_levels_not_of_numbers() -> None:
    messages = [
        "claim mj_0007_c004 has weight 3 but its latest label is not true; a claim enters the answer key only while every annotator's latest label is true",
        "a reviewed game_review answer key needs a weighted claim besides the turning-point claims; otherwise coverage only repeats the turning-point score",
    ]
    notes = card_notes(messages=messages, numbers=NUMBERS)
    assert notes.cards["mj_0007_c004"] == ["This claim is in the answer key but isn't labelled True: label it True or set it to Not key."]
    assert "Essential, Important or Nice to have" in notes.summary[1]
    assert not any(phrase in " ".join(notes.summary) for phrase in ("weight 3", "weighted", "1 to 3", "weights 1"))
