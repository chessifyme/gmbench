import json

from bench.grading.extract import StatedVerdict, extract_stated_verdict, extract_turning_points, extract_verdict
from bench.paths import PROMPTS_DIR
from tests.grading_fakes import GR_TASK, FakeJson


def test_extract_verdict_maps_label_and_skips_empty() -> None:
    extractor = FakeJson(payload=json.dumps({"verdict": "mistake", "quote": "Nxd5 was a mistake"}))
    assert extract_verdict(answer_text="Nxd5 was a mistake because of Nxf7.", extractor=extractor) == "mistake"
    assert "Nxd5 was a mistake because of Nxf7." in extractor.calls[0][1]
    silent = FakeJson(payload=None, raise_if_called=True)
    assert extract_verdict(answer_text="   ", extractor=silent) == "none_stated"


def system_prompt(name: str) -> str:
    from bench.paths import PROMPTS_DIR

    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def test_extract_turning_points_maps_move_text_to_ply() -> None:
    extractor = FakeJson(payload=json.dumps({"items": [{"ply": 18, "label": "blunder"}, {"ply": 6, "label": "unlabelled"}, {"ply": 99, "label": "mistake"}, {"ply": 18, "label": "mistake"}, {"ply": 0, "label": "blunder"}]}))
    found = extract_turning_points(answer_text="The mistakes were 3...Bg4 and 9...b5.", task=GR_TASK, extractor=extractor)
    assert [(m.ply, m.label) for m in found] == [(6, "unlabelled"), (18, "blunder")]
    assert "missed_win" in system_prompt("extract_turning_points.md")
    system, user = extractor.calls[0]
    assert "ply 18: 9...b5" in user and "ply 6: 3...Bg4" in user and "The mistakes were 3...Bg4 and 9...b5." in user
    assert "ply" in system


def test_extract_turning_points_empty_answer_returns_empty() -> None:
    assert extract_turning_points(answer_text="", task=GR_TASK, extractor=FakeJson(payload=None, raise_if_called=True)) == []


def test_extraction_prompts_are_neutral() -> None:
    for name in ("extract_verdict.md", "extract_turning_points.md"):
        assert "stockfish" not in (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()


def test_the_verdict_extractor_reports_no_verdict_for_an_answer_that_leaves_two_labels_open() -> None:
    prompt = (PROMPTS_DIR / "extract_verdict.md").read_text(encoding="utf-8")
    assert "stronger one" not in prompt
    assert "leaves the move between two labels without settling on one" in prompt and "answer none_stated" in prompt


def test_extract_stated_verdict_keeps_the_quote_and_needs_it_in_the_answer() -> None:
    answer = "Nxd5 was a mistake because of Nxf7."
    stated = extract_stated_verdict(answer_text=answer, extractor=FakeJson(payload=json.dumps({"verdict": "mistake", "quote": "Nxd5 was a mistake"})))
    assert stated == StatedVerdict(verdict="mistake", quote="Nxd5 was a mistake", error=None)
    missing = extract_stated_verdict(answer_text=answer, extractor=FakeJson(payload=json.dumps({"verdict": "mistake", "quote": "it was a blunder"})))
    assert missing.verdict == "none_stated" and missing.quote == "it was a blunder" and "not found in the answer" in (missing.error or "")
    no_quote = extract_stated_verdict(answer_text=answer, extractor=FakeJson(payload=json.dumps({"verdict": "mistake", "quote": None})))
    assert no_quote.verdict == "none_stated" and "without a quote" in (no_quote.error or "")
    none_stated = extract_stated_verdict(answer_text=answer, extractor=FakeJson(payload=json.dumps({"verdict": "none_stated", "quote": None})))
    assert none_stated == StatedVerdict(verdict="none_stated", quote=None, error=None)


def test_an_annotation_glyph_right_after_the_move_gives_its_label() -> None:
    from bench.grading.extract import annotated_label

    text = "19. Bxe7?? lost the knight. Then 23...Qxa2? grabbed a pawn, and 25…f6?! weakened g6. Later 30.Rd1!? was fine, and 31. Kh1! held."
    assert annotated_label(answer_text=text, ply=37, san="Bxe7") == "blunder"
    assert annotated_label(answer_text=text, ply=46, san="Qxa2") == "mistake"
    assert annotated_label(answer_text=text, ply=50, san="f6") == "inaccuracy"
    assert annotated_label(answer_text=text, ply=59, san="Rd1") is None
    assert annotated_label(answer_text=text, ply=61, san="Kh1") is None
    assert annotated_label(answer_text=text, ply=38, san="Bxe7") is None
    assert annotated_label(answer_text="23. Qe6+?? hung the queen", ply=45, san="Qe6+") == "blunder"
    assert annotated_label(answer_text="119. Bxe7?? is another game", ply=37, san="Bxe7") is None
    assert annotated_label(answer_text="After 19. Bxe7 the game turned: 19. Bxe7?? allowed Qxh2+.", ply=37, san="Bxe7") == "blunder"


def test_annotations_replace_the_extracted_labels_and_add_one_to_an_unlabelled_moment() -> None:
    from bench.engine import PlyEval
    from bench.grading.extract import annotated_labels

    plies = [PlyEval(ply=p, move_san=san, cp_white_before=0, cp_white_after=0, best_san=None, win_pct_drop=0.0, label="best") for p, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1)]
    text = "2.Nf3?? was not the problem, but 1...e5? and 2...Nc6 were."
    assert annotated_labels(labels={2: "missed_win", 3: "unlabelled", 4: "mistake"}, answer_text=text, plies=plies) == {2: "mistake", 3: "blunder", 4: "mistake"}
