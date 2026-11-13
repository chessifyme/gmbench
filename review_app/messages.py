"""The checks' messages as a grandmaster reads them: on the claim's card, by the card's number, in chess words. The checks
themselves (bench.validate, bench.models) keep their own wording for the logs and the workbook."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from bench.models import CLAIM_ID, ReferenceClaim

NEW_ROW = re.compile(r"new claim row (\d+)")
CLAIM_REF = re.compile(rf"(?:claim )?({CLAIM_ID.pattern})")
VERDICT_WORDS = {"best": "the best move", "good": "a good move", "inaccuracy": "an inaccuracy", "mistake": "a mistake", "blunder": "a blunder"}
ID = CLAIM_ID.pattern


def _verdict(value: str) -> str:
    return VERDICT_WORDS.get(value, value)


def _band(value: str) -> str:
    return value.replace("_", " ").capitalize()


ABOUT_WORDS = {"task_position": "the task position", "after_move": "the position after the move"}
OUTCOME_WORDS = {"wins_material": "wins material", "loses_material": "loses material", "mates": "mates", "gets_mated": "gets mated", "equalises": "equalises", "wins": "wins", "loses": "loses"}
MATERIAL_WORDS = {"level": "material is level", "white_up": "White is up material", "black_up": "Black is up material"}


def tag_words(*, claim: ReferenceClaim) -> str:
    """What a claim's kind and structured fields make the checks read it as, in chess words ("" for a claim without a checkable
    kind): the tag a reviewer can clear when the words say something else."""
    where = ABOUT_WORDS.get(claim.about or "", "the position")
    if claim.kind == "eval_band":
        return f"an evaluation of {where} ({_band(claim.band or '')})"
    if claim.kind == "material":
        return f"a material count of {where} ({MATERIAL_WORDS.get(claim.band or '', claim.band or '')})"
    if claim.kind == "best_move":
        return f"naming {claim.move} the best move in {where}"
    if claim.kind == "threat":
        return f"naming the threat {claim.move} in {where}"
    if claim.kind == "mate":
        mover = "White" if (claim.mate_in or 0) > 0 else "Black"
        return f"{mover} mating in {abs(claim.mate_in or 0)} from {where}"
    if claim.kind == "line_outcome":
        return f"the line {claim.line} from {where}, in which {(claim.side or 'the mover').capitalize()} {OUTCOME_WORDS.get(claim.outcome or '', claim.outcome or '')}"
    return ""


REWORDINGS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(rf"claim ({ID}) (?:names (\S+) as better than|says (\S+) is better than) (\S+) but the sheet verdict is best"),
        lambda m: f"This claim says {m[2] or m[3]} is better than {m[4]}, but your verdict says {m[4]} is the best move.",
    ),
    (re.compile(rf"claim ({ID}) calls (\S+) the best move but the sheet verdict is (\w+)"), lambda m: f"This claim says {m[2]} is best, but your verdict says it's {_verdict(m[3])}."),
    (re.compile(rf"claim ({ID}) (?:has weight \d but its annotators disagree|has reviewers who disagree, so it needs adjudication first)"), lambda m: "The reviewers disagree on this claim, so it can't carry a weight until an adjudicator settles it."),
    (
        re.compile(rf"claim ({ID}) (?:has weight \d but its latest label is not true|is not labelled True by every reviewer; only a True|is not labelled yet; only a True)"),
        lambda m: "This claim is in the answer key but isn't labelled True: label it True or set it to Not key.",
    ),
    (re.compile(rf"claim ({ID}) needs adjudication"), lambda m: "The reviewers disagree on this claim; an adjudicator has to settle it before it can carry a weight."),
    (re.compile(rf"claim ({ID}) is compound: (.*)"), lambda m: f"This claim says more than one thing ({m[2].rstrip('.')}): split it into separate claims."),
    (re.compile(rf"claim ({ID}) move '([^']+)' is not legal"), lambda m: f"This claim is tagged with the move {m[2]}, which is not legal in the position. If the claim does not name that move, clear the tag."),
    (re.compile(rf"claim ({ID}) line has illegal move '([^']+)'"), lambda m: f"This claim is tagged with a line that has an illegal move, {m[2]}. If the claim does not give that line, clear the tag."),
    (
        re.compile(rf"claim ({ID}) credits (\w+) \((\w+)\) but the reference has ([+-]\d+) cp for White"),
        lambda m: f"This claim is tagged as an evaluation of {ABOUT_WORDS.get(m[3], 'the position')} ({_band(m[2])}), but the engine gives {int(m[4]) / 100:+.2f} for White. If the claim does not say that, clear the tag.",
    ),
    (re.compile(rf"claim ({ID})(?:: a new wording needs a label| has a new wording but no label)"), lambda m: "You reworded this claim: pick a label for the new wording."),
    (re.compile(rf"claim ({ID})(?:: a comment goes on your label| has a comment but no label)"), lambda m: "A comment goes with a label: pick a label for this claim."),
    (re.compile(rf"claim ({ID}): its text cannot be blanked"), lambda m: "This claim's text can't be blank; if the claim is wrong, label it False."),
    (re.compile(rf"claim ({ID}) calls (\S+) (an? [\w ]+?) but the sheet's turning point labels it (an? [\w ]+)"), lambda m: f"This claim calls {m[2]} {m[3]}, but your turning point calls it {m[4]}."),
    (re.compile(r"question .+ presupposes '([^']+)' but the sheet verdict is (\w+)"), lambda m: f"The question takes “{m[1]}” for granted, but your verdict says it's {_verdict(m[2])}: ask it neutrally."),
    (re.compile(r"\w+ sheet needs a verdict"), lambda m: "Pick a verdict."),
    (re.compile(r"the answer key is empty"), lambda m: "No claim is in the answer key yet: mark at least one True claim Essential, Important or Nice to have."),
    (
        re.compile(r"a reviewed game_review answer key needs a weighted claim besides the turning-point claims"),
        lambda m: "The key needs a claim besides the turning points: mark a True claim that explains a turning point Essential, Important or Nice to have.",
    ),
)


PLAIN_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsheet [a-z]{2}_\d{4}: "), ""),
    (re.compile(r"\bthe sheet verdict\b"), "your verdict"),
    (re.compile(r"\bthe sheet's\b"), "your"),
    (re.compile(r"\b(?:on|in) the sheet\b"), "in this task"),
    (re.compile(r"\bsheets\b"), "tasks"),
    (re.compile(r"\bsheet\b"), "task"),
)


def plain_words(*, text: str) -> str:
    """A check's message without the data model's word "sheet": a grandmaster reviews a task, its claims and the verdict."""
    for pattern, words in PLAIN_WORDS:
        text = pattern.sub(words, text)
    return text


@dataclass
class Notes:
    """The messages for the top of the page, and the same ones per card (by claim id, or "new1", "new2" for new claims)."""

    summary: list[str] = field(default_factory=list)
    cards: dict[str, list[str]] = field(default_factory=dict)


def card_notes(*, messages: list[str], numbers: dict[str, int]) -> Notes:
    """Each message reworded where a common one is recognised, its claim ids replaced by the card numbers the reviewer sees,
    and filed under the first card it names. numbers maps a claim id, or "new<k>" for the k-th new claim, to its card number."""
    notes = Notes()
    for message in messages:
        key, text = None, message
        for pattern, say in REWORDINGS:
            found = pattern.search(message)
            if found is not None:
                key, text = (found[1] if pattern.groups and CLAIM_ID.fullmatch(found[1] or "") else None), say(found)
                break
        if key is None:
            named = CLAIM_ID.search(message)
            new = NEW_ROW.search(message)
            key = named[0] if named else (f"new{new[1]}" if new else None)
        text = plain_words(text=text)
        text = CLAIM_REF.sub(lambda m: f"claim {numbers[m[1]]}" if m[1] in numbers else m[0], text)
        text = NEW_ROW.sub(lambda m: f"claim {numbers[f'new{m[1]}']}" if f"new{m[1]}" in numbers else m[0], text)
        if key is not None and key in numbers:
            notes.cards.setdefault(key, []).append(text)
            notes.summary.append(f"Claim {numbers[key]}: {text}" if text.startswith(("This claim", "You reworded", "A comment", "The reviewers")) else text)
        else:
            notes.summary.append(text)
    return notes
