import io

import chess
import chess.pgn
from dataclasses import dataclass

from bench.config import BenchConfig, EvalBands, LabelThresholds
from bench.engine import Line, PlyEval, label_for_drop, win_pct
from bench.models import (
    MAX_BEGINNER_LINE_PLIES,
    POSITION_FAMILIES,
    About,
    Band,
    Color,
    ClaimKind,
    ClaimSource,
    LineOutcome,
    ReferenceClaim,
    Sheet,
    Task,
    TurningPoint,
    Verdict,
    Weight,
    claim_id_for,
)
from bench.questions import move_label
from bench.reference import ERROR_LABELS, Reference, assert_labels_current, is_users_ply, material_for_board, moment_label

LINE_PLIES = 6
ENGINE_SOURCE = ClaimSource(type="engine", name="stockfish_19", answer_ref=None)
LABEL_WORDS = {"inaccuracy": "inaccuracy", "mistake": "mistake", "blunder": "blunder", "missed_win": "missed win"}

BAND_TEXT: dict[str, str] = {
    "equal": "The position is roughly equal",
    "white_slightly_better": "White is slightly better",
    "white_better": "White is clearly better",
    "white_large_edge": "White has a large advantage",
    "white_winning": "White is winning",
    "black_slightly_better": "Black is slightly better",
    "black_better": "Black is clearly better",
    "black_large_edge": "Black has a large advantage",
    "black_winning": "Black is winning",
}


@dataclass(frozen=True)
class EngineFact:
    """A claim the drafter reads off the reference analysis, before it gets its id in the sheet."""

    text: str
    weight: Weight
    kind: ClaimKind
    about: About | None = "task_position"
    move: str | None = None
    line: str | None = None
    band: str | None = None
    outcome: LineOutcome | None = None
    mate_in: int | None = None

    def claim(self, *, task_id: str, number: int) -> ReferenceClaim:
        return ReferenceClaim(
            claim_id=claim_id_for(task_id=task_id, number=number),
            text=self.text,
            kind=self.kind,
            move=self.move,
            band=self.band,
            line=self.line,
            outcome=self.outcome,
            mate_in=self.mate_in,
            side=None,
            about=self.about,
            source=ENGINE_SOURCE,
            labels=[],
            weight=self.weight,
            duplicate_of=None,
        )


def engine_claims(*, task_id: str, facts: list[EngineFact], first: int = 1) -> list[ReferenceClaim]:
    return [fact.claim(task_id=task_id, number=number) for number, fact in enumerate(facts, start=first)]


def band_for_cp(*, cp_white: int, bands: EvalBands) -> Band:
    magnitude = abs(cp_white)
    if magnitude < bands.slightly_better_cp:
        return "equal"
    side = "white" if cp_white > 0 else "black"
    if magnitude < bands.better_cp:
        return f"{side}_slightly_better"  # type: ignore[return-value]
    if magnitude < bands.large_edge_cp:
        return f"{side}_better"  # type: ignore[return-value]
    if magnitude < bands.winning_cp:
        return f"{side}_large_edge"  # type: ignore[return-value]
    return f"{side}_winning"  # type: ignore[return-value]


def outcome_for(*, cp_end_mover: int, mate_mover: int | None, material_delta_mover: int, winning_cp: int, equal_cp: int) -> LineOutcome | None:
    """A line's outcome for the side to move. "wins" and "loses" start at the winning band (winning_cp) and "equalises" ends where
    the equal band does (equal_cp, eval_bands.slightly_better_cp), so a line's words never say more than the evaluation band of the same position."""
    if mate_mover is not None:
        return "mates" if mate_mover > 0 else "gets_mated"
    if material_delta_mover >= 100 and cp_end_mover >= 100:
        return "wins_material"
    if material_delta_mover <= -100 and cp_end_mover <= -100:
        return "loses_material"
    if cp_end_mover >= winning_cp:
        return "wins"
    if cp_end_mover <= -winning_cp:
        return "loses"
    if abs(cp_end_mover) < equal_cp:
        return "equalises"
    return None


def _mover_sign(*, task: Task) -> int:
    return 1 if task.side_to_move == "white" else -1


def plies_before(*, board: chess.Board) -> int:
    """The plies played before a position, from its FEN's move number and side to move."""
    return 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)


def _ply_count(*, task: Task) -> int:
    """Plies before the task position: from the moves so far, or, without them, from the FEN's move number and side to move."""
    if task.pgn is None:
        return plies_before(board=chess.Board(task.fen or ""))
    game = chess.pgn.read_game(io.StringIO(task.pgn))
    return 0 if game is None else len(list(game.mainline_moves()))


def _line_text(*, board: chess.Board, sans: list[str]) -> str:
    moves = []
    probe = board.copy()
    for san in sans:
        move = probe.parse_san(san)
        moves.append(move)
        probe.push(move)
    return board.variation_san(moves)


def _material_delta_mover(*, board: chess.Board, sans: list[str], sign: int) -> int:
    probe = board.copy()
    before = material_for_board(board=probe).balance_cp
    for san in sans:
        probe.push_san(san)
    return sign * (material_for_board(board=probe).balance_cp - before)


def verdict_from_reference(*, task: Task, reference: Reference, thresholds: LabelThresholds) -> Verdict:
    if reference.position is None or reference.after_move is None or task.move is None:
        raise ValueError(f"{task.task_id} needs a position reference with the analysis after {task.move}")
    sign = _mover_sign(task=task)
    top = reference.position.lines[0]
    after = reference.after_move.lines[0] if reference.after_move.lines else None
    board = chess.Board(task.fen or "")
    board.push_san(task.move)
    cp_after = after.cp_white if after is not None else (-1000 if board.turn == chess.WHITE else 1000) if board.is_checkmate() else 0
    drop = max(0.0, win_pct(cp=sign * top.cp_white) - win_pct(cp=sign * cp_after))
    return label_for_drop(drop=drop, is_top_move=top.move_san == task.move, thresholds=thresholds)


def _line_plies(*, task: Task) -> int:
    return MAX_BEGINNER_LINE_PLIES if task.audience == "beginner" else LINE_PLIES


def _band_for(*, cp_white: int, bands: EvalBands) -> Band:
    return band_for_cp(cp_white=cp_white, bands=bands)


def _after_text(*, label: str, text: str) -> str:
    tail = "t" + text[1:] if text.startswith("The ") else text
    return f"After {label}, {tail}"


def _band_point(*, task: Task, reference: Reference, bands: EvalBands, ply_before: int) -> EngineFact | None:
    if task.move is None:
        analysis, about = reference.position, "task_position"
    else:
        analysis, about = reference.after_move, "after_move"
    if analysis is None:
        raise ValueError(f"{task.task_id} reference lacks the analysis for the {about} evaluation")
    if not analysis.lines:
        return None
    band = _band_for(cp_white=analysis.lines[0].cp_white, bands=bands)
    text = BAND_TEXT[band] if about == "task_position" else _after_text(label=move_label(ply_before=ply_before, san=task.move or ""), text=BAND_TEXT[band])
    return EngineFact(text=text, weight=2, kind="eval_band", about=about, band=band)  # type: ignore[arg-type]


def reference_line_outcome(*, task: Task, board: chess.Board, sans: list[str], cp_end_white: int, mate_white: int | None, bands: EvalBands) -> LineOutcome | None:
    """The outcome the drafter gives a reference line played from `board`, read for the task's side to move."""
    sign = _mover_sign(task=task)
    return outcome_for(
        cp_end_mover=sign * cp_end_white,
        mate_mover=None if mate_white is None else sign * mate_white,
        material_delta_mover=_material_delta_mover(board=board, sans=sans, sign=sign),
        winning_cp=bands.winning_cp,
        equal_cp=bands.slightly_better_cp,
    )


def _line_outcome_point(*, task: Task, board: chess.Board, sans: list[str], cp_end_white: int, mate_white: int | None, weight: Weight, lead: str, bands: EvalBands) -> EngineFact | None:
    outcome = reference_line_outcome(task=task, board=board, sans=sans, cp_end_white=cp_end_white, mate_white=mate_white, bands=bands)
    if outcome is None:
        return None
    wording = {
        "wins_material": "wins material",
        "loses_material": "loses material",
        "mates": "leads to mate",
        "gets_mated": "gets mated",
        "equalises": "keeps the balance",
        "wins": "leads to a winning position",
        "loses": "leads to a lost position",
    }[outcome]
    return EngineFact(text=f"{lead}{_line_text(board=board, sans=sans)} {wording}", weight=weight, kind="line_outcome", line=" ".join(sans), outcome=outcome)


def _position_points(*, task: Task, reference: Reference, bands: EvalBands, verdict: Verdict | None) -> list[EngineFact]:
    if reference.position is None or not reference.position.lines:
        raise ValueError(f"{task.task_id} has no position lines in its reference")
    board = chess.Board(task.fen or "")
    top = reference.position.lines[0]
    ply_before = _ply_count(task=task)
    points: list[EngineFact] = []
    if task.family == "move_judgement":
        if verdict is None:
            raise ValueError(f"{task.task_id} is a move_judgement task, so its points need the verdict")
        points.extend(_move_judgement_points(task=task, reference=reference, board=board, top=top, ply_before=ply_before, verdict=verdict, bands=bands))
    elif task.family == "move_choice":
        points.append(EngineFact(text=f"{move_label(ply_before=ply_before, san=top.move_san)} is the strongest move", weight=3, kind="best_move", move=top.move_san))
    band_point = _band_point(task=task, reference=reference, bands=bands, ply_before=ply_before)
    if band_point is not None:
        points.append(band_point)
    if task.family in POSITION_FAMILIES:
        main = _line_outcome_point(task=task, board=board, sans=top.pv_san[: _line_plies(task=task)], cp_end_white=top.cp_white, mate_white=top.mate, weight=2, lead="The main line ", bands=bands)
        if main is not None and main.outcome in ("wins_material", "mates"):
            points.append(main)
    balance = reference.material.balance_cp
    if abs(balance) >= 100:
        side = "White" if balance > 0 else "Black"
        points.append(EngineFact(text=f"{side} is ahead in material", weight=1, kind="material", band="white_up" if balance > 0 else "black_up"))
    return points


def _move_judgement_points(*, task: Task, reference: Reference, board: chess.Board, top: Line, ply_before: int, verdict: Verdict, bands: EvalBands) -> list[EngineFact]:
    """The judged move is the best move when the verdict says so (it may be within the tolerance of another first choice), or else
    the engine's move was better; and the line after the judged move, when its outcome is clear."""
    if task.move is None or reference.after_move is None:
        raise ValueError(f"{task.task_id} needs a move and the analysis after it")
    played_label = move_label(ply_before=ply_before, san=task.move)
    reply = reference.after_move.lines[0] if reference.after_move.lines else None
    line = None
    if reply is not None:
        sans = [task.move] + reply.pv_san[: _line_plies(task=task) - 1]
        line = _line_outcome_point(task=task, board=board, sans=sans, cp_end_white=reply.cp_white, mate_white=reply.mate, weight=3, lead="", bands=bands)
    lines = [] if line is None else [line]
    if verdict == "best":
        return [EngineFact(text=f"{played_label} is the strongest move", weight=3, kind="best_move", move=task.move), *lines]
    better_weight: Weight = 2 if line is not None else 3
    return [*lines, EngineFact(text=f"{move_label(ply_before=ply_before, san=top.move_san)} was the better move", weight=better_weight, kind="best_move", move=top.move_san)]


def choose_turning_points(*, reference: Reference, user_color: Color, cfg: BenchConfig) -> list[TurningPoint]:
    if reference.game is None:
        raise ValueError(f"{reference.task_id} has no game analysis in its reference")
    plies, rules = reference.game.plies, cfg.turning_points
    labelled = [(p, moment_label(plies=plies, index=i, thresholds=cfg.labels, winning_cp=cfg.eval_bands.winning_cp)) for i, p in enumerate(plies) if is_users_ply(ply=p.ply, user_color=user_color)]
    chosen = [(p, label) for p, label in labelled if label in ERROR_LABELS and (p.win_pct_drop >= rules.sheet_min_drop or (label == "missed_win" and p.win_pct_drop >= rules.missed_win_min_drop))]
    chosen.sort(key=lambda pair: pair[0].win_pct_drop, reverse=True)
    return [TurningPoint(ply=p.ply, move=p.move_san, label=label) for p, label in sorted(chosen[: rules.sheet_max_points], key=lambda pair: pair[0].ply)]  # type: ignore[arg-type]


def turning_point_text(*, ply: int, move: str, label: str) -> str:
    word = LABEL_WORDS[label]
    article = "an" if word[0] in "aeiou" else "a"
    return f"{move_label(ply_before=ply - 1, san=move)} was {article} {word}"


def turning_point_fact(*, ply: PlyEval, label: str) -> EngineFact:
    weight: Weight = 3 if label == "blunder" else 2
    return EngineFact(text=turning_point_text(ply=ply.ply, move=ply.move_san, label=label), weight=weight, kind="other", about=None, move=ply.move_san)


def ply_for(*, reference: Reference, ply: int) -> PlyEval:
    if reference.game is None:
        raise ValueError(f"{reference.task_id} has no game analysis in its reference")
    found = [p for p in reference.game.plies if p.ply == ply]
    if len(found) != 1:
        raise ValueError(f"{reference.task_id} has no ply {ply} in its game analysis")
    return found[0]


def review_user_color(*, task: Task) -> Color:
    if task.user_color is None:
        raise ValueError(f"{task.task_id} is a game review without the user's colour")
    return task.user_color


def _game_points(*, task: Task, reference: Reference, cfg: BenchConfig) -> tuple[list[TurningPoint], list[EngineFact]]:
    turning_points = choose_turning_points(reference=reference, user_color=review_user_color(task=task), cfg=cfg)
    return turning_points, [turning_point_fact(ply=ply_for(reference=reference, ply=tp.ply), label=tp.label) for tp in turning_points]


def draft_sheet(*, task: Task, reference: Reference, cfg: BenchConfig) -> Sheet:
    """A draft sheet holds the engine's claims, each weighted and true by default; plan-level claims come later, from the
    grandmaster's review of claims pooled from model answers and from the grandmaster's own."""
    verdict: Verdict | None = None
    turning_points: list[TurningPoint] = []
    if task.family == "game_review":
        assert_labels_current(reference=reference, labelling=cfg.reference_labelling())
        turning_points, facts = _game_points(task=task, reference=reference, cfg=cfg)
    else:
        if task.family == "move_judgement":
            verdict = verdict_from_reference(task=task, reference=reference, thresholds=cfg.labels)
        facts = _position_points(task=task, reference=reference, bands=cfg.eval_bands, verdict=verdict)
    return Sheet(
        task_id=task.task_id,
        verdict=verdict,
        turning_points=turning_points,
        claims=engine_claims(task_id=task.task_id, facts=facts),
        last_claim_number=len(facts),
        notes="",
        author=None,
        reviewed_by=None,
        signed_at=None,
    )
