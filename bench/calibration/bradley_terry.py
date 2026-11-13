import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ELO_SCALE = 400.0
PRIOR_WEIGHT = 0.2
TOLERANCE_ELO = 1e-6
MAX_ITERATIONS = 100
MAX_HALVINGS = 40
CI_LOW_PERCENT = 2.5
CI_HIGH_PERCENT = 97.5
LOGIT_TO_ELO = ELO_SCALE / math.log(10.0)


@dataclass(frozen=True, kw_only=True)
class Outcome:
    """One comparison of entrant a with entrant b: a's share of it (1 a win, 0.5 a tie, 0 a loss), its weight and the task it comes from."""

    a: str
    b: str
    share_a: float
    weight: float
    cluster: str

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError(f"an outcome compares {self.a} with itself")
        if not 0.0 <= self.share_a <= 1.0:
            raise ValueError(f"the share of {self.a} over {self.b} is {self.share_a}; a share lies between 0 and 1")
        if not self.weight > 0.0:
            raise ValueError(f"the outcome of {self.a} over {self.b} has weight {self.weight}; a weight must be positive")


def win_matrix(*, names: Sequence[str], outcomes: Sequence[Outcome]) -> np.ndarray:
    """wins[i, j] is the weighted share entrant i took from entrant j."""
    index = {name: i for i, name in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    for outcome in outcomes:
        for name in (outcome.a, outcome.b):
            if name not in index:
                raise ValueError(f"{name} is not among the rated entrants ({', '.join(names)})")
        a, b = index[outcome.a], index[outcome.b]
        wins[a, b] += outcome.weight * outcome.share_a
        wins[b, a] += outcome.weight * (1.0 - outcome.share_a)
    return wins


def _log_likelihood(*, theta: np.ndarray, wins: np.ndarray) -> np.ndarray:
    gaps = theta[:, :, None] - theta[:, None, :]
    return -(wins * np.logaddexp(0.0, -gaps)).sum(axis=(1, 2))


def _assert_connected(*, names: Sequence[str], total: np.ndarray) -> None:
    reached = {0}
    frontier = [0]
    while frontier:
        i = frontier.pop()
        for j in np.flatnonzero(total[i] > 0):
            if int(j) not in reached:
                reached.add(int(j))
                frontier.append(int(j))
    if len(reached) < len(names):
        apart = sorted(names[i] for i in range(len(names)) if i not in reached)
        raise ValueError(f"no chain of outcomes links {', '.join(apart)} with {names[0]}; their ratings are not defined")


def fit_win_matrices(*, names: Sequence[str], wins: np.ndarray, anchor: float, prior_weight: float = PRIOR_WEIGHT, context: str = "") -> np.ndarray:
    """Maximum-likelihood Bradley-Terry ratings on the Elo scale for a stack of win matrices (draws x n x n); every row has mean `anchor`.

    A virtual tie of `prior_weight` between every pair of entrants keeps the fit finite when one entrant takes every outcome.
    A fit that has not settled to within TOLERANCE_ELO after MAX_ITERATIONS damped Newton steps is refused, never reported.
    """
    n = len(names)
    if wins.ndim != 3 or wins.shape[1:] != (n, n):
        raise ValueError(f"expected win matrices of shape (draws, {n}, {n}), got {wins.shape}")
    if n < 2:
        raise ValueError(f"a Bradley-Terry fit needs at least two entrants, got {', '.join(names) or 'none'}")
    if prior_weight < 0.0:
        raise ValueError(f"the prior weight is {prior_weight}; it must be zero or positive")
    off_diagonal = 1.0 - np.eye(n)
    observed = wins + wins.transpose(0, 2, 1)
    for draw, total in enumerate(observed):
        silent = [names[i] for i in range(n) if total[i].sum() <= 0.0]
        if silent:
            raise ValueError(f"{', '.join(silent)} has no outcome{context.format(draw=draw)}; every rated entrant needs at least one")
    shaped = wins * off_diagonal + prior_weight / 2.0 * off_diagonal
    totals = shaped + shaped.transpose(0, 2, 1)
    if prior_weight == 0.0:
        for total in totals:
            _assert_connected(names=names, total=total)
    theta = np.zeros((wins.shape[0], n))
    likelihood = _log_likelihood(theta=theta, wins=shaped)
    for _ in range(MAX_ITERATIONS):
        gaps = theta[:, :, None] - theta[:, None, :]
        win_chance = np.exp(-np.logaddexp(0.0, -gaps))
        loss_chance = np.exp(-np.logaddexp(0.0, gaps))
        gradient = (shaped * loss_chance - shaped.transpose(0, 2, 1) * win_chance).sum(axis=2)
        spread = totals * win_chance * loss_chance
        information = -spread
        information[:, range(n), range(n)] = spread.sum(axis=2)
        step = np.zeros_like(theta)
        step[:, 1:] = np.linalg.solve(information[:, 1:, 1:], gradient[:, 1:, None])[:, :, 0]
        size = np.ones(theta.shape[0])
        for _ in range(MAX_HALVINGS):
            trial = theta + size[:, None] * step
            trial_likelihood = _log_likelihood(theta=trial, wins=shaped)
            worse = trial_likelihood < likelihood - 1e-12 * (1.0 + np.abs(likelihood))
            if not worse.any():
                break
            size[worse] /= 2.0
        theta, likelihood = trial, trial_likelihood
        moved = np.abs(size[:, None] * step).max(axis=1) * LOGIT_TO_ELO
        if moved.max() < TOLERANCE_ELO:
            ratings = theta * LOGIT_TO_ELO
            return ratings - ratings.mean(axis=1, keepdims=True) + anchor
    unsettled = int((moved >= TOLERANCE_ELO).sum())
    raise ValueError(
        f"the Bradley-Terry fit of {', '.join(names)} did not converge after {MAX_ITERATIONS} iterations in {unsettled} of {len(moved)} fit(s) "
        f"(the last step still moved a rating by {moved.max():.3g} Elo-scale points); an entrant that takes every outcome needs the prior"
    )


def fit(*, names: Sequence[str], outcomes: Sequence[Outcome], anchor: float, prior_weight: float = PRIOR_WEIGHT) -> dict[str, float]:
    ratings = fit_win_matrices(names=names, wins=win_matrix(names=names, outcomes=outcomes)[None], anchor=anchor, prior_weight=prior_weight)[0]
    return {name: float(rating) for name, rating in zip(names, ratings, strict=True)}
