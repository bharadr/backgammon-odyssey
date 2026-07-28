import math
import random

from engine.board import Board
from coach.analysis import AfterstateEvaluator
from coach.scoring import cubeless_equity


# Cubeless equity is in [-3, 3], so the largest possible gap between two moves
# is 6; an epsilon >= 6 therefore keeps every move (i.e. no blunder floor).
NO_FLOOR = 6.0


class SkillAgent:
    """A tunable-strength Agent. It scores the position's *own* legal
    afterstates with an AfterstateEvaluator (gnubg, a trained net, ...) and
    softmax-samples among them by equity.

    Scoring *our* afterstates -- rather than trusting the backend's move
    generation -- guarantees it returns a legal afterstate the rest of the
    engine (e.g. describe_move) can reconstruct.

    - temperature (tau): 0 -> always the best move (superhuman); larger ->
      weaker/more varied. ~0.02-0.10 is the interesting range.
    - epsilon: a blunder floor. Moves worse than the best by more than epsilon
      are dropped before sampling. Defaults to NO_FLOOR (6.0 >= max gap) -> off.
    - rng: injected random.Random for reproducibility/testability.
    """

    def __init__(self, evaluator: AfterstateEvaluator, temperature: float,
                 rng: random.Random, epsilon: float = NO_FLOOR):
        self._evaluator = evaluator
        self._temperature = temperature
        self._rng = rng
        self._epsilon = epsilon

    def __call__(self, board: Board, dice: tuple[int, int],
                 afterstates: set[Board]) -> Board:
        """Pick one of the given `afterstates` to play.

        Score every afterstate by equity (via the injected evaluator), then:
        at temperature 0, return the highest-equity one; otherwise drop those
        more than `epsilon` below the best (the blunder floor) and softmax-
        sample among the rest, so stronger moves are chosen more often. Always
        returns one of the `afterstates` it was given (never a fresh board).
        `board`/`dice` are unused -- the afterstates already encode the play.

        Precondition: `afterstates` is non-empty -- play_turn forfeits the
        dance before any agent is consulted, so we never score an empty set.
        """
        # sorted() gives a deterministic order so a seeded rng is reproducible
        scored = [(a, cubeless_equity(self._evaluator.evaluate_afterstate(a)))
                  for a in sorted(afterstates)]
        best_equity = max(equity for _, equity in scored)

        if self._temperature <= 0:
            return max(scored, key=lambda pair: pair[1])[0]   # return the board with max equity

        candidates = [(a, e) for a, e in scored
                      if best_equity - e <= self._epsilon]
        weights = _softmax_weights([e for _, e in candidates], self._temperature)
        return self._rng.choices([a for a, _ in candidates], weights=weights)[0]


def _softmax_weights(equities: list[float], temperature: float) -> list[float]:
    """Unnormalised softmax weights, exp((e - max) / tau). Subtracting the max
    keeps exp from overflowing at small temperatures (every exponent <= 0);
    random.choices normalises internally, so we leave them unnormalised."""
    hottest = max(equities)
    return [math.exp((e - hottest) / temperature) for e in equities]
