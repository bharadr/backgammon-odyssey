from dataclasses import dataclass
from typing import Optional, Protocol

from engine.board import Board


@dataclass(frozen=True)
class OutcomeDist:
    """Pure outcome-probability data, oriented to the player being analysed
    (the side to move), NOT the opponent. Fields use the cumulative
    convention:

        win >= win_gammon >= win_backgammon

    `win` is P(win at all); `win_gammon` is P(win by a gammon or more);
    `win_backgammon` is P(win by a backgammon); the lose_* fields mirror that
    for losses. This is a data-representation convention only — providers
    normalise their engine's output into it. It deliberately does NOT know how
    to price itself into an equity: valuation is a separate policy (see
    coach/scoring.py), because cubeless money, match equity, and cube-aware
    equity all score the same distribution differently.
    """
    win: float
    win_gammon: float
    win_backgammon: float
    lose_gammon: float
    lose_backgammon: float


@dataclass(frozen=True)
class MoveAnalysis:
    after_state: Board                 # resulting position, in the mover's perspective
    outcome: OutcomeDist              # oriented to the player who made the move
    equity: float                     # scalar valuation supplied by the provider
                                      #   (higher = better for the mover)
    notation: Optional[str] = None    # human-readable, e.g. "8/5 6/5" (best-effort)


@dataclass(frozen=True)
class Analysis:
    position: Board
    dice: tuple[int, int]
    moves: tuple[MoveAnalysis, ...]   # ranked best-first (descending equity)

    @property
    def best(self) -> MoveAnalysis:
        return self.moves[0]

    def equity_loss(self, move: MoveAnalysis) -> float:
        """How much equity `move` gives up versus the best play (>= 0)."""
        return self.best.equity - move.equity


class AnalysisProvider(Protocol):
    """Pluggable source of analysis. The coach depends only on this interface,
    never on a specific backend (gnubg-nn, a trained net, ...)."""

    def analyze(self, position: Board, dice: tuple[int, int]) -> Analysis: ...


class AfterstateEvaluator(Protocol):
    """Scores a single afterstate -- the position after the mover played, so
    the opponent is on roll -- from the *mover's* perspective.

    Separate from AnalysisProvider so an agent can score a position's *own*
    legal afterstates directly, without the backend re-generating moves."""

    def evaluate_afterstate(self, board: Board) -> OutcomeDist: ...
