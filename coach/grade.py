"""Classify a chosen checker play by how much equity it gave up.

Pure and separate from the LLM: a `Verdict` is the instant, engine-only feedback
shown for every move (the quiz's and the in-game coach's "did I get it right?"
beat). Thresholds follow the usual gnubg/XG cubeless bands.
"""
from dataclasses import dataclass

from coach.evidence import Evidence

DOUBTFUL = 0.020
ERROR = 0.040
BLUNDER = 0.080


@dataclass(frozen=True)
class Verdict:
    label: str            # Best play / Good / Doubtful / Error / Blunder
    symbol: str
    rank: int
    of_n: int
    equity_loss: float
    line: str             # ready-to-print one-liner


def _classify(loss: float) -> tuple[str, str]:
    if loss <= 0:
        return "Best play", "✓"       # ✓
    if loss < DOUBTFUL:
        return "Good", "·"            # ·
    if loss < ERROR:
        return "Doubtful", "?"
    if loss < BLUNDER:
        return "Error", "!"
    return "Blunder", "✗"             # ✗


def grade(evidence: Evidence) -> Verdict:
    """Grade the chosen play. `evidence` must not be a dance (no play to grade)."""
    chosen = evidence.chosen
    loss = chosen.equity_loss
    label, symbol = _classify(loss)
    if loss <= 0:
        line = f"{symbol} {label}!  (best of {chosen.of_n})"
    else:
        line = (f"{symbol} {label} -- your play ranked {chosen.rank} of "
                f"{chosen.of_n}, equity lost {loss:.3f}")
    return Verdict(label=label, symbol=symbol, rank=chosen.rank,
                   of_n=chosen.of_n, equity_loss=loss, line=line)
