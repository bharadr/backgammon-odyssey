"""Scoring policies: how to price an OutcomeDist into a scalar equity.

Kept separate from the analysis data types on purpose. A provider that
already supplies an equity (gnubg does) never needs this. A provider that
only produces a distribution can opt into a policy here. Match-equity and
cube-aware valuations would live alongside `cubeless_equity` as their own
functions, without touching OutcomeDist.
"""
from coach.analysis import OutcomeDist


def cubeless_equity(d: OutcomeDist) -> float:
    """Cubeless money equity in [-3, 3] for a cumulative OutcomeDist:
    2*win - 1 + win_gammon + win_backgammon - lose_gammon - lose_backgammon.
    """
    return (2.0 * d.win - 1.0
            + d.win_gammon + d.win_backgammon
            - d.lose_gammon - d.lose_backgammon)
