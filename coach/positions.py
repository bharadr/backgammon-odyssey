"""A small curated set of instructive mid-game positions for the demo.

Each is a legal position (15 checkers a side) in the mover's perspective
(positive = the student on roll). They span the main strategic themes so that
whatever the random roll turns up, there is usually a real decision to coach.
"""
from typing import NamedTuple

from engine.board import Board


class CuratedPosition(NamedTuple):
    name: str
    theme: str
    board: Board


POSITIONS: tuple[CuratedPosition, ...] = (
    CuratedPosition(
        name="Priming battle",
        theme="Both sides are building primes with checkers trapped behind them.",
        board=Board(
            points=(-2, 0, 0, 1, 2, 2, 2, 2, 0, 0, -1, -3,
                    3, 1, 0, 0, -2, -2, -2, -2, -1, 0, 0, 2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Blitz",
        theme="Opponent is on the bar against a strong, growing home board.",
        board=Board(
            points=(-2, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, -4,
                    3, -2, 0, 0, -3, 0, -3, 0, 0, 0, 0, 2),
            bar_count=0, opp_bar_count=1, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Holding game",
        theme="You hold the 20-point anchor while narrowly ahead in the race.",
        board=Board(
            points=(0, 0, 1, 0, 2, 3, 2, 2, 0, 0, -2, -3,
                    3, -1, -2, -2, -2, 0, -3, 2, 0, 0, 0, 0),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Bear-off race",
        theme="Near-pure race: efficient bear-in and bear-off matter most.",
        board=Board(
            points=(2, 3, 3, 2, 2, 2, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, -1, 0, -3, -2, -3, -2, -2, -2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Back game",
        theme="You hold deep 23/24 anchors and are timing a late shot.",
        board=Board(
            points=(0, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, -3,
                    3, -2, -2, 0, -2, -2, -2, -2, 0, 0, 2, 2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Bear-in under fire",
        theme="You're bringing checkers home, but an opponent anchor on your "
              "3-point threatens shots as you clear the outfield.",
        board=Board(
            points=(2, 2, -2, 2, 3, 3, 0, 2, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, -3, -3, -3, -2, 0, -2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Hit or point",
        theme="The opponent left a blot on your bar point -- hit loose and risk "
              "the return shot, or make a point and play safe?",
        board=Board(
            points=(-2, 0, 0, 1, 1, 2, -1, 2, 0, 0, 2, -3,
                    3, -2, 0, 2, 0, 0, -3, -2, -2, 0, 0, 2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Mutual holding game",
        theme="Both sides hold an advanced anchor in the other's home; it's a "
              "timing battle over who breaks first.",
        board=Board(
            points=(0, 0, 2, 2, -2, 3, 0, 3, 0, 0, 0, -3,
                    3, -2, -2, 0, -3, 0, -3, 2, 0, 0, 0, 0),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Running game",
        theme="You lead the race with two back checkers free to run -- break "
              "contact and race, or hold for safety?",
        board=Board(
            points=(0, 0, 2, 2, 2, 3, 2, 2, 0, -2, 0, -3,
                    0, -2, -2, 0, -3, 0, -3, 0, 0, 0, 0, 2),
            bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
    CuratedPosition(
        name="Blitz defense",
        theme="You're on the bar against a strong four-point board; enter and "
              "anchor before you get shut out.",
        board=Board(
            points=(0, 0, 2, 0, 2, 3, 0, 3, 0, 0, 0, -3,
                    3, -2, 0, 0, -2, 0, -2, -2, -2, -2, 0, 1),
            bar_count=1, opp_bar_count=0, off_count=0, opp_off_count=0,
        ),
    ),
)
