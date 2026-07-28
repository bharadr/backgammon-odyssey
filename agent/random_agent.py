import random

from engine.board import Board
from engine.game import Agent


def random_agent(rng: random.Random) -> Agent:
    """An agent that picks uniformly at random among the legal afterstates.

    The rng is injected (not the global `random` module) so play is
    reproducible when it's seeded. sorted() fixes the candidate order before
    the draw, so the choice depends only on the rng, not on set iteration
    order.

    This is the baseline opponent: any smarter agent (value-net, gnubg-backed)
    plugs into the same `Agent` seam by scoring the afterstates instead of
    choosing blindly.
    """
    def choose(board: Board, dice: tuple[int, int], afterstates: set[Board]) -> Board:
        return rng.choice(sorted(afterstates))  # ignores board and dice
    return choose
