import random
from enum import IntEnum
from typing import Callable, NamedTuple

from .board import Board, flip, starting_board
from .moves import generate_moves


# An agent decides a turn: given the board (from its own perspective) and the
# set of legal afterstates, it returns the one it wants to play. This is the
# seam that keeps agents (random, TD, ...) swappable behind one interface.
Agent = Callable[[Board, set[Board]], Board]


class Outcome(IntEnum):
    """Value of a win, in points. The int value IS the point count, so an
    Outcome doubles as its own multiplier for scoring/equity."""
    SINGLE = 1
    GAMMON = 2
    BACKGAMMON = 3


class GameResult(NamedTuple):
    winner: int          # 0 or 1
    outcome: Outcome
    final_board: Board   # terminal position, from the winner's perspective


def roll_dice(rng: random.Random) -> tuple[int, int]:
    return (rng.randint(1, 6), rng.randint(1, 6))


def is_win(board: Board) -> bool:
    """True when the current player has borne off all 15 checkers."""
    return board.off_count == 15


def classify_win(board: Board) -> Outcome:
    """Classify a won position, judged from the WINNER's perspective (so the
    loser is the opponent: negative points, opp_* counts).

    Must be called on a board where `is_win` holds. Order matters:
      - single: the loser has borne off at least one checker.
      - backgammon: the loser has borne off none AND still has a checker in
        the winner's home board (my indices 0-5) or on the bar.
      - gammon: the loser has borne off none but is otherwise out of my home.
    """
    if board.opp_off_count >= 1:
        return Outcome.SINGLE
    loser_in_my_home = any(p < 0 for p in board.points[:6])
    if board.opp_bar_count > 0 or loser_in_my_home:
        return Outcome.BACKGAMMON
    return Outcome.GAMMON


def play_turn(board: Board, dice: tuple[int, int], agent: Agent) -> tuple[Board, bool]:
    # after_states from generate_moves; empty -> the dance (return board unchanged, False);
    # else let the agent pick one (return it, True).
    after_states = generate_moves(board, dice)
    if not after_states:
        return (board, False)
    new_board = agent(board, after_states)
    return (new_board, True)


def play_game(agents: tuple[Agent, Agent], rng: random.Random,
              max_turns: int = 1000) -> GameResult:
    """Play a full game between two agents, player 0 moving first.

    The board is always held from the current mover's perspective; between
    turns it is flipped so the next player sees themselves as "me". Dice come
    from `rng`, so games are reproducible when the rng is seeded.
    """
    board = starting_board()
    current = 0
    for _ in range(max_turns):
        dice = roll_dice(rng)
        board, _moved = play_turn(board, dice, agents[current])
        # Check the win BEFORE flipping: the mover who just played is the one
        # whose off_count can reach 15, and that shows in this board's POV.
        if is_win(board):
            return GameResult(winner=current, outcome=classify_win(board),
                              final_board=board)
        board = flip(board)          # hand the board to the opponent, their POV
        current = 1 - current
    raise RuntimeError("game exceeded max_turns without terminating")