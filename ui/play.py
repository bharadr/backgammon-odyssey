import random
from typing import Callable

from engine.board import flip, render, starting_board
from engine.game import (
    Agent,
    GameResult,
    classify_win,
    is_win,
    play_turn,
    roll_dice,
)
from engine.notation import describe_move
from agent.human_agent import HumanAgent
from agent.skill_agent import SkillAgent
from coach.gnubg_provider import GnubgProvider


def play_interactive(human: Agent, opponent: Agent, rng: random.Random,
                     output_fn: Callable[[str], None] = print,
                     max_turns: int = 1000) -> GameResult:
    """Drive a full interactive game: you are player 0, the opponent is 1.

    The board is held in the current mover's perspective and flipped between
    turns (as in play_game), but here we only ever *render* it on your turn --
    so you always see it from your own seat. The opponent's play is announced
    as notation in the opponent's own point numbering (standard convention);
    the board you see next turn reflects its result.
    """
    board = starting_board()
    current = 0
    for _ in range(max_turns):
        dice = roll_dice(rng)
        if current == 0:
            output_fn(render(board))
            output_fn(f"Your roll: {dice[0]}-{dice[1]}")
            board, moved = play_turn(board, dice, human)
            if not moved:
                output_fn("No legal moves -- you forfeit the turn.")
        else:
            output_fn(f"\nOpponent rolls {dice[0]}-{dice[1]}.")
            before = board
            board, moved = play_turn(board, dice, opponent)
            if moved:
                output_fn(f"Opponent plays: {describe_move(before, board, dice)}")
            else:
                output_fn("Opponent has no legal moves and forfeits the turn.")

        # win is checked BEFORE the flip: only the mover who just played can
        # have borne off all 15, and that shows in this board's perspective.
        if is_win(board):
            outcome = classify_win(board)
            who, verb = ("You", "win") if current == 0 else ("Opponent", "wins")
            output_fn(f"\n{who} {verb} -- {outcome.name.lower()} ({int(outcome)} pt)!")
            return GameResult(winner=current, outcome=outcome, final_board=board)

        board = flip(board)
        current = 1 - current
    raise RuntimeError("game exceeded max_turns without terminating")


# Uncalibrated starting strength; tune toward a target equity-loss/move later.
OPPONENT_TEMPERATURE = 0.1


def main() -> None:
    rng = random.Random()
    opponent = SkillAgent(GnubgProvider(), OPPONENT_TEMPERATURE, rng)
    print("Backgammon -- you are X (moving toward off), opponent is O "
          "(gnubg-backed). Good luck!\n")
    play_interactive(HumanAgent(), opponent, rng)


if __name__ == "__main__":
    main()
