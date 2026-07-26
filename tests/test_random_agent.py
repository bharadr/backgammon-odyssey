import random

from engine.board import starting_board
from engine.moves import generate_moves
from agent.random_agent import random_agent


def test_random_agent_picks_a_legal_afterstate():
    board = starting_board()
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    choice = random_agent(random.Random(0))(board, dice, afterstates)
    assert choice in afterstates

def test_random_agent_is_reproducible_under_a_seed():
    board = starting_board()
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    a = random_agent(random.Random(7))
    b = random_agent(random.Random(7))
    # same seed -> identical sequence of choices
    assert [a(board, dice, afterstates) for _ in range(10)] == \
           [b(board, dice, afterstates) for _ in range(10)]
