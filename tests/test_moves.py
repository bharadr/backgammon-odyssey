from engine.board import starting_board, is_valid
from engine.moves import generate_moves
from tests.test_board import midgame_boards

def test_opening_63_all_valid():
    moves = generate_moves(starting_board(), (6, 3))
    assert moves, "opening 6-3 must have legal moves"
    assert all(is_valid(b) for b in moves)

def test_bar_board_63_all_valid():
    b1 = midgame_boards()[0]
    moves = generate_moves(b1, (6, 3))
    assert all(is_valid(b) for b in moves)