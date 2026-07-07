# tests/test_board.py
from engine.board import Board, starting_board, flip, pip_count, is_valid


# --- Fixtures: boards to test against ---------------------------------

def midgame_boards() -> list[Board]:
    """Hand-crafted boards with bar/off checkers, for exercising
    the cases the starting position can't reach.
    Each must be legal: 15 checkers per side, counted across
    points + bar + off."""
    b1 = Board(
        points=(
            0,  0,  2,  0,  4,  3,    # me: 2+4+3 = 9 in my home region
            0,  2,  0,  0,  0,  3,    # me: +5 → my total 14 ✓
            -2,  0, -3,  0,  0, -4,    # opp: -9
            0, -2,  0, -2,  0,  0,    # opp: -4 → opp total -13 ✓
        ),
        bar_count=1, opp_bar_count=2,
        off_count=0, opp_off_count=0,
    )
    b2 = Board(
        points=(
            0,  0,  2,  0,  4,  3,    # me: 2+4+3 = 9 in my home region
            0,  3,  0,  0,  0,  0,    # me: +3 → my total 12 ✓
            -2,  0, -3,  0,  0, -4,    # opp: -9
            0, -2,  0, -2,  0,  0,    # opp: -4 → opp total -13 ✓
        ),
        bar_count=0, opp_bar_count=0,
        off_count=3, opp_off_count=2,
    )
    b3 = Board(
        points=(
            0,  0,  2,  0,  4,  3,    # me: 2+4+3 = 9 in my home region
            0,  0,  0,  0,  0,  0,    
            0,  0, 0,  0,  0, 0,    
            0, -2,  0, -3,  0,  0,    
        ),
        bar_count=1, opp_bar_count=2,
        off_count=5, opp_off_count=8,
    )
    return [b1, b2, b3]


ALL_BOARDS = [starting_board()] + midgame_boards()


# --- Layer 0 invariants ------------------------------------------------

def test_starting_board_is_valid():
    assert is_valid(starting_board())

def test_starting_pip_count():
    assert pip_count(starting_board()) == 167

def test_flip_is_involution():
    for b in ALL_BOARDS:
        assert flip(flip(b)) == b

def test_flip_conserves_pip_counts():
    # my pip count before flip == "their" pip count after
    def opp_pip_count(b: Board) -> int:
        total = 0
        for idx, count in enumerate(b.points):
            if count < 0:
                total += (24 - idx) * -count
        total += (25 * b.opp_bar_count)
        return total
    for b in ALL_BOARDS:
        assert pip_count(flip(b)) == opp_pip_count(b)

def test_flip_conserves_checker_totals():
    for b in ALL_BOARDS:
        assert is_valid(b)
        assert is_valid(flip(b))

def test_midgame_boards_are_valid():
    # guards against bugs in the test fixtures themselves
    for b in midgame_boards():
        assert is_valid(b)