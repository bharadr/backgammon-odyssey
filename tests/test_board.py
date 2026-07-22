# tests/test_board.py
from engine.board import Board, starting_board, flip, pip_count, is_valid, render


# --- Fixtures: boards to test against ---------------------------------

def midgame_boards_with_pips() -> list[tuple[Board, int]]:
    """Hand-crafted boards paired with their pre-computed pip count.

    Each board must be legal: 15 checkers per side, counted across
    points + bar + off. The pip count sits in the return tuple beside
    each board on purpose — editing a board's points forces you to
    re-derive its count in the same place, so the two can't drift apart.

    Pip count = sum of (idx+1)*count over my points, plus 25 per checker
    on the bar. Borne-off checkers contribute 0.
    """
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
            -2,  0, -1,  0,  0, -6,    # opp: -9
            0, -2,  0, -2,  0,  0,    # opp: -4 → opp total -13 ✓
        ),
        bar_count=0, opp_bar_count=0,
        off_count=3, opp_off_count=2,
    )
    b3 = Board(
        points=(
            0,  0,  1,  0,  7,  1,    # me: 1+7+1 = 9 in my home region
            0,  0,  0,  0,  0,  0,    
            0,  0, 0,  0,  0, 0,    
            0, -2,  0, -3,  0,  0,    
        ),
        bar_count=1, opp_bar_count=2,
        off_count=5, opp_off_count=8,
    )
    return [
        (b1, 121),   # idx2·3 + idx4·5 + idx5·6 + idx7·8 + idx11·12 + bar·25
        (b2, 68),    # idx2·3 + idx4·5 + idx5·6 + idx7·8
        (b3, 69),    # idx2·3 + idx4·5 + idx5·6 + bar·25
    ]


def midgame_boards() -> list[Board]:
    return [b for b, _ in midgame_boards_with_pips()]


ALL_BOARDS = [starting_board()] + midgame_boards()


# --- Layer 0 invariants ------------------------------------------------

def test_starting_board_fields():
    # The standard opening from my perspective (I move 23 -> 0): my
    # checkers on the 24/13/8/6-points (idx 23/12/7/5), opponent mirrored,
    # nothing on the bar or borne off. Spelled out here independently so an
    # accidental edit to starting_board() is caught rather than mirrored.
    b = starting_board()
    assert b.points == (
        -2, 0, 0, 0, 0, 5,
        0, 3, 0, 0, 0, -5,
        5, 0, 0, 0, -3, 0,
        -5, 0, 0, 0, 0, 2,
    )
    assert b.bar_count == 0
    assert b.opp_bar_count == 0
    assert b.off_count == 0
    assert b.opp_off_count == 0

def test_starting_board_is_valid():
    assert is_valid(starting_board())

def test_starting_pip_count():
    assert pip_count(starting_board()) == 167


def test_midgame_boards_are_valid():
    # guards against bugs in the test fixtures themselves
    for b in midgame_boards():
        assert is_valid(b)

def test_midgame_pip_counts():
    for b, expected_count in midgame_boards_with_pips():
        assert pip_count(b) == expected_count


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

