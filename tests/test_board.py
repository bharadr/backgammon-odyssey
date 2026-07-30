# tests/test_board.py
import re

from engine.board import Board, starting_board, flip, pip_count, is_valid, render, _cell, ROWS


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


# --- render (display) -------------------------------------------------

_BAR_OFF_BOARD = Board(points=(0, 0, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, -4, -3, -2, -2, 0, 0),
                       bar_count=2, opp_bar_count=1, off_count=2, opp_off_count=3)

_ANSI = re.compile(r"\033\[[0-9;]*m")


def test_render_shows_pips_race_bar_and_off():
    out = render(starting_board(), color=False)
    lines = out.splitlines()
    assert lines[0] == "You (X): 167 pips     Opp (O): 167 pips     (even race)"
    assert any(line.strip() == "BAR" for line in lines)   # the bar column is drawn
    assert "Off -- you: 0   opp: 0" in out

def test_cell_stacks_symbols_and_overflows_to_a_count():
    assert _cell(0, 0) == ""                                   # empty point
    assert [_cell(3, r) for r in range(ROWS)] == ["X", "X", "X", "", ""]
    assert [_cell(ROWS, r) for r in range(ROWS)] == ["X"] * ROWS   # full column, no count
    # ROWS+1 is the first overflow: symbols fill ROWS-1 rows, then the count
    assert [_cell(ROWS + 1, r) for r in range(ROWS)] == ["X", "X", "X", "X", "6"]
    assert [_cell(-8, r) for r in range(ROWS)] == ["O", "O", "O", "O", "8"]  # opponent
    assert _cell(8, ROWS) == ""                                # past the drawn rows


def test_render_race_status_reflects_the_pip_difference():
    # my 5-stack on the 6-point (10 pips) + 2 on the bar (50) vs the opp's
    # 11 checkers deep in their home; I trail badly.
    out = render(_BAR_OFF_BOARD, color=False)
    assert out.splitlines()[0] == "You (X): 106 pips     Opp (O): 78 pips     (you trail by 28)"
    assert "Off -- you: 2   opp: 3" in out

def test_color_render_keeps_the_layout_identical():
    # color must only inject ANSI codes, never shift a column: stripping the
    # codes from the colored render must reproduce the plain render exactly.
    for b in (starting_board(), _BAR_OFF_BOARD):
        colored = render(b, color=True)
        assert "\033[" in colored                          # color was actually applied
        assert _ANSI.sub("", colored) == render(b, color=False)

