import os
import sys
from typing import NamedTuple


class Board(NamedTuple):
    points: tuple[int, ...]
    bar_count: int
    opp_bar_count: int
    off_count: int
    opp_off_count: int


def starting_board() -> Board:
    starting_points = (
        -2, 0, 0, 0, 0, 5,
        0, 3, 0, 0, 0, -5,
        5, 0, 0, 0, -3, 0,
        -5, 0, 0, 0, 0, 2,
    )
    return Board(points=starting_points, bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0)

def flip(board: Board) -> Board:
    new_points = [-x for x in reversed(board.points)]
    return Board(points=tuple(new_points), 
                 bar_count=board.opp_bar_count, 
                 opp_bar_count=board.bar_count, 
                 off_count=board.opp_off_count, 
                 opp_off_count=board.off_count)

def pip_count(board: Board) -> int:
    total = 0
    for idx, count in enumerate(board.points):
        if count > 0:
            total += (idx + 1) * count
    total += (25 * board.bar_count)
    return total

def is_valid(board: Board) -> bool:
    c1 = len(board.points) == 24 
    c2 = board.bar_count >= 0 and board.opp_bar_count >= 0
    c3 = board.off_count >= 0 and board.opp_off_count >= 0
    c4 = board.bar_count + board.off_count + sum([p for p in board.points if p > 0]) == 15
    c5 = board.opp_bar_count + board.opp_off_count + sum([-p for p in board.points if p < 0]) == 15
    return all([c1, c2, c3, c4, c5])


# ANSI colors. Layout is always computed on the PLAIN glyph lengths, and codes
# are injected only around the visible character, so color never shifts a cell.
_CYAN, _RED, _DIM, _GREEN, _RESET = "\033[36m", "\033[31m", "\033[2m", "\033[32m", "\033[0m"
ROWS = 5  # max visible checkers per point; taller stacks show a count
MAX_FIELD_WIDTH = 4  # column width for all field cells

def _cell(count: int, row: int) -> str:
    """The glyph in visual row `row` (0 = nearest the point's label) of a point
    holding `count` checkers.

    A point is drawn as a column of up to ROWS symbols -- X for mine (count > 0),
    O for the opponent (count < 0). A stack too tall to draw one-per-checker is
    truncated: symbols fill the first ROWS-1 rows and the final row shows the
    total instead (e.g. 8 checkers -> X, X, X, X, "8"). Empty rows are "".
    """
    if count == 0 or row >= ROWS:
        return ""                              # nothing here, or past the drawn rows

    height = abs(count)
    symbol = "X" if count > 0 else "O"
    too_tall = height > ROWS

    if too_tall and row == ROWS - 1:
        return str(height)                     # final drawn row shows the count
    symbol_rows = ROWS - 1 if too_tall else height
    return symbol if row < symbol_rows else ""

def _supports_color() -> bool:
    return not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def _paint(text: str, code: str, on: bool) -> str:
    return f"{code}{text}{_RESET}" if on and text else text


def _checker_field(count: int, row: int, on: bool) -> str:
    """A MAX_FIELD_WIDTH board cell for the `row`-th checker of a stack (X cyan, O red)."""
    glyph = _cell(count, row)
    return " " * (MAX_FIELD_WIDTH - len(glyph)) + _paint(glyph, _CYAN if count > 0 else _RED, on)


def _bar_field(count: int, row: int, on: bool) -> str:
    """The width-3 center (bar) cell for a checker row."""
    glyph = _cell(count, row)
    pad = 3 - len(glyph)
    return " " * (pad // 2) + _paint(glyph, _CYAN if count > 0 else _RED, on) + " " * (pad - pad // 2)


def _label_field(text: str, on: bool) -> str:
    """Formatted label that should be left-indented by MAX_FIELD_WIDTH - len(text) spaces."""
    return " " * (MAX_FIELD_WIDTH - len(text)) + _paint(text, _DIM, on)


def _row(fields: list[str], center: str) -> str:
    """Assemble one board line: six left columns, the center (bar) column, six
    right. Every field has the width of MAX_FIELD_WIDTH, so color never shifts the layout."""
    return f"{''.join(fields[:6])}  {center}  {''.join(fields[6:])}"


# The board is a stack of named row-entities, top to bottom (see `render`).
_TOP_POINTS = tuple(range(12, 24))       # indices for points 13..24 (left -> right)
_BOTTOM_POINTS = tuple(range(11, -1, -1))   # indices for points 12..1  (left -> right)


def _pip_header(board: Board, on: bool) -> str:
    """Top line: each side's pip count and who leads the race."""
    my_pips, opp_pips = pip_count(board), pip_count(flip(board))
    diff = opp_pips - my_pips
    if diff > 0:
        race = _paint(f"you lead by {diff}", _GREEN, on)
    elif diff < 0:
        race = _paint(f"you trail by {-diff}", _RED, on)
    else:
        race = _paint("even race", _DIM, on)
    return (f"You ({_paint('X', _CYAN, on)}): {my_pips} pips     "
            f"Opp ({_paint('O', _RED, on)}): {opp_pips} pips     ({race})")


def _index_row(indices: tuple[int, ...], on: bool) -> str:
    """A row of point-number labels (1-24), blank center."""
    return _row([_label_field(str(i + 1), on) for i in indices], "   ")


def _divider_row(on: bool) -> str:
    """The dashed rule bordering the checker area."""
    return _row([_label_field("---", on) for _ in range(12)], _paint("---", _DIM, on))


def _upper_checker_rows(board: Board, on: bool) -> list[str]:
    """Opponent's half: top stacks grow downward; opp bar checkers in the center."""
    return [_row([_checker_field(board.points[i], row, on) for i in _TOP_POINTS],
                 _bar_field(-board.opp_bar_count, row, on))
            for row in range(ROWS)]


def _bar_row(on: bool) -> str:
    """The center divider between the two halves, carrying the BAR label."""
    return _row([" " * MAX_FIELD_WIDTH] * 12, _paint("BAR", _DIM, on))


def _lower_checker_rows(board: Board, on: bool) -> list[str]:
    """My half: bottom stacks grow upward; my bar checkers in the center."""
    return [_row([_checker_field(board.points[i], row, on) for i in _BOTTOM_POINTS],
                 _bar_field(board.bar_count, row, on))
            for row in range(ROWS - 1, -1, -1)]


def _bear_off_row(board: Board) -> str:
    """Bottom line: how many checkers each side has borne off."""
    return f"Off -- you: {board.off_count}   opp: {board.opp_off_count}"


def render(board: Board, color: bool | None = None) -> str:
    """ASCII board from the current player's perspective, assembled as a stack
    of named row-entities. I am X (moving 24->1, bearing off past 1); the
    opponent is O. Point labels are 1-24; arrays stay 0-23 internally. The
    center column is the bar -- opponent's hit checkers above BAR, mine below.

    `color` adds ANSI color; None (default) auto-detects a terminal, so piped
    output and tests stay plain. Layout is identical either way.
    """
    on = _supports_color() if color is None else color
    return "\n".join([
        _pip_header(board, on),
        "",
        _index_row(_TOP_POINTS, on),
        _divider_row(on),
        *_upper_checker_rows(board, on),
        _bar_row(on),
        *_lower_checker_rows(board, on),
        _divider_row(on),
        _index_row(_BOTTOM_POINTS, on),
        "",
        _bear_off_row(board),
    ])