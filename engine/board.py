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

ROWS = 5  # max visible checkers per point; taller stacks show a count


def _cell(count: int, row: int) -> str:
    """Symbol for the `row`-th checker (0 = nearest the labels) of a stack.

    X = mine (positive), O = opponent (negative), blank = nothing.
    Stacks taller than ROWS show the total count in the last row.
    """
    n = abs(count)
    sym = "X" if count > 0 else "O"
    if n == 0 or row >= ROWS:
        return ""
    if n <= ROWS:
        return sym if row < n else ""
    # overflow: symbols in rows 0..ROWS-2, total count in row ROWS-1
    return sym if row < ROWS - 1 else str(n)


def render(board: Board) -> str:
    """ASCII board, always from the current player's perspective.

    Top row: indices 12..23 (left to right). Bottom row: 11..0.
    I am X, moving 23 -> 0 and bearing off past 0.
    """
    W = 4  # column width

    def fmt(cells: list[str]) -> str:
        return "".join(f"{c:>{W}}" for c in cells)

    top_idx = list(range(12, 24))
    bot_idx = list(range(11, -1, -1))

    lines = [fmt([str(i) for i in top_idx])]
    lines.append(fmt(["---"] * 12))
    for row in range(ROWS):  # top stacks grow downward
        lines.append(fmt([_cell(board.points[i], row) for i in top_idx]))
    lines.append("")
    for row in range(ROWS - 1, -1, -1):  # bottom stacks grow upward
        lines.append(fmt([_cell(board.points[i], row) for i in bot_idx]))
    lines.append(fmt(["---"] * 12))
    lines.append(fmt([str(i) for i in bot_idx]))
    lines.append("")
    lines.append(
        f"Bar: me {board.bar_count}, opp {board.opp_bar_count}   "
        f"Off: me {board.off_count}, opp {board.opp_off_count}"
    )
    return "\n".join(lines)