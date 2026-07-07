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