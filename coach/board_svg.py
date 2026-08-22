"""An SVG rendering of a Board for the web UI -- crisp, scalable, pure-Python.

`board_svg(board, dice)` returns a standalone <svg> string: 24 triangular points,
checkers as discs (tall stacks show a count), the center bar, an off tray, a
pip/race header, and -- if a roll is given -- the dice as die faces.

Orientation matches engine.board.render: top row is points 13-24 (left->right),
bottom row 12-1, you are X (home in the bottom-right), the opponent is O.
"""
from engine.board import Board, flip, pip_count

# palette
_ME, _ME_EDGE = "#1a5fb4", "#0d3a75"
_OPP, _OPP_EDGE = "#c01c28", "#7a1219"
_LIGHT, _DARK = "#e6d2ad", "#b1885b"
_BG, _EDGE, _INK = "#f4e8d0", "#5a3e2b", "#2b2b2b"

# geometry. A full point holds CAP discs (~190px from the edge), so BOARD_H is
# sized to leave a middle band tall enough for the dice to clear the tallest
# stacks; POINT_H matches that stack height so a full point fills its triangle.
MARGIN, POINT_W, POINT_H = 16, 44, 190
BAR_W, OFF_W, CHECKER_R = 40, 40, 19
TOP_PAD, BOTTOM_PAD, BOARD_H = 30, 30, 460
CAP = 5                                   # max discs drawn before a stack shows a count
BOARD_TOP = TOP_PAD
BOARD_BOTTOM = TOP_PAD + BOARD_H
WIDTH = 2 * MARGIN + 12 * POINT_W + BAR_W + OFF_W
HEIGHT = BOARD_TOP + BOARD_H + BOTTOM_PAD
BAR_X = MARGIN + 6 * POINT_W              # left edge of the center bar
BAR_CX = BAR_X + BAR_W / 2               # bar centre-line (pip counts sit here)

# pip layout per die face, as (x, y) fractions of the die square
_PIPS = {
    1: [(.5, .5)],
    2: [(.28, .28), (.72, .72)],
    3: [(.28, .28), (.5, .5), (.72, .72)],
    4: [(.28, .28), (.72, .28), (.28, .72), (.72, .72)],
    5: [(.28, .28), (.72, .28), (.5, .5), (.28, .72), (.72, .72)],
    6: [(.28, .28), (.72, .28), (.28, .5), (.72, .5), (.28, .72), (.72, .72)],
}


def _col_x(c: int) -> float:
    """Left edge of point-column c (0-11), leaving room for the center bar."""
    return MARGIN + c * POINT_W + (BAR_W if c >= 6 else 0)


def _stack(cx: float, base_y: float, direction: int, count: int) -> list[str]:
    """A column of checkers at cx, growing from base_y (direction +1 down / -1 up).
    `count` is signed: >0 = mine (X, blue), <0 = opponent (O, red)."""
    n = abs(count)
    if n == 0:
        return []
    side = "me" if count > 0 else "opp"
    fill, edge = (_ME, _ME_EDGE) if count > 0 else (_OPP, _OPP_EDGE)
    els = []
    for k in range(min(n, CAP)):
        cy = base_y + direction * (CHECKER_R + k * 2 * CHECKER_R)
        els.append(f'<circle class="chk {side}" cx="{cx:.1f}" cy="{cy:.1f}" r="{CHECKER_R}" '
                   f'fill="{fill}" stroke="{edge}" stroke-width="2"/>')
        if n > CAP and k == CAP - 1:      # overflow: label the outermost disc
            els.append(f'<text x="{cx:.1f}" y="{cy + 5:.1f}" text-anchor="middle" '
                       f'font-size="16" font-weight="bold" fill="#fff">{n}</text>')
    return els


def _points_and_checkers(board: Board) -> list[str]:
    els = []
    for c in range(12):
        x = _col_x(c)
        cx = x + POINT_W / 2
        top_i, bot_i = 12 + c, 11 - c
        # top point (triangle down) + its label + checkers
        els.append(f'<polygon points="{x},{BOARD_TOP} {x + POINT_W},{BOARD_TOP} '
                   f'{cx},{BOARD_TOP + POINT_H}" fill="{_DARK if c % 2 == 0 else _LIGHT}" '
                   f'stroke="{_EDGE}" stroke-width="1"/>')
        els.append(f'<text x="{cx}" y="{BOARD_TOP - 6}" text-anchor="middle" '
                   f'font-size="12" fill="{_INK}">{top_i + 1}</text>')
        els += _stack(cx, BOARD_TOP, +1, board.points[top_i])
        # bottom point (triangle up) + its label + checkers
        els.append(f'<polygon points="{x},{BOARD_BOTTOM} {x + POINT_W},{BOARD_BOTTOM} '
                   f'{cx},{BOARD_BOTTOM - POINT_H}" fill="{_LIGHT if c % 2 == 0 else _DARK}" '
                   f'stroke="{_EDGE}" stroke-width="1"/>')
        els.append(f'<text x="{cx}" y="{BOARD_BOTTOM + 16}" text-anchor="middle" '
                   f'font-size="12" fill="{_INK}">{bot_i + 1}</text>')
        els += _stack(cx, BOARD_BOTTOM, -1, board.points[bot_i])
    return els


def _bar(board: Board) -> list[str]:
    return ([f'<rect x="{BAR_X}" y="{BOARD_TOP}" width="{BAR_W}" height="{BOARD_H}" fill="{_EDGE}"/>']
            + _stack(BAR_CX, BOARD_TOP, +1, -board.opp_bar_count)     # opponent's hits, top
            + _stack(BAR_CX, BOARD_BOTTOM, -1, board.bar_count))      # my hits, bottom


def _off_tray(board: Board) -> list[str]:
    ox = MARGIN + 12 * POINT_W + BAR_W
    cx = ox + OFF_W / 2
    return [
        f'<rect x="{ox}" y="{BOARD_TOP}" width="{OFF_W}" height="{BOARD_H}" '
        f'fill="#00000012" stroke="{_EDGE}"/>',
        f'<text x="{cx}" y="{BOARD_TOP + 18}" text-anchor="middle" font-size="11" fill="{_OPP}">off</text>',
        f'<text x="{cx}" y="{BOARD_TOP + 40}" text-anchor="middle" font-size="20" '
        f'font-weight="bold" fill="{_OPP}">{board.opp_off_count}</text>',
        f'<text x="{cx}" y="{BOARD_BOTTOM - 26}" text-anchor="middle" font-size="20" '
        f'font-weight="bold" fill="{_ME}">{board.off_count}</text>',
        f'<text x="{cx}" y="{BOARD_BOTTOM - 10}" text-anchor="middle" font-size="11" fill="{_ME}">off</text>',
    ]


def _die(x: float, y: float, size: float, value: int) -> list[str]:
    els = [f'<rect class="die" x="{x}" y="{y}" width="{size}" height="{size}" rx="6" '
           f'fill="#fff" stroke="{_INK}" stroke-width="2"/>']
    for fx, fy in _PIPS[value]:
        els.append(f'<circle class="pip" cx="{x + fx * size:.1f}" cy="{y + fy * size:.1f}" '
                   f'r="{size * 0.09:.1f}" fill="{_INK}"/>')
    return els


def _pip_counts(board: Board) -> list[str]:
    """Each side's pip count over the bar -- opponent (red) top, you (blue) bottom.
    Colour identifies the side, so no X/O labels or lead/trail hint are needed."""
    return [
        f'<text x="{BAR_CX}" y="{BOARD_TOP - 10}" text-anchor="middle" font-size="18" '
        f'font-weight="bold" fill="{_OPP}">{pip_count(flip(board))}</text>',
        f'<text x="{BAR_CX}" y="{BOARD_BOTTOM + 22}" text-anchor="middle" font-size="18" '
        f'font-weight="bold" fill="{_ME}">{pip_count(board)}</text>',
    ]


def _dice(dice) -> list[str]:
    """The two dice, rolled into the board's right half (vertically centred)."""
    if not dice:
        return []
    d, gap = 34, 12
    right_cx = MARGIN + 9 * POINT_W + BAR_W          # centre of the right-hand half
    x0 = right_cx - (2 * d + gap) / 2
    y = BOARD_TOP + BOARD_H / 2 - d / 2
    return _die(x0, y, d, dice[0]) + _die(x0 + d + gap, y, d, dice[1])


def board_svg(board: Board, dice: tuple[int, int] | None = None) -> str:
    """A standalone <svg> string for `board`, with the dice drawn if given."""
    els = [f'<rect x="{MARGIN - 6}" y="{BOARD_TOP - 2}" '
           f'width="{12 * POINT_W + BAR_W + OFF_W + 12}" height="{BOARD_H + 4}" '
           f'fill="{_BG}" stroke="{_EDGE}" stroke-width="3"/>']
    els += _points_and_checkers(board)
    els += _bar(board)
    els += _off_tray(board)
    els += _pip_counts(board)
    els += _dice(dice)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="100%" style="max-width:{WIDTH}px;height:auto">'
            + "".join(els) + "</svg>")
