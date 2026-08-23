"""An SVG rendering of a Board for the web UI -- crisp, scalable, pure-Python.

`board_svg(board, dice)` returns a standalone <svg> string: 24 triangular points,
checkers as discs (tall stacks show a count), the center bar, an off tray, a
pip/race header, and -- if a roll is given -- the dice as die faces.

Orientation matches engine.board.render: top row is points 13-24 (left->right),
bottom row 12-1, you are X (home in the bottom-right), the opponent is O.
"""
from engine.board import Board, flip, pip_count
from engine.moves import BAR_IDX, OFF

# palette
_ME, _ME_EDGE = "#1a5fb4", "#0d3a75"
_OPP, _OPP_EDGE = "#c01c28", "#7a1219"
_LIGHT, _DARK = "#e6d2ad", "#b1885b"
_BG, _EDGE, _INK = "#f4e8d0", "#5a3e2b", "#2b2b2b"
_HL = "#4caf50"     # highlight overlay for legal destinations (semi-transparent green)
_DIE_USED = "#c9c9c9"   # greyed face of a spent die

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
        f'fill="{_LIGHT}" stroke="{_EDGE}"/>',
        f'<text x="{cx}" y="{BOARD_TOP + 18}" text-anchor="middle" font-size="11" fill="{_OPP}">off</text>',
        f'<text x="{cx}" y="{BOARD_TOP + 40}" text-anchor="middle" font-size="20" '
        f'font-weight="bold" fill="{_OPP}">{board.opp_off_count}</text>',
        f'<text x="{cx}" y="{BOARD_BOTTOM - 26}" text-anchor="middle" font-size="20" '
        f'font-weight="bold" fill="{_ME}">{board.off_count}</text>',
        f'<text x="{cx}" y="{BOARD_BOTTOM - 10}" text-anchor="middle" font-size="11" fill="{_ME}">off</text>',
    ]


def _die(x: float, y: float, size: float, value: int, used: bool = False) -> list[str]:
    face, ink = (_DIE_USED, "#8a8a8a") if used else ("#fff", _INK)
    els = [f'<rect class="die" x="{x}" y="{y}" width="{size}" height="{size}" rx="6" '
           f'fill="{face}" stroke="{ink}" stroke-width="2"/>']
    for fx, fy in _PIPS[value]:
        els.append(f'<circle class="pip" cx="{x + fx * size:.1f}" cy="{y + fy * size:.1f}" '
                   f'r="{size * 0.09:.1f}" fill="{ink}"/>')
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


def _dice_faces(dice: tuple[int, int], used: list[int]) -> list[tuple[int, bool]]:
    """The die faces to draw as (value, is_used): four for doubles, two otherwise.
    Doubles grey the first `len(used)`; a non-double greys each face whose value
    has been spent."""
    if dice[0] == dice[1]:
        return [(dice[0], i < len(used)) for i in range(4)]
    faces, pool = [], list(used)
    for value in dice:
        spent = value in pool
        if spent:
            pool.remove(value)
        faces.append((value, spent))
    return faces


def _dice(dice, used: list[int]) -> list[str]:
    """The dice, rolled into the board's right half; spent ones greyed out."""
    if not dice:
        return []
    faces = _dice_faces(dice, used)
    d, gap = 34, 10
    total = len(faces) * d + (len(faces) - 1) * gap
    x0 = (MARGIN + 9 * POINT_W + BAR_W) - total / 2      # centred in the right half
    y = BOARD_TOP + BOARD_H / 2 - d / 2
    els = []
    for i, (value, is_used) in enumerate(faces):
        els += _die(x0 + i * (d + gap), y, d, value, is_used)
    return els


def _highlight_marks(highlight: set[int]) -> list[str]:
    """A light, semi-transparent green overlay on each legal destination."""
    fill = f'fill="{_HL}" fill-opacity="0.45"'
    marks = []
    for p in highlight:
        if p == OFF:
            ox = MARGIN + 12 * POINT_W + BAR_W
            marks.append(f'<rect class="hl" x="{ox}" y="{BOARD_TOP}" width="{OFF_W}" '
                         f'height="{BOARD_H}" {fill}/>')
        elif p == BAR_IDX:
            marks.append(f'<rect class="hl" x="{BAR_X}" y="{BOARD_TOP}" width="{BAR_W}" '
                         f'height="{BOARD_H}" {fill}/>')
        else:                                    # a point triangle
            c = p - 12 if p >= 12 else 11 - p
            x = _col_x(c)
            cx = x + POINT_W / 2
            if p >= 12:                          # top point
                pts = f"{x},{BOARD_TOP} {x + POINT_W},{BOARD_TOP} {cx},{BOARD_TOP + POINT_H}"
            else:                                # bottom point
                pts = f"{x},{BOARD_BOTTOM} {x + POINT_W},{BOARD_BOTTOM} {cx},{BOARD_BOTTOM - POINT_H}"
            marks.append(f'<polygon class="hl" points="{pts}" {fill}/>')
    return marks


def point_at(x: float, y: float) -> int | None:
    """The point a pixel click lands on, in the WIDTH x HEIGHT board image: a point
    index 0-23, BAR_IDX (the bar), OFF (the bear-off tray), or None if outside.
    The inverse of the board's layout -- upper half of a column is its top point,
    lower half its bottom point."""
    if not (BOARD_TOP <= y <= BOARD_BOTTOM):
        return None
    if BAR_X <= x < BAR_X + BAR_W:
        return BAR_IDX
    off_x = MARGIN + 12 * POINT_W + BAR_W
    if off_x <= x < off_x + OFF_W:
        return OFF
    for c in range(12):
        if _col_x(c) <= x < _col_x(c) + POINT_W:
            top = y < BOARD_TOP + BOARD_H / 2
            return (12 + c) if top else (11 - c)
    return None


def board_svg(board: Board, dice: tuple[int, int] | None = None,
              highlight: set[int] | None = None, used: list[int] | None = None) -> str:
    """A standalone <svg> string for `board`: dice drawn if given (four for
    doubles; `used` die values greyed out), legal `highlight` destinations tinted."""
    els = [f'<rect x="{MARGIN - 6}" y="{BOARD_TOP - 2}" '
           f'width="{12 * POINT_W + BAR_W + OFF_W + 12}" height="{BOARD_H + 4}" '
           f'fill="{_BG}" stroke="{_EDGE}" stroke-width="3"/>']
    els += _points_and_checkers(board)
    els += _bar(board)
    els += _off_tray(board)
    els += _highlight_marks(highlight or set())
    els += _pip_counts(board)
    els += _dice(dice, used or [])
    # Explicit width/height (not width="100%") so the SVG rasterizes correctly for
    # click-mapping; the browser still scales it down to fit its container.
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}">'
            + "".join(els) + "</svg>")
