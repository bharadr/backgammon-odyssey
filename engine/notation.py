from .board import Board
from .moves import BAR_IDX, move_one


def _find_path(board: Board, target: Board, dice: list[int]):
    """A legal sequence of single-die hops from `board` to `target` using
    `dice` (in some order), as a list of (src, dst, hit) tuples -- or None if
    `target` is unreachable.

    move_one enforces the bar rule and the bear-off gate, so every hop this
    yields is legal by construction; there is no ordering logic to write here.
    When several paths reach the same board they differ only cosmetically (a
    hitless intermediate), so the first one found is a fine description.
    """
    if board == target:
        return []
    for die in sorted(set(dice)):            # distinct values; collapses doubles
        remaining = list(dice)
        remaining.remove(die)                # consume one occurrence of this value
        # the bar rule: a checker on the bar must enter before anything else moves
        if board.bar_count > 0:
            candidates = [BAR_IDX]
        else:
            candidates = [i for i in range(24) if board.points[i] > 0]
        for src in candidates:
            hopped = move_one(board, src, die)
            if hopped is None:
                continue
            rest = _find_path(hopped, target, remaining)
            if rest is not None:             # this hop leads to the target
                hit = hopped.opp_bar_count > board.opp_bar_count
                return [(src, src - die, hit)] + rest
            # else: dead end -- fall through and try the next candidate/die
    return None


def _src_label(src: int) -> str:
    return "bar" if src == BAR_IDX else str(src + 1)      # 0-based index -> 1-24 point


def _dst_label(dst: int) -> str:
    return "off" if dst < 0 else str(dst + 1)             # dst < 0 means borne off


def describe_move(before: Board, after: Board, dice: tuple[int, int]) -> str:
    """Human-readable notation for the play turning `before` into `after` with
    `dice` -- e.g. "8/5 6/5", "bar/21 13/11", "6/off 5/off". A hit is marked
    with '*' (e.g. "11/7*"). `after` must be a legal afterstate of `before`.
    """
    hops = [dice[0]] * 4 if dice[0] == dice[1] else [dice[0], dice[1]]
    path = _find_path(before, after, hops)
    if path is None:
        raise ValueError("`after` is not reachable from `before` with these dice")

    parts = []
    for src, dst, hit in path:
        move = f"{_src_label(src)}/{_dst_label(dst)}"
        if hit:
            move += "*"
        parts.append(move)
    return " ".join(parts)
