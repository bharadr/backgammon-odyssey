from .board import Board


BAR_IDX = 24
OFF = -1        # sentinel destination for a bear-off hop

def _max_play_pile(board: Board, dice: tuple[int, int], walk) -> set[tuple]:
    """Explore `dice` under the max-play + larger-die rules, returning the winning
    pile of (item, depth) leaves.

    `walk` is the tree-walker: `extend` (leaves are boards) or `_extend_paths`
    (leaves are hop paths). The selection depends only on the depth field, so it's
    identical for both -- this is the single home of the subtle ordering rules.
    """
    if dice[0] == dice[1]:
        # Doubles: four identical dice, ordering is meaningless -- one exploration,
        # four levels deep. The larger-die rule can't apply (one die value).
        return walk(board, [dice[0]] * 4)

    # Non-doubles: (hi, lo) and (lo, hi) can reach different leaves, because the
    # intermediate position after the first die differs. Each call implicitly
    # records which die went first. Depth = number of dice consumed.
    hi, lo = max(dice[0], dice[1]), min(dice[0], dice[1])
    hi_pile = walk(board, [hi, lo])
    lo_pile = walk(board, [lo, hi])
    max_depth_hi = max(d for _, d in hi_pile)
    max_depth_lo = max(d for _, d in lo_pile)

    if max_depth_hi == 1 and max_depth_lo == 1:
        # Only single dice playable, and both are -> law: play the larger. Every
        # depth-1 leaf in hi_pile played `hi` first, so keeping hi_pile IS the
        # larger-die rule. (Must precede the equality test, or lo-only leaks.)
        return hi_pile
    if max_depth_lo == max_depth_hi:
        return hi_pile | lo_pile                # equal depth: both orderings lawful
    # Unequal depth: max-play dominates even the larger-die preference -- if only
    # lo-first can use both dice, you are forced to play lo first.
    return hi_pile if max_depth_hi > max_depth_lo else lo_pile


def _max_depth_only(pile: set[tuple]) -> set:
    """Keep only the deepest (max-play) leaves; {} on the dance (max depth 0).
    The walkers always return >=1 leaf, so max() is safe on a non-empty pile."""
    max_depth = max(d for _, d in pile)
    if max_depth == 0:
        return set()
    return {item for item, depth in pile if depth == max_depth}


def generate_moves(board: Board, dice: tuple[int, int]) -> set[Board]:
    """The set of legal afterstates: every distinct position reachable
    by playing `dice` from `board`.

    Each element is a resulting `Board`, not a move description. The set
    excludes the unchanged input — a returned board always differs from
    `board` — and reflects the max-play rule, so every element uses as
    many dice as the rules allow.

    Returns {} when no legal play exists (the "dance"); the caller must
    treat the empty set as a forfeited turn.
    """
    return _max_depth_only(_max_play_pile(board, dice, extend))

def extend(board: Board, dice: list[int], depth: int = 0) -> set[tuple[Board, int]]:
    """Explore the move tree for `dice` played in this exact order.

    Returns every LEAF of the tree as (board, depth), where depth is the
    number of dice consumed to reach it. A board is a leaf in one of two
    ways: all dice were played (leaf by success), or the next die has no
    legal hop from any candidate (leaf by blockage). Boards with a legal
    continuation are waypoints, never recorded — if you can keep playing,
    you must (the max-play rule; enforced by the caller via depths).

    Always returns at least one tuple: a fully blocked root comes back as
    {(board, 0)} — the "dance" — which generate_moves translates to {}.
    """
    if not dice:
        return {(board, depth)}  # leaf by success: every die consumed

    die, rest = dice[0], dice[1:]  # play the head now; rest goes to children

    # The bar rule is absolute: any checker on the bar freezes all other
    # candidates.
    checker_candidates = [BAR_IDX] if board.bar_count > 0 else \
        [idx for idx in range(23, -1, -1) if board.points[idx] > 0]

    results: set[tuple[Board, int]] = set()
    for src in checker_candidates:
        new_board = move_one(board, src, die)
        if new_board is not None:
            # Descend into the child: the NEXT die is played on the board
            # that results from this hop, not on the original.
            results |= extend(new_board, rest, depth + 1)

    if not results:
        # No candidate could play `die`: this node has no children, which
        # makes it a leaf by blockage.
        return {(board, depth)}
    return results


def _extend_paths(board: Board, dice: list[int], path: tuple = ()) -> set[tuple[tuple, int]]:
    """Like `extend`, but records the hop PATH taken to each leaf.

    Returns {(path, depth)} for every leaf, where `path` is a tuple of hops and
    a hop is (src, dst, die): src is a point index (24 = bar), dst a point index
    or OFF (bear-off), die the pips used. depth == len(path). Mirrors `extend`'s
    leaf-by-success / leaf-by-blockage structure exactly.
    """
    if not dice:
        return {(path, len(path))}                       # leaf by success

    die, rest = dice[0], dice[1:]
    candidates = [BAR_IDX] if board.bar_count > 0 else \
        [idx for idx in range(23, -1, -1) if board.points[idx] > 0]

    results: set[tuple[tuple, int]] = set()
    for src in candidates:
        new_board = move_one(board, src, die)
        if new_board is not None:
            dst = src - die
            hop = (src, dst if dst >= 0 else OFF, die)
            results |= _extend_paths(new_board, rest, path + (hop,))

    if not results:
        return {(path, len(path))}                       # leaf by blockage
    return results


def generate_move_paths(board: Board, dice: tuple[int, int]) -> set[tuple]:
    """Every legal max-play as a hop PATH -- a tuple of (src, dst, die) hops.

    The path-carrying twin of `generate_moves`: same rules (max-play, larger-die,
    doubles) but it returns the ordered hop sequences instead of afterstates, and
    keeps BOTH orderings of a non-double roll so a UI can let either checker move
    first. Applying a returned path reproduces one of `generate_moves`' afterstates.
    All returned paths share the max-play length; empty on the dance.
    """
    return _max_depth_only(_max_play_pile(board, dice, _extend_paths))


def move_one(board: Board, src: int, die: int) -> Board | None:
    """Play one checker from `src` (24 = the bar) by `die` pips.

    Returns the resulting board, or None if the hop is illegal.
    All movement law lives here; extend() only iterates candidates.
    """
    dst = src - die
    new_points = list(board.points)
    if src == BAR_IDX:
        # The bar's "stack" lives in bar_count, not the points array
        new_bar_count = board.bar_count - 1
    else:
        new_bar_count = board.bar_count
        new_points[src] -= 1  # source decrement, done once for all cases

    if 0 <= dst < 24:
        # --- Normal hop: destination is on the board -------------------
        dest = board.points[dst]
        if dest < -1:                   # two or more opponents: blocked
            return None
        opp_bar_count_change = 0
        if dest == -1:                  # lone opponent checker: a blot
            # The hit: their checker leaves the board for THEIR bar.
            new_points[dst] = 1
            opp_bar_count_change = 1
        else:                           # open point or my own stack
            new_points[dst] += 1
        return Board(points=tuple(new_points),
                    bar_count=new_bar_count,
                    opp_bar_count=board.opp_bar_count + opp_bar_count_change,
                    off_count=board.off_count,
                    opp_off_count=board.opp_off_count)

    # --- Bear-off: destination is past the edge (dst < 0) --------------
    # Gate: legal only when all 15 of my checkers are home or off.
    # Judged on board.points (the position BEFORE this move), with the
    # p > 0 filter — opponent checkers squatting in my home don't count.
    bearing_off_valid = board.bar_count == 0 and \
        sum(val for val in board.points[:6] if val > 0) + board.off_count == 15
    if not bearing_off_valid:
        return None
    # Overshoot (dst < -1) is legal only for the rearmost checker; an exact
    # roll (dst == -1) is always fine once home.
    if dst < -1 and any(board.points[i] > 0 for i in range(src + 1, 6)):
        return None
    return Board(points=tuple(new_points),
                bar_count=new_bar_count,
                opp_bar_count=board.opp_bar_count,
                off_count=board.off_count + 1,
                opp_off_count=board.opp_off_count)