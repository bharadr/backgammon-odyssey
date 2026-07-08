from .board import Board


BAR_IDX = 24

def generate_moves(board: Board, dice: tuple[int, int]) -> set[Board]:
    """All legal afterstates for `board` given `dice` that differ from
    the original board.

    Returns {} when no legal move exists (the "dance") — the caller must
    treat an empty set as a forfeited turn. Every returned board therefore
    differs from `board`: this function returns *moves*, not afterstates.
    """
    if dice[0] == dice[1]:
        # Doubles: four identical dice, so ordering is meaningless —
        # one exploration, four levels deep. The larger-die rule can't
        # apply (there's only one die value), so no special cases below.
        pile = extend(board, [dice[0]] * 4)
    else:
        # Non-doubles: (hi, lo) and (lo, hi) can reach different boards,
        # because the intermediate position after the first die differs.
        # Crucially, each pile *implicitly* records which die was played
        # first — that fact is encoded by the call, not stored in the data.
        hi, lo = max(dice[0], dice[1]), min(dice[0], dice[1])
        hi_pile = extend(board, [hi, lo])
        lo_pile = extend(board, [lo, hi])

        # Depth = number of dice consumed. Max-play rule: you must play
        # as many dice as possible, so deeper piles dominate shallower ones.
        max_depth_hi = max(d for _, d in hi_pile)
        max_depth_lo = max(d for _, d in lo_pile)

        if max_depth_hi == 1 and max_depth_lo == 1:
            # The ONE asymmetric cell: only single dice are playable, and
            # both are. Law: you must play the larger. Every depth-1 board
            # in hi_pile played `hi` first (by construction of the call),
            # so keeping hi_pile alone IS the larger-die rule.
            # NOTE: this test must precede the general equality test below,
            # or it gets shadowed and illegal lo-only moves leak through.
            pile = hi_pile
        elif max_depth_lo == max_depth_hi:
            # Equal depth (0 or 2): both orderings equally lawful — merge.
            pile = hi_pile | lo_pile
        else:
            # Unequal depth: max-play dominates everything, even the
            # larger-die preference — if only the lo-first ordering can
            # use both dice, you are FORCED to play lo first.
            pile = hi_pile if max_depth_hi > max_depth_lo else lo_pile

    # extend() always returns ≥1 tuple (a fully blocked branch returns the
    # original board at depth 0), so max() is safe on a non-empty pile.
    max_depth = max(d for _, d in pile)
    if max_depth == 0:
        return set()  # the dance: no legal move anywhere
    return {b for b, depth in pile if depth == max_depth}

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

    if dst >= 0 and dst < 24:
        # --- Normal hop: destination is on the board -------------------
        opp_bar_count_change = 0
        if board.points[dst] >= 0:      # open point or my own stack
            new_points[dst] += 1
        elif board.points[dst] == -1:   # lone opponent checker: a blot
            # The hit: their checker leaves the board for THEIR bar.
            new_points[dst] = 1
            opp_bar_count_change = 1
        else:                           # two or more opponents: blocked
            return None
        return Board(points=tuple(new_points),
                    bar_count=new_bar_count,
                    opp_bar_count=board.opp_bar_count + opp_bar_count_change,
                    off_count=board.off_count,
                    opp_off_count=board.opp_off_count)
    else:
        # --- Bear-off: destination is past the edge --------------------
        off_count_change = 0
        # Gate: legal only when all 15 of my checkers are home or off.
        # Judged on board.points (the position BEFORE this move), with the
        # p > 0 filter — opponent checkers squatting in my home don't count.
        # A bar checker fails this sum automatically, so no separate check.
        bearing_off_valid = sum(val for val in board.points[:6] if val > 0) + board.off_count == 15
        if not bearing_off_valid:
            return None
        else:
            if dst == -1:
                # Exact roll: die == src + 1 pips. Always legal once home.
                off_count_change = 1
            elif dst < -1:
                # Overshoot: Legal only for the rearmost checker
                if any(board.points[i] > 0 for i in range(src + 1, 6)):
                    return None
                else:
                    off_count_change = 1
        return Board(points=tuple(new_points),
                    bar_count=new_bar_count,
                    opp_bar_count=board.opp_bar_count,
                    off_count=board.off_count + off_count_change,
                    opp_off_count=board.opp_off_count)