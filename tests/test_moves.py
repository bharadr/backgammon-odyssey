from engine.board import Board, starting_board, is_valid
from engine.moves import BAR_IDX, extend, generate_moves, move_one
from tests.test_board import midgame_boards


def mk(points: dict[int, int] | None = None, *, bar: int = 0, opp_bar: int = 0,
       off: int = 0, opp_off: int = 0) -> Board:
    """A board that's empty except for the given {index: count} points.

    Deliberately minimal: move_one acts locally, so a test only needs to
    populate the source, the destination, and (for bear-off) the home board.
    Some of these boards are intentionally NOT legal 15-checker positions —
    that's how we isolate a single branch (e.g. the bar-count gate).
    """
    pts = [0] * 24
    for idx, cnt in (points or {}).items():
        pts[idx] = cnt
    return Board(points=tuple(pts), bar_count=bar, opp_bar_count=opp_bar,
                 off_count=off, opp_off_count=opp_off)

def test_opening_63_all_valid():
    moves = generate_moves(starting_board(), (6, 3))
    assert moves, "opening 6-3 must have legal moves"
    assert all(is_valid(b) for b in moves)

def test_bar_board_63_all_valid():
    b1 = midgame_boards()[0]
    moves = generate_moves(b1, (6, 3))
    assert all(is_valid(b) for b in moves)


# --- move_one: one checker, one die, all movement law -----------------
# One minimal board per leaf of move_one's decision tree. Assertions
# target only the fields that should change, so the intent stays legible.

def test_hop_to_open_point():
    # non-bar src, empty destination: source -1, destination +1
    b = mk({23: 2, 18: 0})
    r = move_one(b, src=23, die=5)  # 23 -> 18
    assert r is not None
    assert r.points[23] == 1
    assert r.points[18] == 1
    assert r.bar_count == 0 and r.opp_bar_count == 0

def test_hop_onto_own_stack():
    # non-bar src, my own checkers already at destination: stack up
    b = mk({13: 1, 12: 3})
    r = move_one(b, src=13, die=1)  # 13 -> 12
    assert r is not None
    assert r.points[13] == 0
    assert r.points[12] == 4

def test_hop_hits_a_blot():
    # lone opponent checker at destination: hit it, send it to their bar
    b = mk({8: 1, 6: -1})
    r = move_one(b, src=8, die=2)  # 8 -> 6
    assert r is not None
    assert r.points[8] == 0
    assert r.points[6] == 1          # the point is now mine
    assert r.opp_bar_count == 1      # their checker went to the bar

def test_hop_blocked_by_two_opponents():
    # two+ opponents hold the point: no move
    b = mk({8: 1, 6: -2})
    assert move_one(b, src=8, die=2) is None

def test_bar_entry_to_open_point():
    b = mk({18: 0}, bar=1)
    r = move_one(b, src=BAR_IDX, die=6)  # enter on the 18-point
    assert r is not None
    assert r.bar_count == 0
    assert r.points[18] == 1

def test_bar_entry_hits_a_blot():
    b = mk({21: -1}, bar=1)
    r = move_one(b, src=BAR_IDX, die=3)  # enter on 21, hitting a blot
    assert r is not None
    assert r.bar_count == 0
    assert r.points[21] == 1
    assert r.opp_bar_count == 1

def test_bar_entry_blocked():
    b = mk({21: -2}, bar=1)
    assert move_one(b, src=BAR_IDX, die=3) is None

def test_bearoff_blocked_by_checker_on_bar():
    # Over-full board (15 home + 1 on bar) so the home sum ALONE would pass.
    # This isolates the bar-count clause of the gate.
    b = mk({5: 15}, bar=1)
    assert move_one(b, src=5, die=6) is None

def test_bearoff_blocked_when_not_all_home():
    # a checker outside the home board (index 10) fails the gate
    b = mk({5: 1, 10: 1})
    assert move_one(b, src=5, die=6) is None  # 5 -> off, but not all home

def test_bearoff_exact_roll():
    b = mk({5: 15})                 # all 15 on the 6-point, all home
    r = move_one(b, src=5, die=6)   # exact: die == src + 1
    assert r is not None
    assert r.points[5] == 14
    assert r.off_count == 1

def test_bearoff_overshoot_from_rearmost():
    # die overshoots and nothing sits further back -> legal
    b = mk({3: 15})                 # all on the 4-point; nothing at 4 or 5
    r = move_one(b, src=3, die=6)   # dst = -3, an overshoot
    assert r is not None
    assert r.points[3] == 14
    assert r.off_count == 1

def test_bearoff_overshoot_not_rearmost_is_illegal():
    # gate passes (all 15 home) but a checker sits further back (index 5),
    # so overshooting off the 4-point is illegal
    b = mk({3: 1, 5: 14})
    assert move_one(b, src=3, die=6) is None


# --- extend: the shape of the move tree, not per-move legality --------
# Each test feeds a board and an ORDERED dice list and asserts the exact
# set of (board, depth) leaves. Boards are kept tiny so the whole tree is
# enumerable by hand. Depth = dice consumed to reach that leaf.

def test_extend_full_consumption_nondouble():
    # One lone checker with two open landing spots: both dice must be
    # played, so the only leaf is at depth 2. Crucially the depth-1
    # waypoint (checker on 17) is NOT in the result — waypoints aren't
    # recorded.
    b = mk({23: 1})
    result = extend(b, [6, 5])       # 23 -> 17 -> 12
    assert result == {(mk({12: 1}), 2)}

def test_extend_full_consumption_doubles():
    # Doubles are just a length-4 dice list to extend; a clear runway
    # consumes all four, leaving one leaf at depth 4.
    b = mk({23: 1})
    result = extend(b, [2, 2, 2, 2])  # 23 -> 21 -> 19 -> 17 -> 15
    assert result == {(mk({15: 1}), 4)}

def test_extend_dead_root_returns_depth_zero():
    # Checker on the bar, entry point blocked: no legal first hop. extend
    # returns the untouched board at depth 0 (the raw material for the
    # "dance"), never an empty set.
    b = mk({21: -2}, bar=1)
    result = extend(b, [3])          # would enter on 21, but it's blocked
    assert result == {(b, 0)}

def test_extend_blockage_midtree_records_intermediate():
    # First die is playable, second is not: the intermediate board is a
    # leaf by blockage at depth 1 (not discarded, not pushed to depth 2).
    b = mk({8: 1, 0: -2})
    result = extend(b, [2, 6])       # 8 -> 6 ok; then 6 -> 0 is blocked
    assert result == {(mk({6: 1, 0: -2}), 1)}

def test_extend_bar_freezes_other_checkers():
    # A checker on the bar freezes everything else: with bar_count > 0 the
    # only candidate is BAR_IDX, so the movable checker on 13 is ignored
    # and the single leaf is the bar entry.
    b = mk({13: 1}, bar=1)
    result = extend(b, [6])          # enter on 18; 13 stays put
    assert result == {(mk({13: 1, 18: 1}), 1)}

def test_extend_branches_over_candidates():
    # Two independently movable checkers and a single die: the for-loop
    # over candidates yields two distinct depth-1 leaves.
    b = mk({23: 1, 6: 1})
    result = extend(b, [3])
    assert result == {
        (mk({20: 1, 6: 1}), 1),      # moved the back checker 23 -> 20
        (mk({23: 1, 3: 1}), 1),      # moved the front checker 6 -> 3
    }

def test_extend_is_order_sensitive():
    # The same dice in different orders reach different trees. Playing the
    # 4 first threads past the block and consumes both dice (depth 2);
    # playing the 2 first is blocked immediately (dead root, depth 0).
    b = mk({6: 1, 4: -2})
    assert extend(b, [4, 2]) == {(mk({0: 1, 4: -2}), 2)}  # 6 -> 2 -> 0
    assert extend(b, [2, 4]) == {(b, 0)}                  # 6 -> 4 blocked

def test_extend_consumes_both_dice_with_a_blocked_candidate():
    # Several candidates, one blocked for the first die, yet both dice get
    # consumed. My checkers on 13 and 5; opponent walls on 10 and 3.
    #   die 3: 13 -> 10 is BLOCKED (wall); 5 -> 2 is open, so only 5 plays.
    #   die 2 (candidates now 13 and 2 -- re-selection): 13 -> 11 or 2 -> 0.
    # Both branches reach depth 2, so no die is wasted.
    b = mk({13: 1, 5: 1, 10: -2, 3: -2})
    result = extend(b, [3, 2])
    assert result == {
        (mk({11: 1, 2: 1, 10: -2, 3: -2}), 2),   # ...then 13 -> 11
        (mk({13: 1, 0: 1, 10: -2, 3: -2}), 2),   # ...then 2 -> 0
    }


# --- generate_moves: selecting among extend's leaves ------------------
# The rules layer. It strips depth, applies the max-play and larger-die
# laws across the two dice orderings, and returns the chosen afterstates
# (or set() for the dance). Boards are minimal; expectations hand-derived.

def test_generate_moves_doubles_play_all_four():
    # Doubles become a length-4 dice list. A clear runway plays all four.
    assert generate_moves(mk({23: 1}), (2, 2)) == {mk({15: 1})}  # 23->21->19->17->15

def test_generate_moves_doubles_partial_when_blocked():
    # Doubles that can't all be played: 23 -> 21 -> 19 -> 17 runs fine, but
    # the fourth 2 (17 -> 15) is blocked by an opponent point. Max-play keeps
    # the 3-deep play. The block is a made point, so this doesn't depend on
    # any (illegal) checker-count trick.
    b = mk({23: 1, 15: -2})
    assert generate_moves(b, (2, 2)) == {mk({17: 1, 15: -2})}  # 23->21->19->17

def test_generate_moves_larger_die_rule():
    # Each die is playable alone but not both (index 3 is walled). Both
    # orderings stall at depth 1, so the larger-die law forces the 6:
    # the result is the 6-first board (checker on 6), NOT the 3-first (on 9).
    result = generate_moves(mk({12: 1, 3: -2}), (6, 3))
    assert result == {mk({6: 1, 3: -2})}
    assert mk({9: 1, 3: -2}) not in result

def test_generate_moves_equal_depth_merges_both_orderings():
    # Both orderings play both dice (depth 2) but reach different boards,
    # so both are offered. Here the order decides which blot gets hit:
    # 3-then-1 hits the blot on 1; 1-then-3 hits the blot on 3.
    result = generate_moves(mk({4: 1, 3: -1, 1: -1}), (1, 3))
    assert result == {
        mk({0: 1, 3: -1}, opp_bar=1),   # played 3 first, hit the blot on 1
        mk({0: 1, 1: -1}, opp_bar=1),   # played 1 first, hit the blot on 3
    }

def test_generate_moves_dance_returns_empty_set():
    # On the bar with both entry points walled: no legal play in either
    # ordering. The dance is the empty set, not {(board, 0)}.
    assert generate_moves(mk({18: -2, 21: -2}, bar=1), (6, 3)) == set()

def test_generate_moves_maxplay_forces_higher_first():
    # Only the 4-first ordering uses both dice (2-first is blocked at once),
    # so max-play discards the shallow 2-first pile entirely.
    assert generate_moves(mk({6: 1, 4: -2}), (4, 2)) == {mk({0: 1, 4: -2})}

def test_generate_moves_maxplay_forces_lower_first():
    # The counterintuitive one: only the 2-first ordering uses both dice,
    # so you are FORCED to play the smaller die first — max-play overrides
    # the larger-die preference.
    assert generate_moves(mk({6: 1, 2: -2}), (4, 2)) == {mk({0: 1, 2: -2})}

def test_generate_moves_maxplay_drops_short_play():
    # A pile with mixed depths: one line plays both dice, another stalls
    # after one. Max-play keeps only the depth-2 plays and drops the
    # depth-1 stall (the checker left on 10).
    result = generate_moves(mk({10: 1, 6: 1, 8: -2, 1: -2}), (3, 2))
    assert result == {
        mk({5: 1, 6: 1, 8: -2, 1: -2}),   # 10->7->5
        mk({7: 1, 4: 1, 8: -2, 1: -2}),   # 10->7, then 6->4
    }
    assert mk({10: 1, 3: 1, 8: -2, 1: -2}) not in result  # the short play