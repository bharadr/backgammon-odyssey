from engine.board import starting_board
from coach.features import features
from tests.test_moves import mk


def test_features_of_starting_board():
    f = features(starting_board())
    # the start is symmetric, so both sides look identical
    for side in (f.me, f.opp):
        assert side.pips == 167
        assert side.point_counts == (0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0,
                                     5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
        assert side.blots == ()                  # no lone checkers
        assert side.points_made == (6, 8, 13, 24)
        assert side.stripped_points == (24,)     # the 24-point has exactly two
        assert side.stacked_points == (6, 13)    # 5 each on the 6- and 13-points
        assert side.anchors == (24,)             # the two back checkers = 24-anchor
        assert side.advanced_anchor == 24        # the only anchor
        assert side.prime_ranges == ()           # made points aren't contiguous
        assert side.longest_prime == 0
        assert side.home_board_made_points == (6,)   # only the 6-point is home (1-6)
        assert side.home_board_points == 1           # property: the count
        assert side.checkers_in_opponent_home == 2   # the two on the 24-point
        assert side.checkers_on_deep_points == 2     # both on the 24-point (deepest)
        assert side.on_bar == 0 and side.borne_off == 0
    assert f.pip_lead == 0


def test_features_of_a_mixed_board():
    # me: blots on 7/24, made 5/6 (home) + 21, a bar checker, 2 borne off.
    # opp: two checkers on the board (a blot + a made point), 1 on the bar, 3 off.
    b = mk({6: 1, 4: 2, 5: 3, 20: 2, 23: 1, 10: -1, 15: -2},
           bar=1, opp_bar=1, off=2, opp_off=3)
    f = features(b)

    # point_counts is the ground truth the rest is derived from (my checkers only)
    assert f.me.point_counts == (0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1)
    assert f.me.pips == 126                       # 101 on points + 25 on the bar
    assert f.me.blots == (7, 24)                  # idx 6 and 23, in 1-24 numbering
    assert f.me.points_made == (5, 6, 21)         # idx 4, 5, 20 (blots don't count)
    assert f.me.stripped_points == (5, 21)        # exactly two on the 5- and 21-points
    assert f.me.stacked_points == ()              # nothing 4+
    assert f.me.anchors == (21,)                  # made point in the opp home (19-24)
    assert f.me.advanced_anchor == 21
    assert f.me.prime_ranges == ((5, 6),)         # 5 and 6 are consecutive
    assert f.me.longest_prime == 2
    assert f.me.home_board_made_points == (5, 6)  # idx 4 and 5 (in 0-5)
    assert f.me.home_board_points == 2            # property: the count
    assert f.me.checkers_in_opponent_home == 3    # idx 20 (2) + idx 23 (1)
    assert f.me.checkers_on_deep_points == 1      # only idx 23 (the 24-point)
    assert f.me.on_bar == 1 and f.me.borne_off == 2

    # opponent's side, computed from flip(b), in the OPPONENT's own numbering
    assert f.opp.pips == 57
    assert f.opp.blots == (14,)                   # opp's lone checker (b idx 10)
    assert f.opp.points_made == (9,)              # opp's made point (b idx 15)
    assert f.opp.stripped_points == (9,)          # that made point has exactly two
    assert f.opp.anchors == () and f.opp.advanced_anchor is None
    assert f.opp.prime_ranges == ()
    assert f.opp.home_board_made_points == () and f.opp.home_board_points == 0
    assert f.opp.checkers_in_opponent_home == 0
    assert f.opp.on_bar == 1 and f.opp.borne_off == 3

    assert f.pip_lead == f.opp.pips - f.me.pips == -69   # I'm well behind


def test_prime_detection_finds_runs_and_the_longest():
    # a 4-prime (points 3-6) and a separate 2-prime (points 9-10); the lone
    # made point on 12 must NOT extend or start a run.
    b = mk({2: 2, 3: 2, 4: 2, 5: 2, 8: 2, 9: 2, 11: 2, 20: -2})
    f = features(b)
    assert f.me.points_made == (3, 4, 5, 6, 9, 10, 12)
    assert f.me.prime_ranges == ((3, 6), (9, 10))   # 12 is isolated -> no run
    assert f.me.longest_prime == 4


def test_longest_prime_when_the_longest_run_is_not_first():
    # a 2-prime (3-4) BEFORE a 4-prime (8-11): longest_prime must scan all runs,
    # not just take the first. A `_range_length(primes[0])` bug would give 2.
    b = mk({2: 2, 3: 2, 7: 2, 8: 2, 9: 2, 10: 2, 20: -2})
    f = features(b)
    assert f.me.prime_ranges == ((3, 4), (8, 11))
    assert f.me.longest_prime == 4


def test_advanced_anchor_picks_the_most_advanced_of_several():
    # anchors on the 20- AND 24-points. advanced_anchor is the LOWEST-numbered
    # (most advanced); with one anchor min/max are indistinguishable, so this
    # is the board that actually pins the field's meaning.
    b = mk({19: 2, 23: 2, 5: -2})   # my points 20 and 24 made; an opp point
    f = features(b)
    assert f.me.anchors == (20, 24)
    assert f.me.advanced_anchor == 20


def test_near_bear_off_side_is_all_empty_defaults():
    # one lone checker on the 3-point, 14 borne off: `made` is empty, so this
    # exercises the _prime_ranges empty-guard and every default together.
    b = mk({2: 1, 20: -2}, off=14)
    f = features(b)
    assert f.me.point_counts == (0, 0, 1) + (0,) * 21
    assert f.me.pips == 3
    assert f.me.blots == (3,)
    assert f.me.points_made == ()
    assert f.me.stripped_points == () and f.me.stacked_points == ()
    assert f.me.anchors == () and f.me.advanced_anchor is None
    assert f.me.prime_ranges == () and f.me.longest_prime == 0
    assert f.me.home_board_made_points == () and f.me.home_board_points == 0
    assert f.me.checkers_in_opponent_home == 0 and f.me.checkers_on_deep_points == 0
    assert f.me.on_bar == 0 and f.me.borne_off == 14
