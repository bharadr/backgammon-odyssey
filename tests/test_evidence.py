import pytest

from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.evidence import build_evidence, _tuple_delta, _point_shifts, _side_delta
from coach.features import SideFeatures
from tests.test_moves import mk


# --- hand-built analysis (no gnubg) ------------------------------------

def _dist(win: float) -> OutcomeDist:
    return OutcomeDist(win, 0.0, 0.0, 0.0, 0.0)

def _move(board, equity, outcome=None, notation=None):
    return MoveAnalysis(after_state=board, outcome=outcome or _dist((equity + 1) / 2),
                        equity=equity, notation=notation)

def _analysis(moves, dice=(3, 1)):
    return Analysis(position=starting_board(), dice=dice, moves=tuple(moves))

# four distinct afterstates, ranked best-first by equity
B1, B2, B3, B4 = mk({5: 2}), mk({6: 2}), mk({7: 2}), mk({8: 2})

def _ranked():
    return _analysis([_move(B1, 0.3, notation="b1"), _move(B2, 0.1, notation="b2"),
                      _move(B3, -0.2, notation="b3"), _move(B4, -0.5, notation="b4")])


# --- rank / loss / alternatives ----------------------------------------

def test_chosen_is_the_best_play():
    ev = build_evidence(_ranked(), B1)
    assert not ev.is_dance
    assert ev.chosen.rank == 1 and ev.chosen.of_n == 4
    assert ev.chosen.equity_loss == 0.0
    assert ev.best.notation == "b1"
    # best and chosen are the same play -> neither repeats in alternatives
    assert [a.notation for a in ev.alternatives] == ["b2", "b3", "b4"]


def test_chosen_blunder_gets_rank_and_positive_loss():
    ev = build_evidence(_ranked(), B3)
    assert ev.chosen.rank == 3
    assert ev.chosen.equity_loss == pytest.approx(0.5)     # 0.3 - (-0.2)
    assert ev.best.notation == "b1" and ev.best.equity_loss == 0.0


def test_alternatives_exclude_best_and_chosen_and_respect_top_n():
    ev = build_evidence(_ranked(), B3, top_n=5)            # window = all 4
    assert [a.notation for a in ev.alternatives] == ["b2", "b4"]
    assert [a.equity_loss for a in ev.alternatives] == pytest.approx([0.2, 0.8])
    # a narrower window drops plays outside the top-n (b4 falls away)
    ev2 = build_evidence(_ranked(), B3, top_n=2)
    assert [a.notation for a in ev2.alternatives] == ["b2"]


def test_chosen_outside_top_n_still_shows_as_chosen_with_top_alternatives():
    # a big blunder ranked below the window: it still populates `chosen`, and the
    # alternatives are just the top plays (best excluded), not the chosen play.
    a = _analysis([_move(mk({2 + i: 2}), 0.5 - 0.1 * i, notation=f"m{i}") for i in range(6)])
    ev = build_evidence(a, mk({7: 2}), top_n=5)            # chose m5 (rank 6 of 6)
    assert ev.chosen.rank == 6 and ev.chosen.of_n == 6
    assert [x.notation for x in ev.alternatives] == ["m1", "m2", "m3", "m4"]


def test_chosen_not_among_legal_moves_raises():
    with pytest.raises(ValueError):
        build_evidence(_ranked(), mk({12: 2}))            # never analysed


def test_dance_yields_no_plays():
    ev = build_evidence(_analysis([], dice=(6, 3)), mk({1: 1}))  # chosen ignored on the dance
    assert ev.is_dance
    assert ev.roll == (6, 3)
    assert ev.best is None and ev.chosen is None
    assert ev.alternatives == ()
    assert ev.outcome_delta is None and ev.feature_delta is None


# --- deltas (chosen - best) --------------------------------------------

def test_outcome_delta_is_chosen_minus_best():
    best_out = OutcomeDist(0.60, 0.20, 0.05, 0.10, 0.02)
    chosen_out = OutcomeDist(0.55, 0.15, 0.03, 0.14, 0.04)
    a = _analysis([_move(B1, 0.3, outcome=best_out), _move(B2, 0.1, outcome=chosen_out)])
    d = build_evidence(a, B2).outcome_delta
    assert d.win == pytest.approx(-0.05)
    assert d.win_gammon == pytest.approx(-0.05)
    assert d.win_backgammon == pytest.approx(-0.02)
    assert d.lose_gammon == pytest.approx(0.04)      # chosen loses more gammons
    assert d.lose_backgammon == pytest.approx(0.02)


def test_feature_delta_captures_a_hit_and_the_point_shift():
    # best play makes the 5-point AND hits (opp to the bar); chosen leaves a
    # lone checker on the 5-point and does not hit.
    best_after = mk({4: 2}, opp_bar=1)     # my 5-point (idx 4) made; opp on bar
    chosen_after = mk({4: 1})              # my 5-point a blot; no hit
    a = _analysis([_move(best_after, 0.3, notation="best"),
                   _move(chosen_after, 0.1, notation="chosen")])
    fd = build_evidence(a, chosen_after).feature_delta

    # my structure: chosen swaps a made point for a blot on the 5-point
    assert fd.me.points_made.removed == (5,) and fd.me.points_made.added == ()
    assert fd.me.blots.added == (5,) and fd.me.blots.removed == ()
    assert fd.me.point_shifts == ((5, -1),)          # 2 -> 1 checker on the 5-point
    assert fd.me.pips == -5                           # one fewer checker, 5 pips each
    # the opponent: the chosen play forgoes the hit, so one fewer on the bar
    assert fd.opp.on_bar == -1
    assert fd.opp.point_shifts == ()
    # the only arithmetic in the packaging layer: pip-lead delta (chosen - best)
    assert fd.pip_lead == -20                         # best hits (+25 opp pips) & keeps a checker


# --- helper units (the parts with real logic, tested in isolation) -----

def test_tuple_delta_diffs_as_sets_preserving_original_order():
    d = _tuple_delta((1, 2, 3), (2, 4))
    assert d.removed == (1, 3)                     # in best, not chosen (best's order)
    assert d.added == (4,)                         # in chosen, not best
    assert bool(d) is True

def test_tuple_delta_is_empty_and_falsy_when_the_sets_match():
    d = _tuple_delta((5, 6), (6, 5))               # same set, different order
    assert d.added == () and d.removed == ()
    assert bool(d) is False


def _counts(by_point: dict) -> tuple:
    c = [0] * 24
    for point, n in by_point.items():
        c[point - 1] = n
    return tuple(c)

def test_point_shifts_reports_changed_points_signed_in_1_24_numbering():
    best = _counts({5: 2, 8: 1})                   # 2 on the 5-point, 1 on the 8-point
    chosen = _counts({5: 1, 6: 1, 8: 1})           # moved one 5 -> 6; 8-point unchanged
    assert _point_shifts(best, chosen) == ((5, -1), (6, 1))   # 8 unchanged -> omitted


def _side(**overrides) -> SideFeatures:
    base = dict(pips=0, point_counts=(0,) * 24, blots=(), points_made=(),
                stripped_points=(), stacked_points=(), anchors=(), advanced_anchor=None,
                home_board_made_points=(), prime_ranges=(), longest_prime=0,
                checkers_in_opponent_home=0, checkers_on_deep_points=0, on_bar=0, borne_off=0)
    base.update(overrides)
    return SideFeatures(**base)

def test_side_delta_wires_each_field_to_its_own_source():
    # Every source field carries a distinct signature, so any mis-wire (e.g.
    # anchors read from stacked_points) produces a wrong delta and fails here.
    best = _side(pips=100, point_counts=tuple(2 if i == 5 else 0 for i in range(24)),
                 blots=(1,), points_made=(6, 8), stripped_points=(6,), stacked_points=(8,),
                 anchors=(20,), home_board_made_points=(6,), prime_ranges=((6, 7),),
                 longest_prime=2, checkers_in_opponent_home=2, checkers_on_deep_points=1,
                 on_bar=1, borne_off=0)
    chosen = _side(pips=90, point_counts=tuple(1 if i in (4, 5) else 0 for i in range(24)),
                   blots=(2,), points_made=(6, 8, 20), stripped_points=(), stacked_points=(8, 13),
                   anchors=(), home_board_made_points=(6, 3), prime_ranges=((5, 7),),
                   longest_prime=3, checkers_in_opponent_home=0, checkers_on_deep_points=0,
                   on_bar=0, borne_off=2)
    d = _side_delta(best, chosen)

    assert d.pips == -10
    assert d.point_shifts == ((5, 1), (6, -1))     # idx4 +1 (pt5), idx5 -1 (pt6)
    assert (d.blots.added, d.blots.removed) == ((2,), (1,))
    assert (d.points_made.added, d.points_made.removed) == ((20,), ())
    assert (d.stripped_points.added, d.stripped_points.removed) == ((), (6,))
    assert (d.stacked_points.added, d.stacked_points.removed) == ((13,), ())
    assert (d.anchors.added, d.anchors.removed) == ((), (20,))
    assert (d.home_board_made_points.added, d.home_board_made_points.removed) == ((3,), ())
    assert (d.prime_ranges.added, d.prime_ranges.removed) == (((5, 7),), ((6, 7),))
    assert d.longest_prime == 1
    assert d.checkers_in_opponent_home == -2
    assert d.checkers_on_deep_points == -1
    assert d.on_bar == -1
    assert d.borne_off == 2


# --- composes with a real analysis (end-to-end, through gnubg) ---------

def test_build_evidence_composes_with_real_analysis():
    from coach.gnubg_provider import GnubgProvider
    analysis = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    worst = analysis.moves[-1]                         # a deliberately bad legal play
    ev = build_evidence(analysis, worst.after_state)

    assert not ev.is_dance
    assert ev.best.rank == 1 and ev.best.equity_loss == 0.0
    assert ev.chosen.rank == len(analysis.moves)
    assert ev.chosen.equity_loss == pytest.approx(analysis.best.equity - worst.equity)
    # afterstate features are the coached player's (mover perspective, no flip)
    assert ev.chosen.features.me.pips > 0
    assert ev.feature_delta is not None
