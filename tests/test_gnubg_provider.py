import pytest

from engine.board import Board, starting_board
from coach.analysis import OutcomeDist
from coach.scoring import cubeless_equity
from coach.gnubg_provider import (
    GnubgProvider,
    _to_mover_perspective,
    board_from_gnubg,
    board_from_position_id,
    board_to_gnubg,
    position_id,
)
from engine.moves import generate_moves
from tests.test_board import midgame_boards
from tests.test_moves import mk

# The canonical gnubg Position ID for the standard starting position.
START_ID = "4HPwATDgc/ABMA"

# gnubg's deterministic 0-ply read of the opening 3-1 best play -- 8/5 6/5,
# make the 5-point -- oriented to the mover.
FIVE_POINT_DIST = OutcomeDist(win=0.5515, win_gammon=0.1735, win_backgammon=0.0127,
                              lose_gammon=0.1244, lose_backgammon=0.00504)
FIVE_POINT_EQUITY = 0.1597
# the afterstate reached by playing 8/5 6/5 from the opening, mover's perspective
FIVE_POINT_AFTER = Board(points=(-2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5,
                                 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2),
                         bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0)


# --- Board <-> gnubg representation ------------------------------------

def test_position_id_conversion_from_start():
    assert position_id(starting_board()) == START_ID
    assert board_from_position_id(START_ID) == starting_board()


def test_board_gnubg_roundtrip():
    # our Board -> gnubg [2][25] -> our Board is the identity for legal boards
    for b in [starting_board(), *midgame_boards()]:
        assert board_from_gnubg(board_to_gnubg(b)) == b


# A position exercising the edge cases the start/midgame boards don't: both
# sides have checkers on the bar AND borne off. Deliberately asymmetric in
# every slot -- distinct points, distinct bar counts (2 vs 1), distinct off
# counts (2 vs 3) -- so a me/opp swap or a mirror-offset error can't hide.
_BAR_OFF_BOARD = Board(
    points=(0, 0, 1, 2, 3, 5,        # mine: 1+2+3+5 = 11 on points
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            -4, -3, -2, -2, 0, 0),   # opp: 4+3+2+2 = 11 on points
    bar_count=2, opp_bar_count=1,    # asymmetric -> catches a bar me/opp swap
    off_count=2, opp_off_count=3,    # totals: me 11+2+2=15, opp 11+1+3=15
)

def test_bar_and_bearoff_maps_to_gnubg():
    # absolute anchor of the mapping (not just a round-trip): the bar lives at
    # index 24, opponent points mirror to 23-j, and off checkers are implied.
    expected = [
        [0, 0, 1, 2, 3, 5] + [0] * 18 + [2],   # me: from my ace point, bar at 24
        [0, 0, 2, 2, 3, 4] + [0] * 18 + [1],   # opp: from their ace point (our 23-j)
    ]
    assert board_to_gnubg(_BAR_OFF_BOARD) == expected
    assert board_from_gnubg(expected) == _BAR_OFF_BOARD   # off rebuilt as 15 - on - bar

def test_bar_and_bearoff_roundtrips_through_gnubg_codec():
    # end-to-end through gnubg's own encoder/decoder (it must not choke on or
    # renormalise a bar/bear-off position)
    assert board_from_position_id(position_id(_BAR_OFF_BOARD)) == _BAR_OFF_BOARD


# --- cubeless_equity scoring policy (separate from the data type) ------

def test_equity_even_position_is_zero():
    assert cubeless_equity(OutcomeDist(0.5, 0.0, 0.0, 0.0, 0.0)) == 0.0

def test_equity_matches_hand_value():
    # a full distribution (all five terms, not just the win term); the
    # expected value is gnubg's own equity for this play.
    assert abs(cubeless_equity(FIVE_POINT_DIST) - FIVE_POINT_EQUITY) < 1e-3


# --- evaluation (the path SkillAgent/coach actually use) ---------------

def test_to_mover_perspective_flips_opponent_probs():
    # afterstate probs arrive opponent-on-roll; the flip swaps win<->lose and
    # my/their gammon+backgammon. Distinct values so no mis-mapping can hide.
    opp = (0.60, 0.20, 0.05, 0.15, 0.03)  # opp: win, win_g, win_bg, lose_g, lose_bg
    d = _to_mover_perspective(opp)
    assert d.win == pytest.approx(0.40)             # 1 - opp_win
    assert d.win_gammon == pytest.approx(0.15)      # opponent's lose_gammon
    assert d.win_backgammon == pytest.approx(0.03)  # opponent's lose_backgammon
    assert d.lose_gammon == pytest.approx(0.20)     # opponent's win_gammon
    assert d.lose_backgammon == pytest.approx(0.05) # opponent's win_backgammon

def test_evaluate_afterstate_known_equity():
    # end-to-end through real gnubg: the 8/5 6/5 afterstate scores ~+0.16 to
    # the mover. (Deterministic 0-ply; small slack for gnubg version drift.)
    outcome = GnubgProvider(plies=0).evaluate_afterstate(FIVE_POINT_AFTER)
    assert abs(cubeless_equity(outcome) - FIVE_POINT_EQUITY) < 0.01


# --- analyze (our move-gen + gnubg eval + our notation) ----------------

# an ASYMMETRIC (4,4) midgame position -- the kind that crashed the old
# best_move-based analyze (it analysed the opponent, producing afterstates our
# engine rejected). generate_moves gives 56 legal plays here.
_ASYM = Board(points=(-1, 0, 0, 0, 0, 5, 0, 3, 0, -1, 0, -5,
                      5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2),
              bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0)

def test_analyze_opening_31():
    result = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    assert len(result.moves) == len(generate_moves(starting_board(), (3, 1)))
    assert set(result.best.notation.split()) == {"8/5", "6/5"}   # make the 5-point
    assert abs(result.best.equity - FIVE_POINT_EQUITY) < 0.01
    eqs = [m.equity for m in result.moves]
    assert eqs == sorted(eqs, reverse=True)                      # ranked best-first

def test_analyze_asymmetric_position_is_legal_and_ranked():
    result = GnubgProvider(plies=0).analyze(_ASYM, (4, 4))
    legal = generate_moves(_ASYM, (4, 4))
    # every analysed afterstate is one of OUR legal moves -- the regression the
    # old version failed (it returned boards our engine considered illegal).
    assert {m.after_state for m in result.moves} == legal
    eqs = [m.equity for m in result.moves]
    assert eqs == sorted(eqs, reverse=True)
    assert all(result.equity_loss(m) >= 0 for m in result.moves)
    assert all(m.notation for m in result.moves)                 # describe_move never raised

def test_analyze_dance_has_no_moves():
    dancing = mk({18: -2, 21: -2}, bar=1)   # on the bar, both entry points walled
    result = GnubgProvider(plies=0).analyze(dancing, (6, 3))
    assert result.moves == ()

def test_analyze_two_ply_differs_from_zero_ply():
    # deeper search re-evaluates: same set of legal moves, but the equities
    # shift (confirms plies actually flows through to gnubg's evaluation).
    board, dice = starting_board(), (3, 1)
    eq0 = {m.after_state: m.equity for m in GnubgProvider(plies=0).analyze(board, dice).moves}
    eq2 = {m.after_state: m.equity for m in GnubgProvider(plies=2).analyze(board, dice).moves}
    assert eq0.keys() == eq2.keys()   # move generation is ply-independent
    assert eq0 != eq2                 # but the evaluations differ with depth
