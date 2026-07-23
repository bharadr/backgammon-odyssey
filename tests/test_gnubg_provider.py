import pytest

from engine.board import Board, is_valid, starting_board
from coach.analysis import OutcomeDist
from coach.scoring import cubeless_equity
from coach.gnubg_provider import (
    GnubgProvider,
    _render,
    _to_mover_perspective,
    board_from_gnubg,
    board_from_position_id,
    board_to_gnubg,
    position_id,
)
from tests.test_board import midgame_boards

# The canonical gnubg Position ID for the standard starting position.
START_ID = "4HPwATDgc/ABMA"

# gnubg's deterministic 0-ply read of the opening 3-1 best play -- 8/5 6/5,
# make the 5-point -- oriented to the mover. Anchors both the scoring formula
# and the live provider.
FIVE_POINT_DIST = OutcomeDist(win=0.5515, win_gammon=0.1735, win_backgammon=0.0127,
                              lose_gammon=0.1244, lose_backgammon=0.00504)
FIVE_POINT_EQUITY = 0.1597
# distinct legal plays for an opening 3-1 (a move-generation fact, not eval)
OPENING_31_MOVE_COUNT = 16


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


# --- The provider on the opening 3-1 -----------------------------------

def test_provider_opening_31_best_is_the_five_point():
    result = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    assert len(result.moves) == OPENING_31_MOVE_COUNT   # every legal 3-1 play

    best = result.best
    assert best.notation == "8/5 6/5"                   # textbook best: make the 5-point
    # deterministic 0-ply values; tight band, small slack for gnubg version drift
    assert abs(best.equity - FIVE_POINT_EQUITY) < 0.01
    assert abs(best.outcome.win - FIVE_POINT_DIST.win) < 0.01

def test_provider_equity_agrees_with_scored_distribution():
    # gnubg's own equity (stored) and cubeless_equity of the flipped
    # distribution should match -- a guard against perspective/flip drift.
    result = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    for m in result.moves:
        assert abs(m.equity - cubeless_equity(m.outcome)) < 1e-2

def test_provider_moves_ranked_and_equity_loss():
    result = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    eqs = [m.equity for m in result.moves]
    assert eqs == sorted(eqs, reverse=True)          # ranked best-first
    assert result.best.equity == max(eqs)            # best really is the top

    losses = [result.equity_loss(m) for m in result.moves]
    assert losses[0] == 0.0                          # the best play loses nothing
    assert losses == sorted(losses)                  # loss grows down the ranking
    assert all(loss >= 0 for loss in losses)         # and is never negative

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


def test_render_notation_including_bar_and_off():
    # the one position-dependent bit of plumbing the opening 3-1 never hits:
    # gnubg encodes the bar as point 25 and a borne-off checker as point 0.
    assert _render((8, 5, 6, 5)) == "8/5 6/5"            # plain point-to-point
    assert _render((25, 20, 13, 11)) == "bar/20 13/11"   # bar entry
    assert _render((6, 0, 5, 0)) == "6/off 5/off"        # bear-off


def test_provider_afterstate_is_legal_and_mover_oriented():
    result = GnubgProvider(plies=0).analyze(starting_board(), (3, 1))
    after = result.best.after_state           # 8/5 6/5, in the mover's perspective
    assert is_valid(after)
    assert after.points[4] == 2               # 5-point made (mine)
    assert after.points[5] == 4               # 6-point: 5 -> 4
    assert after.points[7] == 2               # 8-point: 3 -> 2
