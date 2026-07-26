import pytest

from engine.board import Board, starting_board
from engine.notation import describe_move
from tests.test_moves import mk


def test_describe_opening_move_two_checkers():
    # opening 3-1 played 8/5 6/5 (make the 5-point): idx7 3->2, idx5 5->4,
    # idx4 0->2. Two hops, no hits. Order of the two hops is cosmetic, so we
    # compare as a set of tokens.
    after = Board(
        points=(-2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5,
                5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2),
        bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0,
    )
    assert set(describe_move(starting_board(), after, (3, 1)).split()) == {"8/5", "6/5"}


def test_describe_bear_off_doubles():
    # all 15 on the 6-point (idx5); double 6s bear four checkers off
    before = Board(points=(0, 0, 0, 0, 0, 15) + (0,) * 18,
                   bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0)
    after = Board(points=(0, 0, 0, 0, 0, 11) + (0,) * 18,
                  bar_count=0, opp_bar_count=0, off_count=4, opp_off_count=0)
    assert describe_move(before, after, (6, 6)) == "6/off 6/off 6/off 6/off"


def test_describe_bar_entry_with_hit():
    # on the bar, opponent blot on idx20; entering with the 4 lands on 20 and
    # hits. The 2 can't be played, so the play is just the entry.
    before = Board(points=(0,) * 20 + (-1, 0, 0, 0),
                   bar_count=1, opp_bar_count=0, off_count=0, opp_off_count=0)
    after = Board(points=(0,) * 20 + (1, 0, 0, 0),
                  bar_count=0, opp_bar_count=1, off_count=0, opp_off_count=0)
    assert describe_move(before, after, (4, 2)) == "bar/21*"


def test_describe_move_through_a_hit():
    # the case that motivated the design: one checker on idx10 plays 4 then 6,
    # hitting a blot on idx6 as it passes through. Sequence is forced, so the
    # notation order is fixed.
    before = Board(points=(0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0) + (0,) * 12,
                   bar_count=0, opp_bar_count=0, off_count=0, opp_off_count=0)
    after = Board(points=(1,) + (0,) * 23,
                  bar_count=0, opp_bar_count=1, off_count=0, opp_off_count=0)
    assert describe_move(before, after, (4, 6)) == "11/7* 7/1"


def test_describe_move_raises_when_target_unreachable():
    # a checker on idx10 can't reach idx3 (7 pips) with two 1s (4 pips max),
    # so _find_path returns None and describe_move surfaces it as an error
    before = mk({10: 1})
    after = mk({3: 1})
    with pytest.raises(ValueError):
        describe_move(before, after, (1, 1))
