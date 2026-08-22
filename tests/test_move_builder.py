import pytest

from engine.board import starting_board
from engine.moves import BAR_IDX, OFF, generate_moves, generate_move_paths
from engine.move_builder import (matching_paths, legal_sources, legal_destinations,
                                 choose_hop, is_complete, apply_hops)
from tests.test_moves import mk

# --- hand-built paths (no engine needed) -------------------------------
# Hops are (src, dst, die) in INDEX space (index = point - 1). The classic
# "make the 5-point" on a 3-1: index 7 (8-point) --3--> index 4 (5-point), and
# index 5 (6-point) --1--> index 4. Both orderings reach the same afterstate, so
# both paths exist -- that's what lets either checker move first.
H_83 = (7, 4, 3)                 # "8/5"
H_61 = (5, 4, 1)                 # "6/5"
FIVE_POINT = {(H_83, H_61), (H_61, H_83)}


# --- matching_paths -----------------------------------------------------

def test_matching_paths_empty_prefix_matches_everything():
    assert matching_paths(FIVE_POINT, ()) == FIVE_POINT

def test_matching_paths_narrows_on_a_prefix():
    assert matching_paths(FIVE_POINT, (H_83,)) == {(H_83, H_61)}

def test_matching_paths_none_when_prefix_is_absent():
    assert matching_paths(FIVE_POINT, ((9, 9, 9),)) == set()


# --- legal_sources ------------------------------------------------------

def test_legal_sources_lists_every_first_hop_source():
    assert legal_sources(FIVE_POINT, ()) == {7, 5}      # either checker may go first

def test_legal_sources_narrows_after_a_hop():
    assert legal_sources(FIVE_POINT, (H_83,)) == {5}    # only the 6/5 remains

def test_legal_sources_empty_when_complete():
    assert legal_sources(FIVE_POINT, (H_83, H_61)) == set()


# --- legal_destinations -------------------------------------------------

def test_legal_destinations_for_a_source():
    assert legal_destinations(FIVE_POINT, (), 7) == {4}
    assert legal_destinations(FIVE_POINT, (), 5) == {4}

def test_legal_destinations_filters_by_source():
    # the empty prefix has two first-hops, but only source 5 leads anywhere now
    assert legal_destinations(FIVE_POINT, (), 99) == set()   # no checker there

def test_legal_destinations_after_a_hop():
    assert legal_destinations(FIVE_POINT, (H_83,), 5) == {4}

def test_legal_destinations_empty_when_complete():
    assert legal_destinations(FIVE_POINT, (H_83, H_61), 7) == set()


# --- choose_hop ---------------------------------------------------------

def test_choose_hop_returns_the_full_hop_with_its_die():
    assert choose_hop(FIVE_POINT, (), 7, 4) == H_83
    assert choose_hop(FIVE_POINT, (), 5, 4) == H_61

def test_choose_hop_after_a_hop():
    assert choose_hop(FIVE_POINT, (H_83,), 5, 4) == H_61

def test_choose_hop_raises_on_an_illegal_destination():
    with pytest.raises(ValueError):
        choose_hop(FIVE_POINT, (), 7, 3)               # 7 -> 3 isn't a next hop

def test_choose_hop_raises_on_a_source_with_no_hop():
    with pytest.raises(ValueError):
        choose_hop(FIVE_POINT, (), 99, 4)


# --- choose_hop: ambiguous bear-off (the determinism guard) -------------
# A checker on the 2-point (index 1) can bear off with the exact 2 OR overshoot
# with the 6; the other die then plays elsewhere. Two paths share the same
# (src=1, dst=OFF) first hop but with different dice -- choose_hop must pick a
# DETERMINISTIC one (the smallest die), never whatever set iteration yields.
BO_2, BO_6 = (1, OFF, 2), (1, OFF, 6)
AMBIG = {(BO_2, (3, OFF, 6)), (BO_6, (3, OFF, 2))}

def test_choose_hop_resolves_ambiguous_bearoff_to_the_smallest_die():
    assert legal_destinations(AMBIG, (), 1) == {OFF}
    assert choose_hop(AMBIG, (), 1, OFF) == BO_2        # smallest die, not set order


# --- bar entry ----------------------------------------------------------

def test_bar_is_a_valid_source():
    enter = (BAR_IDX, 18, 6)
    paths = {(enter, (5, 2, 3))}
    assert legal_sources(paths, ()) == {BAR_IDX}
    assert choose_hop(paths, (), BAR_IDX, 18) == enter


# --- is_complete --------------------------------------------------------

def test_is_complete_tracks_the_max_play_length():
    assert not is_complete(FIVE_POINT, ())
    assert not is_complete(FIVE_POINT, (H_83,))
    assert is_complete(FIVE_POINT, (H_83, H_61))


# --- apply_hops (and undo) ----------------------------------------------

def test_apply_hops_folds_moves_onto_the_board():
    board = mk({7: 1, 5: 1})                            # a checker on the 8- and 6-points
    assert apply_hops(board, (H_83, H_61)) == mk({4: 2})   # both land on the 5-point

def test_apply_hops_partial_is_undo():
    board = mk({7: 1, 5: 1})
    assert apply_hops(board, (H_83,)) == mk({4: 1, 5: 1})  # only the first hop -- i.e. undo
    assert apply_hops(board, ()) == board                  # nothing applied


# --- end-to-end against the real engine ---------------------------------

def test_click_through_reconstructs_a_legal_play():
    # Drive the whole state machine on real paths: repeatedly pick a source and a
    # destination until complete, then check the built play is legal.
    for dice in [(3, 1), (6, 6), (5, 2), (6, 5)]:
        board = starting_board()
        paths = generate_move_paths(board, dice)
        hops = ()
        while not is_complete(paths, hops):
            src = min(legal_sources(paths, hops))
            dst = min(legal_destinations(paths, hops, src))
            hops += (choose_hop(paths, hops, src, dst),)
        assert hops in paths                              # built a real legal path
        assert apply_hops(board, hops) in generate_moves(board, dice)   # legal afterstate
