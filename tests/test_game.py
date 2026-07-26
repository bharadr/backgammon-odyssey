import random

import pytest

from engine.board import is_valid, starting_board
from engine.game import (
    Outcome,
    classify_win,
    is_win,
    play_game,
    play_turn,
)
from engine.moves import generate_moves
from agent.random_agent import random_agent
from tests.test_moves import mk


# --- classify_win: single / gammon / backgammon -----------------------
# Boards are from the WINNER's perspective (off_count == 15); the loser is
# the opponent (negative points, opp_* counts).

def test_classify_single():
    # loser has borne off at least one checker -> at most 1 point
    b = mk({23: -12}, off=15, opp_off=3)
    assert classify_win(b) == Outcome.SINGLE

def test_classify_gammon():
    # loser bore off none, but is out of my home and off the bar
    b = mk({23: -15}, off=15)
    assert classify_win(b) == Outcome.GAMMON

def test_classify_backgammon_via_home():
    # loser bore off none and still has a checker in my home board (0-5)
    b = mk({3: -15}, off=15)
    assert classify_win(b) == Outcome.BACKGAMMON

def test_classify_backgammon_via_bar():
    # loser bore off none and still has a checker on the bar
    b = mk({23: -14}, off=15, opp_bar=1)
    assert classify_win(b) == Outcome.BACKGAMMON

def test_classify_single_short_circuits_backgammon():
    # loser bore off a checker AND has one in my home: single still wins,
    # because bearing off any checker caps the result at 1 point. This
    # guards the ordering of the checks.
    b = mk({3: -14}, off=15, opp_off=1)
    assert classify_win(b) == Outcome.SINGLE


# --- is_win -----------------------------------------------------------

def test_is_win_true_at_fifteen_off():
    assert is_win(mk(off=15))

def test_is_win_false_below_fifteen():
    assert not is_win(mk({0: 1}, off=14))


# --- play_turn: the dance vs. the move --------------------------------

def _explode(board, dice, afterstates):
    raise AssertionError("agent must not be consulted on a dance")

def test_play_turn_dance_forfeits_without_consulting_agent():
    # on the bar with both entry points walled: no legal play
    b = mk({18: -2, 21: -2}, bar=1)
    assert play_turn(b, (6, 3), _explode) == (b, False)

def test_play_turn_returns_agents_choice():
    # the move path must return exactly what the agent picked, flagged True.
    # The agent here uses min() only because we need a *deterministic* rule
    # whose choice we can predict and assert against -- the specific board it
    # picks is irrelevant, any fixed rule would do.
    board = starting_board()
    dice = (6, 3)
    deterministic_pick = lambda b, dice, afterstates: min(afterstates)
    expected = min(generate_moves(board, dice))
    assert play_turn(board, dice, deterministic_pick) == (expected, True)


# --- play_game: full-game integration ---------------------------------

def test_play_game_terminates_with_legal_winner():
    for seed in range(20):
        rng = random.Random(seed)
        agent = random_agent(rng)
        result = play_game((agent, agent), rng)
        assert result.winner in (0, 1)
        assert isinstance(result.outcome, Outcome)
        # the terminal position must be a legal board with the winner
        # (whose perspective it's in) having borne off all 15 checkers
        assert is_valid(result.final_board)
        assert result.final_board.off_count == 15

def test_play_game_hits_safety_cap():
    # a one-turn cap can't finish a game, so the guard must fire
    rng = random.Random(0)
    agent = random_agent(rng)
    with pytest.raises(RuntimeError):
        play_game((agent, agent), rng, max_turns=1)
