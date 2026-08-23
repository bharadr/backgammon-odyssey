import random

import gradio as gr

from engine.board import Board, starting_board
from engine.moves import generate_move_paths
from engine.move_builder import apply_hops
from coach.gnubg_provider import GnubgProvider
from coach.game_app import (new_state, _advance, _review_submission, _render,
                            build_app)


def _trivial_opponent(board, dice, afterstates):
    """Deterministic Agent: always the sorted-first legal afterstate."""
    return sorted(afterstates)[0]


# --- new_state ---------------------------------------------------------------

def test_new_state_starts_your_turn_from_the_opening_position():
    st = new_state(random.Random(0))
    assert st["board"] == starting_board()
    assert st["phase"] == "build" and st["hops"] == () and st["source"] is None
    assert st["paths"] == generate_move_paths(st["board"], st["dice"])
    assert st["verdict"] == "" and st["coach"] == "" and len(st["log"]) == 1


# --- _advance: opponent plays, view returns to YOUR seat --------------------

def test_advance_plays_opponent_and_hands_the_board_back_in_your_seat():
    rng = random.Random(3)
    st = {**new_state(rng), "hops": (), "phase": "dance_me"}   # you forfeit, opp replies
    before_moves = len(st["log"])
    out = _advance(st, rng, _trivial_opponent)

    assert out["phase"] in ("build", "dance_me")               # never gameover here
    assert out["hops"] == () and out["source"] is None         # your move is reset
    assert len(out["log"]) > before_moves                      # opponent's play was logged
    if out["phase"] == "build":                                # board is yours again:
        assert out["paths"] == generate_move_paths(out["board"], out["dice"])


def test_advance_detects_an_opponent_win_and_ends_the_game():
    # Your afterstate: opponent has 14 off and its last checker on its ace point
    # (your 24-point, index 23) -- flipped to its seat, any roll bears it off.
    pts = [0] * 24
    pts[0] = 15          # your 15 checkers, not borne off -> you haven't won
    pts[23] = -1         # opponent's last checker, on its ace
    after = Board(points=tuple(pts), bar_count=0, opp_bar_count=0,
                  off_count=0, opp_off_count=14)
    st = {**new_state(random.Random(0)), "board": after, "hops": ()}

    out = _advance(st, random.Random(0), _trivial_opponent)
    assert out["phase"] == "gameover"
    assert "Opponent wins" in out["verdict"]


# --- _review_submission: grades your play, spots your win -------------------

def test_review_submission_grades_the_move_and_enters_review():
    prov = GnubgProvider(plies=0)
    board, dice = starting_board(), (3, 1)
    path = sorted(generate_move_paths(board, dice))[0]
    st = {**new_state(random.Random(0)), "board": board, "dice": dice, "hops": path}

    out = _review_submission(st, prov)
    assert out["phase"] == "review"
    assert out["verdict"].startswith("### Verdict")
    assert out["evidence"] is not None


def test_review_submission_detects_your_win():
    # You have 14 off and one checker on your ace; bearing it off wins.
    pts = [0] * 24
    pts[0] = 1
    pts[23] = -15        # opponent parked on its ace (your 24) -- a legal 15/15 split
    board = Board(points=tuple(pts), bar_count=0, opp_bar_count=0,
                  off_count=14, opp_off_count=0)
    dice = (1, 3)
    path = sorted(generate_move_paths(board, dice))[0]
    assert apply_hops(board, path).off_count == 15            # this move bears off the last
    st = {**new_state(random.Random(0)), "board": board, "dice": dice, "hops": path}

    out = _review_submission(st, GnubgProvider(plies=0))
    assert out["phase"] == "gameover"
    assert "You win" in out["verdict"]


# --- rendering + construction ----------------------------------------------

def test_render_emits_one_value_per_output_and_passes_state_through():
    st = new_state(random.Random(0))
    rendered = _render(st)
    assert len(rendered) == 10           # must match build_app's `out` list length
    assert rendered[5] is st             # the gr.State passthrough


def test_build_app_constructs_with_stubs():
    app = build_app(provider=GnubgProvider(plies=0), llm=lambda system, user: "x",
                    rng=random.Random(0), opponent=_trivial_opponent)
    assert isinstance(app, gr.Blocks)
