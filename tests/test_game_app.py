import random

import gradio as gr

from engine.board import Board, starting_board
from engine.moves import generate_move_paths
from engine.move_builder import apply_hops
from coach.gnubg_provider import GnubgProvider
from agent.skill_agent import SkillAgent
from coach.game_app import (new_state, _advance, _review_submission, _render, _charts,
                            _mirror_notation, build_app, DIFFICULTY, DEFAULT_LEVEL)


def _trivial_opponent(board, dice, afterstates):
    """Deterministic Agent: always the sorted-first legal afterstate."""
    return sorted(afterstates)[0]


# --- new_state ---------------------------------------------------------------

def test_new_state_starts_from_the_opening_position():
    st = new_state(random.Random(0))
    assert st["board"] == starting_board()
    assert st["hops"] == () and st["source"] is None
    assert st["verdict"] == "" and st["coach"] == "" and len(st["log"]) == 1
    assert st["verdicts"] == [] and st["stats"] == []
    if st["phase"] == "build":                       # you won the opening
        assert st["paths"] == generate_move_paths(st["board"], st["dice"])
    else:                                            # opponent opens (resolved by new_game)
        assert st["phase"] == "opp_first"


def test_opening_roll_is_never_doubles_and_first_player_is_random():
    first = []
    for seed in range(200):
        st = new_state(random.Random(seed))
        assert st["dice"][0] != st["dice"][1]        # opening can never be doubles
        first.append(st["phase"] == "build")         # did you go first?
    assert 0.3 < sum(first) / len(first) < 0.7       # roughly a 50/50 coin flip


def test_opponent_opening_is_resolved_into_your_turn():
    opp_first = next(st for s in range(50)
                     if (st := new_state(random.Random(s)))["phase"] == "opp_first")
    out = _advance(opp_first, random.Random(0), _trivial_opponent, opp_dice=opp_first["dice"])
    assert out["phase"] in ("build", "dance_me")     # now it's your turn
    assert len(out["log"]) > len(opp_first["log"])   # opponent's opening move was logged


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


def test_review_submission_records_each_verdict_for_the_report_card():
    prov = GnubgProvider(plies=0)
    board, dice = starting_board(), (3, 1)
    st = {**new_state(random.Random(0)), "board": board, "dice": dice,
          "hops": sorted(generate_move_paths(board, dice))[0]}
    out = _review_submission(st, prov)
    assert len(out["verdicts"]) == len(st["verdicts"]) + 1


def test_render_shows_the_report_card_at_game_over():
    # reach gameover via the bear-off win, then render
    pts = [0] * 24
    pts[0], pts[23] = 1, -15
    board = Board(points=tuple(pts), bar_count=0, opp_bar_count=0,
                  off_count=14, opp_off_count=0)
    dice = (1, 3)
    st = {**new_state(random.Random(0)), "board": board, "dice": dice,
          "hops": sorted(generate_move_paths(board, dice))[0]}
    over = _review_submission(st, GnubgProvider(plies=0))

    coach_panel = _render(over)[4]                    # index 4 == coach_view
    assert "Report card" in coach_panel
    assert "Moves coached: 1" in coach_panel


# --- rendering + construction ----------------------------------------------

def test_render_emits_one_value_per_output_and_passes_state_through():
    st = new_state(random.Random(0))
    rendered = _render(st)
    assert len(rendered) == 14           # must match build_app's `out` list length
    assert rendered[5] is st             # the gr.State passthrough


def test_stats_accumulate_and_feed_the_charts():
    prov = GnubgProvider(plies=0)
    board, dice = starting_board(), (3, 1)
    st = {**new_state(random.Random(0)), "board": board, "dice": dice,
          "hops": sorted(generate_move_paths(board, dice))[0]}
    out = _review_submission(st, prov)

    (s,) = out["stats"]                              # exactly one coached move so far
    assert s["move"] == 1
    assert 0.0 <= s["win"] <= 1.0
    assert s["decisions"] == 1                       # opening 3-1 has a real choice
    assert s["cum_loss"] >= 0 and s["err"] == s["cum_loss"] / s["decisions"]

    win_df, cum_df, err_df = _charts(out)
    assert len(win_df) == len(cum_df) == len(err_df) == 1
    assert list(win_df["move"]) == [1]
    assert _charts(new_state(random.Random(0)))[0].empty   # no rows before your first move


def test_mirror_notation_flips_points_into_your_numbering():
    assert _mirror_notation("13/7") == "12/18"          # 25-13, 25-7
    assert _mirror_notation("11/7*") == "14/18*"        # hit marker preserved
    assert _mirror_notation("bar/21 13/11") == "bar/4 12/14"   # bar preserved
    assert _mirror_notation("6/off") == "19/off"        # off preserved
    # mirroring is an involution: applying it twice is the identity
    assert _mirror_notation(_mirror_notation("8/5 6/5*")) == "8/5 6/5*"


def test_difficulty_levels_are_ordered_strong_to_weak_and_include_the_default():
    assert DEFAULT_LEVEL in DIFFICULTY
    taus = list(DIFFICULTY.values())
    # weaker levels first -> temperature strictly decreases toward Expert
    assert taus == sorted(taus, reverse=True)
    assert DIFFICULTY["Expert"] < DIFFICULTY["Intermediate"] < DIFFICULTY["Beginner"]


def test_a_level_backed_opponent_advances_a_turn():
    rng = random.Random(5)
    opp = SkillAgent(GnubgProvider(plies=0), DIFFICULTY[DEFAULT_LEVEL], rng)
    st = {**new_state(rng), "hops": (), "phase": "dance_me"}
    out = _advance(st, rng, opp)
    assert out["phase"] in ("build", "dance_me", "gameover")
    assert len(out["log"]) > len(st["log"])


def test_build_app_constructs_with_stubs():
    app = build_app(provider=GnubgProvider(plies=0), llm=lambda system, user: "x",
                    rng=random.Random(0), opponent=_trivial_opponent)
    assert isinstance(app, gr.Blocks)
