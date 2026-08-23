import random

import gradio as gr

from coach.app import _click, _status, _highlights, _new_state, build_app


# make-the-5-point paths (both orderings), in index space -- see test_move_builder
FIVE_POINT = {((7, 4, 3), (5, 4, 1)), ((5, 4, 1), (7, 4, 3))}


def _state(hops=(), source=None, reviewed=False, paths=FIVE_POINT):
    return {"paths": paths, "hops": hops, "source": source, "reviewed": reviewed,
            "board": None, "dice": (3, 1), "analysis": None, "name": "", "theme": ""}


class _StubProvider:
    def analyze(self, position, dice):
        return "ANALYSIS"


# --- _click: the move-building state transitions -----------------------

def test_click_arms_a_legal_source():
    assert _click(_state(), 7)["source"] == 7

def test_click_ignores_an_illegal_source():
    assert _click(_state(), 99)["source"] is None

def test_click_a_destination_commits_a_hop():
    result = _click(_state(source=7), 4)
    assert result["hops"] == ((7, 4, 3),) and result["source"] is None

def test_click_another_source_reselects():
    assert _click(_state(source=7), 5)["source"] == 5      # 5 is also a legal source

def test_click_elsewhere_deselects():
    assert _click(_state(source=7), 99)["source"] is None

def test_click_is_inert_once_reviewed():
    st = _state(source=7, reviewed=True)
    assert _click(st, 4) == st

def test_click_is_inert_on_the_dance():
    st = _state(paths=set())
    assert _click(st, 7) == st                             # no paths -> nothing to do

def test_click_off_board_is_inert():
    assert _click(_state(), None)["source"] is None


# --- _highlights ------------------------------------------------------

def test_highlights_only_destinations_once_a_source_is_armed():
    assert _highlights(_state()) == set()                  # nothing highlighted idle
    assert _highlights(_state(source=7)) == {4}            # only the destination


# --- _status ----------------------------------------------------------

def test_status_messages_track_the_phase():
    assert "checker" in _status(_state())                  # idle: pick a source
    assert "destination" in _status(_state(source=7))      # armed
    assert "Submit" in _status(_state(hops=((7, 4, 3), (5, 4, 1))))  # complete
    assert _status(_state(reviewed=True)) == ""
    assert "dance" in _status(_state(paths=set())).lower()


# --- _new_state + construction ----------------------------------------

def test_new_state_has_the_expected_shape():
    st = _new_state(_StubProvider(), random.Random(0))
    assert set(st) >= {"board", "dice", "analysis", "paths", "hops", "source", "reviewed"}
    assert st["hops"] == () and st["source"] is None and st["reviewed"] is False
    assert isinstance(st["paths"], set)

def test_build_app_constructs_with_stubs():
    app = build_app(provider=_StubProvider(), llm=lambda system, user: "x",
                    rng=random.Random(0))
    assert isinstance(app, gr.Blocks)
