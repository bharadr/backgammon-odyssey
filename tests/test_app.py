import random
from types import SimpleNamespace as NS

import gradio as gr

from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.app import (_board_html, _play_button_updates, _disable_all_buttons,
                       _new_round_data, build_app)
from coach.positions import POSITIONS
from tests.test_moves import mk


class _StubProvider:
    def __init__(self, analysis=None):
        self._analysis = analysis
    def analyze(self, position, dice):
        return self._analysis


def _move(board, equity, notation):
    return MoveAnalysis(after_state=board, outcome=OutcomeDist((equity + 1) / 2, 0, 0, 0, 0),
                        equity=equity, notation=notation)


def test_board_html_is_plain_monospace():
    html = _board_html(starting_board())
    assert html.startswith("<pre") and html.endswith("</pre>")
    assert "You (X): 167 pips" in html
    assert "\033[" not in html                    # no ANSI escapes in the browser


def test_play_button_updates_shows_and_enables_k_plays_hides_the_rest():
    menu = [NS(notation="a"), NS(notation="b")]
    ups = _play_button_updates(menu, pool_size=4)
    assert ups[0]["value"] == "a" and ups[0]["visible"] is True and ups[0]["interactive"] is True
    assert ups[1]["value"] == "b" and ups[1]["visible"] is True
    assert ups[2]["visible"] is False and ups[3]["visible"] is False


def test_disable_all_buttons_greys_out_every_button():
    ups = _disable_all_buttons(pool_size=3)
    assert len(ups) == 3
    assert all(u["interactive"] is False for u in ups)


def test_new_round_data_sorts_the_menu_by_notation_not_equity():
    # equity order is z (best), a (worst); the menu must come back a, z
    a = Analysis(position=starting_board(), dice=(3, 1),
                 moves=(_move(mk({5: 2}), 0.30, "z"), _move(mk({6: 2}), 0.10, "a")))
    position, dice, analysis, menu = _new_round_data(_StubProvider(a), random.Random(0))
    assert [m.notation for m in menu] == ["a", "z"]
    assert position in POSITIONS
    assert len(dice) == 2 and all(1 <= d <= 6 for d in dice)


def test_build_app_constructs_with_stubs():
    app = build_app(provider=_StubProvider(), llm=lambda system, user: "x",
                    rng=random.Random(0))
    assert isinstance(app, gr.Blocks)
