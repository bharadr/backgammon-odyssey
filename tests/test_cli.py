import random

from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.cli import run_demo
from tests.test_moves import mk


class _StubProvider:
    def __init__(self, analysis):
        self._analysis = analysis
    def analyze(self, position, dice):
        return self._analysis

class _StubLLM:
    def __init__(self, reply):
        self.reply = reply
    def __call__(self, system, user):
        return self.reply

def _scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)

def _move(board, equity, notation):
    return MoveAnalysis(after_state=board, outcome=OutcomeDist((equity + 1) / 2, 0, 0, 0, 0),
                        equity=equity, notation=notation)

def _ranked_analysis():
    return Analysis(position=starting_board(), dice=(3, 1),
                    moves=(_move(mk({5: 2}), 0.30, "a"),
                           _move(mk({6: 2}), 0.10, "b"),
                           _move(mk({7: 2}), -0.20, "c")))


def test_run_demo_lists_plays_and_critiques_the_choice():
    out = []
    run_demo(_StubProvider(_ranked_analysis()), _StubLLM("here is why b is worse"),
             random.Random(0), input_fn=_scripted(["2"]), output_fn=out.append)
    text = "\n".join(out)
    # menu shown in stable notation order, no equities to spoil the choice
    assert "1. a" in text and "2. b" in text and "3. c" in text
    assert "equity" not in text.split("You played")[0].split("legal plays")[1]
    # the chosen play (b) is ranked and critiqued
    assert "You played: b" in text
    assert "rank 2 of 3" in text
    assert "equity lost 0.200" in text
    assert "here is why b is worse" in text


def test_menu_is_ordered_by_notation_not_equity_and_choice_maps_to_true_rank():
    # equity order (best-first) is z, m, a; notation order is a, m, z. The menu
    # must use NOTATION order so the ranking isn't given away by position.
    a = Analysis(position=starting_board(), dice=(3, 1),
                 moves=(_move(mk({5: 2}), 0.30, "z-best"),      # best equity, sorts last
                        _move(mk({6: 2}), 0.10, "m-mid"),
                        _move(mk({7: 2}), -0.20, "a-worst")))   # worst equity, sorts first
    out = []
    run_demo(_StubProvider(a), _StubLLM("..."), random.Random(0),
             input_fn=_scripted(["1"]), output_fn=out.append)   # pick menu slot #1

    text = "\n".join(out)
    menu = text.split("Your legal plays:")[1].split("You played")[0]
    assert menu.index("a-worst") < menu.index("m-mid") < menu.index("z-best")
    # slot #1 is the worst-equity play, and it's graded by its TRUE rank (3 of 3)
    assert "You played: a-worst  (rank 3 of 3" in text


def test_run_demo_reprompts_until_valid():
    out = []
    run_demo(_StubProvider(_ranked_analysis()), _StubLLM("ok"),
             random.Random(0), input_fn=_scripted(["99", "abc", "1"]), output_fn=out.append)
    text = "\n".join(out)
    assert "Please enter a number." in text          # "abc"
    assert "Enter a number from 1 to 3." in text      # "99"
    assert "You played: a" in text                    # finally picked 1


def test_run_demo_states_the_dance_without_calling_the_llm():
    analysis = Analysis(position=starting_board(), dice=(6, 3), moves=())
    out = []
    run_demo(_StubProvider(analysis), _StubLLM("SHOULD NOT NARRATE"),
             random.Random(0), input_fn=_scripted([]), output_fn=out.append)
    text = "\n".join(out)
    assert "dance" in text.lower()
    assert "You roll:" in text                        # the position is still shown
    assert "SHOULD NOT NARRATE" not in text           # a dance needs no coaching
