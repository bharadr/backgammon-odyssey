from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.game_coach import GameCoach
from tests.test_moves import mk

BEST, CHOSEN = mk({5: 2}), mk({6: 2})


def _analysis(loss: float):
    return Analysis(position=starting_board(), dice=(3, 1),
                    moves=(MoveAnalysis(BEST, OutcomeDist(0.6, 0, 0, 0, 0), 0.30, "8/5 6/5"),
                           MoveAnalysis(CHOSEN, OutcomeDist(0.5, 0, 0, 0, 0), 0.30 - loss, "13/11")))

class _StubProvider:
    """Returns a queued analysis per call (regardless of args)."""
    def __init__(self, analyses):
        self._it = iter(analyses)
    def analyze(self, position, dice):
        return next(self._it)

class _StubLLM:
    def __init__(self, reply="EXPLANATION"):
        self.reply, self.calls = reply, 0
    def __call__(self, system, user):
        self.calls += 1
        return self.reply

class _Input:
    """Scripted keypress; records how many times it was asked."""
    def __init__(self, answer=""):
        self.answer, self.calls = answer, 0
    def __call__(self, prompt=""):
        self.calls += 1
        return self.answer


def test_review_grades_every_move_but_auto_narrates_only_blunders():
    out, llm, keys = [], _StubLLM("WHY IT'S BAD"), _Input("")   # user just hits Enter
    coach = GameCoach(_StubProvider([_analysis(0.05), _analysis(0.15)]),
                      llm, output_fn=out.append, input_fn=keys, narrate_threshold=0.08)

    coach.review(starting_board(), (3, 1), CHOSEN)   # 0.05 -> Error, prompt but no '?'
    coach.review(starting_board(), (3, 1), CHOSEN)   # 0.15 -> Blunder, auto-narrated

    text = "\n".join(out)
    assert "Error" in text and "Blunder" in text
    assert llm.calls == 1                            # only the blunder was narrated
    assert "WHY IT'S BAD" in text
    assert keys.calls == 2                           # the mistake's ? offer + the blunder's pause


def test_question_mark_explains_a_lesser_mistake_on_demand():
    out, llm, keys = [], _StubLLM("HERE IS WHY"), _Input("?")
    coach = GameCoach(_StubProvider([_analysis(0.05)]), llm,
                      output_fn=out.append, input_fn=keys, narrate_threshold=0.08)

    coach.review(starting_board(), (3, 1), CHOSEN)   # Error, below threshold
    assert keys.calls == 1                           # was offered the choice
    assert llm.calls == 1                            # and asked for the explanation
    assert "HERE IS WHY" in "\n".join(out)


def test_best_play_neither_pauses_nor_narrates():
    out, llm, keys = [], _StubLLM(), _Input("")
    coach = GameCoach(_StubProvider([_analysis(0.0)]), llm,
                      output_fn=out.append, input_fn=keys)

    coach.review(starting_board(), (3, 1), BEST)     # nailed it -> flow straight on
    assert keys.calls == 0 and llm.calls == 0
    assert "Best play" in "\n".join(out)


def test_report_card_summarizes_the_session():
    out = []
    coach = GameCoach(_StubProvider([_analysis(0.0), _analysis(0.05), _analysis(0.15)]),
                      _StubLLM(), output_fn=out.append, input_fn=_Input(""))
    coach.review(starting_board(), (3, 1), BEST)     # best (loss 0)
    coach.review(starting_board(), (3, 1), CHOSEN)   # 0.05
    coach.review(starting_board(), (3, 1), CHOSEN)   # 0.15
    coach.report_card()

    text = "\n".join(out)
    assert "Moves coached: 3" in text
    assert "Best play found: 1/3 (33%)" in text      # integer-division rate
    assert "Avg equity lost/move: 0.067" in text     # (0 + 0.05 + 0.15) / 3
    assert "lost 0.150" in text                      # worst move


def test_report_card_omits_the_worst_line_when_every_move_was_best():
    out = []
    coach = GameCoach(_StubProvider([_analysis(0.0), _analysis(0.0)]),
                      _StubLLM(), output_fn=out.append, input_fn=_Input(""))
    coach.review(starting_board(), (3, 1), BEST)
    coach.review(starting_board(), (3, 1), BEST)
    coach.report_card()

    text = "\n".join(out)
    assert "Best play found: 2/2 (100%)" in text
    assert "Avg equity lost/move: 0.000" in text
    assert "Worst" not in text                       # nobody erred -> no worst-move line


def test_report_card_is_silent_with_no_moves():
    out = []
    GameCoach(_StubProvider([]), _StubLLM(), output_fn=out.append,
              input_fn=_Input("")).report_card()
    assert out == []
