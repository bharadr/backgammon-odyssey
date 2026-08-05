import pytest

from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.evidence import build_evidence
from coach.grade import grade, _classify, DOUBTFUL, ERROR, BLUNDER
from tests.test_moves import mk

BEST, CHOSEN = mk({5: 2}), mk({6: 2})


def _evidence(loss: float):
    """Evidence whose chosen play gives up exactly `loss` equity (0 == best)."""
    a = Analysis(position=starting_board(), dice=(3, 1),
                 moves=(MoveAnalysis(BEST, OutcomeDist(0.6, 0, 0, 0, 0), 0.30, "best"),
                        MoveAnalysis(CHOSEN, OutcomeDist(0.5, 0, 0, 0, 0), 0.30 - loss, "chosen")))
    return build_evidence(a, BEST if loss == 0 else CHOSEN)


# --- _classify: the band -> (label, symbol) logic ----------------------

def test_classify_maps_each_band_to_its_label_and_symbol():
    assert _classify(-0.01) == ("Best play", "✓")   # <= 0 (defensive; loss is never < 0)
    assert _classify(0.00) == ("Best play", "✓")
    assert _classify(0.01) == ("Good", "·")
    assert _classify(0.03) == ("Doubtful", "?")
    assert _classify(0.06) == ("Error", "!")
    assert _classify(0.15) == ("Blunder", "✗")


def test_classify_thresholds_fall_into_the_worse_band():
    # each cutoff belongs to the WORSE band (the better side uses a strict `<`);
    # tested against the constants directly, so no float subtraction can blur it.
    assert _classify(DOUBTFUL) == ("Doubtful", "?")
    assert _classify(ERROR) == ("Error", "!")
    assert _classify(BLUNDER) == ("Blunder", "✗")
    # a hair below each cutoff stays in the better band
    assert _classify(DOUBTFUL - 1e-6)[0] == "Good"
    assert _classify(ERROR - 1e-6)[0] == "Doubtful"
    assert _classify(BLUNDER - 1e-6)[0] == "Error"


# --- grade: wiring _classify into a Verdict + building the line --------

def test_grade_best_play_builds_its_line_and_fields():
    v = grade(_evidence(0.0))
    assert v.label == "Best play" and v.symbol == "✓"
    assert v.rank == 1 and v.of_n == 2 and v.equity_loss == 0.0
    assert v.line == "✓ Best play!  (best of 2)"


def test_grade_mistake_builds_its_line_and_fields():
    v = grade(_evidence(0.15))
    assert v.label == "Blunder" and v.symbol == "✗"
    assert v.rank == 2 and v.of_n == 2
    assert v.equity_loss == pytest.approx(0.15)
    assert v.line == "✗ Blunder -- your play ranked 2 of 2, equity lost 0.150"
