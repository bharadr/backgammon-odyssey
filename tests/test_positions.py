from engine.board import is_valid
from coach.positions import POSITIONS


def test_all_curated_positions_are_valid():
    # 15 checkers a side, non-negative bars/offs -- exactly what is_valid checks.
    assert len(POSITIONS) == 10
    for p in POSITIONS:
        assert is_valid(p.board), f"{p.name} is not a legal position"


def test_positions_are_live_and_labelled():
    names = [p.name for p in POSITIONS]
    assert len(set(names)) == len(names)          # distinct names for the demo menu
    for p in POSITIONS:
        assert p.name and p.theme                 # both are shown to the student
        # a quiz position must be an unfinished game (nobody has borne off all 15)
        assert p.board.off_count < 15 and p.board.opp_off_count < 15
