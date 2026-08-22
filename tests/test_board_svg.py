from engine.board import starting_board
from coach.board_svg import board_svg
from tests.test_moves import mk


def test_board_svg_is_a_self_contained_svg_element():
    svg = board_svg(starting_board())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")


def test_pip_counts_are_shown_over_the_bar_without_a_lead_hint():
    svg = board_svg(starting_board())
    assert svg.count(">167<") == 2                     # both sides' counts, over the bar
    for phrase in ("lead by", "trail by", "even race", "You (X)", "Opp (O)"):
        assert phrase not in svg                       # colour alone identifies the side


def test_checkers_are_drawn_per_side_under_the_stack_cap():
    # 3 of mine (1 + 2), 1 opponent -- all small stacks, so disc count == checkers
    svg = board_svg(mk({0: 1, 5: 2, 23: -1}))
    assert svg.count('class="chk me"') == 3
    assert svg.count('class="chk opp"') == 1


def test_tall_stack_is_capped_and_labelled_with_a_count():
    svg = board_svg(mk({5: 8}))                 # 8 on one point
    assert svg.count('class="chk me"') == 5     # capped at CAP discs
    assert ">8<" in svg                         # ...and the total is written on it


def test_dice_render_two_faces_with_the_right_number_of_pips():
    svg = board_svg(starting_board(), dice=(3, 5))
    assert svg.count('class="die"') == 2
    assert svg.count('class="pip"') == 3 + 5


def test_no_dice_drawn_when_no_roll_given():
    assert 'class="die"' not in board_svg(starting_board())
