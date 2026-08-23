import pytest

from engine.board import starting_board
from engine.moves import BAR_IDX, OFF
from coach.board_svg import board_svg, point_at, WIDTH, HEIGHT, _DIE_USED
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


def test_doubles_render_four_dice():
    svg = board_svg(starting_board(), dice=(2, 2))
    assert svg.count('class="die"') == 4
    assert svg.count('class="pip"') == 2 * 4


def test_used_dice_are_greyed_out():
    assert board_svg(starting_board(), dice=(2, 2)).count(f'fill="{_DIE_USED}"') == 0
    # two of the four doubles spent -> two greyed faces
    svg = board_svg(starting_board(), dice=(2, 2), used=[2, 2])
    assert svg.count(f'fill="{_DIE_USED}"') == 2


def test_no_dice_drawn_when_no_roll_given():
    assert 'class="die"' not in board_svg(starting_board())


# --- highlight (source + legal destinations) ---------------------------

def test_highlight_tints_destinations_with_translucent_green():
    svg = board_svg(starting_board(), highlight={4, OFF})
    assert svg.count('class="hl"') == 2            # a triangle for pt 4, a rect for OFF
    assert 'fill-opacity="0.45"' in svg            # semi-transparent overlay, not an outline
    assert 'class="hl"' not in board_svg(starting_board())


# --- point_at (pixel click -> point), the inverse of the layout --------

def test_point_at_maps_columns_to_top_and_bottom_points():
    assert point_at(38, 100) == 12                 # col 0, upper half -> 13-point (idx 12)
    assert point_at(38, 400) == 11                 # col 0, lower half -> 12-point (idx 11)
    assert point_at(562, 100) == 23                # col 11, upper -> 24-point (idx 23)
    assert point_at(562, 400) == 0                 # col 11, lower -> 1-point (idx 0)

def test_point_at_bar_and_off():
    assert point_at(300, 200) == BAR_IDX           # the center bar
    assert point_at(600, 200) == OFF               # the bear-off tray

def test_point_at_outside_the_board_is_none():
    assert point_at(38, 5) is None                 # above the board
    assert point_at(5, 200) is None                # left of the first column


# --- rasterization (needs system libcairo; skipped if unavailable) -----

def test_board_image_rasterizes_to_expected_size_and_is_not_blank():
    # Needs system libcairo. Import via coach.raster (which sets up the loader path
    # first); skip only if libcairo genuinely isn't installed on this machine.
    try:
        from coach.raster import board_image, SCALE
    except OSError as e:
        pytest.skip(f"libcairo not installed: {e}")
    img = board_image(starting_board(), (3, 1)).convert("RGB")
    assert img.size == (WIDTH * SCALE, HEIGHT * SCALE)
    assert len(img.getcolors(maxcolors=1_000_000)) > 1
