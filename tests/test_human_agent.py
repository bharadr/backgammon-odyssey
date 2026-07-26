from engine.board import starting_board
from engine.moves import generate_moves
from engine.notation import describe_move
from agent.human_agent import HumanAgent
from tests.test_moves import mk


def _agent(inputs):
    """A HumanAgent driven by a scripted list of input strings, collecting
    its output lines into `out`."""
    it = iter(inputs)
    out = []
    agent = HumanAgent(input_fn=lambda prompt="": next(it),
                       output_fn=lambda line="": out.append(str(line)))
    return agent, out


def test_returns_the_chosen_afterstate():
    board = starting_board()
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    agent, _ = _agent(["2"])
    # HumanAgent lists sorted(afterstates); picking "2" is the 2nd (index 1)
    assert agent(board, dice, afterstates) == sorted(afterstates)[1]


def test_menu_lists_every_legal_move():
    # the menu must actually show each legal play's notation, not just count
    board = starting_board()
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    agent, out = _agent(["1"])
    agent(board, dice, afterstates)

    text = "\n".join(out)
    for a in afterstates:
        assert describe_move(board, a, dice) in text


def test_reprompts_on_invalid_input():
    board = starting_board()
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    agent, out = _agent(["abc", "999", "1"])   # garbage, out-of-range, then valid
    result = agent(board, dice, afterstates)
    assert result == sorted(afterstates)[0]
    assert sum("Invalid" in line for line in out) == 2   # re-prompted twice


def test_forced_move_is_auto_played_without_prompting():
    # exactly one legal play (larger-die rule): must play the 6, index blocked
    board = mk({12: 1, 3: -2})
    dice = (6, 3)
    afterstates = generate_moves(board, dice)
    assert len(afterstates) == 1

    out = []

    def explode(_prompt=""):
        raise AssertionError("must not prompt when the move is forced")

    agent = HumanAgent(input_fn=explode, output_fn=lambda line="": out.append(str(line)))
    # afterstates is a set (not indexable); next(iter(...)) gets its sole element
    assert agent(board, dice, afterstates) == next(iter(afterstates))
    assert any("auto-played" in line for line in out)   # announced, not silent
