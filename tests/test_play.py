import random

from engine.board import is_valid
from engine.game import Outcome
from agent.random_agent import random_agent
from ui.play import play_interactive


def test_play_interactive_terminates_with_legal_winner():
    # Drive the loop with a random agent in BOTH seats (no scripted human
    # input needed): this exercises the human-turn branch (render + "Your
    # roll") and the opponent branch (announce + describe_move), and checks
    # the game runs to a legal, well-formed result.
    rng = random.Random(0)
    agent = random_agent(rng)
    out = []
    result = play_interactive(agent, agent, rng,
                              output_fn=lambda line="": out.append(str(line)))

    assert result.winner in (0, 1)
    assert isinstance(result.outcome, Outcome)
    assert is_valid(result.final_board)
    assert result.final_board.off_count == 15

    text = "\n".join(out)
    assert "Your roll" in text                       # human-turn branch ran
    assert "Opponent plays" in text or "forfeits" in text  # opponent branch ran
    assert "win" in text.lower()                     # a result was announced
