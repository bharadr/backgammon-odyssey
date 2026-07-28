import random

from engine.board import is_valid
from engine.game import Outcome
from agent.random_agent import random_agent
from agent.skill_agent import SkillAgent
from coach.gnubg_provider import GnubgProvider
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


def test_full_game_vs_gnubg_skill_agent():
    # End-to-end guard against the integration seam. The opponent is the
    # gnubg-backed SkillAgent, so EVERY opponent move is run through
    # describe_move -- exactly the path that crashed when a gnubg-chosen
    # afterstate wasn't legal in our engine (asymmetric hit/doubles positions).
    # Several seeds to hit varied positions; seed 11 is the one that first
    # exposed the wrong-player bug. Reaching the asserts (no ValueError from
    # describe_move, no RuntimeError from the turn cap) is the real check.
    for seed in [11, 3, 42, 7, 0]:
        rng = random.Random(seed)
        human = random_agent(rng)                    # stands in for the human seat
        opponent = SkillAgent(GnubgProvider(), 0.1, rng)
        out = []
        result = play_interactive(human, opponent, rng,
                                  output_fn=lambda line="": out.append(str(line)))

        assert result.winner in (0, 1)
        assert is_valid(result.final_board)
        assert result.final_board.off_count == 15
        assert any("Opponent plays" in line for line in out)  # describe_move ran
