"""`python -m coach` -- run the interactive coaching quiz.

Set ANTHROPIC_API_KEY for a narrated explanation; without it the demo still
runs and prints the exact engine evidence the coach would narrate.
"""
import random

from coach.cli import run_demo
from coach.gnubg_provider import GnubgProvider
from coach.llm import make_llm


def main():
    run_demo(GnubgProvider(plies=0), make_llm(), rng=random.Random())


if __name__ == "__main__":
    main()
