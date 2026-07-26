from typing import Callable

from engine.board import Board
from engine.notation import describe_move


class HumanAgent:
    """An `Agent` that lists the legal plays and reads the human's choice.

    It presents the moves only; the surrounding loop renders the board and the
    roll (so the board still shows on a dance, when the agent isn't consulted).
    I/O is injected (`input_fn`/`output_fn`, defaulting to the builtins) so the
    agent can be driven with scripted input in tests -- no real stdin/stdout.
    """

    def __init__(self,
                 input_fn: Callable[[str], str] = input,
                 output_fn: Callable[[str], None] = print):
        self._input = input_fn
        self._output = output_fn

    def __call__(self, board: Board, dice: tuple[int, int],
                 afterstates: set[Board]) -> Board:
        # afterstates is never empty here: play_turn handles the dance before
        # consulting the agent.
        options = sorted(afterstates)  # stable, reproducible ordering
        labelled = [(describe_move(board, a, dice), a) for a in options]

        # A forced move isn't worth prompting for -- just play it.
        if len(labelled) == 1:
            label, only = labelled[0]
            self._output(f"Only legal play: {label} (auto-played)")
            return only

        for i, (label, _) in enumerate(labelled, start=1):
            self._output(f"{i:>3}) {label}")

        while True:
            raw = self._input(f"Pick a move [1-{len(labelled)}]: ")
            choice = _parse_choice(raw, len(labelled))
            if choice is not None:
                return labelled[choice - 1][1]
            self._output("Invalid choice, try again.")


def _parse_choice(raw: str, n: int) -> int | None:
    """Parse a 1-based menu selection; None if it isn't an integer in [1, n]."""
    try:
        choice = int(raw.strip())
    except (ValueError, AttributeError):
        return None
    return choice if 1 <= choice <= n else None
