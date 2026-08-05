"""B5: the interactive coaching demo.

Flow: pick a curated board + a random roll, show every legal play (notation
only, no equities -- no spoilers), let the student choose the one they'd play,
then reveal the engine's ranking and the coach's critique of THAT choice.

Output is assembled into a few labelled blocks (pure string builders) so
`run_demo` reads as orchestration. Its provider, llm, rng, and I/O are injected,
so the whole flow is testable with stubs; `coach.__main__` wires the real ones.
"""
import random
from typing import Callable

from engine.board import Board, render
from engine.game import roll_dice
from coach.analysis import AnalysisProvider, MoveAnalysis
from coach.evidence import build_evidence, Evidence
from coach.explain import explain, render_evidence, LLM
from coach.positions import POSITIONS, CuratedPosition


def _position_block(position: CuratedPosition, board: Board, dice: tuple[int, int]) -> str:
    """The position as the student first sees it: theme, board, and the roll."""
    return "\n".join([
        f"=== {position.name} ===",
        position.theme,
        "",
        render(board),
        "",
        f"You roll: {dice[0]}-{dice[1]}",
    ])


def _move_menu_block(menu: list[MoveAnalysis]) -> str:
    """The numbered legal plays -- notation only, so the ranking isn't spoiled."""
    return "\n".join(["Your legal plays:"]
                     + [f"  {i}. {move.notation}" for i, move in enumerate(menu, 1)])


def _feedback_block(chosen: MoveAnalysis, evidence: Evidence, narration: str) -> str:
    """The reveal: how the chosen play ranked, the raw evidence, and the coach's why."""
    return "\n".join([
        f"You played: {chosen.notation}  (rank {evidence.chosen.rank} of "
        f"{evidence.chosen.of_n}, equity lost {evidence.chosen.equity_loss:.3f})",
        "",
        "=== What the coach sees ===",
        render_evidence(evidence),
        "",
        "=== Coach ===",
        narration,
    ])


def _read_choice(input_fn: Callable[[str], str], output_fn: Callable[[str], None], n: int) -> int:
    """Prompt until the student enters an integer in [1, n]."""
    while True:
        raw = input_fn(f"Pick the move you would play [1-{n}]: ")
        try:
            choice = int(raw)
        except (ValueError, TypeError):
            output_fn("Please enter a number.")
            continue
        if 1 <= choice <= n:
            return choice
        output_fn(f"Enter a number from 1 to {n}.")


def run_demo(provider: AnalysisProvider, llm: LLM, rng: random.Random,
             input_fn: Callable[[str], str] = input,
             output_fn: Callable[[str], None] = print,
             positions: tuple[CuratedPosition, ...] = POSITIONS) -> None:
    """Run one coaching round: show a position, take the student's play, critique it."""
    position = rng.choice(positions)
    board = position.board
    dice = roll_dice(rng)
    output_fn(_position_block(position, board, dice))

    analysis = provider.analyze(board, dice)
    if not analysis.moves:
        # a dance is self-explanatory from the board + roll -- no coaching needed
        output_fn("\nNo legal move with this roll -- you dance (forfeit the turn).")
        return

    menu = sorted(analysis.moves, key=lambda move: move.notation or "")
    output_fn("\n" + _move_menu_block(menu))

    chosen = menu[_read_choice(input_fn, output_fn, len(menu)) - 1]
    evidence = build_evidence(analysis, chosen.after_state)
    output_fn("\n" + _feedback_block(chosen, evidence, explain(evidence, llm)))
