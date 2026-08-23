"""The coach that watches a live game and critiques the human's checker plays.

Grades EVERY move instantly (engine-only: analyze + grade) and narrates only the
ones at/above `narrate_threshold` (an LLM call -- the real teaching moments), so
the game stays fast. Keeps a scoreboard for an end-of-game report card.

Dependencies (provider, llm, output) are injected, so it's testable with stubs;
`ui.play.main` wires the real gnubg provider + Anthropic LLM.
"""
from typing import Callable

from engine.board import Board
from coach.analysis import AnalysisProvider
from coach.evidence import build_evidence, Evidence
from coach.explain import explain, LLM
from coach.grade import grade, Verdict, BLUNDER


def report_card_text(verdicts: list[Verdict]) -> str:
    """End-of-game stats block (no banner), shared by the terminal and GUI: moves
    coached, decisions, best-play rate, error rate + estimated PR, and the single
    worst move. Returns '' when no moves were coached.

    The error rate / PR count only real DECISIONS -- a forced move (one legal play)
    is excluded, matching the gnubg convention; PR is approximated as rate x 500.
    """
    if not verdicts:
        return ""
    n = len(verdicts)
    decisions = [v for v in verdicts if v.of_n > 1]      # exclude forced moves
    d = len(decisions)
    worst = max(verdicts, key=lambda v: v.equity_loss)
    lines = [f"Moves coached: {n}   Decisions (non-forced): {d}"]
    if d:
        best = sum(1 for v in decisions if v.equity_loss <= 0)
        rate = sum(v.equity_loss for v in decisions) / d
        lines.append(f"Best play found: {best}/{d} ({100 * best // d}%)")
        lines.append(f"Error rate: {rate:.3f}/move  (~PR {rate * 500:.0f})")
    if worst.equity_loss > 0:
        lines.append(f"Worst: {worst.label.lower()} (rank {worst.rank} of "
                     f"{worst.of_n}, lost {worst.equity_loss:.3f})")
    return "\n".join(lines)


class GameCoach:
    """Reviews the human's plays during one game and reports on it at the end."""

    def __init__(self, provider: AnalysisProvider, llm: LLM,
                 output_fn: Callable[[str], None] = print,
                 input_fn: Callable[[str], str] = input,
                 narrate_threshold: float = BLUNDER):
        """Wire the coach's collaborators.

        `provider` ranks the legal plays (gnubg in production); `llm` produces the
        narrated explanation; `output_fn`/`input_fn` are the console I/O (injected
        for testing). A move is narrated automatically once its equity loss
        reaches `narrate_threshold` (default: the blunder cutoff).
        """
        self._provider = provider
        self._llm = llm
        self._out = output_fn
        self._ask = input_fn
        self._narrate_threshold = narrate_threshold
        self._verdicts: list[Verdict] = []

    def review(self, position: Board, dice: tuple[int, int], chosen_after: Board) -> None:
        """Grade the human's play and give feedback for one turn.

        `position` is the board they moved from, `dice` the roll, `chosen_after`
        the afterstate they picked. Prints the instant verdict, then: narrates a
        blunder automatically (with a pause to read it); offers the explanation on
        demand (`?`) for a lesser mistake; or flows straight on for the best play.
        The verdict is recorded for the report card.
        """
        analysis = self._provider.analyze(position, dice)
        evidence = build_evidence(analysis, chosen_after)
        verdict = grade(evidence)
        self._verdicts.append(verdict)
        self._out(verdict.line)

        if verdict.equity_loss >= self._narrate_threshold:
            self._narrate(evidence)
            self._prompt("  [Enter] to continue: ")          # let the blunder be read
        elif verdict.equity_loss > 0:
            if self._prompt("  [Enter] continue  ·  [?] explain: ").strip() == "?":
                self._narrate(evidence)
        # best play (equity_loss == 0): no pause, flow on

    def _narrate(self, evidence: Evidence) -> None:
        """Print the LLM's full natural-language explanation of the play."""
        self._out("")
        self._out(explain(evidence, self._llm))

    def _prompt(self, text: str) -> str:
        """Ask the human for input, treating EOF (piped/no stdin) as 'continue'."""
        try:
            return self._ask(text)
        except EOFError:
            return ""

    def report_card(self) -> None:
        """Print end-of-game stats: moves coached, best-play rate, average equity
        lost per move, and the single worst move. Silent if no moves were seen."""
        text = report_card_text(self._verdicts)
        if text:
            self._out("")
            self._out("=== Report card ===")
            self._out(text)
