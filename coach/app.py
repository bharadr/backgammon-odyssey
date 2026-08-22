"""Gradio GUI for the coaching quiz.

A presentation shell over the existing pipeline -- it adds no backgammon or coach
logic. A round is: pick a curated position + roll, click a legal play, then see
the resulting board and the coach's verdict + explanation.

    new round: rng.choice(POSITIONS) -> roll_dice -> provider.analyze
    on pick:   build_evidence -> grade -> explain

The UI has two modes: CHOOSING (original board, plays enabled) and REVIEWED
(afterstate board, verdict shown, plays disabled, Retry offered). Retry re-plays
the SAME position (no re-roll); New position draws a fresh one.

Run:  python -m coach.app   (set ANTHROPIC_API_KEY for narrated explanations)
"""
import random

import gradio as gr

from engine.game import roll_dice
from coach.analysis import AnalysisProvider, MoveAnalysis
from coach.board_svg import board_svg
from coach.evidence import build_evidence
from coach.explain import explain, LLM
from coach.grade import grade
from coach.gnubg_provider import GnubgProvider
from coach.llm import make_llm
from coach.positions import POSITIONS

MAX_PLAYS = 40       # fixed pool of play-buttons (curated positions never exceed this)
PLAYS_PER_ROW = 5    # laid out as a tidy grid rather than one long row


def _status_md(has_moves: bool) -> str:
    """A caption under the board -- empty normally, the dance note when stuck."""
    return "" if has_moves else "*No legal move -- you dance (forfeit the turn).*"


def _play_button_updates(menu: list[MoveAnalysis], pool_size: int = MAX_PLAYS) -> list:
    """Relabel + enable the first len(menu) buttons; hide the rest."""
    updates = []
    for i in range(pool_size):
        if i < len(menu):
            updates.append(gr.update(value=menu[i].notation or "(unnamed)",
                                     visible=True, interactive=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates


def _disable_all_buttons(pool_size: int = MAX_PLAYS) -> list:
    """Grey out every play-button (labels/visibility unchanged) after a choice."""
    return [gr.update(interactive=False) for _ in range(pool_size)]


def _new_round_data(provider: AnalysisProvider, rng: random.Random):
    """Pick a position + roll and rank the plays. The menu is notation-sorted, so a
    play's position in the list never gives away its equity ranking."""
    position = rng.choice(POSITIONS)
    dice = roll_dice(rng)
    analysis = provider.analyze(position.board, dice)
    menu = sorted(analysis.moves, key=lambda move: move.notation or "")
    return position, dice, analysis, menu


def build_app(provider: AnalysisProvider | None = None, llm: LLM | None = None,
              rng: random.Random | None = None) -> gr.Blocks:
    """Assemble the Gradio app. Collaborators are injected for testing; defaults
    wire the real gnubg provider + Anthropic LLM."""
    provider = provider or GnubgProvider()
    llm = llm or make_llm()
    rng = rng or random.Random()

    def new_round():
        position, dice, analysis, menu = _new_round_data(provider, rng)
        return (f"### {position.name}\n{position.theme}",
                board_svg(position.board, dice), _status_md(bool(menu)),
                "", "", gr.update(visible=False), (analysis, menu),
                *_play_button_updates(menu))

    def retry(state):
        analysis, menu = state
        return (board_svg(analysis.position, analysis.dice), _status_md(True),
                "", "", gr.update(visible=False), *_play_button_updates(menu))

    def on_pick(i: int, state):
        analysis, menu = state
        chosen = menu[i]
        evidence = build_evidence(analysis, chosen.after_state)
        return (board_svg(chosen.after_state, analysis.dice),
                f"### Verdict\n{grade(evidence).line}",
                f"### Coach\n{explain(evidence, llm)}",
                gr.update(visible=True), *_disable_all_buttons())

    with gr.Blocks(title="Backgammon Coach") as app:
        gr.Markdown("# Backgammon Coach")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                header = gr.Markdown()
                board_view = gr.HTML()
                status = gr.Markdown()
                gr.Markdown("**Your play:**")
                play_buttons = []                       # one fixed pool, laid out as a grid
                while len(play_buttons) < MAX_PLAYS:
                    with gr.Row():
                        row_n = min(PLAYS_PER_ROW, MAX_PLAYS - len(play_buttons))
                        play_buttons += [gr.Button(visible=False, min_width=110)
                                         for _ in range(row_n)]
                with gr.Row():
                    retry_btn = gr.Button("Retry position", visible=False)
                    new_btn = gr.Button("New position", variant="primary")
            with gr.Column(scale=2):
                verdict_view = gr.Markdown()
                coach_view = gr.Markdown()

        after_new = [header, board_view, status, verdict_view, coach_view, retry_btn, state] + play_buttons
        after_retry = [board_view, status, verdict_view, coach_view, retry_btn] + play_buttons
        after_pick = [board_view, verdict_view, coach_view, retry_btn] + play_buttons

        app.load(new_round, outputs=after_new)
        new_btn.click(new_round, outputs=after_new)
        retry_btn.click(retry, inputs=[state], outputs=after_retry)
        for i, btn in enumerate(play_buttons):
            btn.click(lambda state, i=i: on_pick(i, state), inputs=[state], outputs=after_pick)

    return app


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
