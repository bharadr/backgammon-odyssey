"""Gradio GUI for the coaching quiz -- click-to-move on a graphical board.

Flow: a curated position + a random roll is shown on a clickable board. You build
your play by clicking a checker (its legal destinations light up green) then a
destination, one hop at a time (Undo/Submit as you go). On Submit the coach grades
and explains your play; Retry re-plays the same position, New draws another.

The board is a `gr.Image` because it's the only component that reports click
coordinates; the trade-off is a brief reload between renders, tinted to the board
colour (`_TINT`) so it's an unobtrusive flicker, not a white flash. The move logic
is the pure `move_builder` layer; this module is the Gradio glue on top.

    new round: rng.choice(POSITIONS) -> roll_dice -> analyze + generate_move_paths
    each click: point_at(pixel / SCALE) -> _click(state, point) -> re-render
    submit:    build_evidence -> grade -> explain

Run:  python -m coach.app   (set ANTHROPIC_API_KEY for narrated explanations)
"""
import random

import gradio as gr

from engine.game import roll_dice
from engine.moves import generate_move_paths
from engine.move_builder import (legal_sources, legal_destinations, choose_hop,
                                 is_complete, apply_hops)
from coach.analysis import AnalysisProvider
from coach.board_svg import point_at, WIDTH
from coach.raster import board_image, SCALE
from coach.evidence import build_evidence
from coach.explain import explain, LLM
from coach.grade import grade
from coach.gnubg_provider import GnubgProvider
from coach.llm import make_llm
from coach.positions import POSITIONS

# Tint the image's reload gap to the board colour instead of white, so the brief
# flash between renders is an unobtrusive same-colour flicker.
_TINT = "<style>#board, #board img { background: #f4e8d0 !important; }</style>"


# --- pure state transition (no Gradio -- unit-testable) --------------------

def _click(state: dict, point) -> dict:
    """Advance the move-building state on a click at `point` (a point index / BAR
    / OFF, or None for off-board). Selecting a legal source arms it; clicking a
    legal destination commits a hop; anything else reselects or clears."""
    if state["reviewed"] or not state["paths"] or point is None:
        return state
    paths, hops = state["paths"], state["hops"]
    if state["source"] is None:
        return {**state, "source": point} if point in legal_sources(paths, hops) else state
    if point in legal_destinations(paths, hops, state["source"]):
        hop = choose_hop(paths, hops, state["source"], point)
        return {**state, "hops": hops + (hop,), "source": None}
    if point in legal_sources(paths, hops):
        return {**state, "source": point}          # switch to another source
    return {**state, "source": None}               # click elsewhere: deselect


def _highlights(state: dict) -> set:
    """Only the legal destinations, and only once a source is selected -- nothing
    is highlighted while idle."""
    if state["source"] is None:
        return set()
    return legal_destinations(state["paths"], state["hops"], state["source"])


def _status(state: dict) -> str:
    if not state["paths"]:
        return "*No legal move with this roll -- you dance (forfeit the turn).*"
    if state["reviewed"]:
        return ""
    if state["source"] is not None:
        return "Click a highlighted destination (or another checker to reselect)."
    if is_complete(state["paths"], state["hops"]):
        return "Move complete -- click **Submit** for the coach's verdict."
    return "Click a checker to start your move."


def _board(state: dict):
    board = apply_hops(state["board"], state["hops"])
    hl = set() if state["reviewed"] else _highlights(state)
    used = [hop[2] for hop in state["hops"]]        # die values spent so far -> greyed
    return board_image(board, state["dice"], hl, used)


def _new_state(provider: AnalysisProvider, rng: random.Random) -> dict:
    position = rng.choice(POSITIONS)
    dice = roll_dice(rng)
    return {"name": position.name, "theme": position.theme,
            "board": position.board, "dice": dice,
            "analysis": provider.analyze(position.board, dice),
            "paths": generate_move_paths(position.board, dice),
            "hops": (), "source": None, "reviewed": False}


def build_app(provider: AnalysisProvider | None = None, llm: LLM | None = None,
              rng: random.Random | None = None) -> gr.Blocks:
    provider = provider or GnubgProvider(plies=2)   # 2-ply: stronger judgement, slower
    llm = llm or make_llm()
    rng = rng or random.Random()

    def new_round():
        st = _new_state(provider, rng)
        return (f"### {st['name']}\n{st['theme']}", _board(st), _status(st), st, "", "",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(visible=False))

    def on_click(state, evt: gr.SelectData):
        x, y = evt.index                            # pixel coords in the SCALE'd image
        state = _click(state, point_at(x / SCALE, y / SCALE))
        return (_board(state), _status(state), state,
                gr.update(interactive=is_complete(state["paths"], state["hops"])),
                gr.update(interactive=bool(state["hops"])))

    def on_undo(state):
        state = {**state, "hops": state["hops"][:-1], "source": None}
        return (_board(state), _status(state), state,
                gr.update(interactive=is_complete(state["paths"], state["hops"])),
                gr.update(interactive=bool(state["hops"])))

    def on_submit(state):
        evidence = build_evidence(state["analysis"], apply_hops(state["board"], state["hops"]))
        state = {**state, "reviewed": True}
        return (_board(state), _status(state), state,
                f"### Verdict\n{grade(evidence).line}", f"### Coach\n{explain(evidence, llm)}",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(visible=True))

    def retry(state):
        state = {**state, "hops": (), "source": None, "reviewed": False}
        return (_board(state), _status(state), state, "", "",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(visible=False))

    with gr.Blocks(title="Backgammon Coach") as app:
        gr.Markdown("# Backgammon Coach")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                header = gr.Markdown()
                board_img = gr.Image(interactive=False, show_label=False,
                                     width=WIDTH, elem_id="board")
                status = gr.Markdown()
                with gr.Row():
                    undo_btn = gr.Button("Undo", interactive=False)
                    submit_btn = gr.Button("Submit", variant="primary", interactive=False)
                with gr.Row():
                    retry_btn = gr.Button("Retry position", visible=False)
                    new_btn = gr.Button("New position")
            with gr.Column(scale=2):
                verdict_view = gr.Markdown()
                coach_view = gr.Markdown()

        # new_round sets the header too; submit/retry leave it, so they omit it.
        round_out = [header, board_img, status, state, verdict_view, coach_view,
                     submit_btn, undo_btn, retry_btn]
        review_out = [board_img, status, state, verdict_view, coach_view,
                      submit_btn, undo_btn, retry_btn]
        move_out = [board_img, status, state, submit_btn, undo_btn]

        app.load(new_round, outputs=round_out)
        new_btn.click(new_round, outputs=round_out)
        board_img.select(on_click, inputs=[state], outputs=move_out)
        undo_btn.click(on_undo, inputs=[state], outputs=move_out)
        submit_btn.click(on_submit, inputs=[state], outputs=review_out)
        retry_btn.click(retry, inputs=[state], outputs=review_out)

    return app


def main():
    build_app().launch(head=_TINT)     # tint the reload flicker to the board colour


if __name__ == "__main__":
    main()
