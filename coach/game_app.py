"""Gradio GUI for a full game against the gnubg-backed agent, with a live coach.

You play X (blue), the opponent plays O (red). Each of your turns you build a play
on the click-to-move board (same widget as the quiz), Submit it, read the coach's
instant verdict (and optionally a narrated Explain), then Continue -- at which
point the opponent rolls and plays, and the board comes back for your next turn.

Perspective is the one tricky part. The app's board is ALWAYS held in your seat
(your checkers positive, bearing off toward your home). For the opponent's turn we
flip to its seat, let `play_turn` + `SkillAgent` choose, check the win in that
seat, then flip back to yours -- so you always see yourself at the bottom.

    your turn:   generate_move_paths -> click-to-move -> Submit (grade) -> Continue
    Continue:    flip -> opponent play_turn -> win? -> flip back -> your next roll

The core is three pure functions (`new_state`, `_review_submission`, `_advance`);
the Gradio callbacks are thin wrappers that render state. Run: python -m coach.game_app
"""
import random

import gradio as gr
import pandas as pd

from engine.board import flip, starting_board
from engine.game import roll_dice, play_turn, is_win, classify_win
from engine.moves import generate_move_paths
from engine.move_builder import is_complete, apply_hops
from engine.notation import describe_move
from agent.skill_agent import SkillAgent
from coach.analysis import AnalysisProvider
from coach.app import _click, _highlights, _TINT   # reuse the quiz's move-building
from coach.board_svg import point_at, WIDTH
from coach.evidence import build_evidence
from coach.explain import explain, LLM
from coach.grade import grade
from coach.game_coach import report_card_text
from coach.gnubg_provider import GnubgProvider
from coach.llm import make_llm
from coach.raster import board_image, SCALE

# Opponent strength: SkillAgent temperature (tau) mapped to skill level, calibrated
# to average equity lost per move (~gnubg PR = loss/move x 500) over sampled 2-ply
# decisions. Lower tau = stronger. See the calibration in the commit that added this.
DIFFICULTY = {
    "Beginner": 0.08,       # ~PR 25
    "Intermediate": 0.045,  # ~PR 11
    "Advanced": 0.03,       # ~PR 6
    "Expert": 0.02,         # ~PR 3
}
DEFAULT_LEVEL = "Intermediate"


def _mirror_notation(notation: str) -> str:
    """Rewrite opponent notation from ITS numbering into YOURS, so its move reads
    against the board you see. Each point p becomes 25 - p (your-point/their-point
    mirror); 'bar', 'off', and the hit marker '*' are preserved. So the opponent's
    '13/7*' shows as '12/18*' -- the checker you actually watch move."""
    def point(token: str) -> str:
        star = "*" if token.endswith("*") else ""
        token = token[:-1] if star else token
        if token in ("bar", "off"):
            return token + star
        return f"{25 - int(token)}{star}"
    return " ".join("/".join(point(t) for t in move.split("/"))
                    for move in notation.split())


# --- pure game state (no Gradio) -------------------------------------------
#
# phase: "build"     -- your turn, building a move
#        "dance_me"  -- your turn, no legal move (Continue to forfeit)
#        "review"    -- move submitted, showing the verdict (Continue for opp)
#        "gameover"  -- someone has won
#
# The board is your pre-move position; `hops` is the move you're building, so the
# displayed board is always apply_hops(board, hops). `reviewed` is carried only so
# the reused `_click` / `_highlights` stay inert outside the build phase.

def new_state(rng: random.Random) -> dict:
    board = starting_board()
    dice = roll_dice(rng)
    paths = generate_move_paths(board, dice)
    return {"board": board, "dice": dice, "paths": paths, "hops": (), "source": None,
            "phase": "build" if paths else "dance_me", "reviewed": not paths,
            "log": ["New game -- you are X (blue). Good luck!"],
            "verdict": "", "coach": "", "evidence": None, "can_explain": False,
            "verdicts": [],     # every graded move, for the end-of-game report card
            "stats": []}        # per-move (win%, cumulative loss, error rate) for the charts


def _review_submission(state: dict, provider: AnalysisProvider) -> dict:
    """Grade the play you just built and move to the review phase (or gameover)."""
    after = apply_hops(state["board"], state["hops"])
    evidence = build_evidence(provider.analyze(state["board"], state["dice"]), after)
    verdict = grade(evidence)
    move = describe_move(state["board"], after, state["dice"])

    n = len(state["verdicts"]) + 1                     # this is your n-th coached move
    cum_loss = (state["stats"][-1]["cum_loss"] if state["stats"] else 0.0) + verdict.equity_loss
    stat = {"move": n, "win": evidence.chosen.outcome.win,   # win% of the play you made
            "cum_loss": cum_loss, "err": cum_loss / n}       # error rate = avg loss/move

    new = {**state, "phase": "review", "reviewed": True, "source": None,
           "evidence": evidence, "can_explain": verdict.equity_loss > 0, "coach": "",
           "verdict": f"### Verdict\n{verdict.line}",
           "verdicts": state["verdicts"] + [verdict],
           "stats": state["stats"] + [stat],
           "log": state["log"] + [f"You play {move} -- {verdict.line}"]}
    if is_win(after):
        outcome = classify_win(after)
        new["phase"] = "gameover"
        new["verdict"] += f"\n\n**You win -- {outcome.name.lower()} ({int(outcome)} pt)!**"
        new["log"] = new["log"] + [f"You win ({outcome.name.lower()})!"]
    return new


def _advance(state: dict, rng: random.Random, opponent) -> dict:
    """Play the opponent's turn, then set up your next turn (or end the game).

    `after` is your afterstate (or your unchanged board if you danced). We flip to
    the opponent's seat to play + judge the win there, then flip back to yours.
    """
    after = apply_hops(state["board"], state["hops"])
    opp_before = flip(after)
    opp_dice = roll_dice(rng)
    opp_after, moved = play_turn(opp_before, opp_dice, opponent)
    move = (_mirror_notation(describe_move(opp_before, opp_after, opp_dice)) if moved
            else "no legal move -- forfeits")
    log = state["log"] + [f"Opponent rolls {opp_dice[0]}-{opp_dice[1]}: {move}."]

    base = {**state, "hops": (), "source": None, "verdict": "", "coach": "",
            "evidence": None, "can_explain": False}
    if is_win(opp_after):
        outcome = classify_win(opp_after)
        return {**base, "board": flip(opp_after), "phase": "gameover", "reviewed": True,
                "verdict": f"### Result\n**Opponent wins -- {outcome.name.lower()} "
                           f"({int(outcome)} pt).**",
                "log": log + [f"Opponent wins ({outcome.name.lower()})."]}

    my_board = flip(opp_after)
    my_dice = roll_dice(rng)
    paths = generate_move_paths(my_board, my_dice)
    if not paths:
        log = log + [f"You roll {my_dice[0]}-{my_dice[1]}: no legal move -- you dance."]
    return {**base, "board": my_board, "dice": my_dice, "paths": paths,
            "phase": "build" if paths else "dance_me", "reviewed": not paths, "log": log}


# --- rendering (state -> the full set of UI outputs) -----------------------

def _status(state: dict) -> str:
    phase, d = state["phase"], state["dice"]
    if phase == "gameover":
        return "Game over -- click **New game** to play again."
    if phase == "dance_me":
        return f"You rolled **{d[0]}-{d[1]}** -- no legal move. Click **Continue**."
    if phase == "review":
        return "Move submitted -- read the verdict, optionally **Explain**, then **Continue**."
    roll = f"You rolled **{d[0]}-{d[1]}**. "     # build
    if state["source"] is not None:
        return roll + "Click a highlighted destination (or another checker to reselect)."
    if is_complete(state["paths"], state["hops"]):
        return roll + "Move complete -- click **Submit move**."
    return roll + "Click a checker to start your move."


_WIN, _CUM, _ERR = "win %", "equity lost (cumulative)", "avg loss / move"


def _charts(state: dict) -> tuple:
    """Three tidy DataFrames (win%, cumulative equity lost, error rate) indexed by
    your move number -- one row per coached move, empty before your first."""
    stats = state["stats"]
    win = pd.DataFrame([{"move": s["move"], _WIN: round(s["win"] * 100, 1)} for s in stats],
                       columns=["move", _WIN])
    cum = pd.DataFrame([{"move": s["move"], _CUM: round(s["cum_loss"], 3)} for s in stats],
                       columns=["move", _CUM])
    err = pd.DataFrame([{"move": s["move"], _ERR: round(s["err"], 4)} for s in stats],
                       columns=["move", _ERR])
    return win, cum, err


def _render(state: dict) -> tuple:
    """Every callback returns this: one function computes the whole UI from state,
    so outputs can never drift out of sync with the returns."""
    phase = state["phase"]
    display = apply_hops(state["board"], state["hops"])
    dice = None if phase == "gameover" else state["dice"]
    hl = _highlights(state) if phase == "build" else set()
    used = [hop[2] for hop in state["hops"]]
    is_build = phase == "build"
    # every move is kept; show all, newest first, in a scrollable panel (see _HEAD)
    log = "### Moves\n" + "\n".join(f"- {line}" for line in reversed(state["log"]))
    coach = state["coach"]
    if phase == "gameover":
        card = report_card_text(state["verdicts"])
        if card:
            coach = f"### Report card\n```\n{card}\n```"
    win_df, cum_df, err_df = _charts(state)
    return (
        board_image(display, dice, hl, used),
        _status(state), log, state["verdict"], coach, state,
        gr.update(interactive=is_build and bool(state["hops"])),                       # undo
        gr.update(interactive=is_build and is_complete(state["paths"], state["hops"])),  # submit
        gr.update(visible=phase in ("review", "dance_me"),
                  value="Continue" if phase == "review" else "Continue (you dance)"),   # continue
        gr.update(visible=phase == "review" and state["can_explain"]),                  # explain
        win_df, cum_df, err_df,                                                         # live charts
    )


def build_app(provider: AnalysisProvider | None = None, llm: LLM | None = None,
              rng: random.Random | None = None, opponent=None) -> gr.Blocks:
    provider = provider or GnubgProvider(plies=2)
    llm = llm or make_llm()
    rng = rng or random.Random()
    fixed_opponent = opponent      # tests may inject a deterministic Agent; else per-level

    def new_game():
        return _render(new_state(rng))

    def on_click(state, evt: gr.SelectData):
        if state["phase"] != "build":
            return _render(state)
        x, y = evt.index
        return _render(_click(state, point_at(x / SCALE, y / SCALE)))

    def on_undo(state):
        if state["phase"] != "build":
            return _render(state)
        return _render({**state, "hops": state["hops"][:-1], "source": None})

    def on_submit(state):
        if state["phase"] != "build" or not is_complete(state["paths"], state["hops"]):
            return _render(state)
        return _render(_review_submission(state, provider))

    def on_continue(state, level):
        if state["phase"] not in ("review", "dance_me"):
            return _render(state)
        opponent = fixed_opponent or SkillAgent(provider, DIFFICULTY[level], rng)
        return _render(_advance(state, rng, opponent))

    def on_explain(state):
        if not state.get("evidence"):
            return _render(state)
        return _render({**state, "coach": f"### Coach\n{explain(state['evidence'], llm)}"})

    with gr.Blocks(title="Backgammon") as app:
        gr.Markdown("# Backgammon\nYou are **X (blue)**, bearing off toward your home; "
                    "the opponent is **O (red)**. Your coach reviews every move.")
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                board_img = gr.Image(interactive=False, show_label=False,
                                     width=WIDTH, elem_id="board")
                status = gr.Markdown()
                with gr.Row():
                    undo_btn = gr.Button("Undo", interactive=False)
                    submit_btn = gr.Button("Submit move", variant="primary", interactive=False)
                continue_btn = gr.Button("Continue", variant="primary", visible=False)
            with gr.Column(scale=2):
                level = gr.Radio(list(DIFFICULTY), value=DEFAULT_LEVEL,
                                 label="Opponent level",
                                 info="Calibrated to gnubg PR: Expert ~3, Advanced ~6, "
                                      "Intermediate ~11, Beginner ~25 (lower is stronger).")
                verdict_view = gr.Markdown()
                explain_btn = gr.Button("Explain this move", visible=False)
                coach_view = gr.Markdown()
                with gr.Accordion("Your stats (live)", open=True):
                    win_plot = gr.LinePlot(x="move", y=_WIN, title="Win % (your play)",
                                           y_lim=[0, 100], height=170)
                    cum_plot = gr.LinePlot(x="move", y=_CUM, title="Cumulative equity lost",
                                           height=170)
                    err_plot = gr.LinePlot(x="move", y=_ERR, title="Error rate (avg loss/move)",
                                           height=170)
                log_view = gr.Markdown(elem_id="movelog")
                gr.Markdown("---")               # separate the reset from the play controls
                new_btn = gr.Button("New game")

        out = [board_img, status, log_view, verdict_view, coach_view, state,
               undo_btn, submit_btn, continue_btn, explain_btn,
               win_plot, cum_plot, err_plot]
        app.load(new_game, outputs=out)
        new_btn.click(new_game, outputs=out)
        board_img.select(on_click, inputs=[state], outputs=out)
        undo_btn.click(on_undo, inputs=[state], outputs=out)
        submit_btn.click(on_submit, inputs=[state], outputs=out)
        continue_btn.click(on_continue, inputs=[state, level], outputs=out)
        explain_btn.click(on_explain, inputs=[state], outputs=out)

    return app


# board tint (from the quiz) plus a scrollable, fixed-height move log
_HEAD = _TINT + "<style>#movelog { max-height: 320px; overflow-y: auto; }</style>"


def main():
    build_app().launch(head=_HEAD)


if __name__ == "__main__":
    main()
