"""B3: build the coach's LLM prompts from an `Evidence` object.

Two prompts: a FIXED `SYSTEM_PROMPT` (the coach's role, rules, and vocabulary)
and a per-position USER prompt that `render_evidence` builds from the evidence.
The LLM is injected as a callable `(system, user) -> str` -- a stub in tests, the
Anthropic SDK in B4 -- so the exact text the model sees is unit-testable offline.
"""
from typing import Protocol

from coach.analysis import OutcomeDist
from coach.evidence import (Evidence, PlayEvidence, AlternativePlay,
                            OutcomeDelta, FeatureDelta, SideDelta, TupleDelta)
from coach.features import SideFeatures


class LLM(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


SYSTEM_PROMPT = """\
You are an expert backgammon coach. You explain, in plain language, WHY the
engine's best checker play is better than the play the student actually made. If
they made the best move, then there is no need to explain.

You are given EXACT, engine-computed facts about the position and the candidate
plays. Treat them as authoritative and complete.

RULES -- follow strictly:
- When the student did not make the best play, name the best play EXPLICITLY in
  the standard checker notation given on the `BEST PLAY:` line (e.g. "play 13/7
  8/7"), early in your answer. Never refer to it only obliquely ("the better
  move", "the engine's choice") -- always state the actual move.
- Reason ONLY from the numbers provided. Do not imagine checkers, points, or
  rolls that are not in the data, and do not recompute or second-guess figures.
- If a fact is not in the data, do not assert it.
- The `blots`, `made`, `anchors`, and `stripped` lists are EXHAUSTIVE and
  authoritative. A point NOT in `blots [...]` is NOT a blot; a point in
  `made [...]` is safely covered. Never infer a blot from the notation -- a move
  landing on a point often covers an existing checker rather than exposing one
  (e.g. 8/4 leaves no blot if 4 is in `made`). When you state how many blots a
  play leaves, COUNT the entries in that play's `blots [...]` list; state no more.
- Be concrete: cite the specific equity loss, the outcome-probability shifts,
  and the structural changes (points made/lost, blots, hits, anchors, primes,
  pips). Lead with the single biggest reason.
- Be concise and instructive: a short paragraph a club player can act on.

PERSPECTIVE:
- "You" = the student (the player on roll). "Opp" = the opponent.
- Each side's points are numbered 1-24 from THAT side's own view, so the
  opponent's 5-point is your 20-point. Never mix the two.
- Deltas are (your play) minus (best play): a negative win delta means your play
  wins less often; "opp on bar -1" means your play hits one fewer checker.

DEFINITIONS:
- pips: total distance left to bear off (lower is better); pip lead > 0 means you
  lead the race.
- outcome %s are CUMULATIVE, from your side: win = P(win at all); gammon =
  P(win a gammon or backgammon); bg = P(win a backgammon); lose-gammon / lose-bg
  mirror this. P(lose at all) = 100% - win, so it is not shown separately.
- blot: a point with exactly one checker (can be hit). made point: 2+ checkers.
- stripped point: a made point with exactly 2 (no spare to spend). stacked: 4+
  (often inflexible). spare: a checker beyond the second on a made point.
- anchor: a made point in the opponent's home (points 19-24); the advanced anchor
  is the most forward one. prime: a run of consecutive made points that blocks.
- home board: your points 1-6. rear/deep checkers: yours on points 19-24 / 23-24.
"""

_DANCE_NOTE = ("You had no legal move with this roll -- a dance (forfeit the "
               "turn). Briefly explain to the student why that happens.")


# --- number formatting ------------------------------------------------------

def _pct(x: float) -> str:
    """A probability as a 1-dp percent: 0.551 -> '55.1%'."""
    return f"{x * 100:.1f}%"


def _signed_pct(x: float) -> str:
    """A probability *change* as a signed percent: -0.05 -> '-5.0%'."""
    return f"{x * 100:+.1f}%"


# --- outcome distribution (absolute, and chosen-minus-best) -----------------

def _outcome_line(dist: OutcomeDist) -> str:
    """One play's cumulative outcome distribution."""
    return (f"win {_pct(dist.win)} (gammon {_pct(dist.win_gammon)}, bg {_pct(dist.win_backgammon)}); "
            f"lose-gammon {_pct(dist.lose_gammon)}, lose-bg {_pct(dist.lose_backgammon)}")


def _outcome_delta_line(delta: OutcomeDelta) -> str:
    """The chosen-minus-best change in each outcome probability."""
    return (f"win {_signed_pct(delta.win)}; win-gammon {_signed_pct(delta.win_gammon)}; "
            f"win-bg {_signed_pct(delta.win_backgammon)}; lose-gammon {_signed_pct(delta.lose_gammon)}; "
            f"lose-bg {_signed_pct(delta.lose_backgammon)}")


# --- one side's structure (absolute, and chosen-minus-best) -----------------

def _side_features_line(sf: SideFeatures) -> str:
    """A side's headline features; only the non-empty ones are shown."""
    parts = [f"pips {sf.pips}"]
    if sf.blots:
        parts.append(f"blots {list(sf.blots)}")
    if sf.points_made:
        parts.append(f"made {list(sf.points_made)}")
    if sf.anchors:
        parts.append(f"anchors {list(sf.anchors)}")
    if sf.longest_prime:
        parts.append(f"longest prime {sf.longest_prime}")
    if sf.home_board_points:
        parts.append(f"home-board points {sf.home_board_points}")
    if sf.on_bar:
        parts.append(f"on bar {sf.on_bar}")
    if sf.borne_off:
        parts.append(f"borne off {sf.borne_off}")
    return ", ".join(parts)


def _format_tuple_delta(name: str, td: TupleDelta) -> str:
    """A set-diff fragment like 'made +[5] -[7]' (only the non-empty side shown)."""
    bits = []
    if td.added:
        bits.append(f"+{list(td.added)}")
    if td.removed:
        bits.append(f"-{list(td.removed)}")
    return f"{name} {' '.join(bits)}"


def _side_delta_line(sd: SideDelta) -> str:
    """A side's change from best to chosen; only the fields that moved appear."""
    parts = []
    if sd.pips:
        parts.append(f"pips {sd.pips:+d}")
    if sd.points_made:
        parts.append(_format_tuple_delta("made", sd.points_made))
    if sd.blots:
        parts.append(_format_tuple_delta("blots", sd.blots))
    if sd.stripped_points:
        parts.append(_format_tuple_delta("stripped", sd.stripped_points))
    if sd.stacked_points:
        parts.append(_format_tuple_delta("stacked", sd.stacked_points))
    if sd.anchors:
        parts.append(_format_tuple_delta("anchors", sd.anchors))
    if sd.home_board_made_points:
        parts.append(_format_tuple_delta("home points", sd.home_board_made_points))
    if sd.prime_ranges:
        parts.append(_format_tuple_delta("primes", sd.prime_ranges))
    if sd.longest_prime:
        parts.append(f"longest prime {sd.longest_prime:+d}")
    if sd.checkers_in_opponent_home:
        parts.append(f"rear checkers {sd.checkers_in_opponent_home:+d}")
    if sd.checkers_on_deep_points:
        parts.append(f"deep checkers {sd.checkers_on_deep_points:+d}")
    if sd.on_bar:
        parts.append(f"on bar {sd.on_bar:+d}")
    if sd.borne_off:
        parts.append(f"borne off {sd.borne_off:+d}")
    return "; ".join(parts) or "no change"


# --- prompt sections (each returns a block of lines; "" means "omit") -------

def _roll_line(ev: Evidence) -> str:
    return f"Roll: {ev.roll[0]}-{ev.roll[1]}"


def _play_block(pe: PlayEvidence, label: str) -> str:
    """A labelled play: notation, equity (+ loss if any), outcome, both sides."""
    equity = f"  equity {pe.equity:+.3f}"
    if pe.equity_loss:
        equity += f"  (equity lost: {pe.equity_loss:.3f})"
    return "\n".join([
        f"{label}: {pe.notation or '(unnamed)'}",
        equity,
        f"  outcome: {_outcome_line(pe.outcome)}",
        f"  you: {_side_features_line(pe.features.me)}",
        f"  opp: {_side_features_line(pe.features.opp)}",
    ])


def _plays_block(ev: Evidence) -> str:
    """The chosen play immediately above the best play, for side-by-side reading."""
    chosen = _play_block(ev.chosen, f"YOUR PLAY (rank {ev.chosen.rank} of {ev.chosen.of_n})")
    return chosen + "\n" + _play_block(ev.best, "BEST PLAY")


def _outcome_delta_block(delta: OutcomeDelta) -> str:
    return "Outcome change (your play minus best):\n  " + _outcome_delta_line(delta)


def _feature_delta_block(fd: FeatureDelta) -> str:
    lines = ["Structural change (your play vs best):"]
    if fd.pip_lead:
        lines.append(f"  pip lead {fd.pip_lead:+d}")
    lines.append(f"  you: {_side_delta_line(fd.me)}")
    lines.append(f"  opp: {_side_delta_line(fd.opp)}")
    return "\n".join(lines)


def _alternatives_block(alternatives: tuple[AlternativePlay, ...]) -> str:
    if not alternatives:
        return ""
    lines = ["Other legal plays (equity lost vs best):"]
    lines += [f"  {a.notation or '(unnamed)'}  -{a.equity_loss:.3f}" for a in alternatives]
    return "\n".join(lines)


def render_evidence(ev: Evidence) -> str:
    """Build the user prompt: the exact, engine-computed facts for this decision.

    On the dance there is no play to compare, so we emit the roll and a note.
    Otherwise it is a sequence of blank-line-separated sections.
    """
    if ev.is_dance:
        return f"{_roll_line(ev)}\n{_DANCE_NOTE}"

    sections = [
        _roll_line(ev),
        _plays_block(ev),
        _outcome_delta_block(ev.outcome_delta),
        _feature_delta_block(ev.feature_delta),
        _alternatives_block(ev.alternatives),
    ]
    return "\n\n".join(section for section in sections if section)


def explain(evidence: Evidence, llm: LLM) -> str:
    """Render `evidence` and ask `llm` for the coaching explanation."""
    return llm(SYSTEM_PROMPT, render_evidence(evidence))
