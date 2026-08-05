"""Assemble the exact bundle the coaching LLM reasons from.

Pure data assembly: given an `Analysis` (already computed by a provider) and the
play the human actually chose, produce an `Evidence` object. No gnubg, no LLM --
only `features()` and arithmetic on the `Analysis` -- so it is fully unit-testable
with a hand-built `Analysis`. Turning `Evidence` into prompt *text* is B3's job.

Perspective note: afterstates stay in the MOVER's perspective (generate_moves
moves my positive checkers; it does not flip), so `features(after).me` is the
player being coached and `.opp` is the opponent. A hit therefore surfaces as
`opp.on_bar` rising -- see the tests.

Every delta is defined as CHOSEN minus BEST, so a delta describes how the human's
play differs from the best one (`blots.added` = blots the chosen play leaves that
the best play does not; `pips` > 0 = chosen leaves more pips to go).
"""
from dataclasses import dataclass
from typing import Optional

from engine.board import Board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.features import features, PositionFeatures, SideFeatures


# --- per-play views ---------------------------------------------------------

@dataclass(frozen=True)
class PlayEvidence:
    """A fully-described play: the best one and the chosen one both use this."""
    notation: Optional[str]
    equity: float
    equity_loss: float                # vs the best play; 0 for the best play
    rank: int                         # 1-based; 1 for the best play
    of_n: int                         # total legal plays
    outcome: OutcomeDist
    features: PositionFeatures        # of the resulting afterstate


@dataclass(frozen=True)
class AlternativePlay:
    """A runner-up shown only to convey the ranking -- no features/outcome."""
    notation: Optional[str]
    equity: float
    equity_loss: float


# --- deltas (chosen - best) -------------------------------------------------

@dataclass(frozen=True)
class OutcomeDelta:
    win: float
    win_gammon: float
    win_backgammon: float
    lose_gammon: float
    lose_backgammon: float


@dataclass(frozen=True)
class TupleDelta:
    """Set-difference of two point/range tuples, order preserved.

    Elements are board points (1-24) for most fields, or `PointRange` runs for
    `prime_ranges`."""
    added: tuple                      # in chosen, not best
    removed: tuple                    # in best, not chosen

    def __bool__(self) -> bool:
        return bool(self.added or self.removed)


@dataclass(frozen=True)
class SideDelta:
    pips: int
    point_shifts: tuple[tuple[int, int], ...]  # (point, chosen-best) for changed points
    blots: TupleDelta
    points_made: TupleDelta
    stripped_points: TupleDelta
    stacked_points: TupleDelta
    anchors: TupleDelta
    home_board_made_points: TupleDelta
    prime_ranges: TupleDelta
    longest_prime: int
    checkers_in_opponent_home: int
    checkers_on_deep_points: int
    on_bar: int
    borne_off: int


@dataclass(frozen=True)
class FeatureDelta:
    me: SideDelta
    opp: SideDelta
    pip_lead: int


# --- the bundle -------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    roll: tuple[int, int]
    is_dance: bool
    best: Optional[PlayEvidence]
    chosen: Optional[PlayEvidence]
    alternatives: tuple[AlternativePlay, ...]
    outcome_delta: Optional[OutcomeDelta]
    feature_delta: Optional[FeatureDelta]


# --- helpers ----------------------------------------------------------------

def _index_of(moves: tuple[MoveAnalysis, ...], after: Board) -> int:
    for i, move in enumerate(moves):
        if move.after_state == after:
            return i
    raise ValueError("chosen play is not among the legal moves for this roll")


def _play_evidence(analysis: Analysis, move: MoveAnalysis, rank: int) -> PlayEvidence:
    return PlayEvidence(
        notation=move.notation,
        equity=move.equity,
        equity_loss=analysis.equity_loss(move),
        rank=rank,
        of_n=len(analysis.moves),
        outcome=move.outcome,
        features=features(move.after_state),
    )


def _alternatives(analysis: Analysis, exclude: set[int], top_n: int) -> tuple[AlternativePlay, ...]:
    """The top-n window, minus the best and chosen plays (so neither repeats)."""
    return tuple(
        AlternativePlay(notation=move.notation, equity=move.equity,
                        equity_loss=analysis.equity_loss(move))
        for i, move in enumerate(analysis.moves[:top_n])
        if i not in exclude
    )


def _outcome_delta(best: OutcomeDist, chosen: OutcomeDist) -> OutcomeDelta:
    return OutcomeDelta(
        win=chosen.win - best.win,
        win_gammon=chosen.win_gammon - best.win_gammon,
        win_backgammon=chosen.win_backgammon - best.win_backgammon,
        lose_gammon=chosen.lose_gammon - best.lose_gammon,
        lose_backgammon=chosen.lose_backgammon - best.lose_backgammon,
    )


def _tuple_delta(best: tuple, chosen: tuple) -> TupleDelta:
    best_set, chosen_set = set(best), set(chosen)
    return TupleDelta(
        added=tuple(x for x in chosen if x not in best_set),
        removed=tuple(x for x in best if x not in chosen_set),
    )


def _point_shifts(best: tuple[int, ...], chosen: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (i + 1, chosen[i] - best[i])
        for i in range(24)
        if chosen[i] != best[i]
    )


def _side_delta(best: SideFeatures, chosen: SideFeatures) -> SideDelta:
    return SideDelta(
        pips=chosen.pips - best.pips,
        point_shifts=_point_shifts(best.point_counts, chosen.point_counts),
        blots=_tuple_delta(best.blots, chosen.blots),
        points_made=_tuple_delta(best.points_made, chosen.points_made),
        stripped_points=_tuple_delta(best.stripped_points, chosen.stripped_points),
        stacked_points=_tuple_delta(best.stacked_points, chosen.stacked_points),
        anchors=_tuple_delta(best.anchors, chosen.anchors),
        home_board_made_points=_tuple_delta(best.home_board_made_points,
                                            chosen.home_board_made_points),
        prime_ranges=_tuple_delta(best.prime_ranges, chosen.prime_ranges),
        longest_prime=chosen.longest_prime - best.longest_prime,
        checkers_in_opponent_home=chosen.checkers_in_opponent_home - best.checkers_in_opponent_home,
        checkers_on_deep_points=chosen.checkers_on_deep_points - best.checkers_on_deep_points,
        on_bar=chosen.on_bar - best.on_bar,
        borne_off=chosen.borne_off - best.borne_off,
    )


def _feature_delta(best: PositionFeatures, chosen: PositionFeatures) -> FeatureDelta:
    return FeatureDelta(
        me=_side_delta(best.me, chosen.me),
        opp=_side_delta(best.opp, chosen.opp),
        pip_lead=chosen.pip_lead - best.pip_lead,
    )


# --- the entry point --------------------------------------------------------

def build_evidence(analysis: Analysis, chosen_after: Board, top_n: int = 5) -> Evidence:
    """Bundle `analysis` and the human's chosen afterstate into `Evidence`.

    `chosen_after` must be one of the analysed legal afterstates (else
    ValueError). On the dance (no legal play) `is_dance` is True and the play
    fields are None.
    """
    if not analysis.moves:
        return Evidence(roll=analysis.dice, is_dance=True, best=None, chosen=None,
                        alternatives=(), outcome_delta=None, feature_delta=None)

    chosen_index = _index_of(analysis.moves, chosen_after)
    best_move = analysis.best
    chosen_move = analysis.moves[chosen_index]

    best = _play_evidence(analysis, best_move, rank=1)
    chosen = _play_evidence(analysis, chosen_move, rank=chosen_index + 1)

    return Evidence(
        roll=analysis.dice,
        is_dance=False,
        best=best,
        chosen=chosen,
        alternatives=_alternatives(analysis, exclude={0, chosen_index}, top_n=top_n),
        outcome_delta=_outcome_delta(best_move.outcome, chosen_move.outcome),
        feature_delta=_feature_delta(best.features, chosen.features),
    )
