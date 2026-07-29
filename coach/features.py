"""Interpretable, exactly computed board features for the checker-play coach.

The engine computes objective board facts. The LLM uses those facts to explain
why one candidate move is strategically preferable to another.

`point_counts` is the canonical structural representation for one side:
index 0 is that player's 1-point and index 23 is that player's 24-point.

The remaining fields are deterministic strategic interpretations derived from
those counts. They are included even when technically derivable because they:

- prevent the LLM from miscounting or mishandling point orientation;
- give stable definitions to coaching concepts;
- make candidate moves easy to compare through explicit feature deltas.

Both sides use their own 1-24 perspective. The opponent's features are computed
by flipping the board and treating the opponent as the side to move.
"""

from dataclasses import dataclass

from engine.board import Board, flip, pip_count


PointRange = tuple[int, int]


@dataclass(frozen=True)
class SideFeatures:
    """Exactly computed features for one player, in that player's perspective."""

    # Race and exact structure
    pips: int
    point_counts: tuple[int, ...]  # Exactly 24 entries; index 0 is the 1-point

    # Point structure
    blots: tuple[int, ...]          # Points containing exactly one checker
    points_made: tuple[int, ...]    # Points containing at least two checkers
    stripped_points: tuple[int, ...]  # Made points containing exactly two
    stacked_points: tuple[int, ...]   # Points containing at least four

    # Contact structure
    anchors: tuple[int, ...]        # Made points in opponent's home: 19-24
    advanced_anchor: int | None     # Lowest-numbered anchor, e.g. 20 over 24
    home_board_made_points: tuple[int, ...]

    # Blocking structure
    prime_ranges: tuple[PointRange, ...]  # Maximal runs of >=2 made points
    longest_prime: int

    # Board strength and rear-checker burden
    checkers_in_opponent_home: int  # Checkers on points 19-24
    checkers_on_deep_points: int    # Checkers specifically on points 23-24

    # Non-point locations
    on_bar: int
    borne_off: int

    @property
    def home_board_points(self):
        return len(self.home_board_made_points)


def _prime_ranges(made: tuple[int, ...]) -> tuple[PointRange, ...]:
    """Return maximal runs of at least two consecutive made points.

    Examples:
        (3, 4, 5, 8, 9) -> ((3, 5), (8, 9))
        (3, 5, 7)       -> ()
    """
    if not made:
        return ()

    ranges: list[PointRange] = []
    start = previous = made[0]

    for point in made[1:]:
        if point == previous + 1:
            previous = point
            continue

        if previous > start:
            ranges.append((start, previous))

        start = previous = point

    if previous > start:
        ranges.append((start, previous))

    return tuple(ranges)


def _range_length(point_range: PointRange) -> int:
    start, end = point_range
    return end - start + 1


def side_features(board: Board) -> SideFeatures:
    """Compute features for the side to move, represented by positive checkers."""
    counts = tuple(max(checkers, 0) for checkers in board.points)

    made = tuple(
        point
        for point, count in enumerate(counts, start=1)
        if count >= 2
    )

    anchors = tuple(point for point in made if 19 <= point <= 24)
    primes = _prime_ranges(made)

    return SideFeatures(
        pips=pip_count(board),
        point_counts=counts,
        blots=tuple(
            point
            for point, count in enumerate(counts, start=1)
            if count == 1
        ),
        points_made=made,
        stripped_points=tuple(
            point
            for point, count in enumerate(counts, start=1)
            if count == 2
        ),
        stacked_points=tuple(
            point
            for point, count in enumerate(counts, start=1)
            if count >= 4
        ),
        anchors=anchors,
        advanced_anchor=min(anchors, default=None),
        prime_ranges=primes,
        longest_prime=max(
            (_range_length(point_range) for point_range in primes),
            default=0,
        ),
        home_board_made_points=tuple(point for point in made if point <= 6),
        checkers_in_opponent_home=sum(counts[18:24]),
        checkers_on_deep_points=sum(counts[22:24]),
        on_bar=board.bar_count,
        borne_off=board.off_count,
    )


@dataclass(frozen=True)
class PositionFeatures:
    """Features for the player being advised and their opponent."""

    me: SideFeatures
    opp: SideFeatures

    # Defined as opponent minus player, so positive always means I am ahead.
    pip_lead: int


def features(board: Board) -> PositionFeatures:
    me = side_features(board)
    opp = side_features(flip(board))

    return PositionFeatures(
        me=me,
        opp=opp,
        pip_lead=opp.pips - me.pips,
    )