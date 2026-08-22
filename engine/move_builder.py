"""Build a checker play one click at a time, by prefix-filtering legal paths.

The UI computes the legal plays once per turn with `generate_move_paths` (each a
tuple of (src, dst, die) hops), then constructs the student's play incrementally:

    click a source  -> `legal_destinations` highlights where it may go
    click a dest    -> `choose_hop` resolves the full hop; append it to hops_so_far
    (re-render with `apply_hops`; undo == drop the last hop)
    Submit enabled  -> when `is_complete`

Everything here is pure -- it reasons only over the pre-generated `paths` and the
hops chosen so far -- so it's fully unit-testable with hand-built paths, no Gradio.

Conventions (matching engine.moves): a point is 0-23; src == BAR_IDX (24) is the
bar; dst == OFF (-1) is bearing off. A hop is (src, dst, die); a path is a tuple
of hops; all paths from `generate_move_paths` share the max-play length.
"""
from engine.board import Board
from engine.moves import BAR_IDX, OFF, move_one

Hop = tuple[int, int, int]      # (src, dst, die)
Path = tuple[Hop, ...]


def matching_paths(paths: set[Path], hops_so_far: tuple[Hop, ...]) -> set[Path]:
    n = len(hops_so_far)
    return {path for path in paths if path[:n] == hops_so_far}


def legal_sources(paths: set[Path], hops_so_far: tuple[Hop, ...]) -> set[int]:
    """Source points that have a legal NEXT hop given `hops_so_far` -- the
    checkers (or the bar) the student may pick up now. Empty once the play is
    complete (no hop left to make)."""
    src_set = set()
    if is_complete(paths, hops_so_far):
        return src_set
    matches = matching_paths(paths, hops_so_far)
    hop_count = len(hops_so_far)
    for path in matches:
        src_set.add(path[hop_count][0])
    return src_set


def legal_destinations(paths: set[Path], hops_so_far: tuple[Hop, ...], source: int) -> set[int]:
    """The destination points a checker from `source` may move to as the next hop
    -- the highlight set after the student clicks `source`. `OFF` means bear off."""
    dest_set = set()
    if is_complete(paths, hops_so_far):
        return dest_set
    matches = matching_paths(paths, hops_so_far)
    hop_count = len(hops_so_far)
    for path in matches:
        if path[hop_count][0] == source:
            dest_set.add(path[hop_count][1])
    return dest_set


def choose_hop(paths: set[Path], hops_so_far: tuple[Hop, ...], source: int, dest: int) -> Hop:
    """The full (src, dst, die) hop for a clicked `source` -> `dest`, ready to
    append to `hops_so_far`. Resolves the die (disambiguating a bear-off
    overshoot). Raises ValueError if source -> dest is not a legal next hop."""
    matches = matching_paths(paths, hops_so_far)
    n = len(hops_so_far)
    candidates = [path[n] for path in matches if path[n][0] == source and path[n][1] == dest]
    if not candidates:
        raise ValueError(f"no legal hop {source} -> {dest}")
    return min(candidates, key=lambda h: h[2])

def is_complete(paths: set[Path], hops_so_far: tuple[Hop, ...]) -> bool:
    max_play_length = len(next(iter(paths)))
    return len(hops_so_far) == max_play_length


def apply_hops(board: Board, hops_so_far: tuple[Hop, ...]) -> Board:
    for src, _dst, die in hops_so_far:
        board = move_one(board, src, die)
    return board