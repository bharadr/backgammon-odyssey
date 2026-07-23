import gnubg_nn

from engine.board import Board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist


# --- Board <-> gnubg's [2][25] representation -------------------------------
# gnubg board: g[0] = side-to-move's checkers, g[1] = opponent's, each from
# that player's OWN perspective; index 0..23 are points (0 = ace point),
# index 24 is the bar. Off checkers are implied (15 - on board - bar).
#
# Our Board: points[i] from my perspective (i=0 my ace, i=23 my 24-point),
# positive = me, negative = opponent. The opponent moves the other way, so
# their point j sits at our index 23 - j.

def board_to_gnubg(board: Board) -> list[list[int]]:
    me = [0] * 25
    opp = [0] * 25
    for i, n in enumerate(board.points):
        if n > 0:
            me[i] = n
        elif n < 0:
            opp[23 - i] = -n
    me[24] = board.bar_count
    opp[24] = board.opp_bar_count
    return [me, opp]


def board_from_gnubg(g: list[list[int]]) -> Board:
    me, opp = g[0], g[1]
    points = tuple(me[i] - opp[23 - i] for i in range(24))
    return Board(
        points=points,
        bar_count=me[24],
        opp_bar_count=opp[24],
        off_count=15 - sum(me[:24]) - me[24],
        opp_off_count=15 - sum(opp[:24]) - opp[24],
    )


def position_id(board: Board) -> str:
    return gnubg_nn.position_id(board_to_gnubg(board))


def board_from_position_id(pid: str) -> Board:
    return board_from_gnubg(gnubg_nn.board_from_position_id(pid))


# --- The provider -----------------------------------------------------------

def _to_mover_perspective(opp_probs) -> OutcomeDist:
    # gnubg evaluates an afterstate with the OPPONENT on roll, so its 5-tuple
    # is the opponent's. Flip it back to the player who made the move: my win
    # is their loss, and my/their gammon+backgammon chances swap.
    opp_win, opp_win_g, opp_win_bg, opp_lose_g, opp_lose_bg = opp_probs
    return OutcomeDist(
        win=1.0 - opp_win,
        win_gammon=opp_lose_g,
        win_backgammon=opp_lose_bg,
        lose_gammon=opp_win_g,
        lose_backgammon=opp_win_bg,
    )


def _render(move_tuple) -> str:
    # gnubg gives a flat sequence of (from, to) point pairs, e.g. (8, 5, 6, 5).
    # 25 = the bar, 0 = borne off. Translate every point, then group the flat
    # list into "from/to" moves separated by spaces.
    def pt(p: int) -> str:
        return {25: "bar", 0: "off"}.get(p, str(p))
    labels = [pt(p) for p in move_tuple]
    return " ".join(f"{labels[i]}/{labels[i + 1]}"
                    for i in range(0, len(labels), 2))


class GnubgProvider:
    """AnalysisProvider backed by the gnubg-nn neural net."""

    def __init__(self, plies: int = 0):
        self.plies = plies

    def analyze(self, position: Board, dice: tuple[int, int]) -> Analysis:
        g_board = board_to_gnubg(position)
        _best, entries = gnubg_nn.best_move(g_board, dice[0], dice[1],
                                            n=self.plies, list=1)
        moves = []
        for key, move_tuple, probs, equity in entries:
            # gnubg stores the afterstate board in the mover's perspective
            # (so no flip here), but evaluates it with the opponent on roll --
            # which is why `probs` below is opponent-perspective and DOES get
            # flipped by _to_mover_perspective. `equity` is gnubg's own
            # (already mover-perspective), so we use it directly rather than
            # re-derive a formula at the analysis layer.
            after_state = board_from_gnubg(gnubg_nn.board_from_position_key(key))
            moves.append(MoveAnalysis(
                after_state=after_state,
                outcome=_to_mover_perspective(probs),
                equity=equity,
                notation=_render(move_tuple),
            ))
        moves.sort(key=lambda m: m.equity, reverse=True)  # best-first (defensive)
        return Analysis(position=position, dice=dice, moves=tuple(moves))
