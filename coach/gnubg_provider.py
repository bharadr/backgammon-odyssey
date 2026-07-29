import gnubg_nn

from engine.board import Board
from coach.analysis import OutcomeDist, MoveAnalysis, Analysis
from engine.moves import generate_moves
from coach.scoring import cubeless_equity
from engine.notation import describe_move

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


class GnubgProvider:
    """AfterstateEvaluator backed by the gnubg-nn neural net -- evaluation
    only. Move generation is the caller's job (our engine's generate_moves);
    gnubg's own move generation (best_move) is deliberately not used, because
    on asymmetric positions it analyses the wrong player."""

    def __init__(self, plies: int = 0):
        self.plies = plies

    def evaluate_afterstate(self, board: Board) -> OutcomeDist:
        """Mover-perspective outcome for an afterstate (opponent on roll next).
        gnubg's `probabilities` returns the opponent's cumulative 5-tuple, so
        we flip it to the mover with `_to_mover_perspective` (no board flip;
        verified against the known 8/5 6/5 equity)."""
        probs = gnubg_nn.probabilities(board_to_gnubg(board), self.plies)
        return _to_mover_perspective(probs)

    def analyze(self, position: Board, dice: tuple[int, int]) -> Analysis:
        """Rank every legal play for `dice` from `position`, best-first.

        Uses OUR generate_moves for the legal afterstates, gnubg for the
        per-afterstate evaluation, and describe_move for notation -- never
        gnubg's best_move (which analyses the wrong player on asymmetric
        boards). `moves` is empty on the dance (no legal play).
        """
        move_analysis_list = []
        for new_board in generate_moves(position, dice):
            outcome_dist = self.evaluate_afterstate(new_board)
            move_analysis_list.append(MoveAnalysis(after_state=new_board,
                    outcome=outcome_dist,
                    equity=cubeless_equity(outcome_dist),
                    notation=describe_move(position, new_board, dice)
                )
            )
        sorted_move_analysis_list = sorted(move_analysis_list, key=lambda x: x.equity, reverse=True)
        return Analysis(
            position=position,
            dice=dice,
            moves=tuple(sorted_move_analysis_list),
        )