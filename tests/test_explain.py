from engine.board import starting_board
from coach.analysis import Analysis, MoveAnalysis, OutcomeDist
from coach.evidence import build_evidence, OutcomeDelta, FeatureDelta, SideDelta, TupleDelta
from coach.features import SideFeatures
from coach.explain import (explain, render_evidence, SYSTEM_PROMPT, _pct, _signed_pct,
                           _outcome_line, _outcome_delta_line, _side_features_line,
                           _side_delta_line, _feature_delta_block)
from tests.test_moves import mk


class _StubLLM:
    """Captures the prompts it is called with and returns a canned reply."""
    def __init__(self, reply="the coach's explanation"):
        self.reply = reply
        self.system = self.user = None

    def __call__(self, system, user):
        self.system, self.user = system, user
        return self.reply


def _dist(win):
    return OutcomeDist(win, 0.0, 0.0, 0.0, 0.0)

def _move(board, equity, outcome=None, notation=None):
    return MoveAnalysis(after_state=board, outcome=outcome or _dist((equity + 1) / 2),
                        equity=equity, notation=notation)

def _side(**overrides) -> SideFeatures:
    base = dict(pips=0, point_counts=(0,) * 24, blots=(), points_made=(),
                stripped_points=(), stacked_points=(), anchors=(), advanced_anchor=None,
                home_board_made_points=(), prime_ranges=(), longest_prime=0,
                checkers_in_opponent_home=0, checkers_on_deep_points=0, on_bar=0, borne_off=0)
    base.update(overrides)
    return SideFeatures(**base)

def _sdelta(**overrides) -> SideDelta:
    empty = TupleDelta((), ())
    base = dict(pips=0, point_shifts=(), blots=empty, points_made=empty,
                stripped_points=empty, stacked_points=empty, anchors=empty,
                home_board_made_points=empty, prime_ranges=empty, longest_prime=0,
                checkers_in_opponent_home=0, checkers_on_deep_points=0, on_bar=0, borne_off=0)
    base.update(overrides)
    return SideDelta(**base)


# A hand-built 2-play analysis with a deliberately clear delta so the render
# assertions are obvious. The boards are MINIMAL (not full 15-checker positions)
# and the notations are just labels -- render only echoes notation and needs the
# feature/equity deltas, which these boards produce:
#   best   = my 5-point MADE (2 on index 4) and the opponent HIT (sent to the bar)
#   chosen = my 5-point a BLOT (1 on index 4), no hit
# => chosen-vs-best: made -[5], blots +[5], opp on bar -1, equity lost 0.20.
def _made_point_vs_blot_evidence():
    best = mk({4: 2}, opp_bar=1)     # 5-point made (idx 4 = point 5); opp on the bar = a hit
    chosen = mk({4: 1})              # 5-point now a lone checker (a blot); no hit
    analysis = Analysis(position=starting_board(), dice=(3, 1),
                        moves=(_move(best, 0.30, notation="8/5 6/5"),
                               _move(chosen, 0.10, notation="13/10 24/23")))
    return build_evidence(analysis, chosen)


# --- prompt plumbing ---------------------------------------------------

def test_explain_passes_prompts_and_returns_the_reply():
    llm = _StubLLM("make your 5-point")
    out = explain(_made_point_vs_blot_evidence(), llm)
    assert out == "make your 5-point"
    assert llm.system == SYSTEM_PROMPT              # constraints/definitions go in system
    assert "Roll: 3-1" in llm.user                  # facts go in the user message


def test_system_prompt_carries_the_rules_perspective_and_vocabulary():
    assert "name the best play EXPLICITLY" in SYSTEM_PROMPT            # no oblique references
    assert "Reason ONLY from the numbers provided" in SYSTEM_PROMPT   # anti-hallucination rule
    assert "opponent's 5-point is your 20-point" in SYSTEM_PROMPT     # perspective rule
    assert "CUMULATIVE" in SYSTEM_PROMPT                             # outcome-% convention
    for term in ("blot", "made point", "stripped", "stacked", "anchor", "prime", "pip"):
        assert term in SYSTEM_PROMPT


# --- number / line formatters ------------------------------------------

def test_percent_formatters():
    assert _pct(0.551) == "55.1%" and _pct(0.0) == "0.0%"
    assert _signed_pct(0.1) == "+10.0%"
    assert _signed_pct(-0.05) == "-5.0%"
    assert _signed_pct(0.0) == "+0.0%"               # zero still carries a sign


def test_outcome_line_renders_the_cumulative_distribution():
    d = OutcomeDist(0.55, 0.17, 0.01, 0.12, 0.005)
    assert _outcome_line(d) == \
        "win 55.0% (gammon 17.0%, bg 1.0%); lose-gammon 12.0%, lose-bg 0.5%"


def test_outcome_delta_line_signs_each_term():
    d = OutcomeDelta(-0.05, -0.05, -0.02, 0.04, 0.02)
    assert _outcome_delta_line(d) == \
        "win -5.0%; win-gammon -5.0%; win-bg -2.0%; lose-gammon +4.0%; lose-bg +2.0%"


# --- side lines (absolute features, and the delta) ---------------------

def test_side_features_line_lists_only_present_features():
    sf = _side(pips=120, blots=(7, 13), points_made=(6, 8), anchors=(20,),
               longest_prime=3, home_board_made_points=(6,), on_bar=1, borne_off=2)
    assert _side_features_line(sf) == (
        "pips 120, blots [7, 13], made [6, 8], anchors [20], longest prime 3, "
        "home-board points 1, on bar 1, borne off 2")
    assert _side_features_line(_side(pips=167)) == "pips 167"   # a bare side: just the race


def test_side_delta_line_labels_and_signs_every_changed_field():
    sd = _sdelta(pips=-5, blots=TupleDelta((7,), ()), points_made=TupleDelta((), (5,)),
                 stripped_points=TupleDelta((6,), ()), stacked_points=TupleDelta((), (8,)),
                 anchors=TupleDelta((20,), ()), home_board_made_points=TupleDelta((), (3,)),
                 prime_ranges=TupleDelta(((5, 7),), ()), longest_prime=2,
                 checkers_in_opponent_home=-1, checkers_on_deep_points=1, on_bar=-1, borne_off=2)
    assert _side_delta_line(sd) == (
        "pips -5; made -[5]; blots +[7]; stripped +[6]; stacked -[8]; anchors +[20]; "
        "home points -[3]; primes +[(5, 7)]; longest prime +2; rear checkers -1; "
        "deep checkers +1; on bar -1; borne off +2")

def test_side_delta_line_reports_no_change_when_nothing_moved():
    assert _side_delta_line(_sdelta()) == "no change"

def test_side_delta_line_shows_both_added_and_removed_for_one_field():
    # a field that both gains and loses points -> "+[..] -[..]" in one fragment
    sd = _sdelta(points_made=TupleDelta((5,), (7,)))   # made the 5-point, gave up the 7
    assert _side_delta_line(sd) == "made +[5] -[7]"


def test_feature_delta_block_omits_the_pip_lead_line_when_unchanged():
    fd = FeatureDelta(me=_sdelta(blots=TupleDelta((7,), ())), opp=_sdelta(), pip_lead=0)
    block = _feature_delta_block(fd)
    assert "pip lead" not in block                  # zero delta -> line dropped
    assert "you: blots +[7]" in block
    assert "opp: no change" in block


# --- render_evidence (the assembled user prompt) -----------------------

def test_render_surfaces_ranking_equity_loss_and_the_key_deltas():
    text = render_evidence(_made_point_vs_blot_evidence())
    assert "YOUR PLAY (rank 2 of 2): 13/10 24/23" in text
    assert "BEST PLAY: 8/5 6/5" in text
    assert "equity +0.100" in text                  # chosen equity, signed
    assert "equity +0.300" in text                  # best equity, signed
    assert "equity lost: 0.200" in text             # 0.30 - 0.10
    assert text.count("equity lost") == 1           # ONLY the chosen play, not the best
    assert "Other legal plays" not in text          # 2 plays -> no alternatives section
    # the mechanistic why: chosen swaps a made point for a blot and skips the hit
    assert "made -[5]" in text
    assert "blots +[5]" in text
    assert "on bar -1" in text                      # opp: the forgone hit
    assert "pip lead -20" in text                   # the pip-lead delta line
    # the strategic why: the outcome shift is shown, signed
    assert "win " in text and "Outcome change" in text


def test_render_lists_alternatives_with_equity_loss():
    a = Analysis(position=starting_board(), dice=(3, 1),
                 moves=(_move(mk({5: 2}), 0.30, notation="a"),
                        _move(mk({6: 2}), 0.10, notation="b"),
                        _move(mk({7: 2}), -0.20, notation="c"),
                        _move(mk({8: 2}), -0.50, notation="d")))
    text = render_evidence(build_evidence(a, mk({7: 2})))   # chose c (rank 3)
    assert "Other legal plays (equity lost vs best):" in text
    assert "  b  -0.200" in text                    # 0.30 - 0.10
    assert "  d  -0.800" in text                    # 0.30 - (-0.50)


def test_render_falls_back_to_unnamed_for_missing_notation():
    # notation is Optional; a None must render as "(unnamed)" in both the play
    # blocks and the alternatives list, not crash or print "None".
    a = Analysis(position=starting_board(), dice=(2, 1),
                 moves=(_move(mk({5: 2}), 0.30, notation=None),    # best
                        _move(mk({6: 2}), 0.10, notation=None),    # chosen (rank 2)
                        _move(mk({7: 2}), -0.20, notation=None)))  # alternative (rank 3)
    text = render_evidence(build_evidence(a, mk({6: 2})))
    assert "YOUR PLAY (rank 2 of 3): (unnamed)" in text
    assert "BEST PLAY: (unnamed)" in text
    assert "  (unnamed)  -0.500" in text            # the alternative row, 0.30 - (-0.20)


def test_render_dance_explains_the_forfeit_and_still_calls_the_llm():
    dance = build_evidence(Analysis(position=starting_board(), dice=(6, 3), moves=()),
                           mk({1: 1}))
    text = render_evidence(dance)
    assert "Roll: 6-3" in text
    assert "a dance" in text.lower()

    llm = _StubLLM()
    assert explain(dance, llm) == "the coach's explanation"
    assert "a dance" in llm.user.lower()            # the dance context reached the model
