import random

from coach.analysis import OutcomeDist
from agent.skill_agent import SkillAgent, _softmax_weights


# --- stubs so the POLICY is tested without gnubg or real boards ---------
#
# SkillAgent never inspects an afterstate's internals -- it only sorts them,
# hands each to the evaluator, and returns the one it chose. So an afterstate
# can be any orderable, hashable token; plain ints keep these tests about the
# policy (rank -> sample -> cap), not board construction.

def _outcome(equity: float) -> OutcomeDist:
    # gammon-free distribution whose cubeless_equity == equity  (2*win - 1)
    return OutcomeDist((equity + 1) / 2, 0.0, 0.0, 0.0, 0.0)

class _StubEvaluator:
    def __init__(self, equities: dict):        # afterstate -> equity
        self._eq = equities
    def evaluate_afterstate(self, afterstate):
        return _outcome(self._eq[afterstate])

def _setup(equities: dict):
    return _StubEvaluator(equities), set(equities)

_IGNORED = (None, None)   # the board/dice args the policy doesn't score on


# --- _softmax_weights (pure) ------------------------------------------

def test_softmax_weights_monotonic_in_equity():
    w = _softmax_weights([0.0, 0.1, 0.2], temperature=0.05)
    assert w[0] < w[1] < w[2]
    assert max(w) == 1.0                       # top equity -> exp(0) == 1

def test_softmax_lower_temperature_is_more_peaked():
    cold = _softmax_weights([0.0, 0.1], 0.02)
    warm = _softmax_weights([0.0, 0.1], 0.2)
    assert cold[1] / cold[0] > warm[1] / warm[0]

def test_softmax_weights_single_element_is_one():
    # the only element is also the max -> exp((e - e) / tau) == exp(0) == 1
    assert _softmax_weights([0.42], temperature=0.05) == [1.0]


# --- SkillAgent policy -------------------------------------------------

def test_zero_temperature_always_plays_the_best():
    ev, afters = _setup({1: 0.3, 2: 0.1, 3: -0.2})
    # temperature 0 is the argmax path: deterministic and rng-independent, so
    # it must return the best move on every seed, every call.
    for seed in range(10):
        agent = SkillAgent(ev, temperature=0.0, rng=random.Random(seed))
        assert agent(None, None, afters) == 1   # the best afterstate, always

def test_reproducible_under_a_seed():
    ev, afters = _setup({1: 0.3, 2: 0.1, 3: -0.2})
    a = SkillAgent(ev, 0.1, random.Random(7))
    b = SkillAgent(ev, 0.1, random.Random(7))
    assert [a(None, None, afters) for _ in range(20)] == \
           [b(None, None, afters) for _ in range(20)]

def test_low_temperature_mostly_picks_best():
    ev, afters = _setup({1: 1.0, 2: 0.0})       # huge gap
    agent = SkillAgent(ev, temperature=0.05, rng=random.Random(1))
    picks = [agent(None, None, afters) for _ in range(200)]
    assert picks.count(1) >= 195

def test_high_temperature_introduces_variety():
    ev, afters = _setup({1: 1.0, 2: 0.0})
    agent = SkillAgent(ev, temperature=1.0, rng=random.Random(1))
    picks = [agent(None, None, afters) for _ in range(200)]
    # warmth makes the weaker move show up often (not a one-off fluke), yet the
    # stronger move is still favoured -- sampling stays equity-weighted.
    assert 0 < picks.count(2) < picks.count(1)

def test_single_afterstate_is_returned_at_any_temperature():
    ev, afters = _setup({7: 0.3})               # only one legal play
    for temp in (0.0, 0.05, 1.0):
        agent = SkillAgent(ev, temperature=temp, rng=random.Random(0))
        assert agent(None, None, afters) == 7

def test_equal_equities_sample_roughly_uniformly():
    # equal equities -> equal weights -> uniform sampling; a bias bug (e.g.
    # always the sorted-first move) would fail this.
    ev, afters = _setup({1: 0.2, 2: 0.2, 3: 0.2})
    agent = SkillAgent(ev, temperature=0.1, rng=random.Random(5))
    picks = [agent(None, None, afters) for _ in range(300)]
    assert all(50 < picks.count(t) < 150 for t in (1, 2, 3))   # ~100 each


# --- epsilon (blunder floor) ------------------------------------------

def test_epsilon_drops_moves_beyond_the_cap():
    ev, afters = _setup({1: 0.5, 2: 0.45, 3: -0.5})   # move 3 is 1.0 below best
    agent = SkillAgent(ev, temperature=1.0, rng=random.Random(2), epsilon=0.1)
    assert 3 not in [agent(None, None, afters) for _ in range(100)]

def test_epsilon_six_keeps_every_move():
    ev, afters = _setup({1: 0.5, 2: 0.45, 3: -0.5})   # gap 1.0 < 6 -> kept
    agent = SkillAgent(ev, temperature=1.0, rng=random.Random(2), epsilon=6.0)
    assert 3 in [agent(None, None, afters) for _ in range(100)]

def test_epsilon_zero_forces_the_best():
    # epsilon 0 keeps only moves tied with the best, so even at high temperature
    # it collapses to always the best -- a distinct path from temperature 0,
    # which never reaches the sampling branch at all.
    ev, afters = _setup({1: 0.3, 2: 0.1, 3: -0.2})
    agent = SkillAgent(ev, temperature=1.0, rng=random.Random(4), epsilon=0.0)
    assert all(p == 1 for p in [agent(None, None, afters) for _ in range(50)])

def test_epsilon_cap_is_inclusive():
    # move 2 is exactly 0.5 below the best; the cap is `best - e <= epsilon`,
    # so epsilon == 0.5 keeps it (equality included) but just under drops it.
    ev, afters = _setup({1: 0.5, 2: 0.0})
    kept = SkillAgent(ev, 1.0, random.Random(3), epsilon=0.5)
    dropped = SkillAgent(ev, 1.0, random.Random(3), epsilon=0.49)
    assert 2 in [kept(None, None, afters) for _ in range(200)]
    assert 2 not in [dropped(None, None, afters) for _ in range(200)]
