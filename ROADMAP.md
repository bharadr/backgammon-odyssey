# Roadmap

Build order for the rest of the project, following the same bottom-up,
test-as-you-go discipline used for the engine. Each layer depends on the
one before it.

## Status

- [x] `engine/board.py` — state, flip, pip count, invariants. Tested.
- [x] `engine/moves.py` — `move_one` / `extend` / `generate_moves`. Tested
      bottom-up (movement law → tree shape → rule selection).
- [x] `engine/game.py` — turn loop, dice, dance/forfeit, win + single/gammon/
      backgammon classification, agent seam. Tested.
- [x] `agent/random_agent.py` — uniform pick behind the Agent seam,
      injected rng. Tested (legal pick + seed-reproducible).
- [ ] `agent/value_net.py`
- [ ] `agent/td_agent.py`
- [ ] `training/self_play.py`
- [ ] `training/train.py`

## `engine/game.py` (next)

Turn loop, dice rolling, player alternation, terminal detection, and
treating an empty `generate_moves` result as a forfeited turn (the dance).

**Interface decision:** keep move selection behind an agent interface so
agents are swappable. `game.py` rolls the dice and calls the agent with the
board + legal afterstates; the agent returns its chosen afterstate. Since
`generate_moves` already returns afterstates, this is the clean seam.

**Terminal detection must classify the outcome, not just the winner** —
this is what the value net's outputs (below) will be trained against:
- single: loser has borne off >= 1 checker
- gammon: loser has borne off 0
- backgammon: loser has borne off 0 AND still has a checker in the
  winner's home board or on the bar

Decide this before writing win detection so it matches the net's head.

## 1. `agent/random_agent.py` — do first

Pick uniformly from `generate_moves`. Trivial, but strategically first: it's
the baseline opponent and the integration-test harness.

Unlocks:
- End-to-end games, which flush out `game.py` bugs unit tests miss
  (turn alternation, win detection, the dance forfeiting a turn).
- A reference win-rate (a trained agent must beat ~50% vs random).
- First real integration test: N random-vs-random games all terminate and
  produce a legal winner.

## 2. `agent/value_net.py` — value function

Board → feature vector (classic TD-Gammon encoding is ~198 units) → small
net.

**Output a multi-outcome head, not a single win probability.** Predict the
full outcome distribution (5-6 outputs), the way modern bots do:

    P(win single), P(win gammon), P(win backgammon),
    P(lose single), P(lose gammon), P(lose backgammon)

From that, cubeless equity is:

    equity = 1*P(Ws) + 2*P(Wg) + 3*P(Wbg)
           - 1*P(Ls) - 2*P(Lg) - 3*P(Lbg)

Why bake this in now even though cube logic is deferred:
- A single win-prob only yields a race-only proxy equity (`2P-1`) that is
  wrong wherever gammons matter.
- The gammon/backgammon splits are the *required input* to cube decisions
  later (take points and doubling windows are highly sensitive to gammon
  rate). Predicting them from the start avoids a retrain when cube coaching
  is added.
- Requires the gammon/backgammon-aware terminal detection in `game.py`
  above; the TD target at game end is the categorical outcome.

Tests (in isolation):
- Outputs are probabilities that sum to 1 (a valid distribution).
- `flip` symmetry: win/lose outcomes swap under `flip`, so
  `equity(board) ≈ -equity(flip(board))`. Write this early — it catches
  encoding bugs.

**Deferred: cube module.** Everything above is *cubeless*. Double/take/drop
decisions are a separate layer on top of these outputs (not a bigger net):
Janowski's cubeful-equity model + cube ownership for money play, or a
match-equity table + score for match play. Addable later without retraining
as long as the net already predicts the gammon splits.

## 3. `agent/td_agent.py` — greedy play over the net

For each afterstate from `generate_moves`, evaluate with `value_net`, pick
the max. (This is why afterstates pay off — the agent just scores boards.)

Tests: with a mock value net, picks the highest-valued afterstate; handles
the dance (no moves) gracefully.

## 4. `training/self_play.py` — generate games

TD agent plays itself, recording the per-game sequence of positions.

Tests: a game produces a well-formed trajectory (legal states, terminal
end, sensible length distribution).

## 5. `training/train.py` — TD(λ) learning loop

Temporal-difference updates over self-play trajectories. Hardest to
unit-test; lean on behavioral checks instead:
- Learning-curve: win-rate vs the random agent climbs over training.
- Cheap regression: a few gradient steps reduce TD error on a fixed batch.

## 6. Coaching layer (the north star)

The end goal is a **coach for the user**, not just a strong self-play agent.
Built on top of the trained net + afterstate scoring:

- **Move analysis:** for a position the user is in, score every legal
  afterstate and rank them. Report equity-loss-per-move
  (`best_equity - chosen_equity`) to flag blunders.
- **Lookahead:** 0-ply (score afterstates directly) is the first cut, but
  the net's estimate is noisy — use ~2-ply lookahead (or rollouts) to
  sharpen the equity before reporting numbers. Afterstate design makes
  n-ply straightforward.
- **LLM explainability layer (wanted):** an LLM that explains *why* the
  2-ply equity numbers come out as they do — especially **why one move
  beats another**, in natural language, not just the raw equity delta. This
  turns the equity engine into an actual teacher. It consumes the net's
  outcome distribution + the per-move equities as structured input and
  narrates the reasoning (e.g. "this play wins fewer gammons but is much
  safer, and the safety is worth more here because...").
- Cube coaching slots in here once the deferred cube module (step 2) exists.

## Testing philosophy shift

Through the engine and agents, prefer minimal hand-derived example tests
(one per branch/leaf) plus property tests (every afterstate `is_valid`,
pip count monotonic, `flip` symmetry). Once in `training/`, exact outputs
stop being hand-derivable — shift to behavioral / learning-curve checks.
