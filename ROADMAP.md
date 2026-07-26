# Roadmap

The project has split into **two tracks** that share the engine:

- **Track 1 — the Coach (the product/goal).** A personal backgammon coach:
  analyze a position, rank the plays, and explain *why* one beats another.
  Its equity oracle is **gnubg-nn** (a pip-installable, superhuman, tested
  neural net), not a from-scratch net — trustworthy numbers now, and a
  mediocre homegrown net would make the coach confidently explain wrong
  moves. Runs on hand-fed positions / positions from playing in-app.
- **Track 2 — the RL benchmark (learning / portfolio).** Backgammon as a
  clean testbed to build and compare RL methods (TD(λ) → policy gradient →
  mini-AlphaZero), benchmarked against gnubg-nn as a superhuman reference.
  No product deadline; this is the "RL chops" track.
- **Track 3 — synthesis.** Point the coach's explainability layer at *your
  trained agents* to explain what they learned / where they diverge from
  gnubg. The novel payoff that justifies the split. Last.

Environment + oracle details live in the `analysis-oracle-and-env` memory
(native arm64 Python 3.12 via uv; `gnubg-nn` API + cumulative equity).

## Status (done)

- [x] `engine/board.py` — state, flip, pip count, invariants. Tested.
- [x] `engine/moves.py` — `move_one` / `extend` / `generate_moves`. Tested.
- [x] `engine/game.py` — turn loop, dice, dance/forfeit, win + single/gammon/
      backgammon classification. `Agent` seam carries `(board, dice,
      afterstates)`. Tested.
- [x] `engine/notation.py` — `describe_move` (targeted DFS via `move_one`,
      hit-aware, 1-24 notation). Tested.
- [x] `agent/random_agent.py` — uniform pick behind the seam. Tested.
- [x] `agent/human_agent.py` — menu + injected-I/O choice, auto-plays forced
      moves. Tested.
- [x] `coach/analysis.py` — `OutcomeDist` / `MoveAnalysis` / `Analysis` +
      `AnalysisProvider` protocol (equity is provider-supplied, not baked in).
- [x] `coach/scoring.py` — `cubeless_equity` (opt-in scoring policy).
- [x] `coach/gnubg_provider.py` — `Board <-> gnubg` converter, `GnubgProvider`.
      Tested (converter anchored, provider validated on the opening).
- [x] `ui/play.py` — interactive game vs the random agent (`python -m
      ui.play`). **Playable (P1 done).**

## Track 1 — the Coach

### P2 — `SkillAgent` (a tunable, beatable opponent)
Wrap `GnubgProvider`: score the legal afterstates and **softmax-sample by
temperature** instead of always taking the best. τ→0 = superhuman; higher τ =
weaker. Optionally cap to moves within ε of best (no catastrophic blunders).
Calibrate τ by measuring equity-loss-per-move vs gnubg-best (intermediate ≈
0.02-0.03/move). Drops into `play_interactive` in place of `random_agent`.
- Note: doesn't need the dice — it scores the afterstates the loop hands it.

### P3 — the explainability layer (the MVP feature)
Turn an `Analysis` into natural language: *why* the best play beats the
alternatives, grounded in the `OutcomeDist` deltas + equity-loss.
- **Prompt an existing LLM (few-shot), do NOT fine-tune** — the analysis is
  done by the engine; the LLM narrates grounded evidence. Fine-tuning's cost
  is a dataset we don't have; prompt + good grounding is enough.
- Guard against the rationalization failure: feed decomposed evidence
  (outcome-distribution deltas, position features like shots/blots/pips), and
  constrain the model to explain only from the numbers, never invent reasons.
- Consider active-recall UX: predict-then-reveal (you guess, then it explains
  the gap) — stickier for building intuition than passive reading.
- Cube coaching slots in here once the cube module (deferred) exists.

## Track 2 — the RL benchmark

Build and compare RL methods on backgammon, each benchmarked against
gnubg-nn. Same bottom-up, test-as-you-go discipline. (Install numpy/torch
into the venv when starting — `uv pip install -r requirements.txt`.)

### `agent/value_net.py` — value function
Board → feature vector (classic TD-Gammon encoding ~198 units) → small net.
**Multi-outcome head, not a single win prob:** predict the cumulative
distribution (win, win_gammon, win_backgammon, lose_gammon, lose_backgammon),
matching gnubg's convention and `OutcomeDist`. Cubeless equity uses
`coach/scoring.py:cubeless_equity`.
- Requires `game.py`'s gammon/backgammon terminal detection (done) as the TD
  target.
- Tests: outputs form a valid distribution; `flip` symmetry
  (`equity(b) ≈ -equity(flip(b))`) — catches encoding bugs early.

### `agent/td_agent.py` — greedy play over the net
Score each afterstate with `value_net`, pick the max. Test with a mock net.

### `training/self_play.py` — generate games
Agent plays itself, recording per-game position trajectories. Test: a game
yields a well-formed trajectory (legal states, terminal end).

### `training/train.py` — TD(λ) learning loop
TD updates over trajectories (see Tesauro's paper + Sutton & Barto's
TD-Gammon case study — read when starting this). Behavioral checks: win-rate
vs random climbs; a few gradient steps reduce TD error on a fixed batch.

### Later method comparisons
Policy-gradient / actor-critic agent; a mini-AlphaZero (self-play + MCTS +
value/policy net). Benchmark all against gnubg-nn: convergence speed, params,
compute, final strength.

## Track 3 — synthesis
Feed *your* trained agents' analyses into the P3 explainability layer:
explain what the agent learned, or narrate where it disagrees with gnubg and
why. Interpretability of a learned RL policy — the strongest frontier signal.

## Deferred: cube module
Everything is cubeless. Double/take/drop is a layer on the same outputs (no
retraining): Janowski's cubeful model + cube ownership for money, or a
match-equity table (gnubg-nn ships `equities.value`) + score for match play.

## Testing philosophy
Engine/agents: minimal hand-derived example tests (one per branch/leaf) +
property tests (afterstates `is_valid`, absolute-anchored converters). In
`training/`, exact outputs stop being hand-derivable — shift to behavioral /
learning-curve checks.
