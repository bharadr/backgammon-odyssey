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
- [x] `coach/gnubg_provider.py` — `Board <-> gnubg` converter, `GnubgProvider`
      with `evaluate_afterstate` (per-board eval) and `analyze` (rebuilt on OUR
      generate_moves + eval + describe_move; never gnubg's best_move, which
      analyses the wrong player on asymmetric boards). Tested, incl. asymmetric
      positions and 0-ply vs 2-ply.
- [x] `agent/skill_agent.py` — `SkillAgent`: softmax/temperature + epsilon
      blunder-floor over an `AfterstateEvaluator`. The reusable policy for ANY
      evaluator (gnubg now, trained nets later). Tested.
- [x] `ui/play.py` — interactive game (`python -m ui.play`), now vs the
      gnubg-backed `SkillAgent`. **Playable (P1 + P2 done).**
- [x] `coach/features.py` — per-side `SideFeatures` for BOTH players (opponent
      via `flip`). Canonical `point_counts` (24-tuple, this side's checkers) +
      derived named concepts: blots, points-made, stripped-points,
      stacked-points, anchors + advanced-anchor, prime-ranges + longest-prime,
      home-board points, checkers-in-opponent-home + on-deep-points, bar, off;
      + `pip_lead`. Named fields (even where derivable) so B2 deltas are plain
      field-diffs. Tested (incl. prime runs). More deferred (see P3).

## Track 1 — the Coach

### P2 — `SkillAgent` (DONE; τ uncalibrated)
Scores legal afterstates via an `AfterstateEvaluator` and **softmax-samples by
temperature** (τ→0 = superhuman; higher = weaker) with an ε blunder-floor.
Wired into `play_interactive` as the opponent at `OPPONENT_TEMPERATURE = 0.1`.
- **Still to do (optional):** calibrate τ by measuring equity-loss-per-move vs
  gnubg-best (intermediate ≈ 0.02-0.03/move); 0.1 is a guess.

### P3 — the coach: analysis + explainability (the MVP feature)

**Analysis backend — DONE:** `GnubgProvider.analyze(board, dice)` returns the
legal plays ranked by equity (our generate_moves + evaluate_afterstate +
describe_move), tested on asymmetric positions and the dance.

**The explainability layer** turns an `Analysis` + the player's chosen move
into a natural-language "why," grounded in exact numbers (never a raw board
the LLM would misread):

- **B1 `coach/features.py` — DONE.** Per-side `SideFeatures` for BOTH players
  (opponent via `flip`): pips, blots, points-made (locations), home-board
  points, checkers-back + deep-back, bar, off; + `pip_lead`. **Decision: the
  coach does NOT render the board** — features are the complete structural
  input, so the LLM never sees a raw board to misread.
- **B2 `coach/evidence.py` (next):** assemble the LLM input from an `Analysis`
  + the chosen move. NO board render. Include: the roll; `best` and `chosen`
  plays (notation, equity; `chosen` also equity-loss + rank "Kth of N"; outcome
  distribution; and `features()` of each resulting board — both sides); the
  other top-5 as `alternatives` (notation + equity + equity-loss only — full
  features on all 5 is noise); chosen-vs-best `deltas` (outcome-dist + key
  feature diffs); a dance flag. Rank/equity-loss live here, not in features.py.
- **B3 `coach/explain.py`:** `explain(evidence, llm)` with an INJECTED `llm`
  callable (stub in tests). Prompt = coach role + hard constraint (explain only
  from the provided numbers; features are authoritative; never invent) +
  few-shot + the evidence. **Few-shot an existing model; do NOT fine-tune.**
- **B4:** real `llm` via the Anthropic SDK (behind the injected interface).
- **B5 `coach/cli.py` (`python -m coach`):** position + dice + your move →
  print ranked analysis + explanation. Manual acceptance: feed a position that
  confused you on Galaxy.
- **Input-format wrinkle to resolve:** we support gnubg Position IDs; Galaxy/XG
  use XGID (different format). Decide: gnubg IDs, an XGID parser, or manual
  board entry.
- **Active-recall UX (optional):** predict-then-reveal.
- Cube coaching slots in here once the deferred cube module exists.

### Deferred coach features (NOT yet added — track these)
The coach sees the Tier-1 features above (both sides), no raw board. When
explanations feel thin, ADD a feature here — never reintroduce the error-prone
board. Stack heights and stripped points are now explicit fields (for clean B2
deltas); spare *counts* stay implicit in `point_counts` (the B3 prompt defines
"spare"). Genuinely-still-missing, roughly by value:

1. **Shot counts** (highest value — the key safety signal): "odds the opponent
   hits one of my blots", as N/36 (%). Needs ENUMERATION the LLM can't do, so
   compute by REUSING the engine, not hand-coded geometry: for the afterstate,
   `opp_view = flip(after)`; for each of the 21 distinct rolls,
   `generate_moves(opp_view, roll)`; a roll hits if any resulting move has
   `opp_bar_count > opp_view.opp_bar_count` (a hit on my checker in the flipped
   frame); sum roll weights (1/36 doubles, 2/36 not). Handles blocks,
   indirects, doubles, and the bar for free. "Can hit" (the threat), not
   "would hit".
2. **Checkers trapped behind a prime** (cross-side): a prime only matters
   relative to the opponent's checkers behind it. Needs BOTH sides' positions
   (my `prime_ranges` + opp `point_counts`), so it lives at the position level,
   not `SideFeatures`. Makes `prime_ranges` actionable.
3. **Race wastage / crossovers.** Pip count alone misses bear-off wastage and
   how many crossovers remain; needs distribution math beyond a pip total.
4. **Builders for a specific point** (low priority): which spares bear on making
   a target point. Tactical/derived; `point_counts` already exposes the raw
   spare material, so add only if explanations need it spelled out.

Also **perspective**: opponent features are in the OPPONENT's own numbering
(their 5-point = my 20-point). B3's prompt MUST state this so the coach doesn't
mis-attribute points.

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
