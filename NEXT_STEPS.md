# Next steps: what we learned and what to test next

Written 2026-10-04 as a reminder for when work resumes. It summarizes the v1 / v2 / v3
experiments, the lessons about measuring, and the ideas worth testing, in the order I would
try them.

**Goal:** a v3 that beats v1 in **80%+** of head-to-head games.
**Where it stands:** v3 is at parity with v1 (about 51%), so the goal needs a different kind of
change than the tuning done so far.

---

## 1. Where we are

| Version | What it is | Result |
|---|---|---|
| **v1** (git tag `v1-baseline`) | Per-trick reward (±1 weighted by whether the team still needs a trick), counterfactual bid training, calibration term | Reference. ~92.8% games with a winner, avg score ~21 in its own self-play. |
| **v2** (`Trump/v2/`) | Team "potential" (distance to 41) + "stakes" inputs, bid reward without v1's calibration term | **Failed.** The potential made a made-bid worth ~7x less than a failed one near score 0, so bids stayed low and tricks stayed low. With identical card play, v1's bidding beat v2's bidding 77 to 23. |
| **v3, warm start** (`Trump/v3/models/`) | v1 weights + the new trick-reward table | ~47.8% vs v1 over 2000 games (no gain, no loss). |
| **v3, from scratch** (`Trump/v3/models_fresh/`) | New trick-reward table + `+1` bid exploration (8%), random weights | **~51.1% ±2** vs v1 over 2400 games (checkpoints 15000 to 26500). Parity, flat since 15000 games. The 50000-game run may have finished since: check its log. |

The v3 trick reward (see `Trump/v3/game.py`): v1's ±1 / ±0.3 base, plus a bonus based on the trick
winner's closeness to 41 (`min(1,(score+bid points)/41)`, 0.5 for negative scores) and an end-game
bonus (+1 when `score + bid points >= 41` and the winner's partner is not negative). The bonus
only counts if the winner still needs the trick; a surplus trick is worth 0.3 plus the value of
denying an opponent who still needs it.

### How big is 80%?
Measured on v1 self-play (150 games): a team gains ~3.2 points per round, the per-round team
score difference has a standard deviation of ~7.1, and a game lasts ~18 rounds. With a rough
normal model, **80% needs about +1.5 points per round per team over v1, roughly a 45-50%
increase in scoring rate.** (72% needs +1.0, 88% needs +2.0.) This is an order of magnitude
only; it ignores the exact race-to-41 rules. It is not known whether 80% is reachable at all
with this game's luck.

---

## 2. Lessons about measuring (so we don't repeat the mistakes)

- **Training-window stats can't show progress.** They include exploration noise (random bids and
  cards, `+1` bids, random starting scores). Follow the **greedy evaluation vs v1** instead.
- **100-game evaluations have a noise of about ±5 points.** Single checkpoints are meaningless;
  average several, or use 500-1000 games for any decision.
- **"Bids made %" is not a quality measure.** v2 matched v1 on it while losing badly, because a
  player who always bids low makes most of his bids. Compare points per round and head-to-head.
- **Isolate components with swap tests.** E.g. v3 bidding + v1 card play vs v1, and v1 bidding +
  v3 card play vs v1; or both teams on the same card play and different bidding nets.
- **Print the training targets for a sample case before any long run.** The v2 flaw (gain much
  smaller than loss) was visible in one printed table.
- **Keep v1's proven mechanisms by default** (counterfactual bid training in particular) and
  change one thing at a time. Removing them in v2 on theory cost a full training run.
- **The tricks a player wins depend on his bid** (the playing net chases the bid, then eases
  off). Bid targets computed from tricks won under a low bid look too pessimistic for higher
  bids; this is why low-bid equilibria are sticky and why the `+1` exploration exists.

---

## 3. Ideas to test, in the order I would try them

### 3.1 Return-based targets for the playing net (best first test)
**Problem:** each card is trained only on the reward of its own trick, so the net never learns
what a card does for later tricks (setting up a suit, saving trumps, drawing out trumps, helping
the partner). The reward table adds hand-made proxies, but the net can't plan.

**Idea:** train each card toward the **reward-to-go**: the sum (optionally discounted, e.g.
gamma 0.9-1.0) of the v3 per-trick rewards from this trick to the end of the round. It reuses the
existing reward table, summed at the end of the round for each player.

**How:** a new subclass of `TrumpGameV3` (new folder, e.g. `Trump/v4/`) that holds each player's
per-trick samples until the round ends, then rewrites each reward as its suffix sum before the
network update. No network change, so v1-shaped nets and the existing evaluation work.

**Test:** train from scratch (own `--models-dir`) and compare with the current fresh run at the
same game counts, using 500+ evaluation games. **Risk:** higher variance (rewards of up to ±3
summed over 13 tricks); if noisy, lower the learning rate, raise the card batch, or use a
discount.

### 3.2 Feasibility test of search (decides whether 80% is realistic)
**Idea:** a card player that, at each decision, samples plausible hidden hands for the other
three players (consistent with the cards played and the suits they failed to follow), simulates
the rest of the round with v1's networks for each candidate card, and plays the best on average.
Bidding can use the same rollouts (also gives honest bid targets, not biased by the bid taken).

**Test (no training):** play it against greedy v1.
- Clearly above v1 (say 65-75%): there is real headroom; capture it by training the networks to
  imitate the search results (expert iteration).
- Only ~52-55%: 80% is probably out of reach for this setup, and we should say so.

**Cost:** rough estimate of about a million network calls per game, so a couple of hours for 100
games. Not measured yet; start with few sampled hands per decision.

### 3.3 Held-out-error test for network size (kept for later)
**Question it answers:** would more layers/width help, or is the remaining error just luck in
the rewards?
1. Generate a fixed dataset (~200k card samples, and bid samples) from self-play.
2. Split it 90% training / 10% held-out.
3. Train the current size (playing net: 2x256) and a bigger one (e.g. 4x512) on the training part.
4. Compare the **error on the held-out part**: clearly lower for the big net means capacity
   matters; equal means the error is mostly irreducible luck; worse means it memorizes noise.

It measures prediction accuracy, not playing strength, so it is only a screening step. It takes
minutes. Run it before any "bigger network" training.

### 3.4 Give the net a better view of the hidden information
- **Auxiliary head** that predicts what is hidden (which suits each opponent has run out of,
  who holds the missing high cards); helps the net represent beliefs.
- **History / sequence input** (LSTM or small Transformer over the cards played in order, with
  who led and who failed to follow suit) instead of unordered per-seat sets.
- **Derived features** such as the highest unplayed card in each suit.

### 3.5 Better architecture for cards
- **Shared card scorer:** score `(state, card)` with one network that takes the card as an input,
  instead of 52 separate output slots, so knowledge transfers between cards (used in published
  Monte-Carlo card-game players, e.g. for Dou Dizhu).
- **Suit-symmetric weights** (share weights across suits) for sample efficiency.
- Actor-critic / temporal-difference learning to cut the variance of pure Monte-Carlo returns
  (the natural step after 3.1).

### 3.6 Train against a frozen v1
The nets currently learn only from self-play. Making one team a frozen v1 aligns the training
signal with the head-to-head measurement (a "league" idea).

### 3.7 Cheap tuning ideas (expect a few points at most)
- Lower the playing-net learning rate or raise `--card-batch` (e.g. 4). Rate is fixed at 1e-3 in
  `TrumpGame.__init__` (v1), so a v3/v4 subclass has to override it.
- Scale the v3 bonus down (divide by 2) to reduce reward variance (max reward is ±3 vs v1's ±1).
- Ablate the v3 reward pieces one at a time: surplus-trick rule only, end-game bonus only,
  closeness term only.
- `--random-start-prob 0` vs 0.3; slower epsilon decay (both epsilons fall to 0.02 within ~300
  games).
- **Mid-game-start evaluation** (random scores 20-38 for both sides). The v3 reward is designed
  for those situations, but every current evaluation starts at 0-0, which dilutes the effect.
  Still unbuilt.
- **Behavior probes:** count in self-play how often a player overtakes a partner who is winning a
  trick he needs, or takes a trick from an opponent near 41, for v1 vs v3.

---

## 4. Suggested order when work resumes

1. Check the end of the 50000-game fresh run (log, and a 500-1000 game evaluation of
   `Trump/v3/models_fresh`). Expect parity.
2. **3.1 return-based targets** (from scratch, own models folder).
3. **3.2 search feasibility test** in parallel, since it needs no training and answers whether
   80% is realistic.
4. Depending on 2 and 3: observation/architecture work (3.4, 3.5) or expert iteration.
5. Keep 3.3 and the cheap tuning (3.7) as side experiments.

---

## 5. Reference

### Layout
- `Trump/` is v1 (`TrumpGame`, `TrumpPlayer`, `BiddingNet`, `PlayingNet`, `train.py`).
- `Trump/v2/` is the failed potential/stakes experiment (kept for the record).
- `Trump/v3/` is the current variant: `game.py` (reward table), `player.py` (`+1` bid
  exploration), `train.py`, `evaluate.py`, `test_rewards.py`, `test_bidding.py`.
- `webapp/` is the local web table; model weights are never committed (`.gitignore`).

### Commands (run from the repository root with the project's `.venv`)
```
# Fresh v3 training (own folder; training resumes from whatever weights are in --models-dir)
python -m Trump.v3.train --games 50000 --models-dir Trump/v3/models_fresh --optimistic-bid-prob 0.08 --report-every 500 --eval-games 100

# Greedy evaluation of saved weights against v1 (seats are swapped every other game)
python -m Trump.v3.evaluate --games 500 --models-dir Trump/v3/models_fresh

# Reward table and +1 exploration checks (nothing is trained or saved)
python -m Trump.v3.test_rewards
python -m Trump.v3.test_bidding

# Play against v1 or v3 (AI model dropdown in the top bar)
python -m webapp.server
```

### Things that cost time before
- A "fresh" run in a folder that already has weights silently continues them: always use a new
  `--models-dir`.
- Evaluation runs while training is saving weights can read a half-written file; re-run it.
