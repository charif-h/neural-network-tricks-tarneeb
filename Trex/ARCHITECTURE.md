# Trex: neural-network architecture proposal

Same philosophy as `Trump/`: two learned decision makers, both trained by
self-play with epsilon-greedy exploration and a shared set of weights for all
four seats, everything encoded **from the deciding player's point of view**
(self / partner / opponent 1 / opponent 2 always in the same slots).

| Trump                | Trex                                                         |
|----------------------|--------------------------------------------------------------|
| `BiddingNet`         | `GameChoiceNet` - the dealer picks 1 of 5 games              |
| `PlayingNet`         | `PlayingNet` (4 trick games) + `TrexNet` (the Trex game)     |

> The four "trick" games (King of Hearts, Queens, Diamonds, Slaps) are all
> "follow suit, highest card wins" and differ only in what is penalised, so
> they share one `PlayingNet` conditioned on a game one-hot. Trex is a
> different game (action = "extend a suit layout", no tricks, forced pass), so I
> recommend its own network `TrexNet`. If you want to stay at strictly two
> networks, `TrexNet` can be folded into `PlayingNet` with a 5th game one-hot and
> a padded input; it works but wastes capacity and mixes very different
> gradients, so I'd only do that as a first prototype.

---

## 1. Game structure to implement (`TrexGame.py`)

```
for kingdom in range(4):                  # dealer = seat holding 7H at kingdom 0, then next seat
    available = {KING, QUEENS, DIAMONDS, SLAPS, TREX}
    for round in range(5):
        deal 13 cards each
        game = dealer_choice(available)   # GameChoiceNet, masked to `available`
        available.remove(game)            # 5th round is forced
        play the game                     # dealer leads the 1st trick (or plays first in Trex)
        team_delta = score of the round   # added to the team totals
```

Round scoring (per team):

| Game      | Ends when                   | Team score                                   |
|-----------|-----------------------------|----------------------------------------------|
| King      | K♥ is played                | -75 to the team whose tricks contain it      |
| Queens    | 4th queen is played         | -25 per queen in team tricks (-100 total)    |
| Diamonds  | 13th diamond is played      | -10 per diamond in team tricks (-130 total)  |
| Slaps     | 13 tricks                   | -15 per trick won (-195 total)               |
| Trex      | 4 players are out of cards  | finish order +200/+150/+100/+50, **summed per team** |

Important: the four negative games are **fixed-sum** (the points are just split
between the two teams) and Trex is fixed-sum too (500 points split), so what
matters is the **difference** between the two teams. The learning target is
therefore always `team_delta = my_team_points - opponent_team_points`, scaled
(divide by 200).

Assumptions I made (please confirm - see section 9):
* Teams are opposite seats (0-2 vs 1-3), like Trump.
* In trick games you must follow suit if you can; the trick winner leads next.
* In Trex, **passing is only allowed when no card is playable** (so pass is a
  forced move, never a network decision).
* Early termination (K♥ played, 4th queen, 13th diamond) stops the round even if
  cards remain in hands.

---

## 2. Card-slot layout

Reuse the Trump trick: a 52-slot vector `suit*13 + (ACE - rank)`, with the suit
order fixed once per round so a physical suit keeps the same slots.

* **Trex**: all four suits are strictly symmetric, so use `order_suits`
  (most valuable / longest suit first) exactly like Trump. The net gets
  suit-permutation invariance for free.
* **Trick games**: hearts and diamonds are special (K♥, diamond penalties), so
  fix the order `[♥, ♦, then the two black suits sorted by hand strength]`.

Everything below that is "52 wide" uses this same layout (hand, legal mask,
played cards, ...).

---

## 3. `GameChoiceNet` (the dealer's choice)

Replaces `BiddingNet`. It is only called once per round, by the dealer, so
there are only 20 decisions per game - data is scarce, so the design focuses
on giving it **good labels** (section 6.1).

### Input (~80 values)

| Slots   | Content                                                                    |
|---------|----------------------------------------------------------------------------|
| 0-51    | dealer's hand (one-hot, symmetric suit order by strength)                  |
| 52-55   | cards per suit / 13                                                        |
| 56-59   | hand features: honor points (A=4,K=3,Q=2,J=1) per suit / 10               |
| 60-64   | games **still available** (5-bit mask, 1 = available)                      |
| 65      | rounds left in this kingdom / 5                                            |
| 66      | kingdoms left in the match / 4                                             |
| 67-68   | team scores: dealer's team, opponents / 500                                |
| 69      | has K♥ (0/1)                                                               |
| 70      | number of queens held / 4                                                  |
| 71      | number of diamonds held / 13                                               |
| 72      | has Jack/Queen/10 of every suit: number of suits with a J / 4 (Trex start) |

The last four are redundant with the hand but are exactly the facts a human
uses to choose; they make the net converge much faster.

### Output

5 values `Q[g]` = expected `team_delta` if I play game `g` with this hand.
Illegal games (already used in this kingdom) are masked with `-inf` before the
argmax, exactly like the legality mask of `PlayingNet`.

Architecture: `Linear(73,128) - ReLU - Linear(128,128) - ReLU - Linear(128,5)`.

### The "greedy trap" and its fix

Choosing `argmax Q[g]` greedily always favours Trex (it is the only positive
game) and burns it in round 1 of every kingdom. A dealer must also
consider **what stays available for the rest of the kingdom**. Two options:

1. **Advantage form (simple, recommended first)**: train `Q[g]` on
   `team_delta - baseline[g]`, where `baseline[g]` is the running average
   of `team_delta` for game g over all hands. The net learns *how much better
   than usual* each game is for this hand, which is what the dealer actually
   wants. Pick `argmax (Q[g] - lambda * future_cost[g])`, with
   `future_cost[g] = baseline[g] normalised` as a first approximation.
2. **Kingdom value function (better)**: a tiny extra head/net
   `V(remaining_games_mask, round_index, score_diff)` trained by TD on the
   kingdom's cumulative score. Then choose
   `argmax_g  Q[g] + V(remaining \ {g})`.

Start with (1); add (2) once the playing nets are decent.

---

## 4. `PlayingNet` (King / Queens / Diamonds / Slaps)

Same shape as Trump's `PlayingNet`: 52 outputs, "value of playing the card in
slot i", masked by the legal-card mask. Plus a few game-specific inputs.

### Input (351 values)

| Slots     | Content                                                                          |
|-----------|----------------------------------------------------------------------------------|
| 0-51      | hand                                                                             |
| 52-103    | legal-cards mask                                                                 |
| 104-155   | "would win the trick now" bit per hand card                                      |
| 156-159   | **game one-hot** (King, Queens, Diamonds, Slaps)                                 |
| 160-164   | led suit one-hot + "I am leading" flag                                           |
| 165-167   | partner winning, opponent winning, players left to act / 3                       |
| 168       | **penalty points currently in this trick** (for my team's perspective if I take it) / 75 |
| 169       | tricks remaining / 13                                                            |
| 170-173   | tricks won: self, partner, opp 1, opp 2 / 13                                     |
| 174       | K♥ already played (0/1)                                                          |
| 175       | queens still to be played / 4                                                    |
| 176       | diamonds still to be played / 13                                                 |
| 177-178   | penalty points already taken this round: my team, opponents (scaled by game max) |
| 179-182   | dealer relation: me / partner / opp 1 / opp 2                                    |
| 183-338   | cards already played by partner / opp 1 / opp 2 (3 x 52)                         |
| 339-350   | **void flags**: for each of partner/opp 1/opp 2, which of the 4 suits they failed to follow |

Why these additions compared to Trump:
* Game one-hot + game-specific progress (K♥ played, queens / diamonds left):
  the same hand of cards is worth completely different things in each game.
* Void flags: in Trump the net can only infer a void from the played-cards
  vectors; here "who can be forced to eat the K♥ / queen" is the core of the
  strategy, so give it explicitly.
* "Penalty points in this trick": lets the net see that *taking* a trick is
  either bad (negative games) or irrelevant, and that dumping a queen on
  the opponent's trick is good.

### Output

52 values, masked by legality. Architecture
`Linear(351,256) - ReLU - Linear(256,256) - ReLU - Linear(256,52)`.

### Strategy notes the net must be able to learn (sanity checks)

* King of Hearts / Queens / Diamonds: dump the dangerous card on an
  *opponent's* winning trick; never on the partner's.
* Slaps: avoid winning tricks; lead low, discard high cards when safe.
* The game ends as soon as the last penalty card falls, so the remaining cards
  have no value: late rounds need the "cards still to come" inputs.

---

## 5. `TrexNet` (the Trex game)

The action is "put card X on the layout". The net outputs 52 values (one per
card slot), masked by the **playable** cards. If the mask is empty the player
passes automatically (no decision, no sample).

### Input (332 values)

| Slots     | Content                                                                          |
|-----------|----------------------------------------------------------------------------------|
| 0-51      | hand                                                                             |
| 52-103    | playable mask (cards I can legally put now)                                      |
| 104-155   | cards already on the table (the whole layout, all four suits)                    |
| 156-159   | per suit: is the Jack (opening card) already played                              |
| 160-167   | per suit: upper-frontier rank and lower-frontier rank (normalised, 0 = closed)   |
| 168-170   | cards left: partner / opp 1 / opp 2, each / 13                                   |
| 171-173   | already finished: partner / opp 1 / opp 2                                        |
| 174       | number of players already out / 4                                                |
| 175-330   | **known-not-held** cards per partner / opp 1 / opp 2 (3 x 52), inferred from passes |
| 331       | team score difference so far / 500                                               |

The key extra feature is the **pass inference**: when a player passes, every
card that was playable at that moment is a card he does *not* hold. That is
strong information (an opponent who passed on a full-open ♠ layout has no ♠
neighbour). Accumulate it in a `(3, 52)` tensor updated in `TrexGame`, same way
`played_by_seat` is tracked in Trump.

Also consider a "hold-back" feature per hand card: *does this card unlock
cards for someone else?* (e.g. playing a Jack opens a suit for everybody). The
net can learn it from the layout vectors but it is cheap to add as a 52-bit
"opens a new suit" vector.

### Output and architecture

52 values, `Linear(332,256) - ReLU - Linear(256,256) - ReLU - Linear(256,52)`.

### Strategy notes

* Because scores are **per team**, the partner finishing first is as good as
  finishing first. The net sees partner/opponent cards-left to learn when to
  help (play what unlocks the partner) versus to block (hold the card the
  opponent is waiting for - Trex's classical tactic).
* Jacks open suits: holding them and playing them late is a valid
  blocking strategy; the `opens a new suit` bit plus `cards left` inputs make it
  learnable.

---

## 6. Training

### 6.1 Labels for the game choice (the important part)

In Trump the bidding network had a cheap counterfactual label (what each bid
would have earned given the tricks actually won). Here the dealer's choice is
not evaluable from one rollout: Trex and the trick games play completely
differently. Since the **deal is known**, we can use true counterfactuals by
replaying the same deal:

```
for each dealt hand (sampled occasionally, e.g. every k-th round):
    for g in available games:
        play the round with game g  (same 4 hands, playing nets greedy
                                     with small epsilon, averaged over N rollouts)
        label[g] = team_delta(g)
    train GameChoiceNet on (state, label)  (MSE on all legal games at once, like counterfactual bids)
```

Cost is ~5x a normal round, only for the dealer's decision, so sample it: do
the full 5-way rollout for 1 round out of 4, or run 2-3 rollouts per game for a
lower variance label. The actual game in the real match is still the
epsilon-greedy choice of the net.

### 6.2 Playing nets

Two sensible reward schemes, in order of preference:

1. **Trick games**: per-trick reward, like `reward_card_players` in Trump:
   `r = -(penalty points my team takes in this trick) + (points the opponents take)`
   divided by the game's max, plus a small terminal bonus for the final
   `team_delta`. Dense rewards, no credit-assignment problem.
2. **Trex**: the real reward arrives only at the end (+200/+150/+100/+50), so
   use n-step TD (n=4-8) targets `Q(s,a) <- r + gamma * max Q(s', .)` with a
   potential-based shaping reward `Phi = -(cards left)/13 * w`
   (potential-based shaping does not change the optimal policy) and an
   immediate reward when a player (or the partner) finishes.

Both use:
* epsilon-greedy with per-net epsilon and decay persisted to JSON (as
  `epsilon_state.json` in Trump);
* MSE loss on the chosen card slot only;
* one shared network for all 4 seats;
* a replay buffer (recent ~50k samples) instead of batch-and-discard,
  because Trex has long rounds and correlated samples.

### 6.3 Curriculum

1. **Phase 1, playing nets, fixed game**: for each game in turn, deal random
   hands and train its playing net (game chosen at random). Expected to
   converge quickly (a few thousand rounds for trick games).
2. **Phase 2, game choice**: freeze the playing nets (or train them at a low
   learning rate) and train `GameChoiceNet` with the counterfactual labels.
3. **Phase 3, joint self-play**: full matches (4 kingdoms x 5 rounds), all nets
   learning, epsilon decayed.

### 6.4 Avoiding self-play collapse

Shared weights for all four seats can drift into a mutual strategy that
only beats itself. Keep a **league**: every N matches snapshot the nets, and
make one team play a randomly sampled old snapshot (or the heuristic bot).
Track win rate against: random legal player, heuristic bot, previous snapshot.

---

## 7. Baselines and evaluation

Needed before you can say the nets "learn" (the Trump README reports an average
score and a win percentage; do the same here).

* `RandomPlayer`: a legal random move.
* `HeuristicPlayer`:
  * trick games: dump K♥ / queens / diamonds on opponents when they are
    winning, else play the lowest card;
  * Trex: play the card that unlocks the fewest opponent moves, hold Jacks;
  * game choice: Trex if the hand has many Jacks/low cards and sequences, K♥
    if you hold fewer than 3 hearts, etc.
* Metrics per training run: average `team_delta` per round per game type, average
  match score difference vs each baseline, distribution of chosen games
  (should be non-degenerate), Trex finishing-position histogram,
  games-ended-early rate (K♥, queens, diamonds).

---

## 8. Suggested code layout (mirrors `Trump/`)

```
Trex/
  ARCHITECTURE.md        <- this file
  TrexGame.py            <- kingdoms, dealer rotation, available-games set, scoring, early termination
  TrexPlayer.py          <- hand, playable_cards() for tricks and for the Trex layout, pass inference
  GameChoiceNet.py       <- encode_game_choice_state + net
  PlayingNet.py          <- encode_playing_state (trick games) + net
  TrexNet.py             <- encode_trex_state + net
  baselines.py           <- RandomPlayer, HeuristicPlayer
  train.py               <- same CLI as Trump: --games, --max-rounds, epsilon args, per-phase flag
  models/                <- game_choice_net.pt, playing_net.pt, trex_net.pt, epsilon_state.json
```

Reuse from `Trump/`: `card_slot`, `order_suits`, `encode_hand`, `HONOR_POINTS`,
`played_by_seat` tracking, epsilon persistence, the `load_models` /
`save_models` pattern.

Implementation order: (1) `TrexGame` + scoring with random players and unit tests
for each game's end condition; (2) baselines; (3) trick-games `PlayingNet`;
(4) `TrexNet`; (5) `GameChoiceNet` with counterfactual labels; (6) league.

---

## 9. Questions / rules to confirm

1. "The next player to his **right**": same seat direction as Trump's
   `(distributer + 1) % 4`, or the opposite one?
2. Is the pass truly **forced** when no card is playable (and forbidden when one
   is)? I assumed yes, so pass is never a network decision.
3. Who plays first in Trex? (I assumed the dealer, who may play any legal card -
   usually a Jack - or must he start with a specific card?)
4. In King of Hearts / Queens / Diamonds, are there additional obligations
   (for example "must lead hearts / must play the penalty card when allowed")
   or is it plain follow-suit?
5. Does the match winner = team with the best total after 20 rounds (no
   elimination)? Then the final training target is the total score difference.
6. The 5th choice of a kingdom is forced; if you want, it is not a network
   decision and produces no training sample.
