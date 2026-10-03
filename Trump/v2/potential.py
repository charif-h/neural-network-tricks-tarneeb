"""
Game-level value of a score line, and the "stakes" derived from it.

The v2 experiment rewards what the game is really about: each team wants its
leading score to reach WIN_SCORE (with a non-negative partner) and wants the
other team's leading score to stay as far from it as possible.

  potential(a, b)         how good the score pair (a, b) of one team is, 0 at best
  state_value(scores, s)  my team's potential minus the opponents', from seat s
  stakes_for(...)         how much each player's bid result swings state_value

Seats are always read from one player's point of view with `seat_order`:
[self, partner, opponent 1, opponent 2] - the same layout the networks use.
"""

WIN_SCORE = 41
LOSE_SCORE = -50

GAP_EXPONENT = 0.5      # < 1 makes the last points before WIN_SCORE worth more than the first
SOFT_WEIGHT = 0.25      # share of the potential given to the team's lower score
NEG_WEIGHT = 2.0        # extra penalty per point below 0 (a negative partner blocks the win)
TERMINAL_BONUS = 1.0    # added to a team that won the game, subtracted from one that lost it

STAKE_NORM = 2.0        # a raw swing of this size counts as a maximal stake
STAKE_FLOOR = 0.25      # smallest stake magnitude, so tricks always matter a little


def seat_order(seat, num_players=4):
    """Seats read from `seat`'s view: [self, partner, opponent 1, opponent 2]."""
    partner = (seat + 2) % num_players
    opponents = [s for s in ((seat + o) % num_players for o in (1, 2, 3)) if s != partner]
    return [seat, partner] + opponents


def gap(score):
    """Normalized distance from WIN_SCORE: 0 once reached, above 1 for a negative score."""
    return max(0.0, (WIN_SCORE - score) / WIN_SCORE)


def team_result(a, b):
    """1 if a team with scores (a, b) has won the game, -1 if it has lost it, else 0."""
    if (a >= WIN_SCORE and b >= 0) or (b >= WIN_SCORE and a >= 0):
        return 1
    if a <= LOSE_SCORE or b <= LOSE_SCORE:
        return -1
    return 0


def potential(a, b):
    """
    Value of one team's score pair: 0 is the best a score line can look like
    before the game is won, more negative is worse.

    The leader's gap to WIN_SCORE counts fully and the other member's gap
    counts SOFT_WEIGHT as much (so a partner who can overtake still matters),
    both through GAP_EXPONENT. Scores below 0 pay an extra linear penalty and
    the game result adds or subtracts TERMINAL_BONUS.
    """
    hi, lo = max(a, b), min(a, b)
    gaps = (gap(hi) ** GAP_EXPONENT + SOFT_WEIGHT * gap(lo) ** GAP_EXPONENT) / (1 + SOFT_WEIGHT)
    negatives = (max(0, -a) + max(0, -b)) / WIN_SCORE
    return -gaps - NEG_WEIGHT * negatives + TERMINAL_BONUS * team_result(a, b)


def state_value(scores, seat):
    """My team's potential minus the opposing team's, for the player at `seat`."""
    me, partner, opp1, opp2 = (scores[s] for s in seat_order(seat, len(scores)))
    return potential(me, partner) - potential(opp1, opp2)


def bid_multiplier(bid):
    """Factor applied to a bid when scoring: 3 for 13, 2 for 7 to 12, 1 otherwise."""
    if bid == 13:
        return 3
    if bid >= 7:
        return 2
    return 1


def stakes_for(scores, bids, seat):
    """
    How much each player's bid result matters to the player at `seat`.

    For every player i, compare the state value if i makes their bid with the
    value if i fails it, everything else held at the current scores. The swing
    is turned into a magnitude in [STAKE_FLOOR, 1] and signed + for my team
    (I want them to make it) and - for the opponents (I want them to fail).

    Returns:
        list: 4 stakes in seat_order(seat): [self, partner, opponent 1, opponent 2]
    """
    order = seat_order(seat, len(scores))
    stakes = []
    for k, i in enumerate(order):
        points = bids[i] * bid_multiplier(bids[i])
        made = list(scores)
        made[i] += points
        failed = list(scores)
        failed[i] -= points
        swing = abs(state_value(made, seat) - state_value(failed, seat))
        magnitude = STAKE_FLOOR + (1 - STAKE_FLOOR) * min(1.0, swing / STAKE_NORM)
        stakes.append(magnitude if k < 2 else -magnitude)
    return stakes


def team_features(scores, seat):
    """
    Four numbers describing where both teams stand, for the bidding network:
    my team's leader gap, the opponents' leader gap, and both potentials.
    """
    me, partner, opp1, opp2 = (scores[s] for s in seat_order(seat, len(scores)))
    return [
        gap(max(me, partner)),
        gap(max(opp1, opp2)),
        potential(me, partner),
        potential(opp1, opp2),
    ]
