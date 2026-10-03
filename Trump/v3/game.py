"""
v3 Trump game: v1's rules, networks and bid reward, with a new trick reward
that makes the players aware of who is close to winning the game.

Only reward_card_players is overridden. Players, networks and everything else
are v1's, so v1 itself is untouched and v3 can start from v1's weights.

The trick reward, from the point of view of the player who played the card:

  - a trick won by me or my partner (W is the winner)
      W still needs the trick : +1.0 + bonus(W)
      W already made his bid  : +0.3 + the largest bonus among the opponents
                                (the value of denying them the trick)
  - a trick lost to an opponent (W is the winner)
      my team still needs one : -1.0 - bonus(W)
      my team made both bids  : -0.3 - bonus(W)

  bonus(x) is 0 if x already made his bid, otherwise
      term(x) + end_game(x)
  with
      term(x)     = min(1, (score + bid points) / 41), or NEGATIVE_SCORE_TERM
                    if x's score is negative
      end_game(x) = 1 if score + bid points >= 41 and x's partner's score is
                    not negative (a negative partner blocks the win), else 0

The waste penalty of v1 (losing a trick while playing a strong card) is kept.
"""

from Trump.PlayingNet import card_strength
from Trump.TrumpGame import TrumpGame

NEGATIVE_SCORE_TERM = 0.5  # term(x) for a player whose score is below 0


def bid_points(bid):
    """Points a bid scores when made (or loses when failed)."""
    return bid * TrumpGame.bid_multiplier(bid)


def bonus(score, bid, tricks_won, partner_score):
    """
    How much the next trick matters for the progress of one player's bid
    towards the end of the game (see the module docstring).

    Args:
        score (int): The player's current score
        bid (int): The player's bid this round
        tricks_won (int): Tricks the player won so far this round
        partner_score (int): The player's partner's current score

    Returns:
        float: 0 if the player already made his bid, else term + end_game
    """
    if tricks_won >= bid:
        return 0.0
    if score < 0:
        return NEGATIVE_SCORE_TERM
    reached = score + bid_points(bid)
    term = min(1.0, reached / TrumpGame.WINNING_SCORE)
    end_game = 1.0 if reached >= TrumpGame.WINNING_SCORE and partner_score >= 0 else 0.0
    return term + end_game


def trick_outcome(seat, winner_seat, scores, bids, tricks_won, need_floor=TrumpGame.CARD_NEED_FLOOR):
    """
    Outcome part of the reward of the player at `seat` for a resolved trick.

    Args:
        seat (int): The seat of the player being rewarded
        winner_seat (int): The seat that won the trick
        scores (list): Every seat's current score
        bids (list): Every seat's bid this round
        tricks_won (list): Every seat's tricks won this round, before this trick is credited
        need_floor (float): Value of a trick nobody on the relevant side needs (0.3)

    Returns:
        float: The reward, before the waste penalty
    """
    def bonus_of(s):
        return bonus(scores[s], bids[s], tricks_won[s], scores[(s + 2) % 4])

    partner = (seat + 2) % 4
    opponents = [s for s in range(4) if s not in (seat, partner)]

    if winner_seat in (seat, partner):
        if tricks_won[winner_seat] < bids[winner_seat]:
            return 1.0 + bonus_of(winner_seat)
        return need_floor + max(bonus_of(o) for o in opponents)

    team_needs = any(tricks_won[s] < bids[s] for s in (seat, partner))
    return -(1.0 if team_needs else need_floor) - bonus_of(winner_seat)


class TrumpGameV3(TrumpGame):
    """
    A TrumpGame whose trick reward follows the table in the module docstring.

    Attributes (on top of TrumpGame):
        optimistic_bid_prob (float): Chance, while bid_learning is on, that a
            TrumpPlayerV3 raises its greedy bid by one (see Trump.v3.player)
    """

    def __init__(self, optimistic_bid_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.optimistic_bid_prob = optimistic_bid_prob

    def reward_card_players(self, played_cards, leader, winner_seat):
        """
        Compute each player's reward for the trick just resolved and record it.

        Uses scores, bids and tricks_won as of just before this trick is
        credited, like v1. The reward is trick_outcome plus v1's waste penalty.

        Args:
            played_cards (list): The 4 cards played this trick, in play order
            leader (int): Seat that led this trick
            winner_seat (int): Seat that won this trick
        """
        if not self.card_learning:
            return

        scores = [p.score for p in self.players]
        tricks_won = [p.tricks_won for p in self.players]

        for k, card in enumerate(played_cards):
            seat = (leader + k) % self.NUM_PLAYERS
            partner_seat = (seat + 2) % self.NUM_PLAYERS

            outcome = self.CARD_TRICK_SCALE * trick_outcome(
                seat, winner_seat, scores, self.bids, tricks_won, self.CARD_NEED_FLOOR)
            won_by_us = winner_seat in (seat, partner_seat)
            waste_penalty = 0.0 if won_by_us else -self.CARD_WASTE_WEIGHT * card_strength(card, self.trump)

            self.players[seat].record_card_reward(outcome + self.CARD_CALIBRATION_SCALE * waste_penalty)
