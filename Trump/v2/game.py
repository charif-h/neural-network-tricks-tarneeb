"""
v2 Trump game: same rules as TrumpGame, with rewards built on the team-level
score potential (see Trump.v2.potential) instead of per-player round points.

  - bid reward : how much the round's score changes moved my team's state
                 value (my team's potential minus the opponents'), for every
                 possible bid at once (counterfactual, as in v1 - see
                 counterfactual_bid_rewards_v2). No calibration term.
  - trick reward: the stakes of every player's bid, times what this trick did
                 for that player (won a needed trick, or made the bid impossible).
"""

import torch

from Trump.PlayingNet import PlayingNet, card_strength
from Trump.BiddingNet import BiddingNet
from Trump.TrumpGame import TrumpGame
from Trump.v2.player import N_BID_INPUTS_V2, N_PLAY_INPUTS_V2
from Trump.v2.potential import bid_multiplier, seat_order, stakes_for, state_value


class TrumpGameV2(TrumpGame):
    """
    Attributes (on top of TrumpGame):
        round_start_scores (list): Every seat's score when bidding ended
        stakes (list): For each seat, the 4 bid stakes from that seat's point
            of view (see potential.stakes_for), fixed once bidding ends
    """

    BID_REWARD_SCALE = 0.5  # keeps bid rewards roughly within [-1, 1]

    def __init__(self, **kwargs):
        kwargs['bid_counterfactual'] = True
        super().__init__(**kwargs)
        # Replace the v1 networks created by TrumpGame with the wider v2 ones,
        # before any player is seated (players grab them from the game).
        self.shared_bid_net = BiddingNet(n_inputs=N_BID_INPUTS_V2)
        self.shared_bid_optimizer = torch.optim.Adam(self.shared_bid_net.parameters(), lr=1e-3)
        self.shared_playing_net = PlayingNet(n_inputs=N_PLAY_INPUTS_V2)
        self.shared_playing_optimizer = torch.optim.Adam(self.shared_playing_net.parameters(), lr=1e-3)
        self.round_start_scores = [0] * self.NUM_PLAYERS
        self.stakes = [[0.0] * 4 for _ in range(self.NUM_PLAYERS)]

    def bidding(self):
        super().bidding()
        self.round_start_scores = [p.score for p in self.players]
        self.stakes = [stakes_for(self.round_start_scores, self.bids, seat)
                       for seat in range(self.NUM_PLAYERS)]

    def reward_players(self, deltas):
        if not self.bid_learning:
            return
        for seat, player in enumerate(self.players):
            player.record_bid_reward(self.counterfactual_bid_rewards_v2(seat, player.tricks_won, player.pending_min_bid))

    def counterfactual_bid_rewards_v2(self, seat, tricks_won, min_bid):
        """
        Reward every possible bid (2 to 13) would have earned for the player at `seat`.

        For each candidate bid, the player's own score change is replaced by
        what that bid would have scored given the tricks actually won, while
        the three other players keep their real score changes; the reward is
        how much that moved this player's team state value (see
        potential.state_value). Training all bids at once is what makes
        bidding learn fast - the v1 experiments showed it is far better than
        training only the bid taken.

        Args:
            seat (int): Seat of the player being rewarded
            tricks_won (int): Tricks the player won this round
            min_bid (int): Lowest bid that was legal for the player this round

        Returns:
            torch.Tensor: Shape (12,), one reward per bid 2..13; NaN for bids
                below min_bid, which were not legal and must not be trained on
        """
        before = state_value(self.round_start_scores, seat)
        rewards = torch.full((12,), float('nan'))
        for bid in range(max(2, min_bid), 14):
            points = bid * bid_multiplier(bid)
            scores = [p.score for p in self.players]
            scores[seat] = self.round_start_scores[seat] + (points if tricks_won >= bid else -points)
            rewards[bid - 2] = self.BID_REWARD_SCALE * (state_value(scores, seat) - before)
        return rewards

    def reward_card_players(self, played_cards, leader, winner_seat):
        if not self.card_learning:
            return

        # Cards are already removed from the hands, so this is what is left after this trick.
        remaining_after = len(self.players[0].hand)
        remaining_before = remaining_after + 1

        # What this trick did for each player's own bid: +1 for winning a trick
        # they needed (CARD_NEED_FLOOR if the bid was already made), -1 if it
        # just made their bid impossible.
        progress = []
        for seat in range(self.NUM_PLAYERS):
            need = self.bids[seat] - self.players[seat].tricks_won
            won_now = seat == winner_seat
            value = 0.0
            if won_now:
                value = 1.0 if need > 0 else self.CARD_NEED_FLOOR
            doomed_before = need > remaining_before
            doomed_after = need - (1 if won_now else 0) > remaining_after
            if doomed_after and not doomed_before:
                value -= 1.0
            progress.append(value)

        for k, card in enumerate(played_cards):
            seat = (leader + k) % self.NUM_PLAYERS
            player = self.players[seat]
            order = seat_order(seat, self.NUM_PLAYERS)

            outcome = self.CARD_TRICK_SCALE * sum(
                stake * progress[s] for stake, s in zip(self.stakes[seat], order))
            won_by_us = winner_seat in order[:2]
            waste_penalty = 0.0 if won_by_us else -self.CARD_WASTE_WEIGHT * card_strength(card, self.trump)

            player.record_card_reward(outcome + self.CARD_CALIBRATION_SCALE * waste_penalty)
