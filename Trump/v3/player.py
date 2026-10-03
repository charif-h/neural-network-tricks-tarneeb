"""
v3 Trump player: v1's player with an extra, targeted bid exploration.

While the bidding network is learning, a greedy (not epsilon-random) bid is
raised by one with probability game.optimistic_bid_prob. Uniform epsilon
exploration mostly produces absurd bids (anything from 2 to 13); a +1 stays
close to what the hand may really support. It also produces the samples the
counterfactual bid targets are missing: the tricks a player wins depend on
his bid (the playing net chases the bid, then eases off), so bids above the
greedy one are otherwise under-sampled and look worse than they are.
"""

import random

import torch

from Trump.BiddingNet import encode_bidding_state
from Trump.TrumpPlayer import TrumpPlayer


class TrumpPlayerV3(TrumpPlayer):
    """
    A TrumpPlayer that sometimes bids one more than the network predicts.

    Attributes (on top of TrumpPlayer):
        last_predicted_bid (int): The bid the network chose in the last call
            to bid(), before any +1 exploration (None if epsilon picked a
            random bid instead)
    """

    def __init__(self, game, partner=None):
        super().__init__(game, partner)
        self.last_predicted_bid = None

    def bid(self, partner_bid, opponent_bids):
        """
        Same as TrumpPlayer.bid, plus the +1 exploration described in the
        module docstring (only while bid_learning is on, so evaluation games
        stay greedy).

        Args:
            partner_bid (int): The partner's bid, or 0 if the partner has not bid yet
            opponent_bids (list): The two opponents' bids, in seat order after
                this player; 0 for an opponent who has not bid yet

        Returns:
            int: A bid between self.min_bid() and self.MAX_BID
        """
        state = encode_bidding_state(self, partner_bid, opponent_bids)
        game = self.game
        self.last_predicted_bid = None

        if game.bid_learning and random.random() < game.bid_epsilon:
            self.bid_value = random.randint(self.min_bid(), self.MAX_BID)
        else:
            with torch.no_grad():
                bid_values = self.bid_net(state)
            bid_values[: self.min_bid() - 2] = float('-inf')
            self.bid_value = max(int(bid_values.argmax()) + 2, self.min_bid())
            self.last_predicted_bid = self.bid_value
            if game.bid_learning and self.bid_value < self.MAX_BID and random.random() < game.optimistic_bid_prob:
                self.bid_value += 1

        if game.bid_learning:
            self.pending_bid = (state, self.bid_value)
            self.pending_min_bid = self.min_bid()
        return self.bid_value
