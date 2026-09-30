"""
Trump player: a Game4Player that can bid on the number of tricks it expects to win.
"""

import random

from Cards.Game4 import Game4Player
import torch
from Trump.BiddingNet import encode_bidding_state, card_slot
from Trump.PlayingNet import encode_playing_state


class TrumpPlayer(Game4Player):
    """
    Player of a TrumpGame.

    On top of the Game4Player attributes, a trump player estimates, from the
    cards in its hand, how many tricks it will win in the round (its bid).
    """

    MAX_BID = 13

    def __init__(self, game, partner=None):
        super().__init__(game, partner)
        self.tricks_won = 0
        self.bid_net = game.shared_bid_net
        self.bid_optimizer = game.shared_bid_optimizer
        self.bid_history = []       # (state, bid, reward) samples ready for training
        self.pending_bid = None     # (state, bid) for this round, waiting for its reward
        self.pending_min_bid = 2    # lowest legal bid when pending_bid was made
        self.playing_net = game.shared_playing_net
        self.playing_optimizer = game.shared_playing_optimizer
        self.card_history = []      # (state, slot, reward) samples ready for training
        self.pending_card = None    # (state, slot) for the trick in progress, waiting for its reward
        self.suit_order = None      # fixed once per round by TrumpGame.bidding(), reused all round

    def min_bid(self):
        """Lowest bid allowed for this player, which rises with the player's score."""
        if self.score >= 30:
            return self.score // 10
        return 2

    def bid(self, partner_bid, opponent_bids):
        """
        Estimate how many tricks this player will win in the round.

        Args:
            partner_bid (int): The partner's bid, or 0 if the partner has not bid yet
            opponent_bids (list): The two opponents' bids, in seat order after
                this player; 0 for an opponent who has not bid yet

        Returns:
            int: A bid between self.min_bid() and self.MAX_BID
        """
        state = encode_bidding_state(self, partner_bid, opponent_bids)

        if self.game.bid_learning and random.random() < self.game.bid_epsilon:
            self.bid_value = random.randint(self.min_bid(), self.MAX_BID)
        else:
            with torch.no_grad():
                bid_values = self.bid_net(state)
            bid_values[: self.min_bid() - 2] = float('-inf')
            self.bid_value = max(int(bid_values.argmax()) + 2, self.min_bid())

        if self.game.bid_learning:
            self.pending_bid = (state, self.bid_value)
            self.pending_min_bid = self.min_bid()
        return self.bid_value

    def record_bid_reward(self, reward):
        """
        Turn this round's pending bid into a training sample, once its reward is known.

        If bidding was aborted and retried, only the last (successful) attempt's
        bid has a pending_bid by the time this is called, since each new bid()
        call overwrites it - so only the round actually played is learned from.

        Args:
            reward (float): Reward computed by the game for this round's bid
        """
        if not self.game.bid_learning or self.pending_bid is None:
            return
        state, bid = self.pending_bid
        self.bid_history.append((state, bid, reward))
        self.pending_bid = None

    def playable_cards(self, played_cards):
        """
        Cards the player is allowed to play in the current trick.

        The player must follow the suite of the first card played if they have
        a card of that suite; otherwise any card in the hand can be played.

        Args:
            played_cards (list): Cards already played in the current trick
        """
        if played_cards:
            same_suite = [c for c in self.hand if c.suite == played_cards[0].suite]
            if same_suite:
                return same_suite
        return list(self.hand)

    def choose_card(self, played_cards):
        """
        Choose the card to play in the current trick.

        Args:
            played_cards (list): Cards already played this trick, in play order

        Returns:
            Card: A card from self.hand
        """
        playable = self.playable_cards(played_cards)
        state = encode_playing_state(self, played_cards)

        if self.game.card_learning and random.random() < self.game.card_epsilon:
            card = random.choice(playable)
        else:
            with torch.no_grad():
                card_values = self.playing_net(state)
            legal_slots = [card_slot(c, self.suit_order) for c in playable]
            masked = torch.full((52,), float('-inf'), dtype=torch.float32)
            masked[legal_slots] = card_values[legal_slots]
            best_slot = int(masked.argmax())
            card = next(c for c in playable if card_slot(c, self.suit_order) == best_slot)

        if self.game.card_learning:
            self.pending_card = (state, card_slot(card, self.suit_order))
        return card

    def record_card_reward(self, reward):
        """
        Turn this trick's pending card into a training sample, once its reward is known.

        Args:
            reward (float): Reward computed by the game for this trick's card
        """
        if not self.game.card_learning or self.pending_card is None:
            return
        state, slot = self.pending_card
        self.card_history.append((state, slot, reward))
        self.pending_card = None
