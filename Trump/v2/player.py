"""
v2 Trump player: same as TrumpPlayer, but its networks also see where both
teams stand (bidding) and the stakes of every player's bid (playing).
"""

import random

import torch

from Trump.BiddingNet import N_INPUTS, encode_bidding_state, card_slot
from Trump.PlayingNet import N_PLAY_INPUTS, encode_playing_state
from Trump.TrumpPlayer import TrumpPlayer
from Trump.v2.potential import team_features

N_BID_INPUTS_V2 = N_INPUTS + 4        # + [my gap, opponents' gap, my potential, opponents' potential]
N_PLAY_INPUTS_V2 = N_PLAY_INPUTS + 4  # + stakes [self, partner, opponent 1, opponent 2]


def encode_bidding_state_v2(player, partner_bid, opponents_bids):
    """encode_bidding_state plus 4 team-standing features (see potential.team_features)."""
    game = player.game
    seat = game.players.index(player)
    extra = torch.tensor(team_features([p.score for p in game.players], seat), dtype=torch.float32)
    return torch.cat([encode_bidding_state(player, partner_bid, opponents_bids), extra])


def encode_playing_state_v2(player, played_cards):
    """encode_playing_state plus the 4 bid stakes computed once bidding ends (see TrumpGameV2.bidding)."""
    game = player.game
    seat = game.players.index(player)
    stakes = torch.tensor(game.stakes[seat], dtype=torch.float32)
    return torch.cat([encode_playing_state(player, played_cards), stakes])


class TrumpPlayerV2(TrumpPlayer):
    """A TrumpPlayer using the v2 state encodings; everything else is inherited."""

    def bid(self, partner_bid, opponent_bids):
        state = encode_bidding_state_v2(self, partner_bid, opponent_bids)

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

    def choose_card(self, played_cards):
        playable = self.playable_cards(played_cards)
        state = encode_playing_state_v2(self, played_cards)

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
