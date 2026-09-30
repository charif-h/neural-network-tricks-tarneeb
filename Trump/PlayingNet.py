"""
Playing network: predicts, for the player about to act, the value of playing
each specific card of its hand in the current trick.

Output layout mirrors the hand encoding: slot i is "the value of playing
whatever card occupies hand-slot i" (see BiddingNet.card_slot), so the same
slot numbering is used for the hand, the legality mask and the network's
output, using player.suit_order - the suit order fixed once for the whole
round at bidding time (see TrumpGame.bidding()), not recomputed per trick.
This keeps a given physical suit's slot position stable across all 13 tricks.
"""

import torch
from torch import nn

from Trump.BiddingNet import HONOR_POINTS, card_slot

N_PLAY_INPUTS = 333  # size of the vector built by encode_playing_state

TRUMP_BONUS = 2  # extra "strength" credit for a trump card - even a low trump can win a trick


class PlayingNet(nn.Module):
    def __init__(self, n_inputs=N_PLAY_INPUTS, n_outputs=52, hidden=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_outputs)
        )

    def forward(self, x):
        return self.layers(x)

    def saveModel(self, path):
        torch.save(self.layers.state_dict(), path)

    def loadModel(self, path):
        self.layers.load_state_dict(torch.load(path))


def card_strength(card, trump):
    """
    How costly it is to lose this card without winning the trick.

    Bridge-style honor points (A=4, K=3, Q=2, J=1, else 0), plus a flat bonus
    for a trump card, since even a low trump is generically valuable - it can
    win tricks no side-suit card can.

    Args:
        card (Card): The card played
        trump (CardSuite): The round's trump suite

    Returns:
        int: The card's strength
    """
    return HONOR_POINTS.get(card.rank.value, 0) + (TRUMP_BONUS if card.suite == trump else 0)


def played_by_seat_vector(game, seat, suit_order):
    """
    Encode the cards a given seat has already played this round as 52 binary values.

    Args:
        game (TrumpGame): The game, for its played_by_seat tracker
        seat (int): The seat whose revealed cards to encode
        suit_order (list): The 4 suits in encoding order (see order_suits)

    Returns:
        torch.Tensor: float32 tensor of shape (52,)
    """
    encoded = torch.zeros(52, dtype=torch.float32)
    for card in game.played_by_seat[seat]:
        encoded[card_slot(card, suit_order)] = 1.0
    return encoded


def encode_playing_state(player, played_cards):
    """
    Encode what a player knows when choosing a card to play, as a tensor.

    As with bidding, everything is seen from the deciding player's point of
    view (partner and opponents always in the same slots), and hand-shaped
    vectors all share player.suit_order, so a given physical suit always
    occupies the same slot range for this player throughout the round.

    Args:
        player (TrumpPlayer): The player about to play a card
        played_cards (list): Cards already played this trick, in play order
            (this player has not played yet, so 0 to 3 cards)

    The encoding scheme is as follows:
        - Slot 0-51   : current hand (one-hot), player.suit_order layout
        - Slot 52-103 : playable-cards mask (which hand cards are legal this trick)
        - Slot 104-155: "would win now" bits - for each hand card, would playing
                        it this instant make it the current leader of the trick
        - Slot 156-160: led suite - one-hot over the 4 suit-slot positions, plus
                        a 5th "I am leading" flag (no card played yet this trick)
        - Slot 161    : is partner currently winning this trick (0/1)
        - Slot 162    : is an opponent currently winning this trick (0/1)
        - Slot 163    : number of players still to act after me this trick, / 3
        - Slot 164-167: tricks won so far: self, partner, opp 1, opp 2, each / 13
        - Slot 168-171: bids: self, partner, opp 1, opp 2, each / 13
        - Slot 172-175: scores: self, partner, opp 1, opp 2, each / 41
        - Slot 176    : tricks remaining in the round (including this one), / 13
        - Slot 177-228: cards played this round by partner (one-hot, 52)
        - Slot 229-280: cards played this round by opponent 1 (one-hot, 52)
        - Slot 281-332: cards played this round by opponent 2 (one-hot, 52)

    Returns:
        torch.Tensor: float32 tensor of shape (N_PLAY_INPUTS,)
    """
    game = player.game
    hand = player.hand
    suit_order = player.suit_order

    seat = game.players.index(player)
    partner_seat = game.players.index(player.partner)
    opponent_seats = [s for s in ((seat + o) % game.NUM_PLAYERS for o in (1, 2, 3))
                      if s != partner_seat]

    own_position = len(played_cards)  # 0 = leading, 1..3 = following
    leader = (seat - own_position) % game.NUM_PLAYERS
    seats_so_far = [(leader + k) % game.NUM_PLAYERS for k in range(own_position)]

    playable = player.playable_cards(played_cards)
    playable_slots = {card_slot(c, suit_order) for c in playable}

    hand_vec = torch.zeros(52, dtype=torch.float32)
    mask_vec = torch.zeros(52, dtype=torch.float32)
    would_win_vec = torch.zeros(52, dtype=torch.float32)
    for card in hand:
        slot = card_slot(card, suit_order)
        hand_vec[slot] = 1.0
        if slot in playable_slots:
            mask_vec[slot] = 1.0
            if game.trick_winner(played_cards + [card]) == own_position:
                would_win_vec[slot] = 1.0

    led_vec = [0.0] * 5
    if played_cards:
        led_vec[suit_order.index(played_cards[0].suite)] = 1.0
    else:
        led_vec[4] = 1.0  # "I am leading" flag

    partner_winning = 0.0
    opponent_winning = 0.0
    if played_cards:
        winning_seat = seats_so_far[game.trick_winner(played_cards)]
        partner_winning = 1.0 if winning_seat == partner_seat else 0.0
        opponent_winning = 1.0 if winning_seat in opponent_seats else 0.0

    control = [
        partner_winning,
        opponent_winning,
        (game.NUM_PLAYERS - 1 - own_position) / (game.NUM_PLAYERS - 1),
    ]

    tricks_won = [player.tricks_won, player.partner.tricks_won] \
        + [game.players[s].tricks_won for s in opponent_seats]
    bids = [game.bids[seat], game.bids[partner_seat]] + [game.bids[s] for s in opponent_seats]
    scores = [player.score, player.partner.score] + [game.players[s].score for s in opponent_seats]
    tricks_remaining = [len(hand) / 13]

    rest = (
        led_vec
        + control
        + [n / 13 for n in tricks_won]
        + [b / 13 for b in bids]
        + [s / 41 for s in scores]
        + tricks_remaining
    )

    revealed = torch.cat([
        played_by_seat_vector(game, partner_seat, suit_order),
        played_by_seat_vector(game, opponent_seats[0], suit_order),
        played_by_seat_vector(game, opponent_seats[1], suit_order),
    ])

    return torch.cat([hand_vec, mask_vec, would_win_vec, torch.tensor(rest, dtype=torch.float32), revealed])
