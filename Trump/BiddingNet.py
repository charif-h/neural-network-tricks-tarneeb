import torch
from torch import nn
from Cards.StandarDeck import StandarDeck
from Cards.Card import Card, CardSuite, CardRank

N_INPUTS = 68  # size of the vector built by encode_bidding_state


class BiddingNet(nn.Module):
    def __init__(self, n_inputs=N_INPUTS, n_outputs=12, hidden=128):
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

# Bridge-style high card points: only the top cards win tricks on their own.
HONOR_POINTS = {14: 4, 13: 3, 12: 2, 11: 1}


def order_suits(hand, trump):
    """
    Order the four suits for the network: the trump first, then the three
    other suits from the most to the least valuable in this hand.

    Side suits are compared by honor points (A=4, K=3, Q=2, J=1), then by
    their cards read from the highest down, then by length, and finally by suit
    id so that the order is always deterministic.

    Args:
        hand (list): The player's Card objects
        trump (CardSuite): The trump suit of the round

    Returns:
        list: The four CardSuite values in encoding order
    """
    def key(suit):
        ranks = sorted((c.rank.value for c in hand if c.suite == suit), reverse=True)
        points = sum(HONOR_POINTS.get(r, 0) for r in ranks)
        return (-points, tuple(-r for r in ranks), -len(ranks), suit.id)

    return [trump] + sorted((s for s in CardSuite if s != trump), key=key)


def card_slot(card, suit_order):
    """
    Index (0-51) of `card` in the 52-slot layout defined by `suit_order`.

    This is the one place that turns a (suit, rank) into a slot number, so
    every hand-shaped tensor (a hand, a legality mask, a revealed-cards
    tracker, a network's card outputs...) addresses a given card the same way.

    Args:
        card (Card): The card to locate
        suit_order (list): The 4 suits in encoding order (see order_suits)

    Returns:
        int: A slot between 0 and 51
    """
    return suit_order.index(card.suite) * 13 + (CardRank.ACE.value - card.rank.value)


def encode_hand(hand, suit_order):
    """
    Encode a hand as 52 binary values: 4 suits x 13 ranks.

    Inside a suit the ranks always go from the Ace (first slot) down to the
    Two (last slot), because ranks are not interchangeable. A slot is 1.0 if
    the player holds that card, else 0.0.

    Args:
        hand (list): The player's Card objects
        suit_order (list): The 4 suits in encoding order (see order_suits)

    Returns:
        torch.Tensor: float32 tensor of shape (52,)
    """
    encoded = torch.zeros(52, dtype=torch.float32)
    for card in hand:
        encoded[card_slot(card, suit_order)] = 1.0
    return encoded


def encode_bidding_state(player, partner_bid, opponents_bids):
    """
    Encode what a player knows before bidding as a tensor for the network.

    Everything is seen from the bidding player's point of view: partner and
    opponents always occupy the same slots, so one network can serve all seats.

    Args:
        player (TrumpPlayer): The player about to bid (gives access to the
            hand, the score, the partner, the game, and its cached suit_order
            - see TrumpGame.bidding(), which sets it before calling bid())
        partner_bid (int): The partner's bid, or 0 if not bid yet
        opponents_bids (list): The two opponents' bids in seat order after the
            player, 0 for an opponent who has not bid yet

    The encoding scheme is as follows:
        - Slot 0-51 : hand, 4 suits x 13 ranks (one-hot), in player.suit_order
                      (trump suit first, then the others from the most to the
                      least valuable - see order_suits)
        - Slot 52-55: number of cards in each suit, same suit order, each divided by 13
        - Slot 56-58: partner bid, opponent 1 bid, opponent 2 bid (each divided by 13)
        - Slot 59-61: "has bid" flag (0 or 1) for each of those three
        - Slot 62-65: scores: own, partner, opp 1, opp 2, each divided by 41
        - Slot 66   : this player's min_bid / 13
        - Slot 67   : number of players who have already bid / 3

    Returns:
        torch.Tensor: float32 tensor of shape (N_INPUTS,)
    """
    game = player.game
    hand = player.hand
    suit_order = player.suit_order

    seat = game.players.index(player)
    partner_seat = game.players.index(player.partner)
    opponent_seats = [s for s in ((seat + o) % game.NUM_PLAYERS for o in (1, 2, 3))
                      if s != partner_seat]

    others_bids = [partner_bid] + list(opponents_bids)
    scores = [player.score, player.partner.score] + [game.players[s].score for s in opponent_seats]
    counts = [sum(1 for c in hand if c.suite == s) for s in suit_order]

    rest = (
        [n / 13 for n in counts]
        + [b / 13 for b in others_bids]
        + [1.0 if b > 0 else 0.0 for b in others_bids]
        + [s / 41 for s in scores]
        + [player.min_bid() / 13]
        + [sum(1 for b in others_bids if b > 0) / 3]
    )
    return torch.cat([encode_hand(hand, suit_order), torch.tensor(rest, dtype=torch.float32)])

if __name__ == "__main__":
    #print(BiddingNet()(torch.zeros(64, dtype=torch.float32)))
    deck = StandarDeck()
    c = Card(CardRank.THREE, CardSuite.DIAMOND)  # Example card with rank 3 and suit 1
    d = Card(CardRank.FOUR, CardSuite.DIAMOND)  # Example card with rank 4 and suit 2
    
    print(deck)
    print(c)
    print(d)
    print(deck.cardstoArray([c, d]))