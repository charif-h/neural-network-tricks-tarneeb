"""
Abstract base classes for four-player card games (Tarneeb, Trex, ...).

Game4 holds the deck and the four players and knows how to deal.
Game4Player holds a player's hand, score, partner and strategy.
Game-specific rules and strategies are supplied by subclasses.
"""

from abc import ABC, abstractmethod

from Cards.StandarDeck import StandarDeck


class Game4(ABC):
    """
    Abstract game of four players played with a standard 52-card deck.

    Attributes:
        deck (StandarDeck): The deck used by the game
        players (list): The Game4Player instances seated at the table (max 4)
        round_number (int): Number of rounds started so far (0 before the first)
        distributer (int): Index of the current round's distributer
        start_player (int): Index of the player who plays first in the current round
    """

    NUM_PLAYERS = 4

    def __init__(self):
        self.deck = StandarDeck(shuffled=True)
        self.players = []
        self.round_number = 0
        self.distributer = None
        self.start_player = None

    def start_round(self, distributer=0):
        """
        Start a new round: distribute the cards and record who plays first.

        The player after the distributer plays first, so the distributer plays last.

        Args:
            distributer (int): Index in self.players of the player distributing the cards
        """
        self.distribute(distributer)
        self.round_number += 1
        self.distributer = distributer
        self.start_player = (distributer + 1) % self.NUM_PLAYERS

    def add_player(self, player):
        """Seat a player at the table. Called by Game4Player.__init__."""
        if len(self.players) >= self.NUM_PLAYERS:
            raise ValueError('The game already has %d players' % self.NUM_PLAYERS)
        self.players.append(player)

    def distribute(self, distributer=0):
        """
        Shuffle a fresh deck and deal all of its cards evenly to the 4 players.

        Cards are dealt one at a time, beginning with the player after the
        distributer, so the distributer is dealt to last and receives the last
        card of the deck. Each player's hand is replaced by the cards received,
        in dealing order, so the last card of a hand is the last card that
        player was dealt.

        Args:
            distributer (int): Index in self.players of the player distributing the cards
        """
        if len(self.players) != self.NUM_PLAYERS:
            raise ValueError('Need %d players to distribute, found %d'
                             % (self.NUM_PLAYERS, len(self.players)))
        if not 0 <= distributer < self.NUM_PLAYERS:
            raise ValueError('distributer must be between 0 and %d' % (self.NUM_PLAYERS - 1))

        self.deck = StandarDeck(shuffled=True)
        for player in self.players:
            player.hand = []

        i = (distributer + 1) % self.NUM_PLAYERS
        while self.deck.cards:
            self.players[i].hand.append(self.deck.draw())
            i = (i + 1) % self.NUM_PLAYERS

    @abstractmethod
    def trick_winner(self, cards):
        """
        Determine which of the four played cards wins the trick.

        Args:
            cards (list): The 4 Card objects played, one per player

        Returns:
            int: Index (0-3) in `cards` of the winning card
        """


class Game4Player(ABC):
    """
    Abstract player of a Game4.

    Attributes:
        game (Game4): The game this player belongs to
        hand (list): Cards held by the player, in the order they were dealt
        score (int): The player's score
        partner (Game4Player or None): The player's partner, if the game has teams
    """

    def __init__(self, game, partner=None):
        self.game = game
        self.hand = []
        self.score = 0
        self.partner = None
        if partner is not None:
            self.set_partner(partner)
        game.add_player(self)

    def set_partner(self, partner):
        """Make `partner` this player's partner (symmetrically)."""
        self.partner = partner
        partner.partner = self

    @abstractmethod
    def choose_card(self, played_cards):
        """
        Strategy: choose the next card to play.

        Args:
            played_cards (list): Cards already played in the current trick

        Returns:
            Card: A card from self.hand
        """
