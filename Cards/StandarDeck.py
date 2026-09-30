"""
Standard deck module for card games.

This module provides a StandardDeck class for managing a 52-card deck
and utility functions for card operations.

Note: The class name 'StandarDeck' is kept for backward compatibility,
but should be 'StandardDeck'. Consider refactoring in future versions.
"""

import logging

from Cards.Card import CardRank
from Cards.Card import CardSuite
from Cards.Card import Card
import random
import numpy as np


class StandarDeck:
    """
    Represents a standard 52-card deck.
    
    The deck contains all combinations of 13 values and 4 types (suits).
    Cards can be distributed and the winner of a hand can be determined.
    
    Attributes:
        cards (list): List of Card objects in the deck
    """
    
    def __init__(self, shuffled=True):
        """
        Initialize a standard 52-card deck.
        
        Args:
            shuffled (bool): If True, shuffle the deck after creation (default: True)
        """
        self.cards = []
        for i in CardRank:
            for j in CardSuite:
                self.cards.append(Card(i, j))
        if shuffled:
            random.shuffle(self.cards)
        logging.info('New standard deck created with %d cards', len(self.cards))

    def __str__(self):
        return f"StandarDeck({self.cards})"

    def draw(self):
        """
        Remove and return the top card of the deck.

        Returns:
            Card: The card removed from the deck

        Raises:
            IndexError: If the deck is empty
        """
        return self.cards.pop()

    def distripute(self, n):
        """
        Distribute n cards from the deck.
        
        Note: Method name 'distripute' is a typo (should be 'distribute').
        Kept for backward compatibility.
        
        Args:
            n (int): Number of cards to distribute
        
        Returns:
            list: List of n Card objects removed from the deck
        """
        return [self.draw() for _ in range(n)]

    @staticmethod
    def cardstoArray(cards, val=1):
        """
        Convert a list of cards to a binary numpy array.
        
        Creates a 52-element array where each position represents a card
        (indexed by cardId). Positions corresponding to cards in the input
        list are set to val, others are 0.
        
        Args:
            cards (list): List of Card objects
            val (int or float): Value to set for cards in the list (default: 1)
        
        Returns:
            np.ndarray: 52-element array representing the cards
        """
        ret = np.zeros(52)
        for c in cards:
            ret[c.cardId()] = val
        return ret