"""
Standard deck module for card games.

This module provides a StandardDeck class for managing a 52-card deck
and utility functions for card operations.

Note: The class name 'StandarDeck' is kept for backward compatibility,
but should be 'StandardDeck'. Consider refactoring in future versions.
"""

import logging

from Cards.Card import CardValue
from Cards.Card import CardType
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
    
    def __init__(self, shuffled=False):
        """
        Initialize a standard 52-card deck.
        
        Args:
            shuffled (bool): If True, shuffle the deck after creation (default: False)
        """
        self.cards = []
        for i in CardValue:
            for j in CardType:
                self.cards.append(Card(i, j))
        if shuffled:
            random.shuffle(self.cards)
        logging.info('New standard deck created with %d cards', len(self.cards))

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
        ret = []
        for i in range(n):
            ret.append(self.cards.pop(0))
        return ret

    def winner(self, playedCards, tarneeb=None):
        """
        Determine the winner of a hand.
        
        The winning card is determined by:
        1. Tarneeb (trump) cards beat non-tarneeb cards
        2. Among cards of the same type, higher value wins
        3. The first card sets the "lead" type
        
        Args:
            playedCards (list): List of Card objects played in the hand
            tarneeb (CardType, optional): The trump suit for this round
        
        Returns:
            int: Index of the winning card in playedCards
        
        Note:
            This method has a side effect of appending playedCards to self.cards,
            which seems incorrect and should be reviewed.
        """
        wincard = playedCards[0]
        winId = 0
        for i in range(1, len(playedCards)):
            if not wincard.largerThan(playedCards[i]):
                wincard = playedCards[i]
                winId = i
        
        # TODO: Review this - appending to cards seems incorrect
        self.cards.append(playedCards)
        return winId


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