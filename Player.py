"""
Player module for card game base functionality.

This module provides a base Player class that can be extended for
specific game implementations.
"""

import random
import numpy as np


class Player:
    """
    Base class representing a player in a card game.
    
    Attributes:
        name (str): The name of the player
        hand (list): List of Card objects in the player's hand
        quiver (list): List of cards won by the player
    """
    
    def __init__(self, name):
        """
        Initialize a Player with a name.
        
        Args:
            name (str): The name of the player
        """
        self.name = name
        self.quiver = []
    def __repr__(self):
        return self.name + ": " + str(self.hand) + "\t" + str(self.quiver)
    def setHand(self, cards):
        """
        Set the player's hand with the given cards.
        
        Args:
            cards (list): List of Card objects
        """
        self.hand = cards
    
    def clearHand(self):
        """Clear all cards from the player's hand."""
        self.hand = []
    
    def filterCardsByType(self, cType):
        """
        Filter cards in hand by a specific type/suit.
        
        Args:
            cType (CardType): The card type to filter by
        
        Returns:
            list: List of cards matching the specified type
        """
        nh = []
        for c in self.hand:
            if(c.type == cType):
                nh.append(c)
        return nh

    def playCard(self, cards=[], *args, **kwargs):
        """
        Play a card from hand, following game rules.
        
        If cards have already been played in this turn, must follow suit
        if possible (play a card of the same type as the first card played).
        
        Args:
            cards (list): List of cards already played in this turn
            *args: Additional arguments for subclass implementations
            **kwargs: Additional keyword arguments for subclass implementations
        
        Returns:
            Card: The card chosen to be played
        """
        crd = self.chooseCard(self.hand)
        if(len(cards) > 0):
            fh = self.filterCardsByType(cards[0].type)
            if(len(fh) > 0):
                crd = self.chooseCard(fh)
        self.hand.pop(self.hand.index(crd))
        return crd

    def chooseCard(self, legalCards):
        """
        Choose a card from the list of legal cards.
        
        Base implementation uses random selection. Override in subclasses
        for more sophisticated strategies.
        
        Args:
            legalCards (list): List of cards that can be legally played
        
        Returns:
            Card: The chosen card
        """
        print('do not ...')
        return random.choice(legalCards)

    def handToArray(self):
        """
        Convert player's hand to a binary numpy array.
        
        Creates a 52-element array where each position represents a card
        (indexed by cardId). Value is 1 if card is in hand, 0 otherwise.
        
        Returns:
            np.ndarray: Binary array representing cards in hand
        """
        ret = np.zeros(52)
        for c in self.hand:
            ret[c.cardId()] = 1
        return ret
