"""
Card module for Tarneeb card game.

This module defines the Card class and related enumerations for card types and values.
"""

from enum import Enum

import numpy as np


class CardType(Enum):
    """
    Enumeration representing the four card types (suits) in a standard deck.
    
    Each type has an ID (0-3) and a symbol (♣, ♦, ♠, ♥).
    """
    CLUB = 0, "♣"
    DIAMOND = 1, "♦"
    SPADE = 2, "♠"
    HEART = 3, "♥"

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = name
        member.id = value
        return member

    def __int__(self):
        return self.id

class CardValue(Enum):
    """
    Enumeration representing the values of cards in a standard deck.
    
    Values range from 2 (lowest) to 14 (Ace, highest).
    """
    ACE = 14
    KING = 13
    QUEEN = 12
    JACK = 11
    TEN = 10
    NINE = 9
    EIGHT = 8
    SEVEN = 7
    SIX = 6
    FIVE = 5
    FOUR= 4
    THREE = 3
    TWO = 2

class Card:
    """
    Represents a playing card with a value and type (suit).
    
    Attributes:
        value (CardValue): The value of the card (2-14, where 14 is Ace)
        type (CardType): The type/suit of the card (CLUB, DIAMOND, SPADE, HEART)
    """
    
    def __init__(self, value, type):
        """
        Initialize a Card with a value and type.
        
        Args:
            value (CardValue): The value of the card
            type (CardType): The type/suit of the card
        """
        self.value = value
        self.type = type
    def __str__(self):
        return str(self.valueChar()) + self.type.value
        #return str(self.value)[10:] + " of " + str(self.type)[9:] + "S"
    def __repr__(self):
        return str(self.valueChar()) + "-" + self.type.value

    def __hash__(self):
        return hash(str(self.type) + str(self.value))

    def __eq__(self, other):
        return self.value == other.value and self.type == other.type

    def __lt__(self, other):
        if self.type.value == other.type.value:
            return self.value.value < other.value.value
        return self.type.value < other.type.value

    def valueChar(self):
        if(self.value.value <= 10):
            return self.value.value
        else:
            return str(self.value)[str(self.value).index(".") + 1]

    def cardId(self):
        return  4*(self.value.value - 2) + self.type.id

    def card_to_matrix(self):
        """
        Convert card to a 4-element numpy array for neural network input.
        
        Creates a one-hot encoded array where the position corresponding to
        the card's type contains the normalized value (value/14), and all
        other positions are 0.
        
        Returns:
            np.ndarray: 4-element array with normalized card value at type position
                       Values range from 2/14 (≈0.14) to 14/14 (1.0)
        """
        card_matrix = np.zeros(4)
        card_matrix[self.type.id] = self.value.value/14
        return card_matrix

    def largerThan(self, nextCard, respectype=True):
        """
        Compare this card with another card to determine if it's larger.
        
        Args:
            nextCard (Card): The card to compare against
            respectype (bool): If True, only compare cards of the same type.
                              If False, compare values regardless of type.
        
        Returns:
            bool: True if this card is larger than nextCard
        
        Note:
            When respectype=True and types differ, returns False (not True as buggy code did)
        """
        if respectype:
            if self.type == nextCard.type:
                return self.value.value > nextCard.value.value
            else:
                return False  # Fixed: was returning True
        else:
            return self.value.value > nextCard.value.value

