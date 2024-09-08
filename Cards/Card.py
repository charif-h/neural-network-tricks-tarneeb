from enum import Enum

import numpy as np


class CardType(Enum):
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
    def __init__(self, value, type):
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
        '''
        4 values matrix, each field for a type, the value of the element is value of the card
        Values varies between [2/14, 14/14]
        '''
        card_matrix = np.zeros(4)
        card_matrix[self.type.id] = self.value.value/14
        return card_matrix

    def largerThan(self, nextCard, respectype=True):
        if(respectype):
            if(self.type == nextCard.type):
                if(self.value.value > nextCard.value.value ):
                    return True
                else:
                    return False
            else:
                True
        else:
            if (self.value.value > nextCard.value.value):
                return True
            else:
                return False

