from enum import Enum
class CardType(Enum):
    CLUB = 0, "♣"
    DIAMOND = 1, "♦"
    HEART = 2, "♥"
    SPADE = 3, "♠"

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
        return str(self.value)[10:] + " of " + str(self.type)[9:] + "S"
    def __repr__(self):

        return str(self.valueChar()) + "-" + self.type.value

    def __eq__(self, other):
        return self.value == other.value and self.type == other.type

    def valueChar(self):
        if(self.value.value <= 10):
            return self.value.value
        else:
            return str(self.value)[str(self.value).index(".") + 1]

    def cardId(self):
        return  4*(self.value.value - 2) + self.type.id

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

