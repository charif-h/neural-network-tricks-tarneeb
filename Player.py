import random
import numpy as np
class Player:
    def __init__(self, name):
        self.name = name
        self.quiver = []
    def __repr__(self):
        return self.name + ": " + str(self.hand) + "\t" + str(self.quiver)
    def setHand(self, cards):
        self.hand = cards
    def clearHand(self):
        self.hand = []
    def filterCardsByType(self, cType):
        nh = []
        for c in self.hand:
            if(c.type == cType):
                nh.append(c)
        return nh

    def playCard(self, cards = []):
        crd = self.chooseCard(self.hand)
        if(len(cards) > 0):
            fh = self.filterCardsByType(cards[0].type)
            if(len(fh) > 0):
                crd = self.chooseCard(fh)
        self.hand.pop(self.hand.index(crd))
        return crd

    def chooseCard(self, ligalCards):
        return random.choice(ligalCards)

    def handToArray(self):
        ret = np.zeros(52)
        for c in self.hand:
            ret[c.cardId()] = 1
        return ret
