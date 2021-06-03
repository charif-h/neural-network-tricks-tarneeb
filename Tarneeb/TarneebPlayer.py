import Player
import GenModel
import numpy as np
import math
import random

from Cards.StandarDeck import cardstoArray


class TarneebPlayer(Player.Player):
    def __init__(self, name):
        self.score = 0
        self.prediction = 2
        self.gamesWon = 0
        self.tours = 0
        self.gerror = 0
        self.predictionModel = GenModel.Model()
        self.playModel = GenModel.Model(inputs=116, outputs=52)
        Player.Player.__init__(self, str(name))

    def __repr__(self):
        return self.name + ": " + str(self.score) + "\n" + str(self.predictionModel.summary())

    def predict(self, scores=[0, 0, 0, 0], predictions=[2, 2, 2, 2], tarneeb=[0, 0, 0, 0]):
        inp = np.concatenate((self.handToArray(), scores, predictions, tarneeb))
        self.inp = inp.reshape(1, 64)
        o = self.predictionModel.predict(x=np.array(self.inp))
        min = max(self.score // 10, 2)
        #self.prediction = random.randint(min, 13)
        if(math.isnan(o[0][0])):
            self.prediction = min
        else:
            self.prediction = int(o[0][0]*(13 - min) + min)
        return self.prediction

    def getResult(self):
        ret = self.prediction
        if ((self.prediction >= 7) and (self.score < 30)):
            ret = ret*2
        if (self.tours < self.prediction):
            return -1*ret
        else:
            return ret

    def zikOver(self):
        self.gerror += abs(self.tours - self.prediction)
        self.score += self.getResult()
        self.tours = 0
        return self.score

    def predictionY(self):
        Y = (self.tours - 2) / (13 - 2)
        return np.array([Y])

    def trainPrediction(self, x = np.array([]), y = np.array([])):
        X = x
        Y = y
        if(len(X) == 0):
            X = self.inp
        if(len(Y) == 0):
            Y = self.predictionY()
        self.predictionModel.fit(X, Y, verbose=0)

    # Cards [52] => 1: the card is with me, -1: the card has been played, 0: the card is not played yet
    # Played cards [52] => 1 The card is played, 0 the card is not played
    # 4x(win-bid) [4] =>
    # scores [4]
    #Tarneeb [4]
    def XplayCard(self, sdcards, scores=[0, 0, 0, 0], bids=[2, 2, 2, 2], tours = [0, 0, 0, 0], Tarneeb=[0, 0, 0, 0]):
        rest = []
        for i in range(len(bids)):
            rest.append(tours[i] - bids[i])
        inp = np.concatenate((self.handToArray(), scores, rest, Tarneeb))
        hand = cardstoArray(self.hand, 1)
        passedCard = cardstoArray(sdcards, -1)
        self.handToArray()
        print(str(inp))
        print(str(hand))
        print(str(passedCard))
        print(str(self.handToArray()))
        #exit()

    def chooseCard(self, ligalCards):
        print(self.XplayCard(ligalCards))
        print(ligalCards)
        exit()
        return random.choice(ligalCards)


