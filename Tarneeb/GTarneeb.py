from termcolor import colored

from Cards.StandarDeck import StandarDeck
from Tarneeb.TarneebPlayer import TarneebPlayer
import numpy as np

def distripute(players, tarneeb):
    scores = []
    for p in players:
        scores.append(p.score/41.0)
    preds = np.zeros(4)
    tbs = np.zeros(4)
    tbs[tarneeb.id] = 1
    for i in range(4):
        players[i].setHand(standardeck.distripute(13))
        players[i].predict(scores=scores, predictions=preds, tarneeb=tbs)
        preds[i] = players[i].prediction
        #print(players[i], players[i].predict())
    return preds.sum()

def clearHands(players):
    standardeck = StandarDeck()
    tarneeb = standardeck.cards[51].type
    for p in players:
        p.prediction = 2
        p.clearHand()

def playGame(players):
    winner = 0
    j = 0
    while len(players[0].hand) > 0:
        j += 1
        #print(str(j) + "----------------------------------------------------------------------------")
        tour = []
        for i in range(4):
            tour.append(players[(i + winner) % 4].playCard(tour))
            #print(tour)
        winner = (standardeck.winner(tour) + winner) % 4
        players[winner].tours += 1
        #for i in range(4):
            #print(players[i])

players = []
for i in range(4):
    players.append(TarneebPlayer("p" + str(i)))

s = []
for z in range(5000):
    print("ZIK", "-"*50, z)
    for p in players:
        p.score = 0
    games = 0
    winner = ""
    finish = False
    while not finish:
        games += 1
        sum = 0
        Xbid = np.array([])
        Ybid = np.array([])
        while sum < 11:
            standardeck = StandarDeck()
            tarneeb = standardeck.cards[51].type
            clearHands(players)
            sum = distripute(players, tarneeb)
        playGame(players)
        for p in players:
            if(p.tours > p.prediction):
                print(colored(str(p.prediction) + ":" + str(p.tours).split(".")[0] + "/" + str(p.getResult() + p.score), "blue"), end="\t")
            elif(p.tours == p.prediction):
                print(str(p.prediction) + ":" + str(p.tours).split(".")[0] + "/" + str(p.getResult() + p.score), end="\t")
            else:
                print(colored(str(p.prediction) + ":" + str(p.tours).split(".")[0] + "/" + str(p.getResult() + p.score), "red"), end="\t")
            p.trainPrediction(p.inp, p.predictionY())
            #train all players on this case
            Xbid = np.append(Xbid, p.inp)
            Ybid = np.append(Ybid, p.predictionY())
            '''for pl in players:
                pl.trainPrediction(p.inp, p.predictionY())'''
            p.zikOver()
            # need to be corrected what happen if 2 players arrived to 41 at the same time
            if(p.score >= 41):
                finish = True
                winner = p.name
                p.gamesWon += 1
        Xbid = Xbid.reshape(4,64)
        #for pl in players:
            #pl.trainPrediction(Xbid, Ybid)
        print()
    s.append(games)
    for pl in players:
        e = str(pl.gerror/games)
        print(e[:5], end="\t")
        pl.gerror = 0
    print()

for p in players:
    print(p)

print(s[:25])
print(s[-25:])

for p in players:
    print(p.name, colored(p.gamesWon, "blue"))

#import GenModel
#GenModel.Model()
