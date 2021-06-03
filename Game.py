from Cards.StandarDeck import StandarDeck
from Player import Player

standardeck = StandarDeck()

def distripute(players):
    for i in range(4):
        players[i].setHand(standardeck.distripute(13))
        print(players[i])

players = []
for i in range(4):
    players.append(Player("p" + str(i)))
distripute(players)
winner = 0
#for j in range(2):
j = 0
while len(players[0].hand) > 0:
    j += 1
    print(str(j) + "----------------------------------------------------------------------------")
    tour = []
    for i in range(4):
        tour.append(players[(i + winner)%4].playCard(tour))
        print(tour)
    winner = (standardeck.winner(tour) + winner)%4
    players[winner].win(tour)
    for i in range(4):
        print(players[i])

for i in range(4):
    print(players[i].name, ": ", players[i].getScoreTarneeb())



