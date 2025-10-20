"""
Simple demonstration of the card game.

This module provides a basic game loop demonstrating the core mechanics
of card distribution, playing, and scoring.
"""

from Cards.StandarDeck import StandarDeck
from Player import Player

standardeck = StandarDeck()


def distripute(players):
    """
    Distribute 13 cards to each of 4 players.
    
    Note: Function name 'distripute' is a typo (should be 'distribute').
    
    Args:
        players (list): List of 4 Player objects
    """
    for i in range(4):
        players[i].setHand(standardeck.distripute(13))
        print(players[i])


# Initialize 4 players
players = []
for i in range(4):
    players.append(Player("p" + str(i)))

# Distribute cards to all players
distripute(players)

# Play the game
winner = 0
j = 0
# Play until all cards are exhausted
while len(players[0].hand) > 0:
    j += 1
    print(str(j) + "----------------------------------------------------------------------------")
    tour = []
    # Each player plays a card in turn
    for i in range(4):
        tour.append(players[(i + winner) % 4].playCard(tour))
        print(tour)
    # Determine winner of this round
    winner = (standardeck.winner(tour) + winner) % 4
    players[winner].win(tour)
    # Display current state
    for i in range(4):
        print(players[i])

# Display final scores

for i in range(4):
    print(players[i].name, ": ", players[i].getScoreTarneeb())



