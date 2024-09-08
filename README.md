# neural-network-tricks-tarneeb

## Objective of the project
Tarneeb and Tricks are very known cards games in the middle-east.
The aim of this project is to develop and train a neural networks to play Tarneeb and Tricks using active and reinforcement learning.

The project is writen in Python and uses principally Keras library.

As Tarneeb is the more simple, we will start with it, and tricks will be added in a later time.

## What is Tarneeb-41?
Tarneeb exists mainly in two versions (41 and 61), we are interrested in the 41 version as it is more popular. So in this document when we talk about Tarneeb, we mean Tarneeb-41.

### The cards and their power
The game is played by the classical 52 cards of four types (♠, ♣, ♢, ♡), two red and two black types.

Cards of the same type are ordered by decreasing strength as follow:
*Ace - King - Queen - Jack - 10 - 9 ... 3 - 2*.

### Players and teams
The game is a duel between two teams of two partners each. Partners set facing each others, so each one is surroundend by the players of the opposite team in both sides.

### Distributing cards
In the first round a random player is chosen to distribute the cards, he must distributes 13 cards to each player starting form the player on his right. before distributing any card, he mixes them and then asks the player on his left to cut the cards in upper and lower group, then he inverses the positions of the groups (puts the upper one down and vice-versa). Now he opens the last card and show it to every body. Now the opposite type of the same color of the open card is called tarneeb: if the card is of type (♢) which is a red type, then the other red (♡) becomes the tarneeb.

After showing the tarneeb to everybody, the player distributes 13 cards to each player.

### How to play

# Playing neural network input
Information needed to choose a card to play
1. tarneeb [0, 0, 0, 1]
2. scores $$[s_0/41, s_1/41, s_2/41, s_3/41]$$
3. bids $$[b_0/13, b_1/13, b_2/13, b_3/13]$$
4. wins $$[w_0/13, w_1/13, w_2/13, w_3/13]$$
5. current turn cards (4 cards)
6. Current player hand cards (13 cards)
7. Winner card win=1 loss=-1 ignore=0
