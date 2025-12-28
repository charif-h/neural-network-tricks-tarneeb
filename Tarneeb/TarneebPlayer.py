"""
TarneebPlayer module for AI player implementation.

This module extends the base Player class with neural network capabilities
for bidding and playing in the Tarneeb card game.
"""

import Player
import GenModel
import numpy as np
import math
import random
from termcolor import colored

from Cards.StandarDeck import cardstoArray


class TarneebPlayer(Player.Player):
    """
    AI player for Tarneeb game using neural networks.
    
    This player uses neural networks to make bidding and playing decisions.
    It trains on game outcomes to improve its strategy over time.
    
    Attributes:
        score (int): Current game score (target is 41)
        bidding (int): Number of tricks bid for current round
        gamesWon (int): Total number of games won
        number_of_won_turns (int): Tricks won in current round
        biddingModel: Neural network for bidding decisions
        playModel: Neural network for card playing decisions
        history: History of game states for training
    """
    
    def __init__(self, name, playModel=None):
        """
        Initialize a TarneebPlayer with neural network models.
        
        Args:
            name (str): Player's name
            playModel: Optional pre-trained model for playing (default: None)
        """
        self.score = 0
        self.bidding = 2
        self.gamesWon = 0
        self.number_of_won_turns = 0
        self.biddingModel = GenModel.Model()
        self.playModel = playModel
        self.history = np.zeros((52, 68))
        Player.Player.__init__(self, str(name))

    def bid(self, scores=[0, 0, 0, 0], biddings=[2, 2, 2, 2], tarneeb=[0, 0, 0, 0]):
        """
        Make a bidding decision using the neural network.
        
        The bid is based on:
        - Player's hand strength
        - Current scores of all players
        - Other players' bids (if known)
        - The tarneeb (trump) suit
        
        Args:
            scores (list): Scaled scores of all players [0-1]
            biddings (list): Bids of other players (scaled by 13)
            tarneeb (list): One-hot encoded tarneeb type
        
        Returns:
            int: Number of tricks bid (2-13)
        """
        res = str(self.bidding) + ":" + str(self.number_of_won_turns).split(".")[0] + "/" + str(self.getResult() + self.score)
        if self.number_of_won_turns > self.bidding:
            print(colored(res, "blue"), end="\t")
        elif self.number_of_won_turns == self.bidding:
            print(res, end="\t")
        else:
            print(colored(res, "red"), end="\t")

    def getResult(self):
        """
        Calculate the score result for the current round.
        
        Scoring rules:
        - If bid is met: gain points equal to bid
        - If bid >= 7 and score < 30: double the points
        - If bid is not met: lose points equal to bid (or doubled)
        
        Returns:
            int: Points to add/subtract from score (negative if bid not met)
        """
        self.hand.sort(reverse=True)
        inp = np.concatenate((self.handToArray(), scores, biddings, tarneeb))
        self.bidding_input = inp.reshape(1, 64)
        o = self.biddingModel.predict(x=np.array(self.bidding_input))
        
        # Minimum bid based on current score
        min_bid = max(self.score // 10, 2)
        
        # Handle NaN predictions
        if math.isnan(o[0][0]):
            self.bidding = min_bid
        else:
            self.bidding = int(o[0][0] * (13 - min_bid) + min_bid)
        
        return self.bidding

    def __repr__(self):
        return (self.name + ": " + str(self.score) + '/' + str(self.number_of_won_turns))

    def print_player_round_results(self):
        """Display the player's round results with color coding."""
        ret = self.bidding
        if (self.bidding >= 7) and (self.score < 30):
            ret = ret * 2
        if self.number_of_won_turns < self.bidding:
            return -1 * ret
        else:
            return ret

    def roundOver(self):
        """
        Process end of round: update score and reset turn counter.
        
        Returns:
            int: Updated total score
        """
        self.score += self.getResult()
        self.number_of_won_turns = 0
        return self.score

    def predictionY(self):
        """
        Calculate the target value for bidding model training.
        
        Returns:
            np.ndarray: Normalized number of tricks won (scaled 0-1)
        """
        Y = (self.number_of_won_turns - 2) / (13 - 2)
        return np.array([Y])

    def trainBidding(self, x=np.array([]), y=np.array([])):
        """
        Train the bidding neural network model.
        
        Args:
            x (np.ndarray): Input features (default: use own bidding input)
            y (np.ndarray): Target values (default: use own prediction)
        """
        X = x
        Y = y
        if len(X) == 0:
            X = self.bidding_input
        if len(Y) == 0:
            Y = self.predictionY()
        self.biddingModel.fit(X, Y, verbose=0, batch_size=4)

    def playCard(self, sdcards, scores=[0, 0, 0, 0], bids=[2, 2, 2, 2], 
                 tours=[0, 0, 0, 0], Tarneeb=[0, 0, 0, 0], *args, **kwargs):
        """
        Play a card following game rules.
        
        Must follow suit if possible (play same type as first card).
        Currently uses random selection, but prepared for neural network integration.
        
        Input representation for neural network:
        - Player bidding (1 value, scaled by 13)
        - Player score (1 value, scaled by 41)
        - Won turns / bidding ratio (1 value)
        - Reserved (1 value)
        - Player hand cards (13 cards * 4 values = 52 values)
        - Already played cards in turn (up to 3 cards * 4 values = 12 values)
        Total: 68 values
        
        Args:
            sdcards (list): Cards already played in this turn
            scores (list): Player scores (scaled)
            bids (list): Player bids (scaled)
            tours (list): Turns won by each player
            Tarneeb (list): Tarneeb type encoding
            *args, **kwargs: Additional arguments
        
        Returns:
            Card: The card chosen to play
        """
        # Prepare played cards matrix (up to 3 cards)
        played_cards = np.zeros(12)
        for card in sdcards:
            played_cards = np.concatenate((played_cards, card.card_to_matrix()))
        played_cards = played_cards[-12:]

        # Player context information
        pc_matrix = np.array([
            self.bidding / 13,
            self.score / 41,
            self.number_of_won_turns / self.bidding if self.bidding > 0 else 0,
            0  # Reserved
        ])

        # Player hand representation
        player_input_matrix = np.zeros(13 * 4)
        for j, h in enumerate(self.hand):
            player_input_matrix[j * 4:j * 4 + 4] = h.card_to_matrix()
        
        input_matrix = np.concatenate([pc_matrix, player_input_matrix, played_cards])
        print('player input matrix', input_matrix.shape, input_matrix)

        # Select card (currently random, can be replaced with NN prediction)
        card = random.choice(self.hand)
        # Must follow suit if possible
        if len(sdcards) > 0 and len(self.filterCardsByType(sdcards[0].type)) > 0:
            card = random.choice(self.filterCardsByType(sdcards[0].type))
        
        self.hand.remove(card)
        return card

    def chooseCard(self, legalCards):
        """
        Choose a card from legal options.
        
        Base implementation uses random selection. This method can be
        enhanced to use neural network predictions.
        
        Args:
            legalCards (list): List of cards that can be legally played
        
        Returns:
            Card: The chosen card
        """
        return random.choice(legalCards)



    def play(self, input):
        print(self.playModel.predict(input))

    def playModelInput(self, turns, scores=[0, 0, 0, 0], biddings=[2, 2, 2, 2], tarneeb=[0, 0, 0, 0], my_id = 0):
        scaled_biddings = np.roll(np.array(biddings)/13, my_id)
        scaled_scores = np.roll(np.array(scores)/41, my_id)
        #Tarneeb
        hand = self.hand
        #played_cards = 3 cards
        #starting_type


        current_input = np.concatenate((tarneeb))


