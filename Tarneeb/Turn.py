"""
Turn module for Tarneeb game.

This module represents a single turn in a Tarneeb game, tracking the cards played,
determining the winner, and computing loss functions for neural network training.
"""

import numpy as np


class Turn():
    """
    Represents a single turn (trick) in a Tarneeb game.
    
    A turn consists of 4 players each playing one card, with one player
    winning based on card values and the tarneeb (trump) suit.
    
    Attributes:
        serial (int): The turn number in the round (1-13)
        starting_player_id (int): ID of the player who led this turn
        played_cards (list): List of 4 Card objects played in order
        tarneeb (CardType): The trump suit for this round
        winCard (Card): The card that won this turn
        winnerId (int): ID of the player who won this turn
        winCardId (int): Index of winning card in played_cards
        loss (dict): Loss values for each card (for training)
    """
    
    def __init__(self, cards, tarneeb, serial=1, starting_player_id=0):
        """
        Initialize a Turn with played cards and game state.
        
        Args:
            cards (list): List of 4 Card objects played in this turn
            tarneeb (CardType): The trump suit for this round
            serial (int): Turn number in the round (default: 1)
            starting_player_id (int): ID of player who led (default: 0)
        """
        self.serial = serial
        self.starting_player_id = starting_player_id
        self.played_cards = cards
        self.loss = {}
        self.tarneeb = tarneeb
        self.winner(self.tarneeb)
        self.playing_loss_function()


    def winner(self, tarneeb):
        """
        Determine the winning card and player for this turn.
        
        The winner is determined by:
        1. Tarneeb (trump) cards beat all non-tarneeb cards
        2. Among cards of the same type, higher value wins
        3. First card played sets the "lead" type
        
        Args:
            tarneeb (CardType): The trump suit for this round
        """
        player_cards = '['
        for i, c in enumerate(self.played_cards):
            player_cards += str((self.starting_player_id + i)%4) + ':'
            if c == self.winCard:
                player_cards += '(' + str(c) + ') '
            else:
                player_cards += str(c) + ' '
        player_cards += ']'
        return str(self.serial) + " " + player_cards

    def __str__(self):
        player_cards = '['
        for i, c in enumerate(self.played_cards):
            player_cards += str((self.starting_player_id + i) % 4) + ':' + str(c) + ' '
        player_cards += ']'
        return str(self.serial) + " " + player_cards + ' winner is ' + str(self.winnerId) + ' with card ' + str(
            self.winCard)

        self.winCard = self.played_cards[0]
        self.winnerId = self.starting_player_id
        for i in range(1, len(self.played_cards)):
            card = self.played_cards[i]
            # Check if this card beats the current winner
            if (card.largerThan(self.winCard) or 
                (card.type == tarneeb and self.winCard.type != tarneeb)):
                self.winCard = card
                self.winCardId = i
                self.winnerId = (i + self.starting_player_id) % 4

    def __repr__(self):
        '''
        - Bidding: scaled (1 value)
        - score: scaled (1 value)
        - won turns: percentage of bidding (1 value)
        - tarneeb!: (4 value) #TODO : need to convert the 1 value (is tarneeb) to 4 values.
        - player hand cards (13*4 values), each card is 4 values (zeros otherwise)
        - already played cards in this turn (4*3), we can have up to 3 played cards

        :param players:
        :return:
        '''
        played_cards = np.zeros(4 * 3)
        turn_input_matrix = np.array([])
        
        for i, c in enumerate(self.played_cards):
            player_id = (self.starting_player_id + i) % 4
            
            # Player context: bidding, score, won turns ratio, is tarneeb
            pc_matrix = np.array([
                players[player_id].bidding / 13,
                players[player_id].score / 41,
                players[player_id].number_of_won_turns / players[player_id].bidding,
                c.type == self.tarneeb
            ])

            # Player's hand representation (13 cards * 4 values each)
            player_input_matrix = np.zeros(13 * 4)
            for j, h in enumerate(players[player_id].hand):
                player_input_matrix[j * 4:j * 4 + 4] = h.card_to_matrix()
            
            # Combine all input components
            input_matrix = np.concatenate([pc_matrix, player_input_matrix, played_cards])
            
            # Update played cards for next player
            if i < 3:
                played_cards[i * 4:i * 4 + 4] = c.card_to_matrix()
            
            turn_input_matrix = np.append(turn_input_matrix, input_matrix)
        
        return turn_input_matrix.reshape(4, 68)

    def playing_loss_function(self):
        """
        Calculate loss values for each card played in this turn.
        
        Uses probability-based loss functions where cards are assigned
        expected win probabilities based on their value. Tarneeb cards
        have higher win probabilities.
        
        The winning card's loss is adjusted based on the difference between
        its win probability and the average of all cards.
        
        Note: These probability values are hard-coded and may need tuning
        based on actual game statistics.
        """
        # Win probabilities for regular cards based on value
        card_win_prob = {
            2: 0.012, 3: 0.013, 4: 0.019, 5: 0.028,
            6: 0.044, 7: 0.068, 8: 0.101, 9: 0.148,
            10: 0.206, 11: 0.28, 12: 0.38, 13: 0.5, 14: 0.65
        }
        
        # Win probabilities for tarneeb (trump) cards - higher than regular
        tarneeb_win_prob = {
            2: 0.175, 3: 0.185, 4: 0.205, 5: 0.223,
            6: 0.257, 7: 0.3, 8: 0.34, 9: 0.41,
            10: 0.47, 11: 0.58, 12: 0.7, 13: 0.85, 14: 1.00
        }

        # Assign base loss to each card
        for c in self.played_cards:
            if c.type == self.tarneeb:
                self.loss[c] = tarneeb_win_prob[c.value.value]
            else:
                self.loss[c] = card_win_prob[c.value.value]

        # Adjust winning card's loss based on comparison with other cards
        self.loss[self.winCard] = (4 * self.loss[self.winCard] - sum(self.loss.values())) * 0.01
        print('turn analysis ', self.tarneeb.value, self.played_cards, self.winCard, self.loss)
