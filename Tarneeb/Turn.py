import numpy as np
class Turn():
    def __init__(self, cards, tarneeb, serial=1, starting_player_id=0):
        self.serial = serial
        #self.biddings = [2, 2, 2, 2]
        #self.scores = [0, 0, 0, 0]
        #self.turns_scores = [0, 0, 0, 0]
        self.starting_player_id = starting_player_id
        self.played_cards = cards
        self.loss = {}
        self.tarneeb = tarneeb
        self.winner(self.tarneeb)
        self.playing_loss_function()


    def __repr__(self):
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

    def winner (self, tarneeb):
        self.winCard = self.played_cards[0]
        self.winnerId = self.starting_player_id
        for i in range(1, len(self.played_cards)):
            if (self.played_cards[i].largerThan(self.winCard)) or (self.played_cards[i].type == tarneeb and self.winCard.type != tarneeb):
                self.winCard = self.played_cards[i]
                self.winCardId = i
                self.winnerId = (i + self.starting_player_id)%4
        #return self.winnerId

    def turn_to_matrices(self, players):
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
            player_id = (self.starting_player_id + i)%4
            pc_matrix = np.array([players[player_id].bidding/13,
                                  players[player_id].score / 41,
                                  players[player_id].number_of_won_turns/players[player_id].bidding,
                                   c.type == self.tarneeb])

            player_input_matrix = np.zeros(13*4)
            for j, h in enumerate(players[player_id].hand):
                player_input_matrix[j*4:j*4+4] = h.card_to_matrix()
            input_matrix = np.concatenate([pc_matrix, player_input_matrix, played_cards])
            if i < 3:
                played_cards[i*4:i*4+4] = c.card_to_matrix()
            #print(c, pc_matrix)
            #print(input_matrix)
            turn_input_matrix = np.append(turn_input_matrix, input_matrix)
        '''
        generates 4 input matrices, each one represents the state of one of the 4 players when playing his card:
        :return:
        '''
        return turn_input_matrix.reshape(4, 68)

    def playing_loss_function(self):
        card_win_prob = {2: 0.012, 3: 0.013, 4: 0.019, 5: 0.028,
                         6: 0.044, 7: 0.068, 8: 0.101, 9: 0.148,
                         10: 0.206, 11: 0.28, 12: 0.38, 13: 0.5, 14: 0.65}
        tarneeb_win_prob = {2: 0.175, 3: 0.185, 4: 0.205, 5: 0.223,
                            6: 0.257, 7: 0.3, 8: 0.34, 9: 0.41,
                            10: 0.47, 11: 0.58, 12: 0.7, 13: 0.85, 14: 1.00}

        for c in self.played_cards:
            if c.type == self.tarneeb:
                self.loss[c] = tarneeb_win_prob[c.value.value]
            else:
                self.loss[c] = card_win_prob[c.value.value]

        self.loss[self.winCard] = (4*self.loss[self.winCard] - sum(self.loss.values()))*0.01
        print('turn analysis ', self.tarneeb.value, self.played_cards, self.winCard, self.loss)

    '''def winproba(self, card):
        if card.type == self.tarneeb:
            prob = (12 + card.value)/26
        else:
            prob = (card.value - 1)/26'''

    '''def loss(self):
        if win == True:
            1/(sum(other cards) - mycard)
        else:
            winner_card - mycard'''
