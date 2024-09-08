import Player
import GenModel
import numpy as np
import math
import random
from termcolor import colored



from Cards.StandarDeck import cardstoArray


class TarneebPlayer(Player.Player):
    def __init__(self, name, playModel=None):
        self.score = 0
        self.bidding = 2
        self.gamesWon = 0
        self.number_of_won_turns = 0
        #self.gerror = 0
        self.biddingModel = GenModel.Model()
        self.playModel = playModel
        self.history = np.zeros((52, 68))
        #self.buildPlayingModel()
        Player.Player.__init__(self, str(name))

    def __repr__(self):
        return (self.name + ": " + str(self.score) + '/' + str(self.number_of_won_turns))

    def print_player_round_results(self):
        res = str(self.bidding) + ":" + str(self.number_of_won_turns).split(".")[0] + "/" + str(self.getResult() + self.score)
        if (self.number_of_won_turns > self.bidding):
            print(colored(res,"blue"), end="\t")
        elif (self.number_of_won_turns == self.bidding):
            print(res, end="\t")
        else:
            print(colored(res,"red"), end="\t")

    def bid(self, scores=[0, 0, 0, 0], biddings=[2, 2, 2, 2], tarneeb=[0, 0, 0, 0]):
        self.hand.sort(reverse=True)
        inp = np.concatenate((self.handToArray(), scores, biddings, tarneeb))
        self.bidding_input = inp.reshape(1, 64)
        o = self.biddingModel.predict(x=np.array(self.bidding_input))
        min = max(self.score // 10, 2)
        if(math.isnan(o[0][0])):
            self.bidding = min
        else:
            self.bidding = int(o[0][0]*(13 - min) + min)
        return self.bidding

    def getResult(self):
        ret = self.bidding
        if ((self.bidding >= 7) and (self.score < 30)):
            ret = ret*2
        if (self.number_of_won_turns < self.bidding):
            return -1*ret
        else:
            return ret

    def roundOver(self):
        #self.gerror += abs(self.number_of_won_turns - self.bidding)
        self.score += self.getResult()
        self.number_of_won_turns = 0
        return self.score

    def predictionY(self):
        Y = (self.number_of_won_turns - 2) / (13 - 2)
        return np.array([Y])

    def trainBidding(self, x = np.array([]), y = np.array([])):
        X = x
        Y = y
        if(len(X) == 0):
            X = self.bidding_input
        if(len(Y) == 0):
            Y = self.predictionY()
        self.biddingModel.fit(X, Y, verbose=0, batch_size=4)

    # Cards [52] => 1: the card is with me, -1: the card has been played, 0: the card is not played yet
    # Played cards [52] => 1 The card is played, 0 the card is not played
    # 4x(win-bid) [4] =>
    # scores [4]
    #Tarneeb [4]
    def playCard(self, sdcards, scores=[0, 0, 0, 0], bids=[2, 2, 2, 2], tours = [0, 0, 0, 0], Tarneeb=[0, 0, 0, 0], *args, **kwargs):
        played_cards = np.zeros(12)
        for card in sdcards:
            played_cards = np.concatenate((played_cards, card.card_to_matrix()))
        played_cards = played_cards[-12:]

        pc_matrix = np.array([self.bidding / 13,
                              self.score / 41,
                              self.number_of_won_turns / self.bidding,
                              0])

        player_input_matrix = np.zeros(13 * 4)
        for j, h in enumerate(self.hand):
            player_input_matrix[j * 4:j * 4 + 4] = h.card_to_matrix()
        input_matrix = np.concatenate([pc_matrix, player_input_matrix, played_cards])

        print('player input matrix', input_matrix.shape, input_matrix)

        card = random.choice(self.hand)
        if len(sdcards) > 0 and len(self.filterCardsByType(sdcards[0].type)) > 0:
            card = random.choice(self.filterCardsByType(sdcards[0].type))
        self.hand.remove(card)
        return card
        '''for i in range(len(bids)):
            rest.append(tours[i] - bids[i])
        inp = np.concatenate((self.handToArray(), scores, rest, Tarneeb))
        hand = cardstoArray(self.hand, 1)
        self.handToArray()
        return super().playCard()'''

    def chooseCard(self, ligalCards):
        '''
        Information needed to play a card:
        1. player hand : array[52 cards]: 1 if the card is in hand; 0 otherwise
        2. other players hands : 3 arrays[52 cards] : 1 if the card has been played by this player; 0 otherwise
        3. biddings array[4]
        4. scores array[4]
        5. turns won by each player array[4]

        May be we can have another representation for 1 and 2:
            for each card:
                - is the card in my hand? 1:yes; 0:no
                - did I play this card? 1:yes; 0:no
                - did the player to the left played this card? 1:yes; 0:no
                - did my partner played this card? 1:yes; 0:no
                - did the player to the right played this card? 1:yes; 0:no

        A third solution is by using LSTM we can record each turn by the following information:
            - biddings: array[4]
            - scores : array[4]
            - turns won by each player till this turn: array[4]
            - first player to open the turn
            - card played by me
            - card played by the player on the left
            - card played by the player on the right
            - card played by my partner
            - the winner card
            - the tarneeb
        We need also to know the cards played in the current turn
        :param ligalCards:
        :return:
        '''
        #print(self.XplayCard(ligalCards))
        #print(ligalCards)
        #exit()
        return random.choice(ligalCards)



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


