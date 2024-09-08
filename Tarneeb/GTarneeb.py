import os

from termcolor import colored
from Cards.StandarDeck import StandarDeck
from Tarneeb.TarneebPlayer import TarneebPlayer
from Tarneeb.Turn import Turn
import numpy as np
import logging
import keras
logging.basicConfig(level = logging.INFO)

# Game >> Round >> Turn
def distripute_and_bid(players, tarneeb):
    # This method distributes the cards to the tarneeb players and compute the bidding of each player
    # It returns the sum of bidding /the round is canceled if the sum is less than 11
    # The player bidding function need the following information:
    #   1. The hand of the player
    #   2. The scores of all players (scaled to 1) : [my_score/41, adv1_score/41, partner_score/41, adv2_score/41]
    #   3. The biddings of other players if known or 0 otherwise : # TODO : Divide by 13
    #   4. The tarneeb: a 4 types matrix with 1 in the tarneeb type field, and 0 in the others: Ex. [0, 1, 0, 0]

    scaled_scores = [] #Scaling score to make 41 become 1 (this is better for neural network input)
    for p in players:
        scaled_scores.append(p.score/41.0)
    bids = np.zeros(4)
    tbs = np.zeros(4)
    tbs[tarneeb.id] = 1
    for i in range(4):
        players[i].setHand(standardeck.distripute(13))
        players[i].bid(scores=scaled_scores, biddings=bids, tarneeb=tbs)
        bids[i] = players[i].bidding
        logging.info(players[i].name + ' bid=' + str(players[i].bidding) + ' on hand ' + str(players[i].hand))
    return bids.sum()

def tarneeb_to_array(tarneeb):
    tbs = np.zeros(4)
    tbs[tarneeb.id] = 1
    return tbs
def clearHands(players):
    standardeck = StandarDeck(shuffled=True)
    tarneeb = standardeck.cards[51].type
    for p in players:
        p.prediction = 2
        p.clearHand()

def playRound(players, tarneeb):
    starter_player = 0
    j = 0
    turns = []
    while len(players[starter_player].hand) > 0:
        j += 1
        current_turn_cards = []
        for i in range(4):
            current_turn_cards.append(players[(i + starter_player) % 4].playCard(current_turn_cards))

        turn = Turn(current_turn_cards, serial=j, starting_player_id=starter_player, tarneeb=tarneeb)
        starter_player = turn.winnerId #next starter

        for c in turn.played_cards:
            k = str(c.value.value)
            if c == turn.winCard:
                k += '-W'
            if c.type == tarneeb:
                k = 'T-' + k
            if k in cards_record.keys():
                cards_record[k] += 1
            else:
                cards_record[k] = 1


        logging.info('turn number ' + str(turn))
        players[turn.winnerId].number_of_won_turns += 1
        turns.append(turn)
        logging.info(str(turns))
        turn.turn_to_matrices(players)

    print('turns = ', turns)
    round_loss = {}
    for turn in turns:
        round_loss.update(turn.loss)
    print(round_loss)
    input = np.zeros([4*13, 68])
    for turn in turns:
        #print(turn.serial, ' ==> ', turn.turn_to_matrices(players).shape)
        input[(turn.serial - 1)*4:turn.serial*4] = turn.turn_to_matrices(players)
    '''print(input.shape)
    print(input)
    exit()'''
    return turns

def buildPlayingModel():
    '''pmodel = keras.models.Sequential()
    pmodel.add(keras.layers.LSTM(52, input_shape=(52, 68)))
    pmodel.add(keras.layers.Dense(4))
    pmodel.compile(loss='mean_squared_error', optimizer='adam')
    Xp = np.random.rand(1, 52, 68)
    Yp = np.random.rand(4)
    print('shapes ', Xp.shape, Yp.shape)
    pmodel.fit(Xp, Yp)'''

    timesteps = 52
    features = 68
    output_dim = 4

    # Create random input and output matrices
    # Input: (number_of_samples, timesteps, features)
    # Output: (number_of_samples, output_dim)
    number_of_samples = 52
    X = np.random.random((number_of_samples, timesteps, features))
    y = np.random.random((number_of_samples, output_dim))

    # Define the LSTM model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(timesteps, features), return_sequences=False))
    model.add(keras.layers.Dense(output_dim, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model on the random data
    model.fit(X, y, epochs=10, batch_size=4)
    return model
    #model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

def loss_t(y_true, y_pred):
    for c in round_loss:
        if c.card_to_matrix() == y_pred:
            print('loss ', c, round_loss[c])
            return round_loss[c]
    return 10


cards_record = {}
round_loss = {}
logging.info('Creating players')
players = []
for i in range(4):
    players.append(TarneebPlayer("p" + str(i)))
logging.info('Creating 4 players ' +  str(players))

NUMBER_OF_TRAINING_GAMES = 2
playModel = buildPlayingModel()

logging.info('training for ' + str(NUMBER_OF_TRAINING_GAMES) + ' games')

for z in range(NUMBER_OF_TRAINING_GAMES):
    logging.info("Game number" + str(z))
    turns = []
    print("Game ", "-"*50, z)
    for p in players:
        p.score = 0
    rounds = 0
    winner = ""
    gameOver = False
    while not gameOver:
        rounds += 1
        logging.info("Game " + str(z) + " round " + str(rounds))
        bidding_sum = 0
        Xbid = np.array([])
        Ybid = np.array([])
        while bidding_sum < 11:
            standardeck = StandarDeck(shuffled=True)
            tarneeb = standardeck.cards[51].type
            clearHands(players)
            bidding_sum = distripute_and_bid(players, tarneeb)
        logging.info('The tarneeb is: ' + str(tarneeb) + ' sum of bidding: ' + str(sum))
        turns = playRound(players, tarneeb)
        print(players)
        for p in players:
            p.print_player_round_results()
            Xbid = np.append(Xbid, p.bidding_input)
            Ybid = np.append(Ybid, p.predictionY())
            p.roundOver()
            # need to be corrected what happen if 2 players arrived to 41 at the same time
            if(p.score >= 41): # TODO: what if two players have 41+ scores?
                gameOver = True
                winner = p.name
                p.gamesWon += 1
                logging.info('game ' + str(z) + ' ended with ' + str(rounds) + ' rounds and player ' + str(winner) + ' is the winner')

        logging.info('Round ' + str(rounds) + ' ended. Training bidding models ...')
        Xbid = Xbid.reshape(4,64)
        for pl in players:
            pl.trainBidding(Xbid, Ybid)

        Xplay = np.zeros((52, 52, 68))
        Yplay = np.zeros((52, 4))
        X = np.zeros((103, 68))
        for i, turn in enumerate(turns):
            X[i*4+51:(i+1)*4+51] = turn.turn_to_matrices(players)
            for j, c in enumerate(turn.played_cards):
                Yplay[4*i + j] = c.card_to_matrix()
        '''for i, turn in enumerate(turns):
            Xplay[i*4:(i+1)*4] = turn.turn_to_matrices(players)
            for j, c in enumerate(turn.played_cards):
                Yplay = c.card_to_matrix()'''
        #Xplay.reshape(1, 52, 68)
        #X = np.stack((np.zeros((51, 68)), X), axis=1)
        for i in range(52):
            Xplay[i] = X[i:i + 52]
        playModel.fit(Xplay, Yplay, verbose=2)
        exit()

print()
for p in players:
    print(p.name, colored(p.gamesWon, "blue"))

print(cards_record)

