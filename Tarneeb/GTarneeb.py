"""
GTarneeb - Main game loop and neural network training for Tarneeb.

This module implements the complete Tarneeb-41 game with neural network
training for both bidding and playing strategies. It manages:
- Game flow (multiple rounds until a player reaches 41 points)
- Card distribution and bidding
- Turn-by-turn play with neural network decisions
- Model training after each round

Usage:
    python Tarneeb/GTarneeb.py

Note: This module contains the training loop. For simple games without
training, see Game.py or example.py.
"""

import os

from termcolor import colored
from Cards.StandarDeck import StandarDeck
from Tarneeb.TarneebPlayer import TarneebPlayer
from Tarneeb.Turn import Turn
import numpy as np
import logging
import keras

logging.basicConfig(level=logging.INFO)


# Game >> Round >> Turn
def distripute_and_bid(players, tarneeb):
    """
    Distribute cards to players and collect their bids.
    
    This method:
    1. Distributes 13 cards to each player
    2. Collects bids from each player using their neural network
    3. Returns the sum of all bids
    
    The round is cancelled if the sum is less than 11.
    
    Args:
        players (list): List of 4 TarneebPlayer objects
        tarneeb (CardType): The trump suit for this round
    
    Returns:
        float: Sum of all player bids
    """

    # Scale scores for neural network input (0-1 range)
    scaled_scores = []
    for p in players:
        scaled_scores.append(p.score / 41.0)
    
    bids = np.zeros(4)
    tbs = np.zeros(4)
    tbs[tarneeb.id] = 1
    
    # Each player receives cards and makes a bid
    for i in range(4):
        players[i].setHand(standardeck.distripute(13))
        players[i].bid(scores=scaled_scores, biddings=bids, tarneeb=tbs)
        bids[i] = players[i].bidding
        logging.info(players[i].name + ' bid=' + str(players[i].bidding) + 
                    ' on hand ' + str(players[i].hand))
    
    return bids.sum()


def tarneeb_to_array(tarneeb):
    """
    Convert tarneeb type to one-hot encoded array.
    
    Args:
        tarneeb (CardType): The trump suit
    
    Returns:
        np.ndarray: 4-element one-hot encoded array
    """
    tbs = np.zeros(4)
    tbs[tarneeb.id] = 1
    return tbs


def clearHands(players):
    """
    Clear all players' hands (used between rounds).
    
    Args:
        players (list): List of TarneebPlayer objects
    """
    standardeck = StandarDeck(shuffled=True)
    tarneeb = standardeck.cards[51].type
    for p in players:
        p.prediction = 2
        p.clearHand()


def playRound(players, tarneeb):
    """
    Play a complete round of 13 turns.
    
    Each turn, 4 cards are played (one per player), and a winner is determined.
    The winner leads the next turn.
    
    Args:
        players (list): List of 4 TarneebPlayer objects
        tarneeb (CardType): The trump suit for this round
    
    Returns:
        list: List of Turn objects representing all turns played
    """
    starter_player = 0
    j = 0
    turns = []
    
    # Play all 13 turns
    while len(players[starter_player].hand) > 0:
        j += 1
        current_turn_cards = []
        
        # Each player plays a card
        for i in range(4):
            current_turn_cards.append(
                players[(i + starter_player) % 4].playCard(current_turn_cards)
            )

        # Create turn object and determine winner
        turn = Turn(current_turn_cards, serial=j, 
                   starting_player_id=starter_player, tarneeb=tarneeb)
        starter_player = turn.winnerId  # Winner leads next turn

        # Record card statistics for analysis
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

    # Collect loss information from all turns
    print('turns = ', turns)
    round_loss = {}
    for turn in turns:
        round_loss.update(turn.loss)
    print(round_loss)
    
    # Prepare input matrices for training (not currently used)
    input_matrix = np.zeros([4 * 13, 68])
    for turn in turns:
        input_matrix[(turn.serial - 1) * 4:turn.serial * 4] = turn.turn_to_matrices(players)
    
    return turns


def buildPlayingModel():
    """
    Build an LSTM neural network model for card playing decisions.
    
    The model uses LSTM to process sequential game state information
    and output card selection probabilities.
    
    Architecture:
    - Input: (timesteps=52, features=68)
    - LSTM layer: 64 units
    - Output: 4 values (card representation)
    
    Returns:
        keras.models.Sequential: Compiled LSTM model
    
    Note: The model is pre-trained on random data for initialization.
          Real training happens during gameplay.
    """
    timesteps = 52  # Maximum number of turns
    features = 68  # Input features per turn
    output_dim = 4  # Card representation dimension
    number_of_samples = 52

    # Create random training data for model initialization
    X = np.random.random((number_of_samples, timesteps, features))
    y = np.random.random((number_of_samples, output_dim))

    # Define the LSTM model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(timesteps, features), 
                                return_sequences=False))
    model.add(keras.layers.Dense(output_dim, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Initial training on random data
    model.fit(X, y, epochs=10, batch_size=4, verbose=0)
    
    return model


def loss_t(y_true, y_pred):
    """
    Custom loss function for card playing (currently unused).
    
    Args:
        y_true: True card value
        y_pred: Predicted card value
    
    Returns:
        float: Loss value
    """
    for c in round_loss:
        if c.card_to_matrix() == y_pred:
            print('loss ', c, round_loss[c])
            return round_loss[c]
    return 10


# Global variables for game statistics
cards_record = {}  # Track which cards win
round_loss = {}  # Loss values from current round

# Initialize players
logging.info('Creating players')
players = []
for i in range(4):
    players.append(TarneebPlayer("p" + str(i)))
logging.info('Creating 4 players ' + str(players))

# Training configuration
NUMBER_OF_TRAINING_GAMES = 2
playModel = buildPlayingModel()

logging.info('training for ' + str(NUMBER_OF_TRAINING_GAMES) + ' games')

# Main training loop
for z in range(NUMBER_OF_TRAINING_GAMES):
    logging.info("Game number " + str(z))
    turns = []
    print("Game ", "-" * 50, z)
    
    # Reset scores for new game
    for p in players:
        p.score = 0
    
    rounds = 0
    winner = ""
    gameOver = False
    
    # Play rounds until someone reaches 41 points
    while not gameOver:
        rounds += 1
        logging.info("Game " + str(z) + " round " + str(rounds))
        
        # Ensure valid bidding (sum >= 11)
        bidding_sum = 0
        Xbid = np.array([])
        Ybid = np.array([])
        
        while bidding_sum < 11:
            standardeck = StandarDeck(shuffled=True)
            tarneeb = standardeck.cards[51].type
            clearHands(players)
            bidding_sum = distripute_and_bid(players, tarneeb)
        
        logging.info('The tarneeb is: ' + str(tarneeb) + 
                    ' sum of bidding: ' + str(bidding_sum))
        
        # Play the round
        turns = playRound(players, tarneeb)
        print(players)
        
        # Process round results and train models
        for p in players:
            p.print_player_round_results()
            Xbid = np.append(Xbid, p.bidding_input)
            Ybid = np.append(Ybid, p.predictionY())
            p.roundOver()
            
            # Check for winner (TODO: Handle ties if multiple players reach 41)
            if p.score >= 41:
                gameOver = True
                winner = p.name
                p.gamesWon += 1
                logging.info('game ' + str(z) + ' ended with ' + str(rounds) + 
                           ' rounds and player ' + str(winner) + ' is the winner')

        # Train bidding models with results from this round
        logging.info('Round ' + str(rounds) + ' ended. Training bidding models ...')
        Xbid = Xbid.reshape(4, 64)
        for pl in players:
            pl.trainBidding(Xbid, Ybid)

        # Prepare training data for playing model
        Xplay = np.zeros((52, 52, 68))
        Yplay = np.zeros((52, 4))
        X = np.zeros((103, 68))
        
        # Collect turn data
        for i, turn in enumerate(turns):
            X[i * 4 + 51:(i + 1) * 4 + 51] = turn.turn_to_matrices(players)
            for j, c in enumerate(turn.played_cards):
                Yplay[4 * i + j] = c.card_to_matrix()
        
        # Reshape for LSTM input
        for i in range(52):
            Xplay[i] = X[i:i + 52]
        
        # Train playing model
        playModel.fit(Xplay, Yplay, verbose=2)
        
        # Exit after first round for testing (remove in production)
        exit()

# Display final results
print()
for p in players:
    print(p.name, colored(p.gamesWon, "blue"))

print("Card statistics:", cards_record)


