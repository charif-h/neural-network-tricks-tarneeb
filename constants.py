"""
Constants for the Tarneeb card game.

This module defines game constants to avoid magic numbers throughout the codebase.
"""

# Game scoring constants
WINNING_SCORE = 41  # Score needed to win a game
MAX_TRICKS_PER_ROUND = 13  # Maximum tricks per round (13 cards per player)
MIN_BID = 2  # Minimum bid a player can make
MAX_BID = 13  # Maximum bid (equal to max tricks)

# Card deck constants
TOTAL_CARDS = 52  # Total cards in a standard deck
CARDS_PER_PLAYER = 13  # Cards dealt to each player
NUM_PLAYERS = 4  # Number of players in Tarneeb
NUM_CARD_TYPES = 4  # Number of suits (CLUB, DIAMOND, SPADE, HEART)
NUM_CARD_VALUES = 13  # Number of card values (2-14, where 14 is Ace)

# Bidding constants
MIN_BID_SUM = 11  # Minimum sum of all bids to start a round
DOUBLE_SCORE_THRESHOLD = 30  # Score threshold below which big bids are doubled
BIG_BID_THRESHOLD = 7  # Bids >= this value can be doubled

# Card value ranges
MIN_CARD_VALUE = 2  # Lowest card value (2)
MAX_CARD_VALUE = 14  # Highest card value (Ace)

# Neural network input dimensions
BIDDING_INPUT_DIM = 64  # Input dimension for bidding model
# Composition: 52 (hand) + 4 (scores) + 4 (biddings) + 4 (tarneeb)

PLAYING_INPUT_DIM = 68  # Input dimension for playing model
# Composition: 4 (player context) + 52 (hand) + 12 (played cards)

# Training constants
DEFAULT_BATCH_SIZE = 4  # Default batch size for neural network training
