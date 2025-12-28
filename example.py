"""
Example usage of the neural network Tarneeb game.

This script demonstrates how to:
1. Create players
2. Set up a simple game
3. Distribute cards
4. Play a basic game loop

For training neural networks, see Tarneeb/GTarneeb.py
"""

from Cards.StandarDeck import StandarDeck
from Player import Player


def simple_game_example():
    """
    Run a simple game with 4 players using random card selection.
    
    This demonstrates the basic game mechanics without neural networks.
    """
    print("=" * 60)
    print("Simple Tarneeb Game Example")
    print("=" * 60)
    
    # Create a shuffled deck
    deck = StandarDeck(shuffled=True)
    
    # Create 4 players
    players = []
    for i in range(4):
        players.append(Player(f"Player_{i}"))
    
    # Distribute 13 cards to each player
    print("\nDistributing cards...")
    for i in range(4):
        players[i].setHand(deck.distripute(13))
        print(f"{players[i].name}: {len(players[i].hand)} cards")
    
    # Play game rounds
    winner_id = 0
    round_num = 0
    
    print(f"\nPlaying {len(players[0].hand)} rounds...")
    
    while len(players[0].hand) > 0:
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        
        # Each player plays a card
        played_cards = []
        for i in range(4):
            current_player = players[(i + winner_id) % 4]
            card = current_player.playCard(played_cards)
            played_cards.append(card)
            print(f"{current_player.name} plays: {card}")
        
        # Determine winner of this round
        winner_id = (deck.winner(played_cards) + winner_id) % 4
        print(f"Winner: {players[winner_id].name}")
    
    print("\n" + "=" * 60)
    print("Game Complete!")
    print("=" * 60)


def neural_network_game_info():
    """
    Display information about the neural network training.
    """
    print("\n" + "=" * 60)
    print("Neural Network Training")
    print("=" * 60)
    print("\nFor neural network training, run:")
    print("  python Tarneeb/GTarneeb.py")
    print("\nThis will:")
    print("  - Create AI players with neural networks")
    print("  - Train bidding models")
    print("  - Train playing models")
    print("  - Track game statistics")
    print("\nNote: Training requires tensorflow/keras to be installed.")
    print("Install dependencies with: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    try:
        simple_game_example()
        neural_network_game_info()
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nMake sure you have installed dependencies:")
        print("  pip install -r requirements.txt")
