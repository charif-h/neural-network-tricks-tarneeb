# Neural Network Tricks Tarneeb

## Objective of the Project

Tarneeb and Tricks are very well-known card games in the Middle East. The aim of this project is to develop and train neural networks to play Tarneeb and Tricks using active and reinforcement learning.

The project is written in Python and uses principally the Keras library with TensorFlow backend.

As Tarneeb is the simpler game, we will start with it, and Tricks will be added at a later time.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/charif-h/neural-network-tricks-tarneeb.git
cd neural-network-tricks-tarneeb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running a Simple Game
```bash
python Game.py
```

### Training the Neural Network
```bash
python Tarneeb/GTarneeb.py
```

## Project Structure

```
neural-network-tricks-tarneeb/
├── Cards/              # Card and deck implementations
│   ├── Card.py         # Card, CardType, and CardValue classes
│   └── StandarDeck.py  # Deck management and utilities
├── Tarneeb/            # Tarneeb game implementation
│   ├── GTarneeb.py     # Main game loop and training
│   ├── TarneebPlayer.py # AI player with neural network
│   └── Turn.py         # Turn representation and logic
├── Player.py           # Base player class
├── GenModel.py         # Neural network model generation
├── Game.py             # Simple game demonstration
└── README.md           # This file
```

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

### How to Play

The game follows standard Tarneeb-41 rules. Players are arranged in teams of two, and each round involves:
1. Card distribution
2. Bidding phase
3. Playing phase with 13 turns
4. Scoring

### Neural Network Architecture

#### Playing Neural Network Input
Information needed to choose a card to play:
1. **Tarneeb type**: One-hot encoded vector [0, 0, 0, 1]
2. **Scores**: Scaled player scores [s₀/41, s₁/41, s₂/41, s₃/41]
3. **Bids**: Scaled bids [b₀/13, b₁/13, b₂/13, b₃/13]
4. **Wins**: Scaled wins [w₀/13, w₁/13, w₂/13, w₃/13]
5. **Current turn cards**: Up to 4 cards (matrix representation)
6. **Player hand cards**: 13 cards (matrix representation)
7. **Winner card**: win=1, loss=-1, ignore=0

## Development

### Code Evaluation
For a comprehensive code evaluation and improvement recommendations, see [CODE_EVALUATION.md](CODE_EVALUATION.md).

### Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Future Work

- [ ] Implement Tricks game variant
- [ ] Add more sophisticated neural network architectures
- [ ] Implement better training strategies
- [ ] Add unit tests
- [ ] Add game visualization
- [ ] Improve model persistence and versioning
- [ ] Add hyperparameter tuning

## License

This project is open source and available for educational purposes.

## Acknowledgments

This project was created to explore reinforcement learning and neural networks in the context of traditional card games.
