"""
Runs Trump games with the bidding and playing networks in learning mode.

Previously saved weights are loaded once at startup (if present) and saved
once at the end, so training accumulates across separate runs of this script
instead of restarting from scratch every time.

Usage:
    python -m Trump.train                                # 1 game
    python -m Trump.train --games 10                      # 10 games back to back
    python -m Trump.train --games 10 --max-rounds 100      # with a smaller safety cap
"""

import argparse
import os

from tqdm import tqdm

from Trump.TrumpGame import TrumpGame
from Trump.TrumpPlayer import TrumpPlayer

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
BID_MODEL_PATH = os.path.join(MODELS_DIR, 'bidding_net.pt')
PLAYING_MODEL_PATH = os.path.join(MODELS_DIR, 'playing_net.pt')
EPSILON_STATE_PATH = os.path.join(MODELS_DIR, 'epsilon_state.json')


def run(num_games=1, max_rounds_per_game=200, bid_epsilon=0.2, bid_epsilon_min=0.02, bid_epsilon_decay=0.9995,
        card_epsilon=0.2, card_epsilon_min=0.02, card_epsilon_decay=0.9995, bid_batch=16, card_batch=1,
        bid_counterfactual=True):
    """
    Play `num_games` full Trump games back to back.

    The shared bidding and playing networks are loaded once before the first
    game and saved once after the last, so their weights keep training across
    every game played here (and across separate runs of this function or
    script) instead of resetting for each new game. The current (decayed)
    bid_epsilon/card_epsilon are persisted the same way (see EPSILON_STATE_PATH),
    so bid_epsilon/card_epsilon/*_decay below only apply as starting values the
    very first time this runs - once epsilon_state.json exists, decay resumes
    from where the last run left off instead of restarting.

    Args:
        num_games (int): Number of games to play
        max_rounds_per_game (int): Safety cap on rounds per game - with the
            networks still training, near-random play can take a long time to
            reach a winning or losing score (see TrumpGame.LOSING_SCORE)
        bid_epsilon (float): Starting exploration rate for the bidding network
        bid_epsilon_min (float): Floor bid_epsilon decays down to
        bid_epsilon_decay (float): Multiplicative decay applied to bid_epsilon
            each completed round (1.0 disables decay)
        card_epsilon (float): Starting exploration rate for the playing network
        card_epsilon_min (float): Floor card_epsilon decays down to
        card_epsilon_decay (float): Multiplicative decay applied to card_epsilon
            each completed round (1.0 disables decay)
        bid_batch (int): Rounds between bidding network updates - each round
            gives 4 samples (one per player), so 16 rounds = 64 samples per step
        card_batch (int): Rounds between playing network updates - each round
            gives up to 52 samples (13 tricks x 4 players)
        bid_counterfactual (bool): Train all bid outputs each round towards the
            reward every bid would have earned (see TrumpGame.counterfactual_bid_rewards)
            instead of only the bid actually taken

    Returns:
        (TrumpGame, list, list): The game, its 4 players, and one result dict
            per game played: {'rounds': int, 'scores': list, 'ended': bool}
            ('ended' is False if max_rounds_per_game was hit instead)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    game = TrumpGame(bid_learning=True, bid_epsilon=bid_epsilon, bid_epsilon_min=bid_epsilon_min,
                      bid_epsilon_decay=bid_epsilon_decay, bid_batch=bid_batch,
                      card_learning=True, card_epsilon=card_epsilon, card_epsilon_min=card_epsilon_min,
                      card_epsilon_decay=card_epsilon_decay, card_batch=card_batch,
                      bid_counterfactual=bid_counterfactual)
    game.load_models(BID_MODEL_PATH, PLAYING_MODEL_PATH)
    game.load_epsilon_state(EPSILON_STATE_PATH)
    players = [TrumpPlayer(game) for _ in range(4)]

    results = []
    games_bar = tqdm(range(num_games), desc='Games', unit='game')
    for _ in games_bar:
        for player in players:
            player.score = 0

        distributer = 0
        rounds_before = game.completed_rounds
        rounds_bar = tqdm(range(max_rounds_per_game), desc='Rounds', unit='round', leave=False)
        for _ in rounds_bar:
            game.start_round(distributer)
            distributer = (game.distributer + 1) % game.NUM_PLAYERS
            rounds_bar.set_postfix(scores=[player.score for player in players])
            if game.is_over():
                break
        rounds_bar.close()

        results.append({
            'rounds': game.completed_rounds - rounds_before,
            'scores': [player.score for player in players],
            'ended': game.is_over(),
        })
        games_bar.set_postfix(scores=results[-1]['scores'], rounds=results[-1]['rounds'])

    game.save_models(BID_MODEL_PATH, PLAYING_MODEL_PATH)
    game.save_epsilon_state(EPSILON_STATE_PATH)
    return game, players, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Trump bidding and playing networks.')
    parser.add_argument('--games', type=int, default=1, help='Number of games to play back to back')
    parser.add_argument('--max-rounds', type=int, default=200, help='Safety cap of rounds per game')
    parser.add_argument('--bid-epsilon', type=float, default=0.2,
                         help='Starting exploration rate for bidding (ignored once epsilon_state.json exists)')
    parser.add_argument('--bid-epsilon-min', type=float, default=0.02, help='Floor bid_epsilon decays down to')
    parser.add_argument('--bid-epsilon-decay', type=float, default=0.9995,
                         help='Per-round multiplicative decay for bid_epsilon (1.0 disables decay)')
    parser.add_argument('--card-epsilon', type=float, default=0.2,
                         help='Starting exploration rate for card play (ignored once epsilon_state.json exists)')
    parser.add_argument('--card-epsilon-min', type=float, default=0.02, help='Floor card_epsilon decays down to')
    parser.add_argument('--card-epsilon-decay', type=float, default=0.9995,
                         help='Per-round multiplicative decay for card_epsilon (1.0 disables decay)')
    parser.add_argument('--bid-batch', type=int, default=16,
                         help='Rounds between bidding network updates (4 samples per round)')
    parser.add_argument('--card-batch', type=int, default=1,
                         help='Rounds between playing network updates (up to 52 samples per round)')
    parser.add_argument('--legacy-bid-reward', action='store_true',
                         help='Train only the bid actually taken (old behavior) instead of all bids')
    args = parser.parse_args()

    game, players, results = run(num_games=args.games, max_rounds_per_game=args.max_rounds,
                                  bid_epsilon=args.bid_epsilon, bid_epsilon_min=args.bid_epsilon_min,
                                  bid_epsilon_decay=args.bid_epsilon_decay, card_epsilon=args.card_epsilon,
                                  card_epsilon_min=args.card_epsilon_min, card_epsilon_decay=args.card_epsilon_decay,
                                  bid_batch=args.bid_batch, card_batch=args.card_batch,
                                  bid_counterfactual=not args.legacy_bid_reward)

    for i, result in enumerate(results, 1):
        print('Game %d: %d round(s), ended=%s, scores=%s'
              % (i, result['rounds'], result['ended'], result['scores']))
    print('Total rounds played (this run): %d' % game.completed_rounds)
    avg_score = sum(sum(result['scores']) for result in results) / (len(results) * len(players))
    avg_rounds = sum(result['rounds'] for result in results) / len(results)
    print('Average score per player: %.2f' % avg_score)
    print('Average rounds per game: %.2f' % avg_rounds)
    # A game has a winner when a player reaches WINNING_SCORE with a non-negative
    # partner (partners sit opposite: seats 0-2 and 1-3); games stopped by the
    # LOSING_SCORE cutoff or by max_rounds don't count.
    won_games = sum(
        any(s >= game.WINNING_SCORE and result['scores'][(i + 2) % 4] >= 0
            for i, s in enumerate(result['scores']))
        for result in results)
    print('Games with a winner (%d+ points): %.1f%% (%d/%d)'
          % (game.WINNING_SCORE, 100 * won_games / len(results), won_games, len(results)))
    print('Epsilon now: bid=%.4f, card=%.4f' % (game.bid_epsilon, game.card_epsilon))
    print('Models saved to:', MODELS_DIR)
