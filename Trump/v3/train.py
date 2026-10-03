"""
Trains the v3 Trump networks: v1's networks and bid reward with the v3 trick
reward (see Trump.v3.game) and the +1 bid exploration (see Trump.v3.player).
Fully separate from Trump.train: weights and epsilon state live in a v3 models
folder (Trump/v3/models/ unless --models-dir says otherwise), never in
Trump/models/.

Training resumes from whatever weights the models folder already holds, so use
a new --models-dir for a fresh run. With --init-from-v1 the v1 weights (and
epsilon state) are copied into an empty models folder first; v1's own files
are only read.

Usage:
    python -m Trump.v3.train --games 5000
    python -m Trump.v3.train --games 50000 --models-dir Trump/v3/models_fresh --optimistic-bid-prob 0.08
    python -m Trump.v3.train --games 5000 --init-from-v1
"""

import argparse
import os
import random
import shutil

from tqdm import tqdm

import Trump.train as v1_train
from Trump.v3.evaluate import evaluate_nets, format_report, load_v1_nets
from Trump.v3.game import TrumpGameV3
from Trump.v3.player import TrumpPlayerV3

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
BID_MODEL_PATH = os.path.join(MODELS_DIR, 'bidding_net.pt')
PLAYING_MODEL_PATH = os.path.join(MODELS_DIR, 'playing_net.pt')
EPSILON_STATE_PATH = os.path.join(MODELS_DIR, 'epsilon_state.json')

RANDOM_START_RANGE = (-10, 38)  # scores drawn for games that start from a random score line


def model_paths(models_dir):
    """The (bidding weights, playing weights, epsilon state) file paths inside a models folder."""
    return (os.path.join(models_dir, 'bidding_net.pt'),
            os.path.join(models_dir, 'playing_net.pt'),
            os.path.join(models_dir, 'epsilon_state.json'))


def init_from_v1(models_dir=MODELS_DIR):
    """
    Copy the v1 weights and epsilon state into a v3 models folder, unless it
    already holds weights (so an ongoing training is never overwritten).
    v1's own files are only read.
    """
    os.makedirs(models_dir, exist_ok=True)
    bid_path, playing_path, epsilon_path = model_paths(models_dir)
    if os.path.exists(bid_path) or os.path.exists(playing_path):
        tqdm.write('Not copying v1 weights: %s already has weights.' % models_dir)
        return
    for source, destination in ((v1_train.BID_MODEL_PATH, bid_path),
                                (v1_train.PLAYING_MODEL_PATH, playing_path),
                                (v1_train.EPSILON_STATE_PATH, epsilon_path)):
        if os.path.exists(source):
            shutil.copy2(source, destination)
    tqdm.write('Started from the v1 weights (copied into %s).' % models_dir)


def checkpoint(game, window_results, games_done, eval_games=100, v1_nets=None, models_dir=MODELS_DIR):
    """
    Save the models and epsilon state, print the stats of the games played
    since the previous checkpoint, then measure the networks greedily.

    The training stats mix in exploration noise (random bids and cards, +1
    bids, random starting scores), which can hide progress. The greedy
    evaluation plays the networks as they are now against v1, without
    exploration or learning, so its numbers are the ones to follow.

    Args:
        game (TrumpGameV3): The game holding the networks and epsilons
        window_results (list): The result dicts of the games in this window
        games_done (int): Games played so far in this run, for the header
        eval_games (int): Greedy games against v1 (0 skips the evaluation)
        v1_nets (tuple): v1's (bidding net, playing net), loaded by run()
        models_dir (str): Folder the weights and epsilon state are saved to
    """
    bid_path, playing_path, epsilon_path = model_paths(models_dir)
    game.save_models(bid_path, playing_path)
    game.save_epsilon_state(epsilon_path)

    n = len(window_results)
    avg_rounds = sum(r['rounds'] for r in window_results) / n
    avg_score = sum(sum(r['scores']) for r in window_results) / (n * game.NUM_PLAYERS)
    # A game has a winner when a player reaches WINNING_SCORE with a non-negative
    # partner (partners sit opposite); games cut off by LOSING_SCORE or by
    # max_rounds don't count.
    won = sum(
        any(s >= game.WINNING_SCORE and r['scores'][(i + 2) % 4] >= 0 for i, s in enumerate(r['scores']))
        for r in window_results)

    tqdm.write('--- Checkpoint after %d games (last %d) ---' % (games_done, n))
    tqdm.write('Average rounds per game: %.2f' % avg_rounds)
    tqdm.write('Average score per player per game: %.2f' % avg_score)
    tqdm.write('Games ending with a winner (%d+ points): %.1f%% (%d/%d)'
               % (game.WINNING_SCORE, 100 * won / n, won, n))
    tqdm.write('Epsilon now: bid=%.4f, card=%.4f' % (game.bid_epsilon, game.card_epsilon))
    tqdm.write('Models saved to: %s' % models_dir)

    if eval_games > 0 and v1_nets is not None:
        result = evaluate_nets((game.shared_bid_net, game.shared_playing_net), v1_nets, eval_games)
        tqdm.write('Greedy evaluation vs v1 (%d games, no exploration):' % eval_games)
        for line in format_report(result):
            tqdm.write('  ' + line)


def run(num_games=1, max_rounds_per_game=200, bid_epsilon=0.2, bid_epsilon_min=0.02, bid_epsilon_decay=0.9995,
        card_epsilon=0.2, card_epsilon_min=0.02, card_epsilon_decay=0.9995, bid_batch=16, card_batch=1,
        random_start_prob=0.3, report_every=500, eval_games=100, start_from_v1=False,
        models_dir=MODELS_DIR, optimistic_bid_prob=0.0):
    """
    Play `num_games` full games with both networks learning.

    Same loop as Trump.train.run, with differences: a fraction
    `random_start_prob` of games start from a random score line (because the
    "someone is close to 41" situations the v3 reward cares about are rare
    when every game starts at 0), greedy bids are raised by one with
    probability `optimistic_bid_prob` (see Trump.v3.player), and the models
    are saved and the stats printed every `report_every` games (see
    checkpoint), with a final checkpoint for any games left over. Each
    checkpoint also plays `eval_games` greedy games against the saved v1
    networks (0 disables it).

    Args:
        start_from_v1 (bool): Copy the v1 weights into the models folder first
            (only if it holds no weights yet, see init_from_v1)
        models_dir (str): Folder the weights and epsilon state are loaded
            from (if present) and saved to. Use a new folder for a fresh run.
        optimistic_bid_prob (float): Chance that a greedy bid is raised by one

    Returns:
        (TrumpGameV3, list, list): The game, its 4 players, and one result dict
            per game: {'rounds': int, 'scores': list, 'ended': bool}
    """
    os.makedirs(models_dir, exist_ok=True)
    bid_path, playing_path, epsilon_path = model_paths(models_dir)
    if start_from_v1:
        init_from_v1(models_dir)

    game = TrumpGameV3(optimistic_bid_prob=optimistic_bid_prob,
                       bid_learning=True, bid_epsilon=bid_epsilon, bid_epsilon_min=bid_epsilon_min,
                       bid_epsilon_decay=bid_epsilon_decay, bid_batch=bid_batch,
                       card_learning=True, card_epsilon=card_epsilon, card_epsilon_min=card_epsilon_min,
                       card_epsilon_decay=card_epsilon_decay, card_batch=card_batch,
                       bid_counterfactual=True)
    resumed = os.path.exists(bid_path) or os.path.exists(playing_path)
    game.load_models(bid_path, playing_path)
    game.load_epsilon_state(epsilon_path)
    tqdm.write('%s weights in %s' % ('Resuming from the existing' if resumed else 'Fresh networks, no', models_dir))
    players = [TrumpPlayerV3(game) for _ in range(4)]
    v1_nets = load_v1_nets() if eval_games > 0 else None

    results = []
    games_bar = tqdm(range(num_games), desc='Games', unit='game')
    for game_index in games_bar:
        randomize = random.random() < random_start_prob
        for player in players:
            player.score = random.randint(*RANDOM_START_RANGE) if randomize else 0

        distributer = 0
        rounds_before = game.completed_rounds
        rounds_bar = tqdm(range(max_rounds_per_game), desc='Rounds', unit='round', leave=False)
        for _ in rounds_bar:
            if game.is_over():  # a random start can already be over
                break
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

        games_done = game_index + 1
        if games_done % report_every == 0 or games_done == num_games:
            window_start = (games_done - 1) // report_every * report_every
            checkpoint(game, results[window_start:], games_done, eval_games, v1_nets, models_dir)
    return game, players, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the v3 Trump networks (separate from v1).')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--max-rounds', type=int, default=200)
    parser.add_argument('--bid-epsilon', type=float, default=0.2)
    parser.add_argument('--bid-epsilon-min', type=float, default=0.02)
    parser.add_argument('--bid-epsilon-decay', type=float, default=0.9995)
    parser.add_argument('--card-epsilon', type=float, default=0.2)
    parser.add_argument('--card-epsilon-min', type=float, default=0.02)
    parser.add_argument('--card-epsilon-decay', type=float, default=0.9995)
    parser.add_argument('--bid-batch', type=int, default=16)
    parser.add_argument('--card-batch', type=int, default=1)
    parser.add_argument('--random-start-prob', type=float, default=0.3,
                        help='Fraction of games that start from a random score line')
    parser.add_argument('--report-every', type=int, default=500,
                        help='Save the models and print the stats every this many games')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Greedy games against v1 played at every checkpoint (0 disables)')
    parser.add_argument('--init-from-v1', action='store_true',
                        help='Start from copies of the v1 weights (only if the models folder has no weights yet)')
    parser.add_argument('--models-dir', default=MODELS_DIR,
                        help='Folder holding this run\'s weights and epsilon state (default: Trump/v3/models). '
                             'Training resumes from the weights found there, so use a new folder for a fresh run.')
    parser.add_argument('--optimistic-bid-prob', type=float, default=0.0,
                        help='Chance, while learning, that a greedy bid is raised by one (0 disables)')
    args = parser.parse_args()

    run(num_games=args.games, max_rounds_per_game=args.max_rounds,
        bid_epsilon=args.bid_epsilon, bid_epsilon_min=args.bid_epsilon_min,
        bid_epsilon_decay=args.bid_epsilon_decay, card_epsilon=args.card_epsilon,
        card_epsilon_min=args.card_epsilon_min, card_epsilon_decay=args.card_epsilon_decay,
        bid_batch=args.bid_batch, card_batch=args.card_batch, random_start_prob=args.random_start_prob,
        report_every=args.report_every, eval_games=args.eval_games, start_from_v1=args.init_from_v1,
        models_dir=args.models_dir, optimistic_bid_prob=args.optimistic_bid_prob)
