"""
Plays the v3 networks against the v1 networks, greedy and without learning.

Both versions use the plain v1 player and network shapes, so they only differ
by their weights. Each game puts one version on seats 0/2 and the other on
seats 1/3; the versions swap sides every other game so seat and dealing
advantages cancel out.

Used on its own to compare the saved weights, and by Trump.v3.train at every
checkpoint to measure the networks being trained (see evaluate_nets).

Usage:
    python -m Trump.v3.evaluate --games 200
"""

import argparse

from tqdm import tqdm

from Trump.BiddingNet import BiddingNet
from Trump.PlayingNet import PlayingNet
from Trump.TrumpGame import TrumpGame
from Trump.TrumpPlayer import TrumpPlayer


def load_nets(bid_path, play_path):
    """Build a (bidding net, playing net) pair and load both from disk."""
    bid_net, play_net = BiddingNet(), PlayingNet()
    bid_net.loadModel(bid_path)
    play_net.loadModel(play_path)
    return bid_net, play_net


def load_v1_nets():
    """The saved v1 networks (Trump/models/)."""
    import Trump.train as v1_train
    return load_nets(v1_train.BID_MODEL_PATH, v1_train.PLAYING_MODEL_PATH)


def load_v3_nets(models_dir=None):
    """The saved v3 networks, from `models_dir` (default: Trump/v3/models/)."""
    from Trump.v3 import train as v3_train
    bid_path, playing_path, _ = v3_train.model_paths(models_dir or v3_train.MODELS_DIR)
    return load_nets(bid_path, playing_path)


def play_one_game(v3_nets, v1_nets, v3_on_even, max_rounds, made):
    """
    Play one greedy game between the two versions.

    Args:
        v3_nets (tuple): (bidding net, playing net) of v3
        v1_nets (tuple): (bidding net, playing net) of v1
        v3_on_even (bool): True to seat v3 on seats 0/2 and v1 on 1/3, False to swap
        max_rounds (int): Safety cap on rounds
        made (dict): {'v1': [made, total], 'v3': [made, total]} bid counters,
            updated in place with every player's bid result each round

    Returns:
        str or None: 'v1' or 'v3' for the winning version, None if nobody won
    """
    game = TrumpGame()  # learning flags default to False: greedy play, no updates
    kinds = []
    for seat in range(4):
        is_v3 = (seat % 2 == 0) == v3_on_even
        player = TrumpPlayer(game)
        player.bid_net, player.playing_net = v3_nets if is_v3 else v1_nets
        kinds.append('v3' if is_v3 else 'v1')

    distributer = 0
    rounds = 0
    while rounds < max_rounds and not game.is_over():
        game.start_round(distributer)
        distributer = (game.distributer + 1) % game.NUM_PLAYERS
        rounds += 1
        for seat, player in enumerate(game.players):
            made[kinds[seat]][0] += player.tricks_won >= game.bids[seat]
            made[kinds[seat]][1] += 1

    # A team wins by reaching the winning score, or when the other team hits the
    # losing cutoff (the training-only ending, counted as a loss for that team).
    for seat in range(2):
        result = game._team_result(seat, (seat + 2) % 4)
        if result != 0:
            return kinds[seat if result > 0 else seat + 1]
    return None


def evaluate_nets(v3_nets, v1_nets, num_games, max_rounds=200, progress=True):
    """
    Play `num_games` greedy games between the two versions.

    Returns:
        dict: {'v3_wins', 'v1_wins', 'no_winner', 'v3_made', 'v1_made'} where
            the *_made values are the share (0-1) of bids that were made
    """
    wins = {'v3': 0, 'v1': 0, None: 0}
    made = {'v3': [0, 0], 'v1': [0, 0]}
    games = range(num_games)
    for g in (tqdm(games, desc='Evaluation', unit='game', leave=False) if progress else games):
        wins[play_one_game(v3_nets, v1_nets, g % 2 == 0, max_rounds, made)] += 1
    return {
        'v3_wins': wins['v3'], 'v1_wins': wins['v1'], 'no_winner': wins[None],
        'v3_made': made['v3'][0] / max(1, made['v3'][1]),
        'v1_made': made['v1'][0] / max(1, made['v1'][1]),
    }


def format_report(result):
    """The evaluation result as the lines printed by this script and by the training checkpoints."""
    decided = result['v3_wins'] + result['v1_wins']
    lines = ['v3 wins: %d | v1 wins: %d | no winner: %d'
             % (result['v3_wins'], result['v1_wins'], result['no_winner'])]
    if decided:
        lines.append('v3 win rate among decided games: %.1f%%' % (100 * result['v3_wins'] / decided))
    lines.append('Bids made: v3 %.1f%% | v1 %.1f%%' % (100 * result['v3_made'], 100 * result['v1_made']))
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play v3 against v1.')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--max-rounds', type=int, default=200)
    parser.add_argument('--models-dir', default=None,
                        help='Folder holding the v3 weights to evaluate (default: Trump/v3/models)')
    args = parser.parse_args()

    for line in format_report(evaluate_nets(load_v3_nets(args.models_dir), load_v1_nets(),
                                            args.games, args.max_rounds)):
        print(line)
