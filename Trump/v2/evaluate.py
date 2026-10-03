"""
Plays the v1 networks against the v2 networks, greedy and without learning.

Each game puts one version on seats 0/2 and the other on seats 1/3; the
versions swap sides every other game so seat and dealing advantages cancel out.

Used on its own to compare the saved weights, and by Trump.v2.train at every
checkpoint to measure the networks being trained (see evaluate_nets).

Usage:
    python -m Trump.v2.evaluate --games 200
"""

import argparse

from tqdm import tqdm

from Trump.BiddingNet import BiddingNet
from Trump.PlayingNet import PlayingNet
from Trump.TrumpPlayer import TrumpPlayer
from Trump.v2.game import TrumpGameV2
from Trump.v2.player import N_BID_INPUTS_V2, N_PLAY_INPUTS_V2, TrumpPlayerV2


def load_nets(bid_kwargs, bid_path, play_kwargs, play_path):
    """Build a (bidding net, playing net) pair and load both from disk."""
    bid_net, play_net = BiddingNet(**bid_kwargs), PlayingNet(**play_kwargs)
    bid_net.loadModel(bid_path)
    play_net.loadModel(play_path)
    return bid_net, play_net


def load_v1_nets():
    """The saved v1 networks (Trump/models/)."""
    import Trump.train as v1_train
    return load_nets({}, v1_train.BID_MODEL_PATH, {}, v1_train.PLAYING_MODEL_PATH)


def load_v2_nets():
    """The saved v2 networks (Trump/v2/models/)."""
    from Trump.v2 import train as v2_train
    return load_nets({'n_inputs': N_BID_INPUTS_V2}, v2_train.BID_MODEL_PATH,
                     {'n_inputs': N_PLAY_INPUTS_V2}, v2_train.PLAYING_MODEL_PATH)


def play_one_game(v1_nets, v2_nets, v1_on_even, max_rounds, made):
    """
    Play one greedy game between the two versions.

    Args:
        v1_nets (tuple): (bidding net, playing net) of v1
        v2_nets (tuple): (bidding net, playing net) of v2
        v1_on_even (bool): True to seat v1 on seats 0/2 and v2 on 1/3, False to swap
        max_rounds (int): Safety cap on rounds
        made (dict): {'v1': [made, total], 'v2': [made, total]} bid counters,
            updated in place with every player's bid result each round

    Returns:
        str or None: 'v1' or 'v2' for the winning version, None if nobody won
    """
    game = TrumpGameV2()  # learning flags default to False: greedy play, no updates
    kinds = []
    for seat in range(4):
        is_v1 = (seat % 2 == 0) == v1_on_even
        if is_v1:
            player = TrumpPlayer(game)
            player.bid_net, player.playing_net = v1_nets
        else:
            player = TrumpPlayerV2(game)
            player.bid_net, player.playing_net = v2_nets
        kinds.append('v1' if is_v1 else 'v2')

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


def evaluate_nets(v1_nets, v2_nets, num_games, max_rounds=200, progress=True):
    """
    Play `num_games` greedy games between the two versions.

    Returns:
        dict: {'v1_wins', 'v2_wins', 'no_winner', 'v1_made', 'v2_made'} where
            the *_made values are the share (0-1) of bids that were made
    """
    wins = {'v1': 0, 'v2': 0, None: 0}
    made = {'v1': [0, 0], 'v2': [0, 0]}
    games = range(num_games)
    for g in (tqdm(games, desc='Evaluation', unit='game', leave=False) if progress else games):
        wins[play_one_game(v1_nets, v2_nets, g % 2 == 0, max_rounds, made)] += 1
    return {
        'v1_wins': wins['v1'], 'v2_wins': wins['v2'], 'no_winner': wins[None],
        'v1_made': made['v1'][0] / max(1, made['v1'][1]),
        'v2_made': made['v2'][0] / max(1, made['v2'][1]),
    }


def format_report(result):
    """The evaluation result as the lines printed by this script and by the training checkpoints."""
    decided = result['v1_wins'] + result['v2_wins']
    lines = ['v1 wins: %d | v2 wins: %d | no winner: %d'
             % (result['v1_wins'], result['v2_wins'], result['no_winner'])]
    if decided:
        lines.append('v2 win rate among decided games: %.1f%%' % (100 * result['v2_wins'] / decided))
    lines.append('Bids made: v2 %.1f%% | v1 %.1f%%' % (100 * result['v2_made'], 100 * result['v1_made']))
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play v1 against v2.')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--max-rounds', type=int, default=200)
    args = parser.parse_args()

    for line in format_report(evaluate_nets(load_v1_nets(), load_v2_nets(), args.games, args.max_rounds)):
        print(line)
