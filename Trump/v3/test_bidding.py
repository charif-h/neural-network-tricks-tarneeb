"""
Checks the +1 bid exploration of TrumpPlayerV3 on real rounds: about the
requested share of greedy bids is raised by exactly one, never above the
maximum bid, never when the probability is 0, and never outside learning.

Nothing is trained or saved.

Usage:
    python -m Trump.v3.test_bidding
"""

from Trump.v3.game import TrumpGameV3
from Trump.v3.player import TrumpPlayerV3


def raised_share(optimistic_bid_prob, bid_learning, rounds=300):
    """Play `rounds` rounds greedily and return (share of bids raised by one, any bid out of range)."""
    game = TrumpGameV3(optimistic_bid_prob=optimistic_bid_prob, bid_learning=bid_learning,
                       bid_epsilon=0.0, bid_epsilon_min=0.0, bid_batch=10 ** 9, bid_counterfactual=True)
    players = [TrumpPlayerV3(game) for _ in range(4)]
    raised = total = 0
    out_of_range = False
    for _ in range(rounds):
        for p in players:
            p.score = 0
        game.start_round(0)
        for seat, p in enumerate(players):
            bid = game.bids[seat]
            raised += bid == p.last_predicted_bid + 1
            out_of_range |= not (p.min_bid() <= bid <= p.MAX_BID) or bid not in (p.last_predicted_bid, p.last_predicted_bid + 1)
            total += 1
    return raised / total, out_of_range


def run():
    ok = True
    for prob, learning, low, high in ((0.08, True, 0.04, 0.13), (0.0, True, 0.0, 0.0), (0.5, False, 0.0, 0.0)):
        share, bad = raised_share(prob, learning)
        good = low <= share <= high and not bad
        ok &= good
        print('%s prob=%.2f bid_learning=%-5s -> %.1f%% of bids raised by one (expected %.0f%%-%.0f%%)%s'
              % ('ok  ' if good else 'FAIL', prob, learning, 100 * share, 100 * low, 100 * high,
                 ' [bid out of range]' if bad else ''))
    return ok


if __name__ == '__main__':
    print('\nAll good.' if run() else '\nSomething failed.')
