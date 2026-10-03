"""
Prints and checks the v3 trick reward on hand-built situations, then plays a
few rounds of TrumpGameV3 to make sure the reward plugs into the game.

Nothing is trained or saved.

Usage:
    python -m Trump.v3.test_rewards
"""

from Trump.TrumpPlayer import TrumpPlayer
from Trump.v3.game import TrumpGameV3, trick_outcome

# Seats: 0 = me, 1 = opponent p2, 2 = my partner p3, 3 = opponent p4.
ME, P2, P3, P4 = 0, 1, 2, 3


def check(label, got, expected):
    ok = abs(got - expected) < 1e-9
    print('%s %-62s %+.3f  (expected %+.3f)' % ('ok  ' if ok else 'FAIL', label, got, expected))
    return ok


def run():
    results = []

    # Everyone still needs tricks. Scores: me 36 (bid 5), p2 38 (bid 3), partner 21 (bid 3), p4 24 (bid 3).
    scores, bids, tricks = [36, 38, 21, 24], [5, 3, 3, 3], [0, 0, 0, 0]
    print('Everyone needs tricks, scores', scores, 'bids', bids)
    results += [
        check('I win (36+5=41, partner >= 0)', trick_outcome(ME, ME, scores, bids, tricks), 3.0),
        check('my partner wins (21+3=24)', trick_outcome(ME, P3, scores, bids, tricks), 1 + 24 / 41),
        check('p2 wins (38+3=41, his partner >= 0)', trick_outcome(ME, P2, scores, bids, tricks), -3.0),
        check('p4 wins (24+3=27)', trick_outcome(ME, P4, scores, bids, tricks), -1 - 27 / 41),
    ]

    # Same, but my partner is negative: my win can no longer end the game.
    scores = [36, 38, -11, 24]
    print('\nPartner at -11, scores', scores)
    results += [
        check('I win (EG blocked by negative partner)', trick_outcome(ME, ME, scores, bids, tricks), 2.0),
        check('my partner wins (negative score -> 0.5)', trick_outcome(ME, P3, scores, bids, tricks), 1.5),
        check('p2 wins (his partner p4 is >= 0)', trick_outcome(ME, P2, scores, bids, tricks), -3.0),
    ]

    # My team made both bids; p2 still needs tricks, p4 made his.
    scores, bids, tricks = [36, 38, 21, 24], [5, 3, 3, 3], [5, 0, 3, 3]
    print('\nMy team made both bids, p2 still needs, p4 made his')
    results += [
        check('I win: 0.3 + denial of p2 (1 + EG 1)', trick_outcome(ME, ME, scores, bids, tricks), 2.3),
        check('p2 wins: -0.3 - his bonus', trick_outcome(ME, P2, scores, bids, tricks), -0.3 - 2.0),
        check('p4 wins (he already made his bid): -0.3 only', trick_outcome(ME, P4, scores, bids, tricks), -0.3),
    ]

    # Surplus trick: I made my bid, my partner still needs tricks.
    tricks = [5, 0, 0, 3]
    print('\nSurplus trick: I made my bid, partner still needs')
    results += [
        check('I win: 0.3 + denial of p2 (p4 made his)', trick_outcome(ME, ME, scores, bids, tricks), 2.3),
        check('my partner wins the trick he needs', trick_outcome(ME, P3, scores, bids, tricks), 1 + 24 / 41),
    ]
    quiet = [10, 10, 10, 10]
    results += [
        check('I win a surplus trick, nobody near 41 (10+3)', trick_outcome(ME, ME, quiet, bids, [5, 0, 0, 0]), 0.3 + 13 / 41),
    ]

    # Losing a trick to an opponent who already made his bid.
    print('\nTrick lost to an opponent who does not need it')
    results += [
        check('team still needs, winner made his bid: -1.0 only', trick_outcome(ME, P4, quiet, bids, [0, 0, 0, 3]), -1.0),
    ]

    # Term capped at 1.
    print('\nTerm cap')
    big = [38, 0, 0, 0]
    results += [
        check('38 + 14 points (bid 7) -> term capped at 1, plus EG 1', trick_outcome(ME, ME, big, [7, 2, 2, 2], [0, 0, 0, 0]), 3.0),
    ]

    print('\n%d/%d checks passed' % (sum(results), len(results)))
    return all(results)


def smoke_test():
    """Play a few learning rounds with untrained v1-shaped networks, without saving anything."""
    game = TrumpGameV3(bid_learning=True, card_learning=True, card_epsilon=0.2, bid_epsilon=0.2,
                       bid_counterfactual=True, bid_batch=10 ** 9, card_batch=10 ** 9)
    players = [TrumpPlayer(game) for _ in range(4)]
    for _ in range(5):
        for p in players:
            p.score = 0
        game.start_round(0)
    samples = [len(p.card_history) for p in players]
    rewards = [r for p in players for _, _, r in p.card_history]
    print('\nSmoke test: %d rounds played, card samples per player %s, reward range [%.2f, %.2f]'
          % (game.completed_rounds, samples, min(rewards), max(rewards)))
    return all(n == 5 * 13 for n in samples)


if __name__ == '__main__':
    rewards_ok = run()
    smoke_ok = smoke_test()
    print('\nAll good.' if rewards_ok and smoke_ok else '\nSomething failed.')
