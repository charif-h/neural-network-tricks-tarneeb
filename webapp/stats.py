"""
Cross-game statistics for the web table, persisted to webapp/stats.json.

Seats: 0 = You, 1 = East, 2 = Partner, 3 = West. Teams: {0, 2} vs {1, 3}.
"""

import json
import os
import threading

STATS_PATH = os.path.join(os.path.dirname(__file__), 'stats.json')


def _empty():
    return {
        'games': 0, 'games_won': 0,
        'rounds': 0,
        'team_points': [0, 0],  # [you + partner, opponents]
        'seats': [{'bid_sum': 0, 'tricks_sum': 0, 'made': 0, 'over': 0, 'under': 0,
                   'points': 0, 'high_bids': 0} for _ in range(4)],
    }


class Stats:
    def __init__(self, path=STATS_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.data = _empty()
        try:
            with open(path) as f:
                loaded = json.load(f)
            if len(loaded.get('seats', [])) == 4:
                self.data = loaded
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.data, f)
        except OSError:
            pass

    def record_round(self, bids, tricks, deltas):
        with self.lock:
            d = self.data
            d['rounds'] += 1
            d['team_points'][0] += deltas[0] + deltas[2]
            d['team_points'][1] += deltas[1] + deltas[3]
            for i in range(4):
                s = d['seats'][i]
                s['bid_sum'] += bids[i]
                s['tricks_sum'] += tricks[i]
                s['points'] += deltas[i]
                s['high_bids'] += 1 if bids[i] >= 7 else 0
                if tricks[i] >= bids[i]:
                    s['made'] += 1
                if tricks[i] > bids[i]:
                    s['under'] += 1   # won more tricks than bid (underbid)
                elif tricks[i] < bids[i]:
                    s['over'] += 1    # failed the bid (overbid)
            self._save()

    def record_game(self, we_won):
        with self.lock:
            self.data['games'] += 1
            self.data['games_won'] += 1 if we_won else 0
            self._save()

    def reset(self):
        with self.lock:
            self.data = _empty()
            self._save()

    def report(self):
        with self.lock:
            d = self.data
            n = d['rounds']
            seats = []
            for s in d['seats']:
                seats.append({
                    'avg_bid': s['bid_sum'] / n if n else None,
                    'avg_tricks': s['tricks_sum'] / n if n else None,
                    'made_pct': 100 * s['made'] / n if n else None,
                    'under_pct': 100 * s['under'] / n if n else None,
                    'over_pct': 100 * s['over'] / n if n else None,
                    'avg_points': s['points'] / n if n else None,
                })
            return {'games': d['games'], 'games_won': d['games_won'], 'rounds': n,
                    'team_points': d['team_points'], 'seats': seats}
