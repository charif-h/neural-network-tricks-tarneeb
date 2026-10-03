"""
Interactive Trump session: one human (seat 0) plus three AI players.

The engine (TrumpGame) is synchronous, so the game runs in a background thread
and the human seat blocks until the browser submits a bid or a card. The
browser polls `snapshot()`; `version` changes whenever something visible does.

Seats: 0 = You (south), 1 = East (opponent, on your right), 2 = Partner
(north), 3 = West (opponent, on your left). Play order is 0 -> 1 -> 2 -> 3,
i.e. counter-clockwise: play (and the deal) passes to the player on the right.
"""

import os
import random
import threading

from Trump.BiddingNet import BiddingNet, order_suits
from Trump.PlayingNet import PlayingNet
from Trump.TrumpGame import TrumpGame
from Trump.TrumpPlayer import TrumpPlayer
from webapp.stats import Stats, stats_path

TRUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Trump')

# Models the AI players can use: name -> folder holding bidding_net.pt and playing_net.pt.
# v1 and v3 share the same player and network shapes, so only the weights differ.
# A model is offered in the page only when both of its weight files exist.
MODELS = {
    'v1': os.path.join(TRUMP_DIR, 'models'),
    'v3': os.path.join(TRUMP_DIR, 'v3', 'models_fresh'),
}
DEFAULT_MODEL = 'v1'


def model_paths(name):
    """The (bidding weights, playing weights) file paths of a model."""
    return os.path.join(MODELS[name], 'bidding_net.pt'), os.path.join(MODELS[name], 'playing_net.pt')


def available_models():
    """Names of the models whose weight files are both present."""
    return [name for name in MODELS if all(os.path.exists(p) for p in model_paths(name))]


def check_weights(name):
    """Load a model's weights into throwaway networks; raises if a file is unreadable (e.g. mid-save)."""
    bid_path, playing_path = model_paths(name)
    BiddingNet().loadModel(bid_path)
    PlayingNet().loadModel(playing_path)

SEAT_NAMES = ['You', 'East', 'Partner', 'West']
SUIT_NAMES = {0: 'clubs', 1: 'diamonds', 2: 'spades', 3: 'hearts'}
RANK_CHARS = {14: 'A', 13: 'K', 12: 'Q', 11: 'J'}

AI_THINK_SECONDS = 0.7
TRICK_PAUSE_SECONDS = 1.6
BIDS_PAUSE_SECONDS = 1.8


class Aborted(Exception):
    """Raised inside the game thread when the session is replaced."""


def card_json(card):
    value = card.rank.value
    return {'id': '%d-%d' % (value, card.suite.id), 'rank': RANK_CHARS.get(value, str(value)),
            'suit': SUIT_NAMES[card.suite.id], 'value': value}


def hand_sort_key(card):
    return (card.suite.id, -card.rank.value)


class WebGame(TrumpGame):
    """TrumpGame that reports progress to a session and paces the AI for a human."""

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.deal_hands = []

    def distribute(self, distributer=0):
        super().distribute(distributer)
        self.deal_hands = [sorted(p.hand, key=hand_sort_key) for p in self.players]

    def bidding(self):
        s = self.session
        s.set(phase='bidding', trick=[], trick_cards=[], trick_winner=None)
        self.bids = [0] * self.NUM_PLAYERS
        for k in range(self.NUM_PLAYERS):
            seat = (self.start_player + k) % self.NUM_PLAYERS
            player = self.players[seat]
            player.suit_order = order_suits(player.hand, self.trump)
            partner_seat = self.players.index(player.partner)
            opponent_seats = [x for x in ((seat + o) % 4 for o in (1, 2, 3)) if x != partner_seat]
            s.set(turn=seat)
            bid = player.bid(self.bids[partner_seat], [self.bids[x] for x in opponent_seats])
            if not player.min_bid() <= bid <= player.MAX_BID:
                raise ValueError('Invalid bid %s from seat %d' % (bid, seat))
            self.bids[seat] = bid
            s.log('%s bids %d' % (SEAT_NAMES[seat], bid))
            s.set()
        s.set(turn=None)
        total = sum(self.bids)
        if total < self.MIN_TOTAL_BID:
            s.log('Bids total %d (< %d): cards are redealt' % (total, self.MIN_TOTAL_BID))
        s.sleep(BIDS_PAUSE_SECONDS)

    def play_tricks(self):
        s = self.session
        for player in self.players:
            player.tricks_won = 0
        self.played_by_seat = [set() for _ in range(self.NUM_PLAYERS)]
        s.set(phase='playing', trick=[], trick_cards=[], trick_winner=None)

        leader = self.start_player
        for _ in range(len(self.players[0].hand)):
            played = []
            s.set(trick=[], trick_cards=[], trick_winner=None)
            for k in range(self.NUM_PLAYERS):
                seat = (leader + k) % self.NUM_PLAYERS
                player = self.players[seat]
                s.set(turn=seat)
                card = player.choose_card(played)
                if card not in player.playable_cards(played):
                    raise ValueError('Seat %d cannot play %s' % (seat, card))
                player.hand.remove(card)
                played.append(card)
                self.played_by_seat[seat].add(card)
                s.set(trick=[(leader + i) % 4 for i in range(len(played))], trick_cards=list(played))
            winner = (leader + self.trick_winner(played)) % self.NUM_PLAYERS
            self.players[winner].tricks_won += 1
            s.set(turn=None, trick_winner=winner)
            s.sleep(TRICK_PAUSE_SECONDS)
            leader = winner
        s.set(trick=[], trick_cards=[], trick_winner=None)

    def update_scores(self):
        deltas = super().update_scores()
        self.session.round_deltas = deltas
        return deltas


class HumanPlayer(TrumpPlayer):
    def __init__(self, game, session):
        super().__init__(game)
        self.session = session

    def bid(self, partner_bid, opponent_bids):
        self.session.set(min_bid=self.min_bid())
        value = self.session.wait_for('bid')
        self.session.set(min_bid=None)
        return value

    def choose_card(self, played_cards):
        legal = self.playable_cards(played_cards)
        self.session.set(legal=[card_json(c)['id'] for c in legal])
        card_id = self.session.wait_for('card')
        self.session.set(legal=[])
        return next(c for c in legal if card_json(c)['id'] == card_id)


class AIPlayer(TrumpPlayer):
    def __init__(self, game, session):
        super().__init__(game)
        self.session = session

    def bid(self, partner_bid, opponent_bids):
        self.session.sleep(AI_THINK_SECONDS)
        if random.random() < self.session.randomness:
            return random.randint(self.min_bid(), self.MAX_BID)
        return super().bid(partner_bid, opponent_bids)

    def choose_card(self, played_cards):
        self.session.sleep(AI_THINK_SECONDS)
        if random.random() < self.session.randomness:
            return random.choice(self.playable_cards(played_cards))
        return super().choose_card(played_cards)


class Session:
    def __init__(self):
        self.cv = threading.Condition(threading.RLock())
        self.abort = threading.Event()
        self.version = 0
        self.state = {}
        self.awaiting = None      # 'bid' | 'card' | 'ack'
        self.action = None
        self.messages = []
        self.round_deltas = []
        self.summary = None
        self.randomness = 0.0   # 0 = AI always greedy, 1 = AI always random
        self.reveal = False     # show the other players' hands
        self.model = DEFAULT_MODEL   # which weights the three AI players use (see MODELS)
        self.stats = Stats(stats_path(self.model))
        self.game = None
        self.thread = None
        self.start()

    # ---- state plumbing -------------------------------------------------
    def start(self):
        self.game = WebGame(self)
        HumanPlayer(self.game, self)
        for _ in range(3):
            AIPlayer(self.game, self)
        self.game.load_models(*model_paths(self.model))
        self.state = {'phase': 'dealing', 'turn': None, 'trick': [], 'trick_cards': [], 'trick_winner': None,
                      'legal': [], 'min_bid': None}
        self.thread = threading.Thread(target=self._run, args=(self.game,), daemon=True)
        self.thread.start()

    def restart(self, model=None):
        """Abandon the current game and start a new one, optionally with another AI model (which also switches the stats)."""
        with self.cv:
            self.abort.set()
            self.cv.notify_all()
        old = self.thread
        if old:
            old.join(timeout=5)
        with self.cv:
            self.abort = threading.Event()
            self.version += 1
            self.messages = []
            self.summary = None
            self.awaiting = None
            self.action = None
            if model is not None and model != self.model:
                self.model = model
                self.stats = Stats(stats_path(model))
            self.start()

    def set_model(self, name):
        """
        Switch the three AI players to another model, which starts a new game.

        Returns:
            (bool, str): Whether the switch happened, and an error message if not
        """
        if name not in available_models():
            return False, 'Model %s is not available' % name
        if name == self.model:
            return True, None
        try:
            check_weights(name)
        except Exception:
            return False, 'Could not load the %s weights (a training save may be in progress), try again' % name
        previous = self.model
        try:
            self.restart(name)
        except Exception:
            self.restart(previous)  # best effort: go back to the model that was working
            return False, 'Could not switch to the %s model' % name
        return True, None

    def stats_report(self):
        report = self.stats.report()
        report['model'] = self.model
        return report

    def update_settings(self, randomness=None, reveal=None):
        with self.cv:
            if randomness is not None:
                self.randomness = min(1.0, max(0.0, float(randomness)))
            if reveal is not None:
                self.reveal = bool(reveal)
            self.version += 1

    def set(self, **changes):
        with self.cv:
            self.state.update(changes)
            self.version += 1

    def log(self, text):
        with self.cv:
            self.messages.append(text)
            self.messages = self.messages[-60:]
            self.version += 1

    def sleep(self, seconds):
        if self.abort.wait(seconds):
            raise Aborted()

    def wait_for(self, kind):
        with self.cv:
            self.awaiting = kind
            self.action = None
            self.version += 1
            while self.action is None and not self.abort.is_set():
                self.cv.wait(timeout=0.5)
            if self.abort.is_set():
                raise Aborted()
            value = self.action
            self.action = None
            self.awaiting = None
            self.version += 1
            return value

    def submit(self, kind, value):
        """Called from an HTTP thread. Returns (ok, error_message)."""
        with self.cv:
            if self.awaiting != kind:
                return False, 'Not waiting for a %s right now' % kind
            if kind == 'bid':
                if not isinstance(value, int) or not (self.state.get('min_bid') or 2) <= value <= 13:
                    return False, 'Invalid bid'
            elif kind == 'card':
                if value not in self.state.get('legal', []):
                    return False, 'That card cannot be played now'
            self.action = value if kind != 'ack' else True
            self.cv.notify_all()
            return True, None

    # ---- game thread ----------------------------------------------------
    def _run(self, game):
        try:
            dealer = random.randrange(4)
            while not game.is_over():
                self.set(trick=[], trick_cards=[])
                self.log('--- Round %d ---' % (game.round_number + 1))
                game.start_round(dealer)
                dealer = (game.distributer + 1) % 4
                self._finish_round(game)
                if game.is_over():
                    break
                self.wait_for('ack')
                self.summary = None
            self.set(phase='game_over')
            self.log('Game over')
        except Aborted:
            pass

    def _finish_round(self, game):
        seats = []
        for i, p in enumerate(game.players):
            seats.append({'name': SEAT_NAMES[i], 'bid': game.bids[i], 'tricks': p.tricks_won,
                          'delta': self.round_deltas[i], 'score': p.score,
                          'hand': [card_json(c) for c in game.deal_hands[i]]})
            self.log('%s: bid %d, won %d -> %+d' % (SEAT_NAMES[i], game.bids[i], p.tricks_won, self.round_deltas[i]))
        self.stats.record_round(game.bids, [p.tricks_won for p in game.players], self.round_deltas)
        self.summary = {'seats': seats, 'over': game.is_over()}
        if game.is_over():
            mine = game._team_result(0, 2)
            theirs = game._team_result(1, 3)
            self.summary['winner'] = 'us' if mine > 0 or theirs < 0 else 'them'
            self.stats.record_game(self.summary['winner'] == 'us')
        self.set(phase='round_over', turn=None)

    # ---- browser view ---------------------------------------------------
    def snapshot(self):
        with self.cv:
            g = self.game
            st = self.state
            players = g.players
            seats = []
            for i, p in enumerate(players):
                seats.append({'name': SEAT_NAMES[i], 'score': p.score, 'bid': g.bids[i],
                              'tricks': p.tricks_won, 'cards': len(p.hand)})
            trick = [{'seat': s, 'card': card_json(c)} for s, c in zip(st.get('trick', []), st.get('trick_cards', []))]
            return {
                'version': self.version,
                'phase': st['phase'],
                'awaiting': self.awaiting,
                'turn': st.get('turn'),
                'dealer': g.distributer,
                'round': g.round_number,
                'trump': None if g.trump is None else SUIT_NAMES[g.trump.id],
                'seats': seats,
                'hand': [card_json(c) for c in sorted(players[0].hand, key=hand_sort_key)],
                'legal': st.get('legal', []) if self.awaiting == 'card' else [],
                'min_bid': st.get('min_bid') if self.awaiting == 'bid' else None,
                'trick': trick,
                'trick_winner': st.get('trick_winner'),
                'randomness': self.randomness,
                'reveal': self.reveal,
                'model': self.model,
                'models': available_models(),
                'other_hands': {i: [card_json(c) for c in sorted(players[i].hand, key=hand_sort_key)]
                                for i in (1, 2, 3)} if self.reveal else None,
                'log': list(self.messages),
                'summary': self.summary if st['phase'] in ('round_over', 'game_over') else None,
            }
