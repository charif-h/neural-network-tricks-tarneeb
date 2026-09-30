"""
Trump game rules built on the abstract Game4.
"""

import json
import os

from Cards.Card import CardSuite
from Cards.Game4 import Game4
from Trump.BiddingNet import BiddingNet, order_suits
from Trump.PlayingNet import PlayingNet, card_strength
import torch

# The trump is the opposite suite of the same color as the last card dealt.
OPPOSITE_SUITE = {
    CardSuite.HEART: CardSuite.DIAMOND,
    CardSuite.DIAMOND: CardSuite.HEART,
    CardSuite.CLUB: CardSuite.SPADE,
    CardSuite.SPADE: CardSuite.CLUB,
}


class TrumpGame(Game4):
    """
    Trump: four players, two teams, with a trump suite.

    Attributes:
        last_suite (CardSuite): Suite of the last card dealt to the distributer
        trump (CardSuite): Trump suite, derived from last_suite
        bids (list): Bid of each player, indexed by seat (0 = has not bid yet)
        bid_learning (bool): Whether players update the shared bidding network
        bid_epsilon (float): Exploration rate used by players while bid_learning is True -
            decays by bid_epsilon_decay every completed round, down to bid_epsilon_min
        bid_epsilon_min (float): Floor bid_epsilon never decays below
        bid_epsilon_decay (float): Multiplicative decay applied to bid_epsilon each round
            (1.0 means no decay, matching the old fixed-epsilon behavior)
        bid_batch (int): Number of completed rounds between bidding network updates
        bid_counterfactual (bool): If True, every round trains all bid outputs
            at once, each towards the reward that bid would have earned given
            the tricks actually won (see counterfactual_bid_rewards), instead
            of only the bid actually taken
        completed_rounds (int): Number of rounds actually played (aborted attempts don't count)
        card_learning (bool): Whether players update the shared playing network
        card_epsilon (float): Exploration rate used by players while card_learning is True -
            decays by card_epsilon_decay every completed round, down to card_epsilon_min
        card_epsilon_min (float): Floor card_epsilon never decays below
        card_epsilon_decay (float): Multiplicative decay applied to card_epsilon each round
            (1.0 means no decay, matching the old fixed-epsilon behavior)
        card_batch (int): Number of completed rounds between playing network updates
        played_by_seat (list): Cards each seat has played so far this round (sets, indexed by seat)
    """

    MIN_TOTAL_BID = 11  # rounds whose bids sum to less than this are aborted
    WINNING_SCORE = 41  # score a player must reach for the game to end
    LOSING_SCORE = -50  # score a player must fall to (or below) for the game to end
                        # (a training-only cutoff to end hopeless games early - not
                        # a real Trump rule - so it counts as a loss for that team)

    # Reward shaping for the bidding network: an outcome term (this round's point
    # swing for the player's team vs the opponents, plus a terminal bonus once the
    # game ends) combined with a calibration term that penalizes bidding far from
    # the number of tricks actually won. UNDER_WEIGHT > OVER_WEIGHT because the
    # real scoring already penalizes a failed bid, so calibration's own job is
    # mainly to discourage chronic underbidding, which the real scoring never
    # punishes on its own.
    CALIBRATION_SCALE = 0.4
    OVER_WEIGHT = 0.3        # calibration weight when the bid failed (miss < 0)
    UNDER_WEIGHT = 1.0       # calibration weight when the bid was made or beaten (miss >= 0)
    CALIBRATION_NORM = 121  # worst-case miss**2 (11**2), keeps calibration on outcome's scale

    # Reward shaping for the playing network: an outcome term (was the trick
    # won by my team, weighted by how much either of us still needs a trick to
    # make our bid) combined with a penalty for losing a trick while holding a
    # strong card (see card_strength in PlayingNet).
    CARD_TRICK_SCALE = 1.0
    CARD_NEED_FLOOR = 0.3     # value of a trick to a player who has already made their bid
    CARD_WASTE_WEIGHT = 0.15
    CARD_CALIBRATION_SCALE = 0.5

    def __init__(self, bid_learning=False, bid_epsilon=0.1, bid_epsilon_min=0.02, bid_epsilon_decay=1.0,
                 bid_batch=1, card_learning=False, card_epsilon=0.1, card_epsilon_min=0.02,
                 card_epsilon_decay=1.0, card_batch=1, bid_counterfactual=False):
        super().__init__()
        self.bid_counterfactual = bid_counterfactual
        self.last_suite = None
        self.trump = None
        self.bids = [0] * self.NUM_PLAYERS
        self.shared_bid_net = BiddingNet()
        self.shared_bid_optimizer = torch.optim.Adam(self.shared_bid_net.parameters(), lr=1e-3)
        self.bid_learning = bid_learning
        self.bid_epsilon = bid_epsilon
        self.bid_epsilon_min = bid_epsilon_min
        self.bid_epsilon_decay = bid_epsilon_decay
        self.bid_batch = bid_batch
        self.completed_rounds = 0
        self.shared_playing_net = PlayingNet()
        self.shared_playing_optimizer = torch.optim.Adam(self.shared_playing_net.parameters(), lr=1e-3)
        self.card_learning = card_learning
        self.card_epsilon = card_epsilon
        self.card_epsilon_min = card_epsilon_min
        self.card_epsilon_decay = card_epsilon_decay
        self.card_batch = card_batch
        self.played_by_seat = [set() for _ in range(self.NUM_PLAYERS)]

    def save_models(self, bid_path, playing_path):
        """
        Save the shared bidding and playing networks' weights to disk.

        Args:
            bid_path (str): File to write the bidding network's weights to
            playing_path (str): File to write the playing network's weights to
        """
        self.shared_bid_net.saveModel(bid_path)
        self.shared_playing_net.saveModel(playing_path)

    def load_models(self, bid_path, playing_path):
        """
        Load previously saved weights into the shared bidding and playing
        networks, if the corresponding file exists (each is loaded
        independently, so having only one of the two is fine).

        Args:
            bid_path (str): File to read the bidding network's weights from
            playing_path (str): File to read the playing network's weights from
        """
        if os.path.exists(bid_path):
            self.shared_bid_net.loadModel(bid_path)
        if os.path.exists(playing_path):
            self.shared_playing_net.loadModel(playing_path)

    def save_epsilon_state(self, path):
        """
        Save the current (decayed) bid_epsilon and card_epsilon to a JSON file.

        Training runs one train.py invocation at a time, each reloading the
        shared networks from disk - without this, bid_epsilon/card_epsilon
        would reset to their CLI-provided starting value on every new
        invocation and the decay accumulated so far would be lost.

        Args:
            path (str): File to write the epsilon values to
        """
        with open(path, 'w') as f:
            json.dump({'bid_epsilon': self.bid_epsilon, 'card_epsilon': self.card_epsilon}, f)

    def load_epsilon_state(self, path):
        """
        Load previously saved bid_epsilon/card_epsilon from a JSON file, if it
        exists, so decay resumes where the last run left off instead of
        restarting from the constructor's starting epsilon.

        Args:
            path (str): File to read the epsilon values from
        """
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        self.bid_epsilon = state.get('bid_epsilon', self.bid_epsilon)
        self.card_epsilon = state.get('card_epsilon', self.card_epsilon)

    def add_player(self, player):
        """Seat a player; once the table is full, partners sit opposite (0-2 and 1-3)."""
        super().add_player(player)
        if len(self.players) == self.NUM_PLAYERS:
            self.players[0].set_partner(self.players[2])
            self.players[1].set_partner(self.players[3])

    def start_round(self, distributer=0):
        """
        Distribute the cards as in Game4, let every player bid, then play the round.

        If the bids sum to less than MIN_TOTAL_BID the round is aborted and a
        new one starts, with the next player (modulo the number of players)
        as distributer, until the bids are high enough. The round is then
        played trick by trick.

        Args:
            distributer (int): Index of the distributer of the first attempted round
        """
        while True:
            super().start_round(distributer)
            self.bidding()
            if sum(self.bids) >= self.MIN_TOTAL_BID:
                break
            distributer = (distributer + 1) % self.NUM_PLAYERS
        self.play_tricks()
        deltas = self.update_scores()
        self.reward_players(deltas)

        self.completed_rounds += 1
        if self.bid_learning and self.completed_rounds % self.bid_batch == 0:
            self.update_bidding_network()
        if self.card_learning and self.completed_rounds % self.card_batch == 0:
            self.update_playing_network()

        if self.bid_learning:
            self.bid_epsilon = max(self.bid_epsilon_min, self.bid_epsilon * self.bid_epsilon_decay)
        if self.card_learning:
            self.card_epsilon = max(self.card_epsilon_min, self.card_epsilon * self.card_epsilon_decay)

    def update_scores(self):
        """
        Update every player's score at the end of a round.

        A player who won at least as many tricks as they bid gains their bid,
        otherwise they lose it. Bids from 7 to 12 count double and a bid of 13
        counts triple (both for gains and losses).

        Returns:
            list: Points gained (or lost) by each player this round, indexed by seat
        """
        deltas = []
        for seat, player in enumerate(self.players):
            bid = self.bids[seat]
            points = bid * self.bid_multiplier(bid)
            delta = points if player.tricks_won >= bid else -points
            player.score += delta
            deltas.append(delta)
        return deltas

    def _team_result(self, seat, partner_seat):
        """
        Whether the two-player team (seat, partner_seat) ended the game as
        winners (1), losers (-1), or neither (0). Only meaningful once
        is_over() is True. The two checks can never both fire for the same
        team: winning requires both scores non-negative (one very high),
        losing requires one score very negative - mutually exclusive.

        Args:
            seat (int): One seat of the team
            partner_seat (int): The other seat of the same team

        Returns:
            int: 1, -1, or 0
        """
        a, b = self.players[seat], self.players[partner_seat]
        if (a.score >= self.WINNING_SCORE and b.score >= 0) or (b.score >= self.WINNING_SCORE and a.score >= 0):
            return 1
        if a.score <= self.LOSING_SCORE or b.score <= self.LOSING_SCORE:
            return -1
        return 0

    def reward_players(self, deltas):
        """
        Compute each player's reward for the round just played and record it.

        The reward combines two terms:
          - outcome: this round's point swing in the player's team's favor
            (their team's gain minus the opponents' gain, scaled down), plus a
            terminal bonus once the round ends the game. The bonus favors
            whichever team actually won or lost (see _team_result) - a team
            that hits LOSING_SCORE counts as having lost even though nobody
            reached WINNING_SCORE, since that ending is a training-only
            cutoff standing in for what would otherwise be a lost cause.
          - calibration: a penalty based on how far tricks_won was from the
            bid (see the class constants above for the reasoning).

        Args:
            deltas (list): Points gained/lost by each player this round (see update_scores)
        """
        if not self.bid_learning:
            return

        if self.bid_counterfactual:
            for player in self.players:
                player.record_bid_reward(self.counterfactual_bid_rewards(player.tricks_won, player.pending_min_bid))
            return

        game_over = self.is_over()
        for seat, player in enumerate(self.players):
            partner_seat = self.players.index(player.partner)
            opponent_seats = [s for s in range(self.NUM_PLAYERS) if s not in (seat, partner_seat)]

            team_gain = deltas[seat] + deltas[partner_seat]
            opponent_gain = sum(deltas[s] for s in opponent_seats)
            outcome = (team_gain - opponent_gain) / self.WINNING_SCORE
            if game_over:
                my_result = self._team_result(seat, partner_seat)
                opp_result = self._team_result(opponent_seats[0], opponent_seats[1])
                if my_result != 0:
                    outcome += 1.0 if my_result > 0 else -1.0
                else:
                    outcome += -1.0 if opp_result > 0 else 1.0

            miss = player.tricks_won - self.bids[seat]
            weight = self.OVER_WEIGHT if miss < 0 else self.UNDER_WEIGHT
            calibration = -weight * miss ** 2 / self.CALIBRATION_NORM

            player.record_bid_reward(outcome + self.CALIBRATION_SCALE * calibration)

    def counterfactual_bid_rewards(self, tricks_won, min_bid):
        """
        Reward every possible bid (2 to 13) would have earned given the tricks
        the player actually won this round.

        Same shape as the single-bid reward, minus the parts the bid doesn't
        control: only the player's own score change (scaled like the outcome
        term above) plus the same calibration term. The opponents' points and
        the terminal bonus are left out on purpose - they're noise for the bid.

        Args:
            tricks_won (int): Tricks the player won this round
            min_bid (int): Lowest bid that was legal for the player this round

        Returns:
            torch.Tensor: Shape (12,), one reward per bid 2..13; NaN for bids
                below min_bid, which were not legal and must not be trained on
        """
        bids = torch.arange(2, 14, dtype=torch.float32)
        multipliers = torch.tensor([self.bid_multiplier(int(b)) for b in bids], dtype=torch.float32)
        points = bids * multipliers
        outcome = torch.where(tricks_won >= bids, points, -points) / self.WINNING_SCORE

        miss = tricks_won - bids
        weight = torch.where(miss < 0, self.OVER_WEIGHT, self.UNDER_WEIGHT)
        calibration = -weight * miss ** 2 / self.CALIBRATION_NORM

        rewards = outcome + self.CALIBRATION_SCALE * calibration
        rewards[bids < min_bid] = float('nan')
        return rewards

    def update_bidding_network(self):
        """
        Train the shared bidding network on every player's collected experience.

        Gathers all (state, bid, reward) samples across the 4 players into one
        batch and takes a single gradient step, then clears every player's
        history. Normally the predicted value of each bid actually taken moves
        towards its observed reward. In bid_counterfactual mode the reward is a
        vector with one entry per bid, and every legal bid's output moves
        towards its own entry.
        """
        samples = [sample for player in self.players for sample in player.bid_history]
        if not samples:
            return

        states = torch.stack([state for state, bid, reward in samples])
        predictions = self.shared_bid_net(states)

        if self.bid_counterfactual:
            rewards = torch.stack([reward for state, bid, reward in samples])
            legal = ~torch.isnan(rewards)
            loss = ((predictions - torch.nan_to_num(rewards)) ** 2)[legal].mean()
        else:
            bid_indices = torch.tensor([bid - 2 for state, bid, reward in samples], dtype=torch.long)
            rewards = torch.tensor([reward for state, bid, reward in samples], dtype=torch.float32)
            predicted_values = predictions.gather(1, bid_indices.unsqueeze(1)).squeeze(1)
            loss = ((predicted_values - rewards) ** 2).mean()

        self.shared_bid_optimizer.zero_grad()
        loss.backward()
        self.shared_bid_optimizer.step()

        for player in self.players:
            player.bid_history = []

    @staticmethod
    def bid_multiplier(bid):
        """Factor applied to a bid when scoring: 3 for 13, 2 for 7 to 12, 1 otherwise."""
        if bid == 13:
            return 3
        if bid >= 7:
            return 2
        return 1

    def is_over(self):
        """
        The game ends when a player reaches WINNING_SCORE with a non-negative
        partner, or when any player's score falls to LOSING_SCORE or below.
        """
        return any(p.score >= self.WINNING_SCORE and p.partner.score >= 0 for p in self.players) \
            or any(p.score <= self.LOSING_SCORE for p in self.players)

    def play_game(self, distributer=0):
        """
        Play rounds until the game is over.

        The distributer moves to the next player (modulo the number of players)
        after each round.

        Args:
            distributer (int): Index of the distributer of the first round
        """
        while not self.is_over():
            self.start_round(distributer)
            distributer = (self.distributer + 1) % self.NUM_PLAYERS

    def play_tricks(self):
        """
        Play the 13 tricks of the round.

        The start player leads the first trick, then the players play in seat
        order. Each player must follow the suite of the first card if they
        can. The winner of a trick (see trick_winner) gains one trick
        (player.tricks_won) and leads the next trick.
        """
        for player in self.players:
            player.tricks_won = 0
        self.played_by_seat = [set() for _ in range(self.NUM_PLAYERS)]

        leader = self.start_player
        for _ in range(len(self.players[0].hand)):
            played_cards = []
            for k in range(self.NUM_PLAYERS):
                seat = (leader + k) % self.NUM_PLAYERS
                player = self.players[seat]
                card = player.choose_card(played_cards)
                if card not in player.playable_cards(played_cards):
                    raise ValueError('Player %d cannot play %s' % (seat, card))
                player.hand.remove(card)
                played_cards.append(card)
                self.played_by_seat[seat].add(card)

            winner_seat = (leader + self.trick_winner(played_cards)) % self.NUM_PLAYERS
            self.reward_card_players(played_cards, leader, winner_seat)
            self.players[winner_seat].tricks_won += 1
            leader = winner_seat

    def reward_card_players(self, played_cards, leader, winner_seat):
        """
        Compute each player's reward for the trick just resolved and record it.

        Uses tricks_won as of just before this trick is credited, so the
        reward reflects how much the trick was needed at the moment it was
        played, not after the fact. The reward combines two terms:
          - outcome: +/- CARD_TRICK_SCALE depending on whether the trick was
            won by the player's team, weighted by how much either the player
            or their partner still needs a trick to make their bid (a trick
            already made is worth less, but not nothing - denying it to the
            opponents still has some value).
          - waste penalty: if the trick was lost, a penalty proportional to
            the strength of the card just played (see PlayingNet.card_strength) -
            losing a strong card without winning costs more than losing a weak one.

        Args:
            played_cards (list): The 4 cards played this trick, in play order
            leader (int): Seat that led this trick
            winner_seat (int): Seat that won this trick
        """
        if not self.card_learning:
            return

        for k, card in enumerate(played_cards):
            seat = (leader + k) % self.NUM_PLAYERS
            player = self.players[seat]
            partner_seat = self.players.index(player.partner)

            my_need = 1.0 if player.tricks_won < self.bids[seat] else self.CARD_NEED_FLOOR
            partner_need = 1.0 if self.players[partner_seat].tricks_won < self.bids[partner_seat] \
                else self.CARD_NEED_FLOOR
            team_need = max(my_need, partner_need)

            won_by_us = winner_seat in (seat, partner_seat)
            outcome = self.CARD_TRICK_SCALE * team_need if won_by_us else -self.CARD_TRICK_SCALE * team_need

            waste_penalty = 0.0 if won_by_us else -self.CARD_WASTE_WEIGHT * card_strength(card, self.trump)

            player.record_card_reward(outcome + self.CARD_CALIBRATION_SCALE * waste_penalty)

    def update_playing_network(self):
        """
        Train the shared playing network on every player's collected experience.

        Gathers all (state, slot, reward) samples across the 4 players into
        one batch, takes a single gradient step so the network's predicted
        value of each card actually played moves towards its observed reward,
        then clears every player's history.
        """
        samples = [sample for player in self.players for sample in player.card_history]
        if not samples:
            return

        states = torch.stack([state for state, slot, reward in samples])
        slot_indices = torch.tensor([slot for state, slot, reward in samples], dtype=torch.long)
        rewards = torch.tensor([reward for state, slot, reward in samples], dtype=torch.float32)

        predictions = self.shared_playing_net(states)
        predicted_values = predictions.gather(1, slot_indices.unsqueeze(1)).squeeze(1)
        loss = ((predicted_values - rewards) ** 2).mean()

        self.shared_playing_optimizer.zero_grad()
        loss.backward()
        self.shared_playing_optimizer.step()

        for player in self.players:
            player.card_history = []

    def bidding(self):
        """
        Let each player bid, starting with the start player.

        Every player is told the partner's bid and the two opponents' bids
        (0 for a player who has not bid yet), and must bid between their
        minimum bid and MAX_BID. The bids are stored in self.bids.
        """
        self.bids = [0] * self.NUM_PLAYERS
        for k in range(self.NUM_PLAYERS):
            seat = (self.start_player + k) % self.NUM_PLAYERS
            player = self.players[seat]
            # Fixed once here and reused for every trick of the round, so a given
            # physical suit keeps the same slot position all round (see PlayingNet).
            player.suit_order = order_suits(player.hand, self.trump)
            partner_seat = self.players.index(player.partner)
            opponent_seats = [s for s in ((seat + o) % self.NUM_PLAYERS for o in (1, 2, 3))
                              if s != partner_seat]

            bid = player.bid(self.bids[partner_seat], [self.bids[s] for s in opponent_seats])
            if not player.min_bid() <= bid <= player.MAX_BID:
                raise ValueError('Invalid bid %s from player %d (allowed: %d to %d)'
                                 % (bid, seat, player.min_bid(), player.MAX_BID))
            self.bids[seat] = bid

    def distribute(self, distributer=0):
        """
        Deal as in Game4 (the distributer is dealt to last), then remember the
        suite of the last card given to the distributer and set the trump to
        the opposite suite of the same color.

        Args:
            distributer (int): Index in self.players of the distributing player
        """
        super().distribute(distributer)
        self.last_suite = self.players[distributer].hand[-1].suite
        self.trump = OPPOSITE_SUITE[self.last_suite]

    def trick_winner(self, cards):
        """
        Determine the winning card of a trick.

        1. The highest trump card wins.
        2. If there is no trump card, the highest card of the first card's
           suite wins.

        Args:
            cards (list): The 4 Card objects played, in playing order

        Returns:
            int: Index in `cards` of the winning card
        """
        candidates = [i for i, c in enumerate(cards) if c.suite == self.trump]
        if not candidates:
            candidates = [i for i, c in enumerate(cards) if c.suite == cards[0].suite]
        return max(candidates, key=lambda i: cards[i].rank.value)

