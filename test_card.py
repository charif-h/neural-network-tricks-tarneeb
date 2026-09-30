"""Unit tests for Cards/Card.py. Run with: python -m unittest test_card -v"""

import unittest

import numpy as np

from Cards.Card import Card, CardSuite, CardRank


class TestCardSuite(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(
            [int(t) for t in (CardSuite.CLUB, CardSuite.DIAMOND, CardSuite.SPADE, CardSuite.HEART)],
            [0, 1, 2, 3],
        )

    def test_symbols(self):
        self.assertEqual(CardSuite.CLUB.value, "♣")
        self.assertEqual(CardSuite.DIAMOND.value, "♦")
        self.assertEqual(CardSuite.SPADE.value, "♠")
        self.assertEqual(CardSuite.HEART.value, "♥")

    def test_id_attribute_matches_int(self):
        for t in CardSuite:
            self.assertEqual(t.id, int(t))

    def test_four_members(self):
        self.assertEqual(len(CardSuite), 4)


class TestCardRank(unittest.TestCase):
    def test_range(self):
        values = sorted(v.value for v in CardRank)
        self.assertEqual(values, list(range(2, 15)))

    def test_face_cards(self):
        self.assertEqual(CardRank.JACK.value, 11)
        self.assertEqual(CardRank.QUEEN.value, 12)
        self.assertEqual(CardRank.KING.value, 13)
        self.assertEqual(CardRank.ACE.value, 14)


class TestCardBasics(unittest.TestCase):
    def setUp(self):
        self.ace_spades = Card(CardRank.ACE, CardSuite.SPADE)
        self.ten_hearts = Card(CardRank.TEN, CardSuite.HEART)
        self.two_clubs = Card(CardRank.TWO, CardSuite.CLUB)

    def test_attributes(self):
        self.assertIs(self.ace_spades.rank, CardRank.ACE)
        self.assertIs(self.ace_spades.suite, CardSuite.SPADE)

    def test_str(self):
        self.assertEqual(str(self.ace_spades), "A♠")
        self.assertEqual(str(self.ten_hearts), "10♥")
        self.assertEqual(str(self.two_clubs), "2♣")

    def test_repr(self):
        self.assertEqual(repr(self.ace_spades), "A-♠")
        self.assertEqual(repr(self.ten_hearts), "10-♥")

    def test_value_char_numbers(self):
        for v in range(2, 11):
            card = Card(CardRank(v), CardSuite.CLUB)
            self.assertEqual(str(card.rankChar()), str(v))

    def test_value_char_faces(self):
        expected = {
            CardRank.JACK: "J",
            CardRank.QUEEN: "Q",
            CardRank.KING: "K",
            CardRank.ACE: "A",
        }
        for value, char in expected.items():
            self.assertEqual(Card(value, CardSuite.CLUB).rankChar(), char)


class TestCardEquality(unittest.TestCase):
    def test_equal_cards(self):
        self.assertEqual(
            Card(CardRank.KING, CardSuite.HEART), Card(CardRank.KING, CardSuite.HEART)
        )

    def test_different_value(self):
        self.assertNotEqual(
            Card(CardRank.KING, CardSuite.HEART), Card(CardRank.QUEEN, CardSuite.HEART)
        )

    def test_different_type(self):
        self.assertNotEqual(
            Card(CardRank.KING, CardSuite.HEART), Card(CardRank.KING, CardSuite.SPADE)
        )

    def test_hash_consistent_with_eq(self):
        a = Card(CardRank.SEVEN, CardSuite.DIAMOND)
        b = Card(CardRank.SEVEN, CardSuite.DIAMOND)
        self.assertEqual(hash(a), hash(b))

    def test_usable_in_set_and_dict(self):
        a = Card(CardRank.SEVEN, CardSuite.DIAMOND)
        b = Card(CardRank.SEVEN, CardSuite.DIAMOND)
        c = Card(CardRank.EIGHT, CardSuite.DIAMOND)
        self.assertEqual(len({a, b, c}), 2)
        self.assertEqual({a: 1}[b], 1)

    def test_full_deck_hashes_are_distinct_entries(self):
        cards = {Card(v, t) for v in CardRank for t in CardSuite}
        self.assertEqual(len(cards), 52)

    def test_eq_with_non_card(self):
        # Desired behaviour: comparing with a non-Card is simply False.
        # Currently raises AttributeError (see code review).
        card = Card(CardRank.ACE, CardSuite.SPADE)
        self.assertFalse(card == None)  # noqa: E711


class TestCardOrdering(unittest.TestCase):
    def test_same_suit_orders_by_value(self):
        low = Card(CardRank.TWO, CardSuite.SPADE)
        high = Card(CardRank.ACE, CardSuite.SPADE)
        self.assertTrue(low < high)
        self.assertFalse(high < low)

    def test_different_suit_orders_by_suit(self):
        club = Card(CardRank.ACE, CardSuite.CLUB)
        diamond = Card(CardRank.TWO, CardSuite.DIAMOND)
        spade = Card(CardRank.TWO, CardSuite.SPADE)
        heart = Card(CardRank.TWO, CardSuite.HEART)
        self.assertTrue(club < diamond < spade < heart)

    def test_equal_cards_not_less_than(self):
        a = Card(CardRank.FIVE, CardSuite.CLUB)
        b = Card(CardRank.FIVE, CardSuite.CLUB)
        self.assertFalse(a < b)

    def test_sorted(self):
        cards = [
            Card(CardRank.KING, CardSuite.CLUB),
            Card(CardRank.TWO, CardSuite.CLUB),
            Card(CardRank.ACE, CardSuite.HEART),
            Card(CardRank.THREE, CardSuite.SPADE),
        ]
        self.assertEqual(
            sorted(cards),
            [
                Card(CardRank.TWO, CardSuite.CLUB),
                Card(CardRank.KING, CardSuite.CLUB),
                Card(CardRank.THREE, CardSuite.SPADE),
                Card(CardRank.ACE, CardSuite.HEART),
            ],
        )


class TestCardId(unittest.TestCase):
    def test_lowest_and_highest(self):
        self.assertEqual(Card(CardRank.TWO, CardSuite.CLUB).cardId(), 0)
        self.assertEqual(Card(CardRank.ACE, CardSuite.HEART).cardId(), 51)

    def test_formula(self):
        card = Card(CardRank.TEN, CardSuite.SPADE)
        self.assertEqual(card.cardId(), 4 * (10 - 2) + 2)

    def test_ids_unique_and_cover_0_to_51(self):
        ids = {Card(v, t).cardId() for v in CardRank for t in CardSuite}
        self.assertEqual(ids, set(range(52)))


class TestCardToMatrix(unittest.TestCase):
    def test_shape(self):
        arr = Card(CardRank.ACE, CardSuite.SPADE).card_to_matrix()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (4,))

    def test_ace_of_spades(self):
        arr = Card(CardRank.ACE, CardSuite.SPADE).card_to_matrix()
        np.testing.assert_allclose(arr, [0, 0, 1.0, 0])

    def test_two_of_clubs(self):
        arr = Card(CardRank.TWO, CardSuite.CLUB).card_to_matrix()
        np.testing.assert_allclose(arr, [2 / 14, 0, 0, 0])

    def test_only_one_nonzero_entry(self):
        for v in CardRank:
            for t in CardSuite:
                arr = Card(v, t).card_to_matrix()
                self.assertEqual(np.count_nonzero(arr), 1)
                self.assertAlmostEqual(arr[t.id], v.value / 14)


class TestLargerThan(unittest.TestCase):
    def setUp(self):
        self.ace_h = Card(CardRank.ACE, CardSuite.HEART)
        self.king_h = Card(CardRank.KING, CardSuite.HEART)
        self.king_s = Card(CardRank.KING, CardSuite.SPADE)
        self.two_s = Card(CardRank.TWO, CardSuite.SPADE)

    def test_same_suit(self):
        self.assertTrue(self.ace_h.largerThan(self.king_h))
        self.assertFalse(self.king_h.largerThan(self.ace_h))

    def test_equal_cards(self):
        self.assertFalse(self.king_h.largerThan(Card(CardRank.KING, CardSuite.HEART)))

    def test_different_suit_respecting_type_is_false(self):
        self.assertFalse(self.ace_h.largerThan(self.two_s))
        self.assertFalse(self.two_s.largerThan(self.ace_h))

    def test_different_suit_ignoring_type_compares_values(self):
        self.assertTrue(self.ace_h.largerThan(self.two_s, respectype=False))
        self.assertFalse(self.two_s.largerThan(self.ace_h, respectype=False))
        self.assertFalse(self.king_h.largerThan(self.king_s, respectype=False))


if __name__ == "__main__":
    unittest.main()
