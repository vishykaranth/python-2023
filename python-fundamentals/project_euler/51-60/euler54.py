from lib.files import read_multiline_file


class HandRank(object):
    HighCard, OnePair, TwoPair, ThreeOfAKind, Straight, \
        Flush, FullHouse, FourOfAKind, StraightFlush, RoyalFlush = range(10)


class Hand(object):
    def __init__(self, hand_rank, cards):
        self.hand_rank = hand_rank
        self.cards = cards

    def __str__(self):
        return "{}: {}".format(self.hand_rank, self.cards)


# Convert a card into it's integer value, e.g convert_value('T') would return "10".
def convert_value(value):
    value_map = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    return value_map[value] if value in value_map else int(value)


def rank_hand(cards):
    cards = str(cards).split()
    values = [convert_value(v[0]) for v in cards]
    suits = [s[1] for s in cards]
    ordered = sorted(values)

    pairs = [v for v in values if values.count(v) == 2]
    has_pair = len(pairs) == 2
    has_two_pair = len(pairs) == 4
    has_three_of_a_kind = len([v for v in values if values.count(v) == 3]) > 0
    has_straight = len(pairs) is 0 and abs(ordered[len(cards) - 1] - ordered[0]) == 4

    # If you haven't already got a straight, check to see if counting aces low would make a straight
    if not has_straight:
        aces_replaced = sorted([1 if v == convert_value('A') else v for v in ordered])
        has_straight = len(pairs) is 0 and abs(aces_replaced[len(cards) - 1] - aces_replaced[0]) == 4

    has_four_of_a_kind = len([v for v in values if values.count(v) >= 4]) > 0
    has_flush = len([s for s in suits if suits.count(s) >= 5]) > 0
    has_full_house = has_pair and has_three_of_a_kind
    has_straight_flush = has_straight and has_flush
    has_royal_flush = has_flush and ordered[0] == convert_value('T')

    if has_royal_flush:
        return Hand(hand_rank=HandRank.RoyalFlush, cards=ordered)
    elif has_straight_flush:
        return Hand(hand_rank=HandRank.StraightFlush, cards=ordered)
    elif has_four_of_a_kind:
        return Hand(hand_rank=HandRank.FourOfAKind, cards=ordered)
    elif has_full_house:
        return Hand(hand_rank=HandRank.FullHouse, cards=ordered)
    elif has_flush:
        return Hand(hand_rank=HandRank.Flush, cards=ordered)
    elif has_straight:
        return Hand(hand_rank=HandRank.Straight, cards=ordered)
    elif has_three_of_a_kind:
        return Hand(hand_rank=HandRank.ThreeOfAKind, cards=ordered)
    elif has_two_pair:
        return Hand(hand_rank=HandRank.TwoPair, cards=ordered)
    elif has_pair:
        return Hand(hand_rank=HandRank.OnePair, cards=ordered)
    else:
        return Hand(hand_rank=HandRank.HighCard, cards=ordered)


def best_hand(one: Hand, two: Hand):
    if one.hand_rank == two.hand_rank:
        # Hands are the same rank, needs some additional logic to work out the best hand
        if one.hand_rank == HandRank.HighCard:
            return rank_highest_card(one.cards, two.cards)
        elif one.hand_rank == HandRank.OnePair:
            return rank_pair(one, two)
        elif one.hand_rank == HandRank.TwoPair:
            return rank_two_pair(one, two)
        elif one.hand_rank == HandRank.ThreeOfAKind:
            return rank_multiple_kinds(one, two)
        elif one.hand_rank == HandRank.Straight:
            return rank_straight(one, two)
        elif one.hand_rank == HandRank.Flush:
            return rank_highest_card(one.cards, two.cards)
        elif one.hand_rank == HandRank.FullHouse:
            return rank_full_house(one, two)
        elif one.hand_rank == HandRank.FourOfAKind:
            return rank_multiple_kinds(one, two)
        elif one.hand_rank == HandRank.StraightFlush:
            return rank_straight(one, two)
        elif one.hand_rank == HandRank.RoyalFlush:
            # Both have a royal flush, you can't have a better royal flush so it's a draw
            return 0
    else:
        # A player's hand is out-right the best hand, so return that player
        return 1 if one.hand_rank > two.hand_rank else 2


def rank_full_house(one: Hand, two: Hand):
    one_trips = sorted([c for c in one.cards if one.cards.count(c) > 2], reverse=True)
    one_pair = sorted([c for c in one.cards if one.cards.count(c) == 2], reverse=True)
    two_trips = sorted([c for c in two.cards if two.cards.count(c) > 2], reverse=True)
    two_pair = sorted([c for c in two.cards if two.cards.count(c) == 2], reverse=True)

    if one_trips[0] == two_trips[0]:
        return 0 if one_pair[0] == two_pair[0] else 1 if one_pair[0] > two_pair[0] else 2

    return 1 if one_trips[0] > two_trips[0] else 2


def rank_straight(one: Hand, two: Hand):
    end = len(one.cards) - 1

    # Edge case of comparing an ace-low straight, convert the ace to lower then perform the comparison
    if one.cards[0] == 2 and one.cards[end] == convert_value('A'):
        one.cards[end] = 1
        one.cards = sorted(one.cards)
    if two.cards[0] == 2 and two.cards[end] == convert_value('A'):
        two.cards[end] = 1
        two.cards = sorted(two.cards)

    return 0 if one.cards[end] == two.cards[end] else 1 if one.cards[end] > two.cards[end] else 2


def rank_multiple_kinds(one: Hand, two: Hand):
    one_kind = sorted([c for c in one.cards if one.cards.count(c) > 2], reverse=True)
    two_kind = sorted([c for c in two.cards if two.cards.count(c) > 2], reverse=True)
    one_cards = [k for k in one.cards if k not in one_kind]
    two_cards = [k for k in two.cards if k not in two_kind]

    if one_kind[0] == two_kind[0]:
        return rank_highest_card(one_cards, two_cards)
    else:
        return 1 if one_kind[2] > two_kind[2] else 2


def rank_two_pair(one: Hand, two: Hand):
    one_pairs = sorted([c for c in one.cards if one.cards.count(c) == 2], reverse=True)
    two_pairs = sorted([c for c in two.cards if two.cards.count(c) == 2], reverse=True)
    one_cards = [k for k in one.cards if k not in one_pairs]
    two_cards = [k for k in two.cards if k not in two_pairs]

    if one_pairs[0] == two_pairs[0]:
        if one_pairs[2] == two_pairs[2]:
            return rank_highest_card(one_cards, two_cards)
        else:
            return 1 if one_pairs[2] > two_pairs[2] else 2
    else:
        return 1 if one_pairs[0] > two_pairs[0] else 2


def rank_pair(one: Hand, two: Hand):
    one_pair = [c for c in one.cards if one.cards.count(c) == 2].pop()
    two_pair = [c for c in two.cards if two.cards.count(c) == 2].pop()

    if one_pair == two_pair:
        return rank_highest_card(one.cards, two.cards)
    else:
        return 1 if one_pair > two_pair else 2


def rank_highest_card(one_cards, two_cards):
    for card_index in range(len(one_cards) - 1, -1, -1):
        if one_cards[card_index] == two_cards[card_index]:
            continue
        return 1 if one_cards[card_index] > two_cards[card_index] else 2

    return 0


hands = read_multiline_file("../input/poker.txt")
print(len([hand for hand in hands if best_hand(rank_hand(hand[:15]), rank_hand(hand[15:])) == 1]))

