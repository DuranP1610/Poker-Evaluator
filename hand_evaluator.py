
from itertools import combinations

RANK_VALUES = {r: i + 2 for i, r in enumerate("23456789TJQKA")}


def rank_of(card_str):
    return RANK_VALUES[card_str[0]]


def suit_of(card_str):
    return card_str[1]


def find_straight_high(rank_list):
    """Highest card of a straight in rank_list, or None. Handles wheel (A-2-3-4-5)."""
    with_wheel = set(rank_list)
    if 14 in with_wheel:
        with_wheel.add(1)
    sorted_ranks = sorted(with_wheel, reverse=True)
    for i in range(len(sorted_ranks) - 4):
        if sorted_ranks[i] - sorted_ranks[i + 4] == 4:
            return sorted_ranks[i]
    return None


def evaluate5(cards):
    """cards: list of 5 card strings. Returns (class_rank, kickers)."""
    ranks = sorted((rank_of(c) for c in cards), reverse=True)
    suits = [suit_of(c) for c in cards]
    is_flush = len(set(suits)) == 1

    straight_high = find_straight_high(ranks)

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
    ordered_ranks = []
    for rank, count in counts:
        ordered_ranks.extend([rank] * count)

    if is_flush and straight_high:
        return 1, [straight_high]
    if counts[0][1] == 4:
        return 2, [counts[0][0], counts[1][0]]
    if counts[0][1] == 3 and counts[1][1] == 2:
        return 3, [counts[0][0], counts[1][0]]
    if is_flush:
        return 4, ranks
    if straight_high:
        return 5, [straight_high]
    if counts[0][1] == 3:
        return 6, ordered_ranks
    if counts[0][1] == 2 and counts[1][1] == 2:
        return 7, [counts[0][0], counts[1][0], counts[2][0]]
    if counts[0][1] == 2:
        return 8, ordered_ranks
    return 9, ranks


def compare_hands(a, b):
    """a, b: (class_rank, kickers) tuples. Returns >0 if a better, <0 if b better, 0 tie."""
    class_a, kickers_a = a
    class_b, kickers_b = b
    if class_a != class_b:
        return class_b - class_a  # lower class_rank = better
    for i in range(max(len(kickers_a), len(kickers_b))):
        ka = kickers_a[i] if i < len(kickers_a) else 0
        kb = kickers_b[i] if i < len(kickers_b) else 0
        if ka != kb:
            return ka - kb
    return 0


def evaluate_hand(all_cards):
    """Best 5-card hand out of 5, 6, or 7 cards. Returns (class_rank, kickers)."""
    if len(all_cards) == 5:
        combos = [all_cards]
    else:
        combos = list(combinations(all_cards, 5))

    best = None
    for combo in combos:
        evaluated = evaluate5(combo)
        if best is None or compare_hands(evaluated, best) > 0:
            best = evaluated
    return best


def normalized_strength(class_rank, kickers):
    """
    Single number, higher = stronger, combining class + full kicker ordering.
    Must exactly match the formula in poker_logic.js's evaluateHand().
    """
    score_frac = 0.0
    for i, k in enumerate(kickers[:5]):
        score_frac += k * (15 ** -(i + 1))
    return ((9 - class_rank) + score_frac) / 9