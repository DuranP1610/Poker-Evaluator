"""
Core equity simulator.

Given the hole cards, the board cards (0-5 of them), and how many
opponents you're facing, estimate your win probability by dealing out
random cards thousands of times and counting outcomes.

"""

import random
from treys import Card, Deck, Evaluator

evaluator = Evaluator()


def make_deck_excluding(known_cards):
    """Full 52-card deck minus any cards already dealt (hole + board)."""
    full_deck = Deck.GetFullDeck()
    known_set = set(known_cards)
    return [c for c in full_deck if c not in known_set]


def simulate_equity(hole_cards, board_cards, num_opponents, num_simulations=5000):
    """
    hole_cards: list of 2 treys card ints, e.g. [Card.new('Ah'), Card.new('Kd')]
    board_cards: list of 0-5 treys card ints (flop/turn/river so far)
    num_opponents: how many opponents you're up against
    num_simulations: how many random deals to run

    Returns dict with win/tie/loss probabilities.
    """
    known = hole_cards + board_cards
    remaining_deck = make_deck_excluding(known)

    wins = 0
    ties = 0
    losses = 0

    cards_needed_for_board = 5 - len(board_cards)

    for _ in range(num_simulations):
        deck_copy = remaining_deck[:]
        random.shuffle(deck_copy)

        draw_idx = 0

        # complete the board
        full_board = board_cards + deck_copy[draw_idx: draw_idx + cards_needed_for_board]
        draw_idx += cards_needed_for_board

        # deal random hole cards to each opponent
        opponent_hands = []
        for _ in range(num_opponents):
            opp_hand = deck_copy[draw_idx: draw_idx + 2]
            draw_idx += 2
            opponent_hands.append(opp_hand)

        # evaluate our hand (lower score = better in treys)
        my_score = evaluator.evaluate(full_board, hole_cards)
        opp_scores = [evaluator.evaluate(full_board, opp) for opp in opponent_hands]

        best_opp_score = min(opp_scores)

        if my_score < best_opp_score:
            wins += 1
        elif my_score == best_opp_score:
            ties += 1
        else:
            losses += 1

    total = num_simulations
    return {
        "win_pct": round(100 * wins / total, 2),
        "tie_pct": round(100 * ties / total, 2),
        "loss_pct": round(100 * losses / total, 2),
    }

