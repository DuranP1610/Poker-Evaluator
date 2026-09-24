"""
Training data generator.

Loops the Monte Carlo equity simulator (equity.py) over thousands of
random (hole cards, board, street, num_opponents) combinations to build
a labeled dataset: features in, simulated win probability out.
"""

import random
import csv
import time
from treys import Card, Deck
from Equity import simulate_equity


RANKS = "23456789TJQKA"
SUITS = "shdc"

# --- tunables ---
NUM_SAMPLES = 30000         # how many training rows to generate
SIMS_PER_SAMPLE = 500       # Monte Carlo trials per row (accuracy vs speed)
MAX_OPPONENTS = 8
OUTPUT_FILE = "poker_dataset.csv"


def random_card_pool():
    """Fresh shuffled 52-card deck as treys ints."""
    deck = Deck.GetFullDeck()
    random.shuffle(deck)
    return deck


def card_to_str(card_int):
    """treys card int -> readable string like 'Ah'."""
    return Card.int_to_str(card_int)


def sample_one_row():
    """Generate one random (hole cards, board, street, opponents) scenario."""
    pool = random_card_pool()

    hole = pool[0:2]

    # pick a random street: 0=preflop, 3=flop, 4=turn, 5=river
    street_options = [0, 3, 4, 5]
    num_board_cards = random.choice(street_options)
    board = pool[2: 2 + num_board_cards]

    num_opponents = random.randint(1, MAX_OPPONENTS)

    return hole, board, num_opponents


def generate_dataset():
    rows = []
    start = time.time()
    SEED = 42
    random.seed(SEED)

    for i in range(NUM_SAMPLES):
        hole, board, num_opponents = sample_one_row()

        result = simulate_equity(
            hole, board, num_opponents, num_simulations=SIMS_PER_SAMPLE
        )

        row = {
            "hole_card_1": card_to_str(hole[0]),
            "hole_card_2": card_to_str(hole[1]),
            "board_card_1": card_to_str(board[0]) if len(board) > 0 else "",
            "board_card_2": card_to_str(board[1]) if len(board) > 1 else "",
            "board_card_3": card_to_str(board[2]) if len(board) > 2 else "",
            "board_card_4": card_to_str(board[3]) if len(board) > 3 else "",
            "board_card_5": card_to_str(board[4]) if len(board) > 4 else "",
            "street": len(board),  # 0, 3, 4, or 5
            "num_opponents": num_opponents,
            "win_pct": result["win_pct"],
            "tie_pct": result["tie_pct"],
            "loss_pct": result["loss_pct"],
        }
        rows.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  {i + 1}/{NUM_SAMPLES} rows done ({elapsed:.1f}s elapsed)")

    return rows


def save_to_csv(rows, filename):
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} rows, {SIMS_PER_SAMPLE} sims each...")
    rows = generate_dataset()
    save_to_csv(rows, OUTPUT_FILE)
    print(f"\nDone. Saved to {OUTPUT_FILE}")
    print(f"Sample row: {rows[0]}")