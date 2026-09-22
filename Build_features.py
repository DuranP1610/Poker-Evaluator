"""
Feature engineering.

Reads the raw dataset (card strings + street + opponents + simulated
equity) produced by generate_dataset.py, and converts it into numeric
features an ML model can actually learn from.

Key idea: preflop, all we know is our two hole cards. Postflop, we can
use treys' own hand evaluator to get an objective hand-strength score
and category (pair, flush, etc.) — a much richer signal than raw cards.
"""

import csv
from treys import Card, Evaluator

evaluator = Evaluator()

RANK_MAP = {r: i + 2 for i, r in enumerate("23456789TJQKA")}  # '2'->2 ... 'A'->14

INPUT_FILE = "poker_dataset.csv"
OUTPUT_FILE = "poker_features.csv"


def rank_of(card_str):
    """'Ah' -> 14, 'Th' -> 10, etc."""
    return RANK_MAP[card_str[0]]


def suit_of(card_str):
    return card_str[1]


def parse_board(row):
    """Collect the 0/3/4/5 board card strings present in a row."""
    board = []
    for i in range(1, 6):
        c = row[f"board_card_{i}"]
        if c:
            board.append(c)
    return board

def compute_features(hole_strs, board_strs, num_opponents):
   
    r1, r2 = rank_of(hole_strs[0]), rank_of(hole_strs[1])
    hole_high, hole_low = max(r1, r2), min(r1, r2)
    suited = int(suit_of(hole_strs[0]) == suit_of(hole_strs[1]))
    pocket_pair = int(r1 == r2)
 
    street = len(board_strs)
 
    # postflop: use treys evaluator for objective hand strength
    if street >= 3:
        hole_ints = [Card.new(c) for c in hole_strs]
        board_ints = [Card.new(c) for c in board_strs]
        score = evaluator.evaluate(board_ints, hole_ints)
        hand_class = evaluator.get_rank_class(score)  # 1=straight flush ... 9=high card
        # normalize score to 0-1 
        normalized_strength = 1 - (score / 7462)
    else:
        # preflop: no evaluator score possible, use neutral placeholders
        hand_class = 0
        normalized_strength = 0.0
 
    return {
        "hole_high": hole_high,
        "hole_low": hole_low,
        "suited": suited,
        "pocket_pair": pocket_pair,
        "street": street,
        "num_opponents": num_opponents,
        "hand_class": hand_class,
        "normalized_strength": round(normalized_strength, 4),
    }
 
def build_features(row):
    hole_strs = [row["hole_card_1"], row["hole_card_2"]]
    board_strs = parse_board(row)
    num_opponents = int(row["num_opponents"])
 
    features = compute_features(hole_strs, board_strs, num_opponents)
    features["win_pct"] = float(row["win_pct"])
    features["tie_pct"] = float(row["tie_pct"])
    features["loss_pct"] = float(row["loss_pct"])
    return features


def process_file(input_file, output_file):
    with open(input_file, newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = [build_features(row) for row in reader]

    fieldnames = list(rows[0].keys())
    with open(output_file, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows

if __name__ == "__main__":
    rows = process_file(INPUT_FILE, OUTPUT_FILE)
    print(f"Processed {len(rows)} rows -> {OUTPUT_FILE}")
    print(f"Sample feature row: {rows[0]}")