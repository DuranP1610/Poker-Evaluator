"""
Run this LOCALLY, using your real poker_features.csv (the one built from
your 30,000-row dataset) — not the small sandbox version.

Trains a compact RandomForest (small enough to export to JS cleanly) and
writes model.js for the static site.

"""

import pandas as pd
import joblib
import m2cgen as m2c
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_FILE = "poker_features.csv"
MODEL_OUTPUT = "equity_model_web.joblib"
JS_OUTPUT = "model.js"

FEATURE_COLUMNS = [
    "hole_high", "hole_low", "suited", "pocket_pair",
    "street", "num_opponents", "hand_class", "normalized_strength",
    "hole_high_x_opponents", "strength_x_opponents",
]
TARGET_COLUMN = "win_pct"

# kept small enough that the exported JS file stays a few MB, not 17MB+
N_ESTIMATORS = 50
MAX_DEPTH = 8

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Web model — MAE: {mae:.2f} percentage points, R^2: {r2:.3f}")

    # sanity check
    sanity_row = pd.DataFrame([{
        "hole_high": 14, "hole_low": 14, "suited": 0, "pocket_pair": 1,
        "street": 0, "num_opponents": 1, "hand_class": 0, "normalized_strength": 0.0,
        "hole_high_x_opponents": 14, "strength_x_opponents": 0.0,
    }])
    pred = model.predict(sanity_row)[0]
    print(f"Sanity check — Pocket Aces preflop vs 1 opponent: {pred:.1f}% (expect ~85%)")

    joblib.dump(model, MODEL_OUTPUT)

    code = m2c.export_to_javascript(model)
    with open(JS_OUTPUT, "w") as f:
        f.write("// Auto-generated from a trained RandomForestRegressor via m2cgen\n")
        f.write("function score(input) {\n")
        f.write(code[code.index("{") + 1: code.rindex("}")])
        f.write("\n}\n")
        f.write("function predictWinPct(featureArray) { return Math.max(0, Math.min(100, score(featureArray))); }\n")

    import os
    size_mb = os.path.getsize(JS_OUTPUT) / (1024 * 1024)
    print(f"\nWrote {JS_OUTPUT} ({size_mb:.2f} MB)")