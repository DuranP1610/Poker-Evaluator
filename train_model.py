"""
Train the win-probability regression model.

Loads poker_features.csv, splits into train/test, fits two models
(a simple linear baseline and a gradient-boosted tree), and compares
them so we have evidence for which one to ship — not just a guess.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_FILE = "poker_features.csv"
MODEL_OUTPUT = "equity_model.joblib"

FEATURE_COLUMNS = [
    "hole_high",
    "hole_low",
    "suited",
    "pocket_pair",
    "street",
    "num_opponents",
    "hand_class",
    "normalized_strength",
    "hole_high_x_opponents",
    "strength_x_opponents",
]
TARGET_COLUMN = "win_pct"


def load_data():
    df = pd.read_csv(INPUT_FILE)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE = {mae:.2f} percentage points, R^2 = {r2:.3f}")
    return mae, r2


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}\n")

    # baseline: linear regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_mae, lin_r2 = evaluate("Linear Regression", lin_model, X_test, y_test)

    # main model: gradient boosting (captures non-linear interactions,
    # e.g. how 'suited' matters differently depending on 'street')
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_mae, gb_r2 = evaluate("Gradient Boosting", gb_model, X_test, y_test)

    # ship whichever performed better
    if gb_mae < lin_mae:
        print("\nGradient Boosting wins — saving that model.")
        best_model = gb_model
    else:
        print("\nLinear Regression wins — saving that model.")
        best_model = lin_model

    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"Saved to {MODEL_OUTPUT}")

    # quick sanity check: pocket aces preflop vs 1 opponent should predict ~85%
    sanity_row = pd.DataFrame([{
        "hole_high": 14, "hole_low": 14, "suited": 0, "pocket_pair": 1,
        "street": 0, "num_opponents": 1, "hand_class": 0, "normalized_strength": 0.0,
        "hole_high_x_opponents": 14, "strength_x_opponents": 0.0,
    }])
    pred = best_model.predict(sanity_row)[0]
    print(f"\nSanity check — Pocket Aces preflop vs 1 opponent: predicted {pred:.1f}% (expect ~85%)")