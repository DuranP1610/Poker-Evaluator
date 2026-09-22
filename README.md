# Live Poker Equity Calculator

A live poker hand equity calculator that combines a Monte Carlo simulator with a trained machine learning model, running entirely in the browser with no backend server.

**[Live demo →](#)** <!-- add your GitHub Pages link here once deployed -->

## What it does

Pick your two hole cards, add the board as it's revealed (flop, turn, river), set how many opponents you're facing, and get:

- **Raw Statistics** — a live Monte Carlo simulation (thousands of randomly dealt outcomes), computed on the spot in the browser.
- **ML model prediction** — an instant estimate from a Random Forest model trained on hundreds of thousands of simulated hands, for comparison against the exact statistical answer.

## Why both?

The Monte Carlo simulation is exact but takes a moment to run. The ML model is instant but is an *approximation* learned from data. Showing both side by side is deliberate — it lets you see where the model's predictions hold up well, and where they don't. See [Known limitations](#known-limitations) below for a genuine gap this surfaced.

## Architecture

```
equity.py              Core Monte Carlo simulator (Python + treys)
generate_dataset.py    Loops the simulator over thousands of random hands
                        -> poker_dataset.csv
hand_evaluator.py       Custom 5/6/7-card hand evaluator (mirrors poker_logic.js
                        exactly, so Python and JS always agree)
build_features.py      Converts raw hands into ML-ready numeric features
                        -> poker_features.csv
train_model.py          Trains + compares Linear Regression vs Gradient Boosting
export_web_model.py    Trains a compact Random Forest and exports it to
                        JavaScript (via m2cgen) -> model.js
poker_logic.js          In-browser hand evaluator, feature builder, and
                        Monte Carlo simulator (JS port of the Python pipeline)
index.html              The static front-end — no backend, no build step
```

**Everything client-side.** `model.js` and `poker_logic.js` run entirely in the visitor's browser. There's no API, no server, and no user data ever leaves the page.

## The pipeline

1. **Simulate** — deal random hands thousands of times per scenario to get ground-truth win probabilities (`equity.py`, `generate_dataset.py`).
2. **Engineer features** — hole card ranks, suited/paired flags, postflop hand strength (via a custom hand evaluator), and interaction terms like `hand_strength × num_opponents` (`build_features.py`).
3. **Train** — compare Linear Regression against Gradient Boosting / Random Forest; keep the best performer (`train_model.py`).
4. **Export** — convert the trained scikit-learn model into plain JavaScript (`export_web_model.py`, using [m2cgen](https://github.com/BayesWitnesses/m2cgen)), so predictions run instantly with zero server round-trip.
5. **Deploy** — a single static `index.html`, publishable for free on GitHub Pages.

## Model performance

Trained on 30,000 simulated hands across all streets (preflop through river):

| Model | MAE (percentage points) | R² |
|---|---|---|
| Linear Regression | ~10.9 | 0.58 |
| Gradient Boosting | ~6.8 | 0.77 |
| Random Forest (deployed, compact) | ~7.1 | 0.76 |

The deployed model trades a small amount of accuracy for a ~10x smaller file size, so it loads quickly in the browser.

## Known limitations

The model's features describe the strength of *your* hand, but not the danger of the *board*. On a board like K♠ Q♠ J♠ (three cards of the same suit), a strong made hand is worth less than the model predicts, because many opponent hands now have live flush equity — something the Monte Carlo simulation captures automatically (it deals real random opponent hands) but the ML model's current features don't represent at all.

This gap is real and measurable — a deeper analysis, comparing the model against ground-truth simulation across different board textures, is documented separately in `notebooks/`.

## Running locally

```bash
git clone https://github.com/DuranP1610/Poker-Evaluator.git
cd Poker-Evaluator
python3 -m http.server 8000
```

Open `http://localhost:8000`.

To regenerate the dataset or retrain the model, you'll need Python 3 with `pandas`, `scikit-learn`, `treys`, `joblib`, and `m2cgen`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn treys joblib m2cgen

python3 generate_dataset.py     # ~15-20 min for 30,000 rows
python3 build_features.py
python3 train_model.py
python3 export_web_model.py
```

## Tech stack

Python (scikit-learn, pandas, treys, m2cgen) for the data pipeline · vanilla JavaScript for the in-browser model and simulator · plain HTML/CSS for the front-end — no frameworks, no build step, no backend.

## Future work

- Board-texture features (flush draw, straight draw, paired board possibilities) to close the gap described above.
- Opponent range modeling, rather than assuming purely random opponent hands.
- A "replay a famous hand" mode showing win probability evolve street-by-street.