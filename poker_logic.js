// Poker hand evaluation + feature building, ported from equity.py / build_features.py
// This mirrors the Python logic exactly so the browser predictions match training data.

const RANKS = "23456789TJQKA";
const RANK_VALUES = { '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14 };
const SUITS = ['s', 'h', 'd', 'c'];

function rankOf(cardStr) { return RANK_VALUES[cardStr[0]]; }
function suitOf(cardStr) { return cardStr[1]; }

function buildFullDeck() {
  const deck = [];
  for (const r of RANKS) for (const s of SUITS) deck.push(r + s);
  return deck;
}

// --- exact 5-card hand ranking ---
// Returns { classRank: 1-9 (1=best), kickers: [...] } where kickers is the
// full ordered list of ranks that matter for tiebreaking, most significant first.
// Hand classes: 1 Straight Flush, 2 Four of a Kind, 3 Full House, 4 Flush,
// 5 Straight, 6 Three of a Kind, 7 Two Pair, 8 Pair, 9 High Card
function evaluate5(cards) {
  const ranks = cards.map(rankOf).sort((a, b) => b - a);
  const suits = cards.map(suitOf);
  const isFlush = suits.every(s => s === suits[0]);

  function findStraightHigh(rankList) {
    const withWheel = rankList.includes(14) ? [...rankList, 1] : rankList;
    const sorted = [...new Set(withWheel)].sort((a, b) => b - a);
    for (let i = 0; i <= sorted.length - 5; i++) {
      if (sorted[i] - sorted[i + 4] === 4) return sorted[i];
    }
    return null;
  }
  const straightHigh = findStraightHigh(ranks);

  const rankCounts = {};
  for (const r of ranks) rankCounts[r] = (rankCounts[r] || 0) + 1;
  // sorted by count desc, then rank desc -> gives correct kicker order for free
  const counts = Object.entries(rankCounts)
    .map(([rank, count]) => ({ rank: parseInt(rank), count }))
    .sort((a, b) => b.count - a.count || b.rank - a.rank);
  const orderedRanks = counts.flatMap(c => Array(c.count).fill(c.rank));

  if (isFlush && straightHigh) return { classRank: 1, kickers: [straightHigh] };
  if (counts[0].count === 4) return { classRank: 2, kickers: [counts[0].rank, counts[1].rank] };
  if (counts[0].count === 3 && counts[1].count === 2) return { classRank: 3, kickers: [counts[0].rank, counts[1].rank] };
  if (isFlush) return { classRank: 4, kickers: ranks };
  if (straightHigh) return { classRank: 5, kickers: [straightHigh] };
  if (counts[0].count === 3) return { classRank: 6, kickers: orderedRanks };
  if (counts[0].count === 2 && counts[1].count === 2) {
    return { classRank: 7, kickers: [counts[0].rank, counts[1].rank, counts[2].rank] };
  }
  if (counts[0].count === 2) return { classRank: 8, kickers: orderedRanks };
  return { classRank: 9, kickers: ranks };
}

function combinations(arr, k) {
  const results = [];
  function helper(start, combo) {
    if (combo.length === k) { results.push([...combo]); return; }
    for (let i = start; i < arr.length; i++) {
      combo.push(arr[i]);
      helper(i + 1, combo);
      combo.pop();
    }
  }
  helper(0, []);
  return results;
}

// compare two {classRank, kickers} hands: returns >0 if a better, <0 if b better, 0 if tie
function compareHands(a, b) {
  if (a.classRank !== b.classRank) return b.classRank - a.classRank; // lower classRank = better
  for (let i = 0; i < Math.max(a.kickers.length, b.kickers.length); i++) {
    const ak = a.kickers[i] || 0, bk = b.kickers[i] || 0;
    if (ak !== bk) return ak - bk;
  }
  return 0;
}

// best 5-card hand out of 5, 6, or 7 cards
function evaluateHand(allCards) {
  const fiveCardCombos = allCards.length === 5 ? [allCards] : combinations(allCards, 5);
  let best = null;
  for (const combo of fiveCardCombos) {
    const evaluated = evaluate5(combo);
    if (!best || compareHands(evaluated, best) > 0) best = evaluated;
  }
  // normalized strength for ML feature use only (not used for win/loss comparison).
  // Must exactly match hand_evaluator.py's normalized_strength() — same formula,
  // same kicker weighting — since the model is trained on Python's values.
  let scoreFrac = 0.0;
  for (let i = 0; i < Math.min(best.kickers.length, 5); i++) {
    scoreFrac += best.kickers[i] * Math.pow(15, -(i + 1));
  }
  const normalizedStrength = ((9 - best.classRank) + scoreFrac) / 9;
  return { classRank: best.classRank, kickers: best.kickers, normalizedStrength };
}

// --- feature building (mirrors compute_features() in build_features.py) ---
function computeFeatures(holeCards, boardCards, numOpponents) {
  const r1 = rankOf(holeCards[0]), r2 = rankOf(holeCards[1]);
  const holeHigh = Math.max(r1, r2), holeLow = Math.min(r1, r2);
  const suited = suitOf(holeCards[0]) === suitOf(holeCards[1]) ? 1 : 0;
  const pocketPair = r1 === r2 ? 1 : 0;
  const street = boardCards.length;

  let handClass = 0, normalizedStrength = 0.0;
  if (street >= 3) {
    const result = evaluateHand([...holeCards, ...boardCards]);
    handClass = result.classRank;
    normalizedStrength = result.normalizedStrength;
  }

  // interaction terms — must mirror build_features.py exactly
  const holeHighXOpponents = holeHigh * numOpponents;
  const strengthXOpponents = Math.round(normalizedStrength * numOpponents * 10000) / 10000;

  // must match FEATURE_COLUMNS order in train_model.py / export_web_model.py exactly:
  // hole_high, hole_low, suited, pocket_pair, street, num_opponents, hand_class,
  // normalized_strength, hole_high_x_opponents, strength_x_opponents
  return [
    holeHigh, holeLow, suited, pocketPair, street, numOpponents, handClass,
    normalizedStrength, holeHighXOpponents, strengthXOpponents,
  ];
}

// --- Monte Carlo simulation (ground truth, runs in-browser) ---
function simulateEquity(holeCards, boardCards, numOpponents, numSimulations = 2000) {
  const known = new Set([...holeCards, ...boardCards]);
  const remainingDeck = buildFullDeck().filter(c => !known.has(c));

  let wins = 0, ties = 0, losses = 0;
  const cardsNeeded = 5 - boardCards.length;

  for (let sim = 0; sim < numSimulations; sim++) {
    const deck = [...remainingDeck];
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }

    let idx = 0;
    const fullBoard = [...boardCards, ...deck.slice(idx, idx + cardsNeeded)];
    idx += cardsNeeded;

    const opponentHands = [];
    for (let o = 0; o < numOpponents; o++) {
      opponentHands.push(deck.slice(idx, idx + 2));
      idx += 2;
    }

    const myEval = evaluateHand([...holeCards, ...fullBoard]);

    let bestOpp = null;
    let tiedWithBest = false;
    for (const opp of opponentHands) {
      const oppEval = evaluateHand([...opp, ...fullBoard]);
      if (!bestOpp || compareHands(oppEval, bestOpp) > 0) {
        bestOpp = oppEval;
        tiedWithBest = false;
      } else if (compareHands(oppEval, bestOpp) === 0) {
        tiedWithBest = true;
      }
    }

    const cmp = compareHands(myEval, bestOpp);
    if (cmp > 0) wins++;
    else if (cmp === 0) ties++; // note: doesn't account for 3-way ties among opponents themselves, rare edge case
    else losses++;
  }

  return {
    winPct: Math.round((wins / numSimulations) * 1000) / 10,
    tiePct: Math.round((ties / numSimulations) * 1000) / 10,
    lossPct: Math.round((losses / numSimulations) * 1000) / 10,
  };
}