// Score LightGBM trees in the browser. Follows LightGBM's own decision rules
// (src/io/tree.cpp: NumericalDecision, CategoricalDecision) so the result matches
// booster.predict(raw_score=True); test/parity.test.mjs checks that on 2,000 test rows.

const ZERO = 1e-35;

export function makeScorer(model) {
  const catIndex = {};
  for (const c of model.categorical) catIndex[c] = new Map(model.categories[c].map((v, i) => [v, i]));

  function encode(row) {
    return model.features.map((f) => {
      if (catIndex[f]) {
        const code = catIndex[f].get(String(row[f]));
        return code === undefined ? -1 : code; // unseen category, like pandas code -1
      }
      const v = row[f];
      return v === null || v === undefined || v === "" ? NaN : Number(v);
    });
  }

  // Trees are compacted by readmit.export: {f, t, l, r, c?, dl?, m?}; a leaf is a number.
  function goesLeft(node, value) {
    const missing = node.m || "None";
    if (node.c) {
      if (Number.isNaN(value)) {
        if (missing === "NaN") return false;
        value = 0;
      }
      const v = Math.trunc(value);
      if (v < 0) return false;
      return String(node.t).split("||").some((t) => Number(t) === v);
    }
    let fval = value;
    if (Number.isNaN(fval) && missing !== "NaN") fval = 0;
    if ((missing === "Zero" && Math.abs(fval) <= ZERO) || (missing === "NaN" && Number.isNaN(fval))) return !!node.dl;
    return fval <= node.t;
  }

  function walk(node, x) {
    while (typeof node !== "number") node = goesLeft(node, x[node.f]) ? node.l : node.r;
    return node;
  }

  return (row) => {
    const x = encode(row);
    let s = 0;
    for (const t of model.trees) s += walk(t, x);
    return s;
  };
}

// sklearn's IsotonicRegression.predict: linear interpolation between thresholds, clipped.
export function calibrate(table, score) {
  const { x, y } = table;
  if (score <= x[0]) return y[0];
  if (score >= x[x.length - 1]) return y[y.length - 1];
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= score) lo = mid; else hi = mid;
  }
  const t = (score - x[lo]) / (x[hi] - x[lo]);
  return y[lo] + t * (y[hi] - y[lo]);
}

// Python's round(): halves go to the even neighbor. 10% of 19,745 is 1974.5, which Python
// rounds to 1974 and Math.round to 1975.
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// Readmissions reached when calling the top `share` of discharges, from labels packed in score order.
export function capacityCurve(packed, n) {
  const bytes = Uint8Array.from(atob(packed), (c) => c.charCodeAt(0));
  const cum = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) cum[i + 1] = cum[i] + ((bytes[i >> 3] >> (7 - (i & 7))) & 1);
  return (share) => {
    const k = Math.min(n, Math.max(1, roundHalfEven(share * n)));
    return { flagged: k, caught: cum[k], positives: cum[n] };
  };
}
