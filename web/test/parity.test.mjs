// The JS tree walker must give LightGBM's own raw scores, and the capacity curves must give
// the recall and precision in metrics.json. READMIT_WEB_DATA points at an export
// (readmit export --out ...); CI exports from a freshly trained model first.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { makeScorer, calibrate, capacityCurve, roundHalfEven } from "../gbdt.mjs";

const dir = process.env.READMIT_WEB_DATA || "test/data";
const read = (f) => JSON.parse(readFileSync(join(dir, f), "utf8"));
const model = read("model.json");

test("raw scores match LightGBM on exported test rows", () => {
  const { rows, raw } = read("parity.json");
  const score = makeScorer(model);
  let worst = 0;
  rows.forEach((row, i) => { worst = Math.max(worst, Math.abs(score(row) - raw[i])); });
  assert.ok(rows.length >= 100);
  assert.ok(worst < 1e-9, `max abs difference ${worst}`);
});

test("calibration is monotone and stays in [0, 1]", () => {
  const xs = model.calibration.x;
  let prev = -1;
  for (let s = xs[0] - 1; s <= xs.at(-1) + 1; s += 0.05) {
    const p = calibrate(model.calibration, s);
    assert.ok(p >= 0 && p <= 1 && p >= prev - 1e-12);
    prev = p;
  }
});

test("capacity curves reproduce recall and precision in metrics.json", () => {
  const cap = read("capacity.json");
  const results = process.env.READMIT_RESULTS || "../results";
  const metrics = JSON.parse(readFileSync(join(results, "metrics.json"), "utf8")).test;
  assert.deepEqual(Object.keys(cap.ranked_labels).sort(), Object.keys(metrics).sort());
  for (const [name, packed] of Object.entries(cap.ranked_labels)) {
    const at = capacityCurve(packed, cap.n);
    assert.equal(at(1).caught, cap.positives);
    for (const pct of [5, 10, 20]) {
      const want = metrics[name][`top${pct}`];
      const got = at(pct / 100);
      assert.equal(got.flagged, want.flagged, `${name} at ${pct}%`);
      assert.ok(Math.abs(got.caught / got.positives - want.recall) < 1e-12, `${name} recall at ${pct}%`);
      assert.ok(Math.abs(got.caught / got.flagged - want.precision) < 1e-12, `${name} precision at ${pct}%`);
    }
  }
});

test("the call list is in risk order and every row has a reason", () => {
  const list = read("call_list.json");
  assert.ok(list.length > 0);
  list.forEach((row, i) => {
    assert.equal(row.rank, i + 1);
    if (i) assert.ok(row.risk <= list[i - 1].risk + 1e-12);
    assert.ok(row.reasons.length > 0);
  });
});

test("rounding matches Python's round()", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 1974.5, 2.4, 2.6].map(roundHalfEven), [0, 2, 2, 1974, 2, 3]);
});
