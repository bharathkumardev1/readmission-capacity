// The JS tree walker must give LightGBM's own raw scores. READMIT_WEB_DATA points at an
// export (readmit export --out ...); CI exports from a freshly trained model first.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { makeScorer, calibrate, capacityCurve } from "../gbdt.mjs";

const dir = process.env.READMIT_WEB_DATA || "data";
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

test("capacity curve agrees with the totals", () => {
  const cap = read("capacity.json");
  for (const packed of Object.values(cap.ranked_labels)) {
    const at = capacityCurve(packed, cap.n);
    assert.equal(at(1).caught, cap.positives);
    assert.ok(at(0.1).caught <= at(0.2).caught);
  }
});
