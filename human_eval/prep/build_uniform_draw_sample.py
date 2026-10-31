"""
human_eval/prep/build_uniform_draw_sample.py

R1 Comment 2 + R2 Comment 2 extension — draw a SECOND 60-pair human-eval
sample uniformly at random from the 5,890-row sweep, complementary to
the original stratified-by-disagreement sample.

Why uniform (not stratified again): the original 60-pair sample was
deliberately stratified across three disagreement buckets (20 frontier
+ 20 agreement + 20 disputed) to maximise power at characterising
WHERE judge and metric diverge. That design is correct for its
inferential target but cannot support a population-level claim about
overall judge-human agreement. A uniform draw gives a direct
population estimate (±13pp binomial CI at n=60), complementing — not
replacing — the stratified result.

Reads:
    results/results_roundtrip.tsv       — the 5,890-row sweep
    results/human_eval_pairs_60.tsv     — original 60 pairs (for exclusion)
    data/core_sample_150.jsonl           — original code + docstring (for artefact lookup)

Writes:
    results/human_eval_pairs_60_uniform.tsv    — 60 new pairs (uniform draw)
    results/human_eval_worksheet_60_uniform.csv — blank rating template
    human_eval/data/pairs_60_uniform_full.jsonl — JSONL with original + reconstructed artefacts

Seed: 20261031  (today's date — deterministic reproducibility)

Run: python3 human_eval/prep/build_uniform_draw_sample.py
"""
from __future__ import annotations

import csv
import json
import random
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_TSV = PROJECT_ROOT / "results" / "results_roundtrip.tsv"
STRATIFIED_PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60.tsv"
SAMPLES_JSONL = PROJECT_ROOT / "data" / "core_sample_150.jsonl"

OUT_PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60_uniform.tsv"
OUT_WORKSHEET_CSV = PROJECT_ROOT / "results" / "human_eval_worksheet_60_uniform.csv"

SEED = 20261031
TARGET_N = 60


def load_sweep() -> list[dict]:
    with SWEEP_TSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_stratified_tuples() -> set[tuple[str, int, int]]:
    """Load (cell_id, sample_idx, path) tuples already in the stratified sample
    so we can exclude them from the uniform draw."""
    seen = set()
    if not STRATIFIED_PAIRS_TSV.exists():
        print(f"  warning: {STRATIFIED_PAIRS_TSV} not found — no exclusions applied")
        return seen
    with STRATIFIED_PAIRS_TSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seen.add((r["cell_id"], int(r["sample_idx"]), int(r["path"])))
    return seen


def main() -> None:
    rows = load_sweep()
    print(f"Loaded {len(rows)} sweep rows")

    excluded = load_stratified_tuples()
    print(f"Excluding {len(excluded)} rows from the stratified sample")

    # Keep only rows that have (a) a valid metric and (b) a judge rating
    # (ratings-eligible pool). Exclude null cells (N1/N2/N3) since those
    # are controls and not representative of pipeline closure.
    eligible = []
    for r in rows:
        if r["cell_id"] in ("N1", "N2", "N3"):
            continue
        if r.get("valid", "").strip().lower() != "true":
            continue
        try:
            mv = float(r["metric_value"])
            jr = int(round(float(r["judge_rating"])))
        except (ValueError, TypeError):
            continue
        if math.isnan(mv) or jr < 0:
            continue
        key = (r["cell_id"], int(r["sample_idx"]), int(r["path"]))
        if key in excluded:
            continue
        eligible.append(r)
    print(f"Eligible pool: {len(eligible)} rows (excludes null cells, invalid rows, "
          f"stratified sample members)")

    # Draw TARGET_N uniformly at random
    rng = random.Random(SEED)
    sampled = rng.sample(eligible, TARGET_N)
    print(f"Sampled {len(sampled)} rows (seed={SEED})")

    # Verify no bucket collapse
    paths = [int(r["path"]) for r in sampled]
    cells = [r["cell_id"] for r in sampled]
    from collections import Counter
    print(f"\nSample path distribution:  {dict(Counter(paths))}")
    print(f"Sample cell distribution:  {dict(sorted(Counter(cells).items()))}")

    # Write the new pairs TSV with same column schema as the stratified one
    # + an added "uniform_pair_id" so worksheets don't collide with the
    # original P001-P060 IDs.
    fieldnames = list(sampled[0].keys()) + ["uniform_pair_id"]
    OUT_PAIRS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PAIRS_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for i, row in enumerate(sampled, 1):
            row = dict(row)
            row["uniform_pair_id"] = f"U{i:03d}"   # U001..U060
            w.writerow(row)
    print(f"\nWrote {OUT_PAIRS_TSV}")

    # Write the blank worksheet CSV for annotators
    with OUT_WORKSHEET_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "original_code", "reconstructed_code", "rating", "justification"])
        for i in range(1, TARGET_N + 1):
            w.writerow([f"U{i:03d}", "", "", "", ""])
    print(f"Wrote {OUT_WORKSHEET_CSV}")

    print(f"\nNext steps:")
    print(f"  1. Run human_eval/prep/build_worksheet.py (adapted for uniform set)")
    print(f"     to populate the original_code + reconstructed_code columns")
    print(f"  2. Share the populated worksheet with BV, GS, AM")
    print(f"  3. Each annotator fills the `rating` (0-4) and `justification` columns")
    print(f"  4. Save as: human_eval/data/annotator_{{BV,GS,AM}}_uniform.csv")
    print(f"  5. Run: python3 human_eval/analysis/compute_uniform_agreement.py")


if __name__ == "__main__":
    main()
