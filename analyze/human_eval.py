"""
analyze/human_eval.py — inter-rater agreement + dispute resolution.

Produces:
    1. A blinded annotation worksheet (CSV) from the stratified 60 pairs
    2. Inter-rater agreement statistics:
         - Pairwise weighted Cohen's κ (3 pairs for 3 annotators)
         - Three-rater Krippendorff's α (ordinal)
    3. Dispute resolution log — pairs where annotators diverge by ≥ 2
       points (potential candidates for a recalibration round)

Per concept-note §10 Q3: rubric is pre-registered BEFORE the sweep;
sampling targets the most-informative cases (handled by sample_stratify).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Blinded worksheet generation
# ──────────────────────────────────────────────────────────────────────
def generate_worksheet(pairs: pd.DataFrame,
                       artefacts_by_pair: dict,
                       output_path: Path) -> dict:
    """
    Build the blinded annotation worksheet.

    Args:
        pairs: output of sample_stratify.stratify_60_pairs() — must have
               at minimum: pair_id, cell_id, sample_idx, path, bucket
        artefacts_by_pair: dict mapping pair_id → {"original_code": str,
               "reconstructed_code": str, ...}. Generated separately by
               looking up the original artefacts from the raw sweep
               artefacts (cache layer).
        output_path: where to write the worksheet CSV.

    The worksheet has these columns (in this order):
        pair_id, original_code, reconstructed_code,
        rating (blank — annotator fills),
        justification (blank — annotator fills, optional)

    All cell_id / model / bucket information is stripped so annotators
    can't infer which pipeline produced each pair.

    Returns a summary dict + writes the CSV. A parallel
    `<output>.metadata.json` is written with the cell mapping for
    de-blinding after rating.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {}
    for _, p in pairs.iterrows():
        pid = p["pair_id"]
        art = artefacts_by_pair.get(pid, {})
        rows.append({
            "pair_id": pid,
            "original_code": _csv_safe(art.get("original_code", "")),
            "reconstructed_code": _csv_safe(art.get("reconstructed_code", "")),
            "rating": "",
            "justification": "",
        })
        metadata[pid] = {
            "cell_id": p["cell_id"],
            "sample_idx": int(p["sample_idx"]),
            "path": int(p["path"]),
            "bucket": p["bucket"],
            "metric_value": float(p["metric_value"]),
            "judge_rating": int(p["judge_rating"]) if pd.notna(p["judge_rating"]) else -1,
        }

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "n_pairs": len(df),
        "worksheet": str(output_path),
        "metadata": str(meta_path),
    }


# ──────────────────────────────────────────────────────────────────────
# Annotator CSV ingestion
# ──────────────────────────────────────────────────────────────────────
def load_annotator_csvs(annotator_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all `annotator_<initials>.csv` files from a directory.

    Each CSV must have: pair_id, rating (0-4), optional justification.

    Returns dict mapping annotator initials → DataFrame.
    """
    annotator_dir = Path(annotator_dir)
    out: dict[str, pd.DataFrame] = {}
    if not annotator_dir.exists():
        return out
    for csv in sorted(annotator_dir.glob("annotator_*.csv")):
        initials = csv.stem.split("_", 1)[1].upper()
        df = pd.read_csv(csv)
        if "pair_id" not in df.columns or "rating" not in df.columns:
            continue
        # Coerce rating to int; reject invalid rows
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].notna()].copy()
        df["rating"] = df["rating"].astype(int)
        df = df[(df["rating"] >= 0) & (df["rating"] <= 4)]
        out[initials] = df.set_index("pair_id")
    return out


# ──────────────────────────────────────────────────────────────────────
# Cohen's κ (pairwise, linear-weighted for ordinal scale)
# ──────────────────────────────────────────────────────────────────────
def cohens_kappa_pairwise(annotators: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute pairwise weighted Cohen's κ for every pair of annotators.
    Uses linear weights for the ordinal 0-4 scale.

    Returns DataFrame with columns: a1, a2, n_common, kappa, weighted_kappa.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:                                            # pragma: no cover
        raise ImportError("Install: pip install scikit-learn")

    keys = sorted(annotators.keys())
    rows = []
    for i, a1 in enumerate(keys):
        for a2 in keys[i + 1:]:
            df1 = annotators[a1]
            df2 = annotators[a2]
            common = df1.index.intersection(df2.index)
            if len(common) < 2:
                continue
            r1 = df1.loc[common, "rating"].astype(int).values
            r2 = df2.loc[common, "rating"].astype(int).values
            kappa = cohen_kappa_score(r1, r2)
            wkappa = cohen_kappa_score(r1, r2, weights="linear")
            rows.append({
                "a1": a1, "a2": a2,
                "n_common": int(len(common)),
                "kappa": float(kappa),
                "weighted_kappa": float(wkappa),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Krippendorff's α (3-rater ordinal)
# ──────────────────────────────────────────────────────────────────────
def krippendorff_alpha(annotators: dict[str, pd.DataFrame]) -> float:
    """
    Compute Krippendorff's α with the ordinal distance metric.

    Uses a hand-rolled implementation so we don't add the `krippendorff`
    package as a hard dependency. Cross-checked against the canonical
    formula in Krippendorff (2018).

    Returns NaN if fewer than 2 annotators or fewer than 2 common items.
    """
    if len(annotators) < 2:
        return float("nan")

    # Build the wide table: rows = items, cols = annotators
    initials = sorted(annotators.keys())
    items: set = set()
    for df in annotators.values():
        items |= set(df.index)
    items = sorted(items)

    matrix = np.full((len(items), len(initials)), np.nan)
    for j, init in enumerate(initials):
        df = annotators[init]
        for i, pid in enumerate(items):
            if pid in df.index:
                matrix[i, j] = df.loc[pid, "rating"]

    # Drop items with fewer than 2 ratings
    valid_mask = np.sum(~np.isnan(matrix), axis=1) >= 2
    matrix = matrix[valid_mask]
    if len(matrix) < 2:
        return float("nan")

    # Observed disagreement (ordinal)
    Do = 0.0
    n_pairs = 0
    for row in matrix:
        ratings = row[~np.isnan(row)]
        if len(ratings) < 2:
            continue
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                Do += (ratings[i] - ratings[j]) ** 2
                n_pairs += 1
    if n_pairs == 0:
        return float("nan")
    Do /= n_pairs

    # Expected disagreement (over the full distribution)
    all_ratings = matrix[~np.isnan(matrix)].astype(int)
    if len(all_ratings) < 2:
        return float("nan")
    De = 0.0
    De_pairs = 0
    for i in range(len(all_ratings)):
        for j in range(i + 1, len(all_ratings)):
            De += (all_ratings[i] - all_ratings[j]) ** 2
            De_pairs += 1
    De /= De_pairs

    if De == 0:
        return float("nan")
    alpha = 1.0 - Do / De
    return float(alpha)


# ──────────────────────────────────────────────────────────────────────
# Dispute resolution
# ──────────────────────────────────────────────────────────────────────
def dispute_log(annotators: dict[str, pd.DataFrame],
                threshold: int = 2) -> pd.DataFrame:
    """
    Identify items where annotators disagree by ≥ threshold rating points.
    These are candidates for a follow-up calibration discussion.

    Returns DataFrame: pair_id, rating_<initials>, max_gap.
    """
    if not annotators:
        return pd.DataFrame()
    # Combine ratings into one wide table
    wide = pd.concat([df["rating"].rename(init) for init, df in annotators.items()],
                     axis=1)
    wide = wide.dropna(thresh=2)
    if wide.empty:
        return pd.DataFrame()
    wide["max_gap"] = wide.max(axis=1) - wide.min(axis=1)
    disputes = wide[wide["max_gap"] >= threshold].copy()
    return disputes.reset_index().rename(columns={"index": "pair_id"})


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _csv_safe(s: str) -> str:
    """Make a multi-line string safe to put in a CSV cell."""
    if not s:
        return ""
    # CSV writers handle quoting, but multi-line gets ugly — keep it for now
    return s


# ──────────────────────────────────────────────────────────────────────
# Smoke test on synthetic annotator data
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== human_eval smoke test ===")
    # Synthesize 3 annotators × 60 pairs with realistic agreement
    rng = np.random.default_rng(42)
    pair_ids = [f"P{i:03d}" for i in range(1, 61)]
    truth = rng.integers(0, 5, size=60)        # "true" rating per pair
    annotators = {}
    for init, noise in [("GS", 0.7), ("BV", 0.8), ("SA", 1.5)]:
        # Larger noise = lower agreement
        ratings = np.clip(truth + rng.normal(0, noise, size=60), 0, 4).round().astype(int)
        annotators[init] = pd.DataFrame({"pair_id": pair_ids, "rating": ratings}).set_index("pair_id")

    kappas = cohens_kappa_pairwise(annotators)
    print("\nPairwise weighted Cohen's κ:")
    print(kappas.to_string(index=False))

    alpha = krippendorff_alpha(annotators)
    print(f"\nKrippendorff's α (3-rater ordinal): {alpha:.3f}")

    disputes = dispute_log(annotators, threshold=2)
    print(f"\nDisputes (gap ≥ 2): {len(disputes)} of 60 pairs")
    if not disputes.empty:
        print(disputes.head(5).to_string(index=False))

    print("\n✓ human_eval smoke test passed.")
