"""
analyze/sample_stratify.py — 60-pair stratified sampler for the human-eval study.

Per concept-note §4.6, the 60 pairs sent to annotators come from three
buckets of 20 each, designed to put annotators on the MOST INFORMATIVE
cases (not a uniform random sample):

    Bucket A — FRONTIER (n=20)
        Closure metric strongly claims success but the configuration is
        deliberately mismatched (cells H4, H5, N1) or has otherwise
        ambiguous closure. These are the cases where the automated
        metric is most likely to be fooled.

    Bucket B — AGREEMENT (n=20)
        Closure metric AND judge LLM both say success, with monotonic
        path-rate consistency. Sanity baseline — should be unanimous
        among annotators.

    Bucket C — DISPUTED (n=20)
        Judge LLM disagrees with the automated metric (judge < 3 but
        metric_value high, OR judge ≥ 3 but metric_value low). The
        most informative cases.

The sampler is deterministic (seeded) so the same DataFrame produces
the same 60 pairs across runs.
"""

from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Optional

# Make project root importable whether run directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analyze.load_results import filter_valid


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
N_PER_BUCKET: int = 20
TOTAL_PAIRS: int = 60
SAMPLER_SEED: int = 4242

FRONTIER_CELLS: set[str] = {"H4", "H5", "N1", "N2", "N3"}


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def stratify_60_pairs(df: pd.DataFrame,
                      n_per_bucket: int = N_PER_BUCKET,
                      seed: int = SAMPLER_SEED) -> pd.DataFrame:
    """
    Sample 20 frontier + 20 agreement + 20 disputed pairs. Returns a
    DataFrame with one row per pair (sample_idx + cell_id + path uniquely
    identifies the artefact triple).

    Each row has a `bucket` column ('frontier' / 'agreement' / 'disputed')
    and a `pair_id` for ordering in the worksheet.
    """
    valid = filter_valid(df)
    if valid.empty:
        return pd.DataFrame(columns=list(df.columns) + ["bucket", "pair_id"])

    rng = random.Random(seed)

    # --- Bucket A: FRONTIER ---
    frontier_pool = valid[
        valid["cell_id"].isin(FRONTIER_CELLS) &
        valid["closure_success"]
    ]
    frontier = _sample_bucket(frontier_pool, n_per_bucket, rng, "frontier")

    # --- Bucket B: AGREEMENT ---
    agree_pool = valid[
        valid["closure_success"] &
        (valid["judge_rating"] >= 3) &
        ~valid["cell_id"].isin(FRONTIER_CELLS)
    ]
    agreement = _sample_bucket(agree_pool, n_per_bucket, rng, "agreement")

    # --- Bucket C: DISPUTED ---
    disputed_high = valid[
        valid["closure_success"] &
        (valid["judge_rating"].notna()) &
        (valid["judge_rating"] >= 0) &
        (valid["judge_rating"] < 3)
    ]
    disputed_low = valid[
        ~valid["closure_success"] &
        (valid["judge_rating"] >= 3)
    ]
    disputed_pool = pd.concat([disputed_high, disputed_low], ignore_index=True)
    disputed = _sample_bucket(disputed_pool, n_per_bucket, rng, "disputed")

    out = pd.concat([frontier, agreement, disputed], ignore_index=True)
    out["pair_id"] = [f"P{i:03d}" for i in range(1, len(out) + 1)]
    return out


def _sample_bucket(pool: pd.DataFrame, n: int, rng: random.Random,
                   bucket_name: str) -> pd.DataFrame:
    """Random sample of n rows from pool. If pool < n, takes everything."""
    if pool.empty:
        empty = pool.head(0).copy()
        empty["bucket"] = pd.Series(dtype=str)
        return empty
    indices = list(pool.index)
    rng.shuffle(indices)
    taken = indices[:n]
    sub = pool.loc[taken].copy()
    sub["bucket"] = bucket_name
    return sub


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analyze.load_results import synthesize_fake_data

    df = synthesize_fake_data(n_samples=30)
    pairs = stratify_60_pairs(df)
    print(f"Selected {len(pairs)} pairs:")
    print(pairs.groupby("bucket").size().to_string())
    print("\nFirst 5 rows:")
    print(pairs.head(5)[["pair_id", "bucket", "cell_id", "sample_idx", "path",
                          "metric_value", "judge_rating"]].to_string(index=False))
