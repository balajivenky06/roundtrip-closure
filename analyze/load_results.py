"""
analyze/load_results.py — TSV loader + derived-column enrichment.

The raw TSV has the 13 columns written by closure_paths.ClosureResult.
For analysis we need several derived columns:

    - benchmark      : 'humaneval' | 'mbpp' | 'livecodebench' | 'humaneval_mutated'
    - cell_stratum   : 'mono' | 'hetero' | 'null'
    - l_spec / l_test / l_code : Ollama tags (resolved from cell_id via doe.py)
    - l_spec_family etc.       : 'Meta' | 'Microsoft' | 'Alibaba' | ...
    - n_families     : how many distinct families across the 3 stages
    - is_homogeneous : True if all 3 stages share a family
    - closure_success : binary version of metric_value (with per-path thresholds)
    - judge_agrees    : binary — does judge_rating agree with closure_success?

This module ALSO provides `synthesize_fake_data` for testing the
analysis pipeline before real Colab results exist.
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make project root importable so we can read config + doe
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_TSV, PIPELINE_MODELS, JUDGE_MODEL, MODELS_BY_OLLAMA_TAG  # noqa: E402
from doe import ALL_CELLS, CELLS_BY_ID                                              # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Closure-success thresholds (per path)
# ──────────────────────────────────────────────────────────────────────
# A row counts as "closure success" if the path's metric exceeds the
# threshold below. Tuned to the SE convention from Chapter 2:
#   Path 1 (mutation kill rate)     — ≥ 0.70  is "strong test suite"
#   Path 2 (reference pass rate)    — = 1.00  (binary in current impl)
#   Path 3 (BERTScore F1)           — ≥ 0.80  is "semantically close"
CLOSURE_THRESHOLDS: dict[int, float] = {
    1: 0.70,
    2: 1.00,
    3: 0.80,
}


# ──────────────────────────────────────────────────────────────────────
# Public loader
# ──────────────────────────────────────────────────────────────────────
def load_tsv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load `results_roundtrip.tsv` and enrich with derived columns.

    Returns an empty DataFrame with the right schema if the file is
    missing or empty — callers can handle that explicitly.
    """
    if path is None:
        path = RESULTS_TSV
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=_full_schema())

    df = pd.read_csv(path, sep="\t")
    df = _enrich(df)
    return df


def _full_schema() -> list[str]:
    """Column list including derived columns."""
    return [
        "cell_id", "sample_idx", "sample_source", "path",
        "metric_name", "metric_value",
        "judge_rating", "judge_justification",
        "valid", "elapsed_s", "cache_hits", "n_llm_calls", "notes",
        # Derived:
        "benchmark", "cell_stratum",
        "l_spec", "l_test", "l_code",
        "l_spec_family", "l_test_family", "l_code_family",
        "n_families", "is_homogeneous",
        "closure_success", "judge_agrees",
    ]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Operates in-place but returns the DataFrame."""
    # Coerce types. Use native int64 (not pandas nullable Int64) because
    # statsmodels' formula API can't interpret Int64Dtype — it would
    # fail with "Cannot interpret 'Int64Dtype()' as a data type" in
    # anova_type3, mixed_effects_logit, etc. sample_idx and path are
    # row-key fields that should never be NaN; judge_rating uses -1 as
    # the parse-fail sentinel, which is a valid int.
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").fillna(-1).astype("int64")
    df["path"] = pd.to_numeric(df["path"], errors="coerce").fillna(-1).astype("int64")
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["judge_rating"] = pd.to_numeric(df["judge_rating"], errors="coerce").fillna(-1).astype("int64")
    df["valid"] = df["valid"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)
    df["cache_hits"] = pd.to_numeric(df.get("cache_hits", 0), errors="coerce").fillna(0).astype(int)
    df["n_llm_calls"] = pd.to_numeric(df.get("n_llm_calls", 0), errors="coerce").fillna(0).astype(int)

    # benchmark
    df["benchmark"] = df["sample_source"].astype(str).str.split("/").str[0]

    # cell-derived columns
    df["cell_stratum"] = df["cell_id"].map(_cell_stratum)
    df["l_spec"] = df["cell_id"].map(lambda c: _cell_field(c, "L_spec"))
    df["l_test"] = df["cell_id"].map(lambda c: _cell_field(c, "L_test"))
    df["l_code"] = df["cell_id"].map(lambda c: _cell_field(c, "L_code"))
    df["l_spec_family"] = df["l_spec"].map(_family_of_tag)
    df["l_test_family"] = df["l_test"].map(_family_of_tag)
    df["l_code_family"] = df["l_code"].map(_family_of_tag)

    def _count_families(row):
        fams = {row["l_spec_family"], row["l_test_family"], row["l_code_family"]}
        fams.discard("(none)")
        return len(fams)
    df["n_families"] = df.apply(_count_families, axis=1)
    df["is_homogeneous"] = df["n_families"] <= 1

    # closure_success (per-path threshold)
    df["closure_success"] = df.apply(_compute_closure_success, axis=1)

    # judge_agrees: judge_rating ≥ 3 agrees with closure_success=True
    df["judge_agrees"] = (
        (df["judge_rating"] >= 3) & df["closure_success"]
    ) | (
        (df["judge_rating"].notna() & (df["judge_rating"] < 3) & ~df["closure_success"])
    )

    return df


# ──────────────────────────────────────────────────────────────────────
# Filtering helpers
# ──────────────────────────────────────────────────────────────────────
def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where valid=False (filter-dropped or empty input)."""
    if df.empty:
        return df
    return df[df["valid"]].copy()


def filter_path(df: pd.DataFrame, path: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["path"] == path].copy()


def filter_benchmark(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["benchmark"] == benchmark].copy()


# ──────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────
def aggregate_per_cell_path(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean / std / n_valid grouped by (cell_id, path)."""
    if df.empty:
        return pd.DataFrame(columns=["cell_id", "path", "n_valid",
                                     "mean", "std", "median"])
    valid = filter_valid(df)
    agg = (
        valid.groupby(["cell_id", "path"])["metric_value"]
        .agg(n_valid="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    return agg


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell summary across all paths (simple unweighted mean)."""
    agg = aggregate_per_cell_path(df)
    if agg.empty:
        return pd.DataFrame(columns=["cell_id", "n_valid", "mean"])
    return (agg.groupby("cell_id")
              .agg(n_valid=("n_valid", "sum"),
                   mean=("mean", "mean"))
              .reset_index())


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────
def _cell_stratum(cell_id: str) -> str:
    cell = CELLS_BY_ID.get(cell_id)
    return cell.stratum if cell else "?"


def _cell_field(cell_id: str, attr: str) -> str:
    cell = CELLS_BY_ID.get(cell_id)
    if cell is None:
        return "(unknown)"
    model = getattr(cell, attr, None)
    return model.ollama_tag if model is not None else "(none)"


def _family_of_tag(tag: str) -> str:
    if not tag or tag in ("(none)", "(unknown)"):
        return "(none)"
    spec = MODELS_BY_OLLAMA_TAG.get(tag)
    return spec.family if spec else "(unknown)"


def _compute_closure_success(row) -> bool:
    metric = row.get("metric_value")
    if metric is None or pd.isna(metric):
        return False
    path = row.get("path")
    if path is None or pd.isna(path):
        return False
    threshold = CLOSURE_THRESHOLDS.get(int(path), 0.5)
    return bool(row.get("valid", False)) and metric >= threshold


# ──────────────────────────────────────────────────────────────────────
# Synthetic data — for testing the analysis chain without real Colab data
# ──────────────────────────────────────────────────────────────────────
def synthesize_fake_data(n_samples: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Build a realistic-shaped DataFrame mirroring what a small pilot run
    would produce. Useful for testing the analysis + plotting pipeline
    BEFORE the real Colab sweep generates data.

    Calibration:
        - Mono cells: closure rate roughly tracks model capability
          (llama3.2 ~ 0.55, qwen3.6 ~ 0.85, qwen3-coder ~ 0.90)
        - Hetero cells: closure rate slightly above the weakest stage's
          mono mean
        - Null cells: closure rate sharply lower
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Per-cell expected closure rate (calibrated against Chapter 2 SQJ findings)
    cell_mu: dict[str, float] = {
        "M1": 0.55, "M2": 0.78, "M3": 0.85, "M4": 0.83, "M5": 0.77, "M6": 0.90,
        "H1": 0.88, "H2": 0.84, "H3": 0.82, "H4": 0.60, "H5": 0.65,
        "H6": 0.79, "H7": 0.71, "H8": 0.83, "H9": 0.89, "H10": 0.85, "H11": 0.83,
        "N1": 0.20, "N2": 0.40, "N3": 0.45,
    }

    benchmarks = ["humaneval"] * 10 + ["mbpp"] * 20  # 1:2 ratio per Ch.2

    for cell in ALL_CELLS:
        mu = cell_mu.get(cell.cell_id, 0.5)
        for idx in range(n_samples):
            for path in (1, 2, 3):
                bench = benchmarks[idx % len(benchmarks)]
                source = f"{bench}/{bench}/{idx}"

                # Inject path-specific drift (Path 2 noisier than Path 1)
                path_noise = {1: 0.10, 2: 0.18, 3: 0.12}[path]
                value = float(np.clip(rng.normal(mu, path_noise), 0.0, 1.0))
                valid = value > 0.05

                # Judge rating roughly tracks value
                if value > 0.85:
                    judge_rating = int(rng.choice([3, 4], p=[0.3, 0.7]))
                elif value > 0.60:
                    judge_rating = int(rng.choice([2, 3], p=[0.4, 0.6]))
                else:
                    judge_rating = int(rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))

                rows.append({
                    "cell_id": cell.cell_id,
                    "sample_idx": idx,
                    "sample_source": source,
                    "path": path,
                    "metric_name": {1: "mutation_kill_rate",
                                    2: "reference_pass_rate",
                                    3: "bertscore"}[path],
                    "metric_value": value,
                    "judge_rating": judge_rating,
                    "judge_justification": f"synth_judge_{judge_rating}",
                    "valid": valid,
                    "elapsed_s": float(rng.uniform(3, 30)),
                    "cache_hits": int(rng.integers(0, 4)),
                    "n_llm_calls": int(rng.integers(2, 5)),
                    "notes": "synthetic",
                })

    df = pd.DataFrame(rows)
    return _enrich(df)


# ──────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== load_results synthetic-data self-test ===")
    df = synthesize_fake_data(n_samples=30)
    print(f"  shape: {df.shape}")
    print(f"  benchmarks: {df['benchmark'].value_counts().to_dict()}")
    print(f"  strata: {df['cell_stratum'].value_counts().to_dict()}")
    print(f"  mean closure_success per stratum:")
    print(df.groupby("cell_stratum")["closure_success"].mean().to_string())
    print("\n✓ synthesize_fake_data works.")
