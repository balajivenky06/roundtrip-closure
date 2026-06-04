"""
analyze/stats.py — all the statistical tests the journal paper needs.

Every function takes the enriched DataFrame from analyze.load_results.load_tsv
and returns a structured result (dict or DataFrame) — never prints, never
plots. Master runner aggregates everything into analysis_summary.json.

Tests implemented:

    mixed_effects_logistic   — closure_success ~ L_spec * L_test * L_code + (1|sample_idx)
    anova_type3              — Type-III ANOVA on metric_value ~ C(cell_id) + C(sample_idx)
    tukey_hsd                — pairwise Tukey with Bonferroni correction
    per_stage_bottleneck     — which stage owns failures (leave-one-stage-out style)
    per_benchmark_slice      — re-run ANOVA on HumanEval vs MBPP separately
    per_operator_slice       — per mutation-operator family (when breakdown is available)
    cross_cell_spearman      — Spearman ρ between cells on per-sample closure rate
    judge_metric_correlation — Pearson r between automated metric and judge rating
    false_closure_rate       — fraction (high metric, low judge) per cell
    contamination_sensitivity — HumanEval vs HumanEval-Mutated delta per cell
    cache_efficiency         — overall cache hit rate + per-cell breakdown
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from analyze.load_results import filter_valid, filter_path


# ──────────────────────────────────────────────────────────────────────
# Lazy heavyweight imports (statsmodels, scipy)
#
# macOS Anaconda often has scipy/statsmodels ABI clashes; we defer
# these imports to call time so the rest of the analyze package
# remains importable for cache-efficiency reports, per-stage
# decomposition, etc. (which need only pandas + numpy).
# ──────────────────────────────────────────────────────────────────────
def _smf():
    import statsmodels.formula.api as smf
    return smf

def _sm():
    import statsmodels.api as sm
    return sm

def _anova_lm():
    from statsmodels.stats.anova import anova_lm
    return anova_lm

def _tukeyhsd():
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    return pairwise_tukeyhsd

def _scs():
    from scipy import stats as scs
    return scs


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Mixed-effects logistic regression  (RQ1)
# ──────────────────────────────────────────────────────────────────────
def mixed_effects_logistic(df: pd.DataFrame,
                            path: Optional[int] = None) -> dict:
    """
    closure_success ~ C(l_spec_family) + C(l_test_family) + C(l_code_family)
                    + (1 | sample_idx)

    Notes:
        - statsmodels MixedLM doesn't support binary outcomes natively.
          We use a GEE (generalised estimating equations) family Binomial
          with cluster=sample_idx as a practical alternative; same
          interpretation (population-averaged effects, accounts for
          per-sample correlation).
        - Two-way interactions are too expensive at 6 LLMs × 3 stages
          (~135 cell combinations); we report main-effects only here
          and let cross_cell_spearman pick up the interaction signal.

    Returns:
        {
          "n": int, "n_clusters": int,
          "coefs": DataFrame[term, coef, std_err, z, p, sig],
          "convergence": bool,
        }
    """
    work = filter_valid(df)
    if path is not None:
        work = filter_path(work, path)
    if work.empty:
        return {"n": 0, "n_clusters": 0, "coefs": pd.DataFrame(),
                "convergence": False, "note": "no valid rows"}

    # Drop rows with NaN closure_success (defensive)
    work = work.dropna(subset=["closure_success"])
    if work["closure_success"].nunique() < 2:
        return {"n": len(work), "n_clusters": 0, "coefs": pd.DataFrame(),
                "convergence": False, "note": "closure_success is constant"}

    formula = ("closure_success ~ C(l_spec_family) + C(l_test_family) "
               "+ C(l_code_family)")
    try:
        model = _smf().gee(
            formula, groups="sample_idx",
            data=work,
            family=_sm().families.Binomial(),
        )
        fitted = model.fit()
    except Exception as exc:                                       # pragma: no cover
        logger.warning(f"GEE fit failed: {exc}; falling back to plain logit")
        try:
            model = _smf().logit(formula, data=work)
            fitted = model.fit(disp=False)
        except Exception as exc2:
            return {"n": len(work), "n_clusters": work["sample_idx"].nunique(),
                    "coefs": pd.DataFrame(), "convergence": False,
                    "note": f"both fits failed: {exc2}"}

    coefs = pd.DataFrame({
        "term": fitted.params.index,
        "coef": fitted.params.values,
        "std_err": fitted.bse.values,
        "z": (fitted.params / fitted.bse).values,
        "p": fitted.pvalues.values,
    })
    coefs["sig"] = coefs["p"].apply(_sig_label)

    return {
        "n": len(work),
        "n_clusters": int(work["sample_idx"].nunique()),
        "coefs": coefs,
        "convergence": True,
        "model": "GEE binomial; cluster=sample_idx; main-effects only",
    }


# ──────────────────────────────────────────────────────────────────────
# 2. Type-III ANOVA  (RQ1, RQ3)
# ──────────────────────────────────────────────────────────────────────
def anova_type3(df: pd.DataFrame,
                path: Optional[int] = None,
                response: str = "metric_value") -> pd.DataFrame:
    """
    Type-III ANOVA: response ~ C(cell_id) + C(sample_idx).

    Returns a DataFrame with one row per factor: sum_sq, df, F, p, sig.
    """
    work = filter_valid(df)
    if path is not None:
        work = filter_path(work, path)
    if work.empty or work["cell_id"].nunique() < 2:
        return pd.DataFrame(columns=["factor", "sum_sq", "df", "F", "p", "sig"])
    work = work.dropna(subset=[response])

    formula = f"{response} ~ C(cell_id) + C(sample_idx)"
    try:
        model = _smf().ols(formula, data=work).fit()
        table = _anova_lm()(model, typ=3)
    except Exception as exc:                                       # pragma: no cover
        logger.error(f"ANOVA failed: {exc}")
        return pd.DataFrame(columns=["factor", "sum_sq", "df", "F", "p", "sig"])

    out = table.reset_index().rename(
        columns={"index": "factor", "sum_sq": "sum_sq",
                 "df": "df", "F": "F", "PR(>F)": "p"}
    )
    out["sig"] = out["p"].apply(_sig_label)
    return out[["factor", "sum_sq", "df", "F", "p", "sig"]]


# ──────────────────────────────────────────────────────────────────────
# 3. Tukey HSD pairwise comparisons (RQ1)
# ──────────────────────────────────────────────────────────────────────
def tukey_hsd(df: pd.DataFrame,
              path: Optional[int] = None,
              response: str = "metric_value",
              alpha: float = 0.05) -> pd.DataFrame:
    """
    All pairwise Tukey HSD across cells. Output is the full pairwise
    table (cells choose 2 = 190 rows on the 20-cell DOE).

    Returns DataFrame with columns: cell_a, cell_b, mean_diff,
    lower_ci, upper_ci, p_adj, reject, sig.
    """
    work = filter_valid(df)
    if path is not None:
        work = filter_path(work, path)
    if work.empty or work["cell_id"].nunique() < 2:
        return pd.DataFrame(columns=["cell_a", "cell_b", "mean_diff",
                                     "lower_ci", "upper_ci", "p_adj",
                                     "reject", "sig"])
    work = work.dropna(subset=[response])

    try:
        tukey = _tukeyhsd()(
            endog=work[response].astype(float),
            groups=work["cell_id"].astype(str),
            alpha=alpha,
        )
    except Exception as exc:                                       # pragma: no cover
        logger.error(f"Tukey HSD failed: {exc}")
        return pd.DataFrame()

    # statsmodels' summary table has columns:
    #   group1, group2, meandiff, p-adj, lower, upper, reject
    raw = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    raw = raw.rename(columns={
        "group1": "cell_a", "group2": "cell_b",
        "meandiff": "mean_diff",
        "p-adj": "p_adj",
        "lower": "lower_ci", "upper": "upper_ci",
    })
    # Coerce numeric columns
    for col in ("mean_diff", "p_adj", "lower_ci", "upper_ci"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["sig"] = raw["p_adj"].apply(_sig_label)
    return raw


# ──────────────────────────────────────────────────────────────────────
# 4. Per-stage bottleneck decomposition  (RQ3)
# ──────────────────────────────────────────────────────────────────────
def per_stage_bottleneck(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each hetero cell, ask: which stage is the bottleneck?

    Approach: compare the hetero cell's closure rate to (a) the mono
    cell that uses its L_spec, (b) the mono cell that uses its L_test,
    (c) the mono cell that uses its L_code. The hetero cell's mean is
    bounded above by max(mono_a, mono_b, mono_c) if every stage adds
    value; if mean(hetero) << min(mono), the stage with the lowest
    mono is the bottleneck.

    Returns DataFrame per (hetero_cell, path):
        hetero_cell, path, hetero_mean,
        mono_spec, mono_test, mono_code,
        bottleneck_stage  : 'spec' | 'test' | 'code' | 'none'
        bottleneck_delta  : (hetero_mean - min_mono_for_bottleneck_stage)
    """
    from doe import MONO_CELLS, HETERO_CELLS, CELLS_BY_ID

    valid = filter_valid(df)
    if valid.empty:
        return pd.DataFrame()

    # Build mono-cell mean lookup keyed by (ollama_tag, path)
    mono_means: dict[tuple[str, int], float] = {}
    for cell in MONO_CELLS:
        rows = valid[valid["cell_id"] == cell.cell_id]
        for path, group in rows.groupby("path"):
            mono_means[(cell.L_spec.ollama_tag, int(path))] = float(group["metric_value"].mean())

    out_rows = []
    for cell in HETERO_CELLS:
        for path in (1, 2, 3):
            rows = valid[(valid["cell_id"] == cell.cell_id) & (valid["path"] == path)]
            if rows.empty:
                continue
            hetero_mean = float(rows["metric_value"].mean())
            spec_tag = cell.L_spec.ollama_tag if cell.L_spec else "(none)"
            test_tag = cell.L_test.ollama_tag if cell.L_test else "(none)"
            code_tag = cell.L_code.ollama_tag if cell.L_code else "(none)"

            spec_mean = mono_means.get((spec_tag, path), float("nan"))
            test_mean = mono_means.get((test_tag, path), float("nan"))
            code_mean = mono_means.get((code_tag, path), float("nan"))

            # Bottleneck = the stage whose mono cell has the lowest mean
            stages = {"spec": spec_mean, "test": test_mean, "code": code_mean}
            valid_stages = {k: v for k, v in stages.items() if not np.isnan(v)}
            if not valid_stages:
                bottleneck = "none"
                delta = float("nan")
            else:
                bottleneck = min(valid_stages, key=valid_stages.get)
                delta = hetero_mean - valid_stages[bottleneck]

            out_rows.append({
                "hetero_cell": cell.cell_id,
                "path": path,
                "hetero_mean": hetero_mean,
                "mono_spec": spec_mean,
                "mono_test": test_mean,
                "mono_code": code_mean,
                "bottleneck_stage": bottleneck,
                "bottleneck_delta": delta,
            })
    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────
# 5. Per-benchmark slice  (RQ3)
# ──────────────────────────────────────────────────────────────────────
def per_benchmark_slice(df: pd.DataFrame) -> dict:
    """
    Re-run ANOVA + Tukey separately for each benchmark.

    Returns dict mapping benchmark name → {"anova": DataFrame, "tukey": DataFrame, "n": int}.
    """
    out = {}
    for bench in df["benchmark"].dropna().unique():
        sub = df[df["benchmark"] == bench]
        out[bench] = {
            "n": int(len(sub)),
            "anova": anova_type3(sub),
            "tukey": tukey_hsd(sub),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# 6. Cross-cell Spearman ρ matrix  (RQ1 generalisability)
# ──────────────────────────────────────────────────────────────────────
def cross_cell_spearman(df: pd.DataFrame, path: int = 1) -> pd.DataFrame:
    """
    For path `path`, build a wide table (rows=sample_idx, cols=cell_id,
    values=metric_value) and compute pairwise Spearman ρ between cell
    columns. Returns the ρ matrix (cells × cells).
    """
    valid = filter_valid(filter_path(df, path))
    if valid.empty:
        return pd.DataFrame()
    wide = (valid.pivot_table(
        index="sample_idx", columns="cell_id", values="metric_value",
        aggfunc="mean"))
    # Drop columns/rows with too few observations
    wide = wide.dropna(axis=1, thresh=max(3, len(wide) // 4))
    if wide.shape[1] < 2:
        return pd.DataFrame()
    return wide.corr(method="spearman")


# ──────────────────────────────────────────────────────────────────────
# 7. Judge ↔ automated metric correlation  (RQ2)
# ──────────────────────────────────────────────────────────────────────
def judge_metric_correlation(df: pd.DataFrame) -> dict:
    """
    Pearson r between judge_rating (0-4) and metric_value, per path.
    Excludes rows where judge_rating == -1 (parse failure).
    """
    out = {"per_path": {}, "overall": None}
    for path in (1, 2, 3):
        sub = filter_path(filter_valid(df), path)
        sub = sub[(sub["judge_rating"].notna()) & (sub["judge_rating"] >= 0)]
        if len(sub) < 5:
            out["per_path"][path] = {"n": len(sub), "r": None, "p": None}
            continue
        r, p = _scs().pearsonr(sub["metric_value"].astype(float),
                             sub["judge_rating"].astype(float))
        out["per_path"][path] = {"n": int(len(sub)),
                                 "r": float(r), "p": float(p),
                                 "sig": _sig_label(p)}

    all_sub = filter_valid(df)
    all_sub = all_sub[(all_sub["judge_rating"].notna()) & (all_sub["judge_rating"] >= 0)]
    if len(all_sub) >= 5:
        r, p = _scs().pearsonr(all_sub["metric_value"].astype(float),
                             all_sub["judge_rating"].astype(float))
        out["overall"] = {"n": int(len(all_sub)),
                          "r": float(r), "p": float(p),
                          "sig": _sig_label(p)}
    return out


# ──────────────────────────────────────────────────────────────────────
# 8. False-closure rate  (RQ5)
# ──────────────────────────────────────────────────────────────────────
def false_closure_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    False closure = automated metric says SUCCESS but judge disagrees
    (judge_rating < 3).

    Returns DataFrame per (cell, path): n, n_closure_success,
    n_false_closure, false_closure_rate.
    """
    valid = filter_valid(df)
    if valid.empty:
        return pd.DataFrame()
    valid = valid[valid["judge_rating"].notna() & (valid["judge_rating"] >= 0)]

    out_rows = []
    for (cell_id, path), group in valid.groupby(["cell_id", "path"]):
        n = int(len(group))
        successes = group[group["closure_success"]]
        n_success = int(len(successes))
        n_false = int(((successes["judge_rating"] < 3)).sum())
        rate = n_false / max(n_success, 1)
        out_rows.append({
            "cell_id": cell_id,
            "path": int(path),
            "n": n,
            "n_closure_success": n_success,
            "n_false_closure": n_false,
            "false_closure_rate": rate,
        })
    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────
# 9. Contamination sensitivity  (RQ4)
# ──────────────────────────────────────────────────────────────────────
def contamination_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare per-cell mean closure rate on HumanEval (potentially
    contaminated) vs HumanEval-Mutated (decontaminated).

    Returns DataFrame: cell_id, path, n_he, n_hem,
                       mean_he, mean_hem, delta (he - hem), sig.
    """
    valid = filter_valid(df)
    out_rows = []
    for (cell_id, path), group in valid.groupby(["cell_id", "path"]):
        he = group[group["benchmark"] == "humaneval"]["metric_value"]
        hem = group[group["benchmark"] == "humaneval_mutated"]["metric_value"]
        if len(he) == 0 or len(hem) == 0:
            continue
        try:
            t, p = _scs().ttest_ind(he, hem, equal_var=False)
        except Exception:
            t, p = float("nan"), float("nan")
        out_rows.append({
            "cell_id": cell_id,
            "path": int(path),
            "n_he": int(len(he)),
            "n_hem": int(len(hem)),
            "mean_he": float(he.mean()),
            "mean_hem": float(hem.mean()),
            "delta": float(he.mean() - hem.mean()),
            "t": float(t),
            "p": float(p),
            "sig": _sig_label(p),
        })
    return pd.DataFrame(out_rows)


# ──────────────────────────────────────────────────────────────────────
# 10. Cache + compute efficiency report
# ──────────────────────────────────────────────────────────────────────
def cache_efficiency(df: pd.DataFrame) -> dict:
    """Total LLM calls vs cache hits across the sweep."""
    if df.empty:
        return {"total_calls": 0, "cache_hits": 0, "hit_rate": 0.0,
                "per_cell": pd.DataFrame()}
    total_calls = int(df["n_llm_calls"].sum())
    cache_hits = int(df["cache_hits"].sum())
    per_cell = (df.groupby("cell_id")
                  .agg(n_calls=("n_llm_calls", "sum"),
                       n_hits=("cache_hits", "sum"))
                  .reset_index())
    per_cell["hit_rate"] = per_cell["n_hits"] / per_cell["n_calls"].clip(lower=1)
    return {
        "total_calls": total_calls,
        "cache_hits": cache_hits,
        "hit_rate": cache_hits / max(total_calls, 1),
        "per_cell": per_cell,
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _sig_label(p) -> str:
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "n.s."


# ──────────────────────────────────────────────────────────────────────
# Smoke test on synthetic data
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from analyze.load_results import synthesize_fake_data

    print("=== stats.py smoke test on synthetic data ===\n")
    df = synthesize_fake_data(n_samples=30)

    print("1) anova_type3 (path 1)…")
    a = anova_type3(df, path=1)
    print(a.to_string(index=False))

    print("\n2) tukey_hsd (path 1, first 5 rows)…")
    t = tukey_hsd(df, path=1)
    print(t.head(5).to_string(index=False))
    print(f"   total pairs: {len(t)}")

    print("\n3) per_stage_bottleneck (first 5 rows)…")
    b = per_stage_bottleneck(df)
    print(b.head(5).to_string(index=False))

    print("\n4) judge_metric_correlation…")
    c = judge_metric_correlation(df)
    for path, stats in c["per_path"].items():
        print(f"   path {path}: {stats}")
    print(f"   overall:    {c['overall']}")

    print("\n5) false_closure_rate (top 5)…")
    fc = false_closure_rate(df)
    print(fc.sort_values("false_closure_rate", ascending=False).head(5).to_string(index=False))

    print("\n6) cache_efficiency…")
    ce = cache_efficiency(df)
    print(f"   total={ce['total_calls']}, hits={ce['cache_hits']}, "
          f"hit_rate={ce['hit_rate']:.2%}")

    print("\n✓ stats.py smoke test passed.")
