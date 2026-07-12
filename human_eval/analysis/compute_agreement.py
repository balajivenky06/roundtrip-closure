"""
human_eval/analysis/compute_agreement.py

Compute inter-rater agreement statistics for the three-annotator human
evaluation study, plus judge-vs-human agreement.

Inputs (all in ../data/):
    annotator_<A>_ratings.tsv, annotator_<B>_ratings.tsv, annotator_<C>_ratings.tsv
        Per-annotator ratings TSVs produced by the Streamlit app.
        Columns: pair_id, rating (0-4), justification, ts_iso, is_revision

    ../results/human_eval_pairs_60.tsv
        The pre-registered 60 pairs with the judge_rating column (LLM judge
        rating on the same 0-4 scale). Used for judge-vs-human agreement.

Outputs (to ../data/):
    agreement_report.json    — machine-readable summary
    agreement_report.tex     — LaTeX table paste-ready for the manuscript
                                addendum

Statistics computed:
    - Krippendorff's α (ordinal distance metric) across all three annotators
    - Pairwise Cohen's κ (linear-weighted) for each of the three pairs
    - Judge-vs-human Cohen's κ, where human = majority vote across the three
      annotators (ties broken toward the higher rating)
    - Per-bucket agreement rate (exact-match proportion) split by the
      pair stratification bucket

Implemented without scipy so this script runs anywhere Python 3.10+ is
installed.

Run:
    python3 human_eval/analysis/compute_agreement.py \
        --annotators ann_alpha ann_beta ann_gamma
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PROJECT_ROOT = HERE.parent.parent
PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60.tsv"


# ────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────
def load_annotator(annotator_id: str) -> dict[str, int]:
    """Return {pair_id -> rating (0..4)} for one annotator."""
    path = DATA_DIR / f"annotator_{annotator_id}_ratings.tsv"
    out: dict[str, int] = {}
    if not path.exists():
        raise FileNotFoundError(f"Ratings not found for {annotator_id}: {path}")
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["pair_id"]] = int(row["rating"])
            except (KeyError, ValueError):
                continue
    return out


def load_pairs() -> list[dict]:
    with PAIRS_TSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ────────────────────────────────────────────────────────────────────────
# Metrics — implemented from first principles
# ────────────────────────────────────────────────────────────────────────
def cohens_kappa_linear(a: list[int], b: list[int], k: int = 5) -> float:
    """Linear-weighted Cohen's κ over an ordinal k-category scale.

    Weights: w[i][j] = 1 - |i - j| / (k - 1)
    """
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    weights = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs = [[0 for _ in range(k)] for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    marg_a = [sum(row) for row in obs]
    marg_b = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    p_o = sum(weights[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    p_e = (
        sum(weights[i][j] * marg_a[i] * marg_b[j] for i in range(k) for j in range(k))
        / (n * n)
    )
    if abs(1.0 - p_e) < 1e-12:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha_ordinal(
    ratings_by_annotator: list[dict[str, int]], k: int = 5
) -> float:
    """Krippendorff's α with the ordinal difference function.

    Ordinal δ²(v, w) = (sum_{c=min(v,w)}^{max(v,w)} n_c - (n_v + n_w)/2)²
    where n_c is the marginal count of rating category c across the entire
    dataset.

    Reference: Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability".
    """
    # Build unit-of-analysis table: {pair_id -> {rating -> count}}
    unit_counts: dict[str, Counter] = defaultdict(Counter)
    all_pair_ids: set[str] = set()
    for tbl in ratings_by_annotator:
        for pid, r in tbl.items():
            all_pair_ids.add(pid)
            unit_counts[pid][r] += 1

    # Marginal totals across categories, restricted to units with ≥2 raters
    valid_units = [pid for pid in all_pair_ids if sum(unit_counts[pid].values()) >= 2]
    marg = Counter()
    for pid in valid_units:
        for c, n in unit_counts[pid].items():
            marg[c] += n

    n_total = sum(marg.values())
    if n_total < 2:
        return float("nan")

    categories = list(range(k))
    # Precompute ordinal δ²(v, w)
    def delta2(v: int, w: int) -> float:
        if v == w:
            return 0.0
        lo, hi = (v, w) if v < w else (w, v)
        s = sum(marg[c] for c in range(lo, hi + 1))
        return (s - (marg[v] + marg[w]) / 2.0) ** 2

    # Observed disagreement D_o
    d_o = 0.0
    for pid in valid_units:
        m_u = sum(unit_counts[pid].values())
        if m_u < 2:
            continue
        # Sum over category pairs
        cats = list(unit_counts[pid].items())
        s_u = 0.0
        for v, nv in cats:
            for w, nw in cats:
                if v == w:
                    s_u += nv * (nw - 1) * delta2(v, w)
                else:
                    s_u += nv * nw * delta2(v, w)
        d_o += s_u / (m_u - 1)

    d_o /= n_total

    # Expected disagreement D_e
    d_e = 0.0
    for v in categories:
        for w in categories:
            d_e += marg[v] * marg[w] * delta2(v, w)
    d_e /= n_total * (n_total - 1)

    if abs(d_e) < 1e-12:
        return float("nan")
    return 1.0 - d_o / d_e


def majority_vote(triples: list[tuple[int, int, int]]) -> list[int]:
    """Majority vote; ties broken toward the higher rating."""
    out = []
    for t in triples:
        counts = Counter(t)
        top = counts.most_common()
        max_n = top[0][1]
        winners = [r for r, n in top if n == max_n]
        out.append(max(winners))
    return out


def per_bucket_agreement_rate(
    ratings_by_annotator: list[dict[str, int]],
    pairs: list[dict],
) -> dict[str, dict]:
    """Fraction of pairs where all three annotators agree exactly, by bucket."""
    by_bucket: dict[str, list[bool]] = defaultdict(list)
    for row in pairs:
        pid = row["pair_id"]
        bucket = row.get("bucket", "unknown")
        try:
            triple = tuple(t[pid] for t in ratings_by_annotator)
        except KeyError:
            continue
        by_bucket[bucket].append(len(set(triple)) == 1)
    return {
        b: {"n": len(vs), "exact_agreement_rate": sum(vs) / len(vs) if vs else 0.0}
        for b, vs in by_bucket.items()
    }


# ────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────
def interpret_alpha(alpha: float) -> str:
    if math.isnan(alpha):
        return "undefined"
    if alpha < 0.667:
        return "insufficient"
    if alpha < 0.8:
        return "tentative"
    return "acceptable"


def render_latex_table(report: dict) -> str:
    def fmt(x: float) -> str:
        return "n/a" if math.isnan(x) else f"{x:.3f}"

    lines = [
        r"% Auto-generated by human_eval/analysis/compute_agreement.py",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Inter-rater and judge--human agreement on the 60-pair",
        r"    human-evaluation sample. Cohen's $\kappa$ is linear-weighted",
        r"    on the 5-level ordinal scale. Krippendorff's $\alpha$ uses",
        r"    the ordinal difference function.}",
        r"  \label{tab:human_eval_agreement}",
        r"  \begin{tabular}{lrl}",
        r"    \toprule",
        r"    Statistic & Value & Interpretation \\",
        r"    \midrule",
        f"    Krippendorff's $\\alpha$ (3 annotators, ordinal) & "
        f"{fmt(report['krippendorff_alpha_ordinal'])} & "
        f"{interpret_alpha(report['krippendorff_alpha_ordinal'])} \\\\",
    ]
    for pair_key, kappa in report["pairwise_cohens_kappa"].items():
        lines.append(
            f"    Cohen's $\\kappa$, {pair_key.replace('_', '--')} & "
            f"{fmt(kappa)} & linear-weighted \\\\"
        )
    lines.append(
        f"    Cohen's $\\kappa$, judge vs.\\ majority human & "
        f"{fmt(report['judge_vs_human_kappa'])} & linear-weighted \\\\"
    )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_report(
    annotator_ids: list[str],
    tables: list[dict[str, int]],
    pairs: list[dict],
) -> dict:
    # Common pair IDs across all annotators
    common_ids = sorted(set.intersection(*(set(t.keys()) for t in tables)))
    triples = [(tables[0][p], tables[1][p], tables[2][p]) for p in common_ids]

    # Pairwise κ
    pairwise: dict[str, float] = {}
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            a = [t[0] if i == 0 else t[1] if i == 1 else t[2] for t in triples]
            b = [t[0] if j == 0 else t[1] if j == 1 else t[2] for t in triples]
            key = f"{annotator_ids[i]}_vs_{annotator_ids[j]}"
            pairwise[key] = cohens_kappa_linear(a, b)

    # Judge vs majority human
    human_majority = majority_vote(triples)
    judge_map: dict[str, int] = {}
    for row in pairs:
        try:
            judge_map[row["pair_id"]] = int(round(float(row["judge_rating"])))
        except (KeyError, ValueError):
            continue
    common_with_judge = [p for p in common_ids if p in judge_map]
    judge_kappa = float("nan")
    if common_with_judge:
        judge_ratings = [judge_map[p] for p in common_with_judge]
        human_for_judge = [
            human_majority[common_ids.index(p)] for p in common_with_judge
        ]
        judge_kappa = cohens_kappa_linear(judge_ratings, human_for_judge)

    # Krippendorff α across all three raters
    alpha = krippendorff_alpha_ordinal(tables)

    # Per-bucket agreement rate
    by_bucket = per_bucket_agreement_rate(tables, pairs)

    # Exact-match rate across three annotators
    exact_match = sum(1 for t in triples if len(set(t)) == 1) / len(triples)

    return {
        "annotators": annotator_ids,
        "n_pairs_common": len(common_ids),
        "krippendorff_alpha_ordinal": alpha,
        "pairwise_cohens_kappa": pairwise,
        "judge_vs_human_kappa": judge_kappa,
        "judge_vs_human_n": len(common_with_judge),
        "exact_agreement_rate_all_three": exact_match,
        "per_bucket_agreement": by_bucket,
    }


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotators", nargs=3, metavar="ID",
        default=["ann_alpha", "ann_beta", "ann_gamma"],
        help="Three annotator IDs (default: ann_alpha ann_beta ann_gamma)",
    )
    args = parser.parse_args()

    tables = [load_annotator(a) for a in args.annotators]
    pairs = load_pairs()

    report = build_report(args.annotators, tables, pairs)

    # Emit JSON
    out_json = DATA_DIR / "agreement_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Emit LaTeX table
    out_tex = DATA_DIR / "agreement_report.tex"
    with out_tex.open("w", encoding="utf-8") as f:
        f.write(render_latex_table(report))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
