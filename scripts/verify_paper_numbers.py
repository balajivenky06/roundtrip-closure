"""
verify_paper_numbers.py — recompute the manuscript's reported numbers from the
released result files and compare each with the value printed in the paper.

    python3 scripts/verify_paper_numbers.py

Every check prints PASS or FAIL with the recomputed and the printed value.
Exit code is 0 only if every check passes. No LLM calls, no cache needed.

Definitions used throughout (as in the paper):
  strict-AND valid closure  = metric_value > 0 and judge_rating >= 3
  V  (canonical)            = valid / all rows of the (cell, path)
  V' (conditional)          = valid / rows with a defined metric
  Y-bar                     = mean metric over rows with a defined metric
  all_dropped               = notes contains "dropped all" (Path 1 test filter)
NOTE: the TSV column `valid` is NOT strict-AND; it flags rows whose metric is
computable. Never use it as the closure outcome.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FAIL = 0


def check(name: str, got, want, tol: float = 0.0015) -> None:
    global FAIL
    if isinstance(want, str):
        ok = str(got) == want
    else:
        ok = abs(float(got) - float(want)) <= tol
    FAIL += not ok
    g = got if isinstance(got, str) else f"{float(got):.4f}"
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} recomputed={g}  paper={want}")


def load(p: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / p, sep="\t")
    d["v"] = (d.metric_value > 0) & (d.judge_rating >= 3)
    d["ok"] = d.metric_value.notna()
    d["ad"] = d.notes.astype(str).str.contains("dropped all")
    d["bm"] = d.sample_source.str.split("/").str[0]
    return d


def main() -> int:
    d = load("results/results_roundtrip.tsv")
    print("== Sweep size and sample (Section 4.2, 5)")
    check("total rows", len(d), 5890, 0)
    core = [json.loads(l) for l in open(ROOT / "data/core_sample_150.jsonl")]
    check("core HumanEval functions", sum(c["source"].startswith("humaneval") for c in core), 40, 0)
    check("core MBPP functions", sum(c["source"].startswith("mbpp") for c in core), 110, 0)

    print("== Table 6: metric mean (count of strict-AND closures)")
    t6 = {"M3": (0.900, 100), "M4": (0.856, 116), "M6": (0.862, 108), "H1": (0.854, 64),
          "H6": (0.882, 53), "H10": (0.764, 22), "H11": (0.872, 61), "N1": (0.429, 23)}
    for c, (y, n) in t6.items():
        x = d[(d.cell_id == c) & (d.path == 1)]
        check(f"Table 6 {c} Path 1 Y-bar", x.metric_value.mean(), y)
        check(f"Table 6 {c} Path 1 n_valid", x.v.sum(), n, 0)
    x = d[(d.cell_id == "H3") & (d.path == 2)]
    check("Table 6/9 H3 Path 2 Y-bar", x.metric_value.mean(), 0.627)
    x = d[(d.cell_id == "H3") & (d.path == 3)]
    check("Table 6/9 H3 Path 3 Y-bar", x.metric_value.mean(), 0.174)

    print("== phi-4 at L_test: Path 1 all_dropped (Section 6.3)")
    for c, n, tot in (("M2", 66, 150), ("H2", 34, 75), ("H10", 41, 75)):
        x = d[(d.cell_id == c) & (d.path == 1)]
        check(f"{c} all_dropped count (of {tot})", x.ad.sum(), n, 0)
    r = load("results_rerun/results_h10_rerun.tsv")
    x = r[(r.cell_id == "H10") & (r.path == 1)]
    check("H10 clean re-run all_dropped (of 70)", x.ad.sum(), 38, 0)

    print("== Section 6.3: test-suite structure from the sweep logs")
    import re as _re
    pat = _re.compile(r"\[(\w+) s=(\d+) p1\] step3 test_filter -> (\d+)/(\d+) tests kept")
    for c, single, ad_s, ad_m in (("M2", 0.400, 0.750, 0.233), ("H2", 0.541, 0.675, 0.176)):
        logp = ROOT / f"logs/cell_{c}_core.log"
        if not logp.exists():
            print(f"SKIP  {c} suite structure (logs/ not present)")
            continue
        rec = {}
        for line in open(logp, errors="ignore"):
            m = pat.search(line)
            if m and m.group(1) == c:
                rec[int(m.group(2))] = (int(m.group(3)), int(m.group(4)))
        x = d[(d.cell_id == c) & (d.path == 1)].set_index("sample_idx")
        j = x.join(pd.DataFrame.from_dict(rec, orient="index", columns=["kept", "n"]), how="inner")
        check(f"{c} log/TSV agreement on all-dropped", ((j.kept == 0) == j.ad).mean(), 1.0, 0)
        check(f"{c} share of single-function suites", (j.n == 1).mean(), single)
        check(f"{c} all-dropped rate, single-function suites", j[j.n == 1].ad.mean(), ad_s)
        check(f"{c} all-dropped rate, multi-function suites", j[j.n >= 2].ad.mean(), ad_m)

    print("== Section 5.5 / Table 13: judge-metric correlation")
    for p, rr, n in ((1, 0.192, 1441), (2, 0.523, 1839), (3, 0.022, 1791)):
        s = d[(d.path == p) & d.ok & (d.judge_rating >= 0)]
        check(f"r Path {p}", stats.pearsonr(s.metric_value, s.judge_rating)[0], rr, 0.0006)
        check(f"n Path {p}", len(s), n, 0)
    s3 = d[(d.path == 3) & d.ok & (d.judge_rating >= 0)]
    for b, rr in (("humaneval", 0.153), ("mbpp", 0.212)):
        t = s3[s3.bm == b]
        check(f"r Path 3 {b}", stats.pearsonr(t.metric_value, t.judge_rating)[0], rr, 0.0015)

    print("== Table 23: Path 3 disagreement by benchmark, chi-square")
    p3 = d[(d.path == 3) & d.ok & (d.judge_rating >= 0)]
    fcc = (p3.metric_value > 0) & (p3.judge_rating < 3)
    mfn = (p3.metric_value <= 0) & (p3.judge_rating >= 3)
    tab = np.array([[fcc[p3.bm == b].sum(), mfn[p3.bm == b].sum()] for b in ("humaneval", "mbpp")])
    check("chi-square (2x2, no continuity correction)",
          stats.chi2_contingency(tab, correction=False)[0], 368.0, 0.05)

    print("== Table 25 / Section 6.4: false closure at the strong bar")
    thr = {1: 0.70, 2: 1.00, 3: 0.80}
    rates = []
    for (c, p), g in d.groupby(["cell_id", "path"]):
        g = g[g.ok & (g.judge_rating >= 0)]
        sx = g[g.metric_value >= thr[p]]
        if len(sx):
            rates.append((c, p, len(sx), (sx.judge_rating < 3).sum()))
    fc = pd.DataFrame(rates, columns=["c", "p", "ns", "nf"])
    fc["r"] = 100 * fc.nf / fc.ns
    check("number of (cell, path) groups", len(fc), 37, 0)
    check("mean JRR (%)", fc.r.mean(), 11.4, 0.05)
    n1 = fc[(fc.c == "N1") & (fc.p == 1)].iloc[0]
    check("N1 Path 1 metric successes", n1.ns, 19, 0)
    check("N1 Path 1 judge rejections", n1.nf, 5, 0)
    check("H10 Path 1 JRR (%)", fc[(fc.c == "H10") & (fc.p == 1)].r.iloc[0], 28.6, 0.05)

    print("== Section 5.3 / Fig. 4: sample-matched Path 1 V (samples 0-74)")
    m = d[(d.path == 1) & (d.sample_idx < 75)]
    vm = m.groupby("cell_id").v.mean()
    for c, v in (("M4", 0.840), ("M6", 0.787), ("H11", 0.813), ("H4", 0.800), ("H10", 0.293), ("H8", 0.707)):
        check(f"matched V {c}", vm[c], v)
    full = d[d.path == 1].groupby("cell_id").v.mean()
    check("full-sample V M4 (150 samples)", full["M4"], 0.773)

    print("== Table 17: complexity quartiles, strict-AND V on H1's 89 samples")
    L = {c["sample_idx"]: sum(1 for l in c["code"].split("\n") if l.strip()) for c in core}
    h1s = set(d[d.cell_id == "H1"].sample_idx)
    q = d[d.cell_id.isin(["H1", "M3"]) & d.sample_idx.isin(h1s)].copy()
    q["q"] = pd.cut(q.sample_idx.map(L), [0, 5, 8, 13, 1000], labels=[1, 2, 3, 4])
    want = {("H1", 1): (0.654, 0.826, 0.684, 0.714), ("M3", 1): (0.654, 0.565, 0.684, 0.762),
            ("H1", 3): (0.769, 0.565, 0.833, 0.381)}
    for (c, p), vals in want.items():
        got = q[(q.cell_id == c) & (q.path == p)].groupby("q", observed=True).v.mean()
        for k, w in enumerate(vals, 1):
            check(f"Table 17 {c} Path {p} Q{k}", got[k], w)

    print("== Table 16: paired before/after on the same 25 HumanEval functions")
    h = load("results_rerun/results_holdout_humaneval_mutated_v2.tsv")
    o = load("results_rerun/results_hem_original_25.tsv")
    t16 = {("H1", 1): (20, 16, 7, 3, 25), ("H1", 2): (20, 19, 1, 0, 25), ("H1", 3): (8, 7, 2, 1, 25),
           ("M3", 1): (16, 15, 3, 2, 25), ("M3", 2): (24, 22, 2, 0, 25), ("M3", 3): (15, 12, 5, 2, 24)}
    for (c, p), (bef, aft, lost, gained, n) in t16.items():
        a = o[(o.cell_id == c) & (o.path == p)].set_index("sample_idx")
        b = h[(h.cell_id == c) & (h.path == p)].set_index("sample_idx")
        idx = sorted(set(a.index) & set(b.index))
        a, b = a.loc[idx], b.loc[idx]
        check(f"Table 16 {c} Path {p} paired n", len(idx), n, 0)
        check(f"Table 16 {c} Path {p} closures before", a.v.sum(), bef, 0)
        check(f"Table 16 {c} Path {p} closures after", b.v.sum(), aft, 0)
        check(f"Table 16 {c} Path {p} lost", (a.v & ~b.v).sum(), lost, 0)
        check(f"Table 16 {c} Path {p} gained", (~a.v & b.v).sum(), gained, 0)

    print("== Table 26: failure types of re-executed phi-4 tests")
    ft = ROOT / "results/phi4_failure_types_long.csv"
    if ft.exists():
        f = pd.read_csv(ft)
        f = f[f.status == "fail"]
        for c, share, n in (("M2", 0.610, 59), ("H2", 0.683, 41), ("H10", 0.589, 56)):
            x = f[f.cell_id == c]
            check(f"{c} failing tests", len(x), n, 0)
            check(f"{c} share of assertion failures", (x.category == "assertion").mean(), share)
    else:
        print("SKIP  failure types (results/phi4_failure_types_long.csv not present)")

    print("== Table 18: closed-weight cells (n = 60)")
    cw = load("results/results_rq4_closed_weight.tsv")
    for c, ad in (("M5_closed", 57), ("M7_gpt", 40), ("M9_gemini", 36), ("H2_closed", 19), ("H9_gemini", 8)):
        x = cw[(cw.cell_id == c) & (cw.path == 1)]
        check(f"{c} Path 1 all_dropped (of 60)", x.ad.sum(), ad, 0)
    x = cw[(cw.cell_id == "H9_gemini") & (cw.path == 1)]
    check("H9_gemini Path 1 V", x.v.mean(), 0.767)

    print("== Section 5.2: Tukey HSD (19 cells with usable rows)")
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        s = d[d.valid.astype(str).str.lower() == "true"]
        t = pairwise_tukeyhsd(s.metric_value, s.cell_id)
        res = pd.DataFrame(t.summary().data[1:], columns=t.summary().data[0])
        check("Tukey cells", s.cell_id.nunique(), 19, 0)
        check("Tukey pairs", len(res), 171, 0)
        check("Tukey significant at 0.05", (res["p-adj"] < 0.05).sum(), 35, 0)
        check("Tukey significant at 0.01", (res["p-adj"] < 0.01).sum(), 33, 0)
    except ImportError:
        print("SKIP  Tukey (statsmodels not installed)")

    print("== Section 5.7 / Appendix C: corrected mixed-effects model (pooled)")
    try:
        sys.path.insert(0, str(ROOT))
        from analyze.mixed_model_refit import load as mm_load, dummies as mm_dummies, fit as mm_fit
        mm = mm_load()
        c = mm[~mm.cell_id.str.startswith("N")].copy()
        sp = c.path.isin([1, 3]).astype(float)
        cd = (c.path == 2).astype(float)
        cols = {**mm_dummies(c, "spec", sp), **mm_dummies(c, "test"), **mm_dummies(c, "code", cd),
                "path=2": (c.path == 2).astype(float), "path=3": (c.path == 3).astype(float)}
        t, _ = mm_fit(c, cols)
        check("mixed model rows", len(c), 5215, 0)
        for term, b in (("test=phi4", -0.159), ("test=llama3.2", -0.122), ("test=mistral-small3.2", -0.099),
                        ("test=gemma4", 0.058), ("code=llama3.2", -0.184), ("spec=llama3.2", -0.096),
                        ("path=2", -0.092), ("path=3", -0.035)):
            check(f"mixed model {term}", t.loc[term, "beta"], b, 0.0006)
    except ImportError as e:
        print(f"SKIP  mixed model ({e})")

    print("== Section 5.11: Krippendorff's alpha (ordinal) on the 60 pairs")
    A = {a: pd.read_csv(ROOT / f"human_eval/data/annotator_{a}_ratings.tsv", sep="\t")
         .drop_duplicates("pair_id", keep="last").set_index("pair_id").rating
         for a in ("A1", "A2", "A3", "E1", "E2")}
    M = pd.DataFrame(A).dropna()

    # ---- Table 8: stage placement (matched samples, cluster bootstrap) ----
    from analyze.stage_placement import main as sp_main
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        SP = {r["cell"]: r for r in sp_main()}
    for c, w, st in [("H11", 0.204, -0.022), ("H7", 0.183, -0.009), ("H1", 0.160, -0.036), ("H6", 0.142, -0.022),
                     ("H8", 0.120, -0.053), ("H9", 0.031, 0.000), ("H4", 0.262, -0.067), ("H5", 0.236, -0.093),
                     ("H2", 0.018, -0.178), ("H3", -0.009, -0.182), ("H10", 0.000, -0.196)]:
        check(f"Table 8 {c} diff vs weakest", SP[c]["weakest"]["diff"], w, 0.0006)
        check(f"Table 8 {c} diff vs strongest", SP[c]["strongest"]["diff"], st, 0.0006)
    check("Table 8: cells beating weakest (CI > 0)", sum(r["weakest"]["lo"] > 0 for r in SP.values()), 7)
    check("Table 8: cells indistinguishable from strongest", sum(r["strongest"]["lo"] <= 0 <= r["strongest"]["hi"] for r in SP.values()), 6)
    check("Table 8: cells exceeding strongest", sum(r["strongest"]["lo"] > 0 for r in SP.values()), 0)
    d75 = d[d.sample_idx < 75].copy()
    d75["v"] = ((d75.metric_value > 0) & (d75.judge_rating >= 3)).astype(float)
    pv = d75.pivot_table(index=["sample_idx", "path"], columns="cell_id", values="v")
    check("H7 vs H2 (same models) diff", (pv.H7 - pv.H2).mean(), 0.165, 0.0006)

    # ---- Section 5.6: stage placement under five judge policies ----
    from analyze.stage_placement import POLICIES, main as sp_run, same_models
    if True:
        for name, (bw, ind_lo, ind_hi) in {"strict-AND rho=2": (7, 5, 5), "strict-AND rho=4": (7, 8, 8),
                                           "metric-only": (6, 4, 4), "judge-only": (7, 6, 6)}.items():
            with contextlib.redirect_stdout(io.StringIO()):
                r = sp_run(10_000, POLICIES[name], quiet=True, write=False)
            check(f"5.6 {name}: beat weakest", sum(x["weakest"]["lo"] > 0 for x in r), bw, 0)
            check(f"5.6 {name}: exceed strongest", sum(x["strongest"]["lo"] > 0 for x in r), 0, 0)
            check(f"5.6 {name}: indistinguishable from strongest",
                  sum(x["strongest"]["lo"] <= 0 <= x["strongest"]["hi"] for x in r), ind_lo, 0)
        for name, want in {"strict-AND rho=4": 0.112, "metric-only": 0.125, "judge-only": 0.170}.items():
            est, lo, _ = same_models(POLICIES[name])
            check(f"5.6 {name}: H7 - H2", est, want, 0.0006)
            check(f"5.6 {name}: H7 - H2 CI lower > 0", float(lo > 0), 1.0, 0)

    # ---- Table 15: interaction ANOVA with rank-correct df (nested least squares) ----
    s15 = d[d.metric_value.notna() & (d.judge_rating >= 0)]

    def _rss(df, terms):
        cols = [np.ones(len(df))]
        for t in terms:
            if isinstance(t, tuple):
                a = pd.get_dummies(df[t[0]], drop_first=True).astype(float)
                b = pd.get_dummies(df[t[1]], drop_first=True).astype(float)
                cols += [(a[x] * b[y]).values for x in a for y in b]
            else:
                cols += [c_.values for _, c_ in pd.get_dummies(df[t], drop_first=True).astype(float).items()]
        M = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(M, df.metric_value.values, rcond=None)
        r = df.metric_value.values - M @ beta
        return r @ r, np.linalg.matrix_rank(M)
    full, rk = _rss(s15, ["cell_id", "path", "sample_idx", ("cell_id", "path")])
    red, rk2 = _rss(s15, ["cell_id", "path", "sample_idx"])
    check("Table 15 n", len(s15), 5071, 0)
    check("Table 15 residual df", len(s15) - rk, 4867, 0)
    check("Table 15 interaction df", rk - rk2, 34, 0)
    check("Table 15 interaction F", ((red - full) / (rk - rk2)) / (full / (len(s15) - rk)), 6.59, 0.006)
    check("Table 15 interaction partial eta2", (red - full) / (red - full + full), 0.044, 0.0006)

    # ---- Appendix C: logit robustness check ----
    from analyze.mixed_model_refit import logit_check
    with contextlib.redirect_stdout(io.StringIO()):
        lg = logit_check()
    for term, want in {"test=phi4": -15.9, "test=llama3.2": -12.4, "test=mistral-small3.2": -10.1, "test=gemma4": 5.8,
                       "code=llama3.2": -17.9, "spec=llama3.2": -9.2, "code=gemma4": -8.5}.items():
        check(f"App C logit AME {term} (pp)", lg.loc[term, "ame_pp"], want, 0.06)
    check("App C logit p code=gemma4", lg.loc["code=gemma4", "p"], 0.017, 0.0006)
    check("App C logit p path=3", lg.loc["path=3", "p"], 0.25, 0.006)

    # ---- Table 24 caption: H10 all-dropped composition ----
    h10 = d[(d.cell_id == "H10") & (d.path == 1)]
    check("H10 Path 1 nominal all-dropped rows", h10.notes.astype(str).str.contains("dropped all").sum(), 41, 0)

    # ---- G3: strict-AND vs either-signal closure ----
    nn = d[~d.cell_id.str.startswith("N")].copy()
    mm, jj = nn.metric_value > 0, nn.judge_rating >= 3
    nn["sa"], nn["un"] = (mm & jj).astype(float), (mm | jj).astype(float)
    gap = nn.groupby(["cell_id", "path"]).apply(lambda t: t.un.mean() - t.sa.mean())
    check("G3 gap min (pp)", 100 * gap.min(), 8.0, 0.06)
    check("G3 gap max (pp)", 100 * gap.max(), 52.0, 0.06)
    check("G3 gap median (pp)", 100 * gap.median(), 25.3, 0.06)

    def alpha(X: np.ndarray) -> float:
        X = np.asarray(X, int)
        O = np.zeros((5, 5))
        m_ = X.shape[1]
        for row in X:
            for i in range(m_):
                for j in range(m_):
                    if i != j:
                        O[row[i], row[j]] += 1 / (m_ - 1)
        nc = O.sum(1)
        N = nc.sum()
        dd = np.array([[(nc[min(a, b):max(a, b) + 1].sum() - (nc[min(a, b)] + nc[max(a, b)]) / 2) ** 2
                        for b in range(5)] for a in range(5)])
        return 1 - (O * dd).sum() / ((np.outer(nc, nc) * dd).sum() / (N - 1))

    check("alpha, 5 annotators", alpha(M.values), 0.193)
    check("alpha, 3 author-team annotators", alpha(M[["A1", "A2", "A3"]].values), 0.376)
    rng = np.random.default_rng(42)
    boot = [alpha(M.values[rng.integers(0, len(M), len(M))]) for _ in range(5000)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    check("alpha bootstrap 95% CI lower (5000 resamples, seed 42)", lo, 0.07, 0.006)
    check("alpha bootstrap 95% CI upper", hi, 0.31, 0.006)

    print(f"\n{'ALL CHECKS PASSED' if FAIL == 0 else f'{FAIL} CHECK(S) FAILED'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
