"""
Mixed-effects linear-probability model of strict-AND closure: original
specification (reproduces the submitted Table 31) and corrected refits.

Original (scripts/build_mixed_logit.py):
    rows with a defined metric and a parsed judge rating; null cells N2/N3
    dropped, N1 kept; all three stage dummies on every path.
Problems (R2.1-type): stage dummies enter paths that never invoke the stage,
and rows whose tests were all dropped are excluded, so the model never sees
test-filter failures.

Corrected specifications:
  A. Per-path models. For each path, V ~ dummies for the two ACTIVE stages
     (spec+test on Paths 1/3; test+code on Path 2) + random sample intercept.
  B. Pooled model with active-stage coding: spec dummies x I(path in {1,3}),
     test dummies on all paths, code dummies x I(path = 2), path effects,
     random sample intercept.
In both: every row of every non-null cell is kept; a row closes only if
metric > 0 and judge >= 3 (filter failures and judge parse failures count as
non-closure, exactly as V in Eq. 14). Null cells (N1-N3) are excluded because
N1's word-shuffled input is not a property of its stage models.
Reference model at every stage: qwen3.6 (as in the paper).

Run: python3 analyze/mixed_model_refit.py
Out: results/mixed_model_refit.json, tables/tab_mixed_refit.tex
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STAGES = {  # (spec, test, code) short names, from doe.py
    "M1": ("llama3.2",) * 3, "M2": ("phi4",) * 3, "M3": ("qwen3.6",) * 3, "M4": ("gemma4",) * 3,
    "M5": ("mistral-small3.2",) * 3, "M6": ("qwen3-coder",) * 3,
    "H1": ("phi4", "qwen3-coder", "qwen3-coder"), "H2": ("qwen3.6", "phi4", "qwen3-coder"),
    "H3": ("gemma4", "mistral-small3.2", "qwen3-coder"), "H4": ("qwen3-coder", "qwen3-coder", "llama3.2"),
    "H5": ("llama3.2", "qwen3-coder", "qwen3-coder"), "H6": ("phi4", "qwen3.6", "phi4"),
    "H7": ("qwen3-coder", "qwen3.6", "phi4"), "H8": ("gemma4", "qwen3-coder", "mistral-small3.2"),
    "H9": ("qwen3-coder", "qwen3.6", "qwen3-coder"), "H10": ("mistral-small3.2", "phi4", "qwen3-coder"),
    "H11": ("phi4", "gemma4", "qwen3-coder"), "N1": ("llama3.2",) * 3,
}
MODELS = ["gemma4", "llama3.2", "mistral-small3.2", "phi4", "qwen3-coder"]  # reference: qwen3.6
REF = "qwen3.6"


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results/results_roundtrip.tsv", sep="\t")
    d["v"] = ((d.metric_value > 0) & (d.judge_rating >= 3)).astype(float)
    st = d.cell_id.map(STAGES)
    d = d[st.notna()].copy()
    d[["spec", "test", "code"]] = pd.DataFrame(d.cell_id.map(STAGES).tolist(), index=d.index)
    return d


def fit(d: pd.DataFrame, cols: dict[str, pd.Series]) -> pd.DataFrame:
    X = pd.DataFrame(cols, index=d.index).astype(float)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    r = MixedLM(d.v.values, X.values, groups=d.sample_idx.values).fit(reml=True, method="lbfgs")
    k = X.shape[1]
    return pd.DataFrame({"beta": r.fe_params[:k], "se": r.bse_fe[:k], "p": r.pvalues[:k]}, index=X.columns), r


def dummies(d, stage, mask=None):
    out = {}
    for m in MODELS:
        x = (d[stage] == m).astype(float)
        if mask is not None:
            x = x * mask
        out[f"{stage}={m}"] = x
    return out


def main() -> None:
    d = load()
    res = {}

    # Original specification (reproduction check)
    o = d[d.metric_value.notna() & (d.judge_rating >= 0)]
    cols = {**dummies(o, "spec"), **dummies(o, "test"), **dummies(o, "code"),
            "path=2": (o.path == 2).astype(float), "path=3": (o.path == 3).astype(float)}
    t, _ = fit(o, cols)
    res["original"] = {"n": len(o), "coef": t.round(4).to_dict("index")}
    print(f"ORIGINAL spec (n={len(o)}):"); print(t.round(3).to_string())

    # Corrected: all rows of non-null cells
    c = d[~d.cell_id.str.startswith("N")].copy()
    print(f"\nCorrected sample: n={len(c)} rows, positive={int(c.v.sum())} ({c.v.mean():.3f})")

    # A. per-path, active stages only
    res["per_path"] = {}
    for p, act in ((1, ("spec", "test")), (2, ("test", "code")), (3, ("spec", "test"))):
        q = c[c.path == p]
        cols = {}
        for s in act:
            cols.update(dummies(q, s))
        t, _ = fit(q, cols)
        res["per_path"][p] = {"n": len(q), "active": act, "coef": t.round(4).to_dict("index")}
        print(f"\nPATH {p} (active {act}, n={len(q)}):"); print(t.round(3).to_string())

    # B. pooled, active-stage coding
    sp = c.path.isin([1, 3]).astype(float)
    cd = (c.path == 2).astype(float)
    cols = {**dummies(c, "spec", sp), **dummies(c, "test"), **dummies(c, "code", cd),
            "path=2": (c.path == 2).astype(float), "path=3": (c.path == 3).astype(float)}
    t, _ = fit(c, cols)
    res["pooled_active"] = {"n": len(c), "coef": t.round(4).to_dict("index")}
    print(f"\nPOOLED active-stage coding (n={len(c)}):"); print(t.round(3).to_string())

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/mixed_model_refit.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()


def build_table() -> None:
    """Rebuild results and write tables/tab_mixed_refit.tex (pooled + per-path)."""
    d = load()
    c = d[~d.cell_id.str.startswith("N")].copy()
    sp = c.path.isin([1, 3]).astype(float)
    cd = (c.path == 2).astype(float)
    cols = {**dummies(c, "spec", sp), **dummies(c, "test"), **dummies(c, "code", cd),
            "path=2": (c.path == 2).astype(float), "path=3": (c.path == 3).astype(float)}
    pooled, rp = fit(c, cols)
    per = {}
    for p, act in ((1, ("spec", "test")), (2, ("test", "code")), (3, ("spec", "test"))):
        q = c[c.path == p]
        cc = {}
        for s in act:
            cc.update(dummies(q, s))
        per[p], _ = fit(q, cc)

    def cell(t, term):
        if term not in t.index:
            return "---"
        b, pv = t.loc[term, "beta"], t.loc[term, "p"]
        star = "^{***}" if pv < 0.001 else "^{**}" if pv < 0.01 else "^{*}" if pv < 0.05 else ""
        num = f"{abs(b):.3f}"
        txt = f"${'-' if b < 0 else '+'}{num}{star}$"
        return f"\\textbf{{{txt}}}" if pv < 0.05 else txt

    nice = {"gemma4": "gemma-4", "llama3.2": "llama3.2", "mistral-small3.2": "mistral-small",
            "phi4": "phi-4", "qwen3-coder": "qwen-coder"}
    lines = []
    for stage in ("spec", "test", "code"):
        for m in MODELS:
            term = f"{stage}={m}"
            lines.append(f"{stage} = {nice[m]} & {cell(pooled, term)} & {cell(per[1], term)} & "
                         f"{cell(per[2], term)} & {cell(per[3], term)} \\\\")
        lines.append("\\midrule")
    lines.append(f"path = 2 & {cell(pooled, 'path=2')} & --- & --- & --- \\\\")
    lines.append(f"path = 3 & {cell(pooled, 'path=3')} & --- & --- & --- \\\\")
    n = {p: int((c.path == p).sum()) for p in (1, 2, 3)}
    tex = r"""\begin{table}[!htbp]
\centering
\small
\caption{Mixed-effects linear-probability models of strict-AND valid closure (Eq.~\ref{eq:mixedlogit}), with a random intercept per sample and qwen3.6 as the reference model at every stage. Coefficients are changes in $\Pr[V = 1]$. Every row of every non-null cell is included, so rows whose generated tests were all dropped count as non-closures. Stage terms enter only on the paths that invoke the stage: the \emph{pooled} model codes spec terms on Paths~1 and~3 and code terms on Path~2 only; the \emph{per-path} models include only the two active stages (``---'': stage not invoked on that path). $n$ = """ + f"{len(c):,}".replace(",", "{,}") + r""" rows pooled (""" + f"{n[1]:,}".replace(",", "{,}") + ", " + f"{n[2]:,}".replace(",", "{,}") + ", " + f"{n[3]:,}".replace(",", "{,}") + r""" per path). Bold: $p < 0.05$ ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$), uncorrected for the number of stage terms.}
\label{tab:mixed_logit}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Term & Pooled & Path 1 & Path 2 & Path 3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/tab_mixed_refit.tex").write_text(tex)
    print("sigma2_sample (pooled):", float(np.asarray(rp.cov_re).ravel()[0]))
    print("positive rate:", round(c.v.mean(), 3), "n:", len(c))


def logit_check() -> pd.DataFrame:
    """Robustness check: logistic regression with the pooled active-stage terms,
    standard errors clustered by sample (Newton-Raphson, sandwich estimator).
    Reports the average marginal effect (pp) of each stage dummy over the rows
    on which the term applies, and the p-value of the logit coefficient."""
    from scipy.stats import norm
    d = load()
    c = d[~d.cell_id.str.startswith("N")].copy()
    sp = c.path.isin([1, 3]).astype(float)
    cd = (c.path == 2).astype(float)
    cols = {**dummies(c, "spec", sp), **dummies(c, "test"), **dummies(c, "code", cd),
            "path=2": (c.path == 2).astype(float), "path=3": (c.path == 3).astype(float)}
    X = pd.DataFrame(cols)
    X.insert(0, "const", 1.0)
    A, y = X.values, c.v.values.astype(float)
    b = np.zeros(A.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-A @ b))
        step = np.linalg.solve(A.T @ (A * (p * (1 - p))[:, None]), A.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-A @ b))
    Hinv = np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))
    g = pd.DataFrame(A * (y - p)[:, None]).groupby(c.sample_idx.values).sum().values
    se = np.sqrt(np.diag(Hinv @ (g.T @ g) @ Hinv))
    rows = {}
    for i, name in enumerate(X.columns[1:], start=1):
        stage = name.split("=")[0]
        appl = (sp.values == 1) if stage == "spec" else (cd.values == 1) if stage == "code" else np.ones(len(A), bool)
        grp = [j for j, nm in enumerate(X.columns) if nm.split("=")[0] == stage]
        A1, A0 = A[appl].copy(), A[appl].copy()
        A1[:, grp] = 0
        A0[:, grp] = 0
        A1[:, i] = 1
        ame = (1 / (1 + np.exp(-A1 @ b)) - 1 / (1 + np.exp(-A0 @ b))).mean()
        rows[name] = {"ame_pp": 100 * ame, "p": 2 * norm.sf(abs(b[i] / se[i]))}
    return pd.DataFrame(rows).T


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "logit":
    print(logit_check().round(4).to_string())

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "table":
    build_table()
