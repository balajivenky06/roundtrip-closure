"""
Stage placement: does a heterogeneous pipeline beat the single-model pipelines
of its own component models?  (RQ1, sample-matched)

All cells were run on sample_idx 0-74; we compare on exactly those samples.
For each hetero cell we compare its strict-AND closure (pooled over the three
paths) with the mono pipeline of each model it uses, and report the paired
difference with a 95% cluster-bootstrap CI that resamples SAMPLES (keeping each
sample's three paths together), B = 10,000, seed 42.

Run:  python3 analyze/stage_placement.py
Out:  results/stage_placement.json, tables/tab_stage_placement.tex
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONO_OF = {"llama3.2": "M1", "phi4": "M2", "qwen3.6": "M3", "gemma4": "M4", "mistral-small3.2": "M5", "qwen3-coder": "M6"}
NICE = {"M1": "llama3.2", "M2": "phi-4", "M3": "qwen3.6", "M4": "gemma-4", "M5": "mistral-s.", "M6": "qwen-coder"}
HET = {  # spec, test, code
    "H1": ("phi4", "qwen3-coder", "qwen3-coder"), "H2": ("qwen3.6", "phi4", "qwen3-coder"),
    "H3": ("gemma4", "mistral-small3.2", "qwen3-coder"), "H4": ("qwen3-coder", "qwen3-coder", "llama3.2"),
    "H5": ("llama3.2", "qwen3-coder", "qwen3-coder"), "H6": ("phi4", "qwen3.6", "phi4"),
    "H7": ("qwen3-coder", "qwen3.6", "phi4"), "H8": ("gemma4", "qwen3-coder", "mistral-small3.2"),
    "H9": ("qwen3-coder", "qwen3.6", "qwen3-coder"), "H10": ("mistral-small3.2", "phi4", "qwen3-coder"),
    "H11": ("phi4", "gemma4", "qwen3-coder"),
}


def main(B: int = 10_000, policy=None, quiet: bool = False, write: bool = True) -> list[dict]:
    d = pd.read_csv(ROOT / "results/results_roundtrip.tsv", sep="\t")
    m, j = d.metric_value.fillna(-1), d.judge_rating.fillna(-9)
    d["v"] = (policy(m, j) if policy else (m > 0) & (j >= 3)).astype(float)
    d = d[d.sample_idx < 75]
    piv = d.pivot_table(index=["sample_idx", "path"], columns="cell_id", values="v")
    rng = np.random.default_rng(42)
    samples = np.array(sorted(d.sample_idx.unique()))
    boot_idx = [rng.choice(samples, size=len(samples), replace=True) for _ in range(B)]
    out = []
    for h, models in HET.items():
        monos = sorted({MONO_OF[m] for m in models})
        avg = {m: piv[m].mean() for m in monos}
        weak, strong = min(monos, key=avg.get), max(monos, key=avg.get)
        rec = {"cell": h, "V": piv[h].mean()}
        for tag, m in (("weakest", weak), ("strongest", strong)):
            x = piv[[h, m]].dropna()
            diff = (x[h] - x[m]).groupby(level=0).sum()          # per-sample sum over paths
            npair = x.groupby(level=0).size()
            est = diff.sum() / npair.sum()
            bs = [diff.reindex(bi).sum() / npair.reindex(bi).sum() for bi in boot_idx]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec[tag] = {"mono": m, "V_mono": avg[m], "diff": est, "lo": lo, "hi": hi}
        out.append(rec)
        w, s = rec["weakest"], rec["strongest"]
        if not quiet:
            print(f"{h:4s} V={rec['V']:.3f} | vs weakest {w['mono']} ({w['V_mono']:.3f}): {w['diff']:+.3f} [{w['lo']:+.3f},{w['hi']:+.3f}]"
                  f" | vs strongest {s['mono']} ({s['V_mono']:.3f}): {s['diff']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}]")
    if write:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results/stage_placement.json").write_text(json.dumps(out, indent=1, default=float))
    return out


POLICIES = {
    "strict-AND rho=3": lambda m, j: (m > 0) & (j >= 3),
    "strict-AND rho=2": lambda m, j: (m > 0) & (j >= 2),
    "strict-AND rho=4": lambda m, j: (m > 0) & (j >= 4),
    "metric-only": lambda m, j: m > 0,
    "judge-only": lambda m, j: j >= 3,
}


def same_models(policy=None, B: int = 10_000) -> tuple[float, float, float]:
    """H7 minus H2 (same three models, different stages), cluster bootstrap."""
    d = pd.read_csv(ROOT / "results/results_roundtrip.tsv", sep="\t")
    d = d[d.sample_idx < 75]
    m, j = d.metric_value.fillna(-1), d.judge_rating.fillna(-9)
    d["v"] = (policy(m, j) if policy else (m > 0) & (j >= 3)).astype(float)
    x = d.pivot_table(index=["sample_idx", "path"], columns="cell_id", values="v")[["H7", "H2"]].dropna()
    diff, n = (x.H7 - x.H2).groupby(level=0).sum(), x.groupby(level=0).size()
    rng = np.random.default_rng(42)
    s = np.array(sorted(d.sample_idx.unique()))
    bs = [diff.reindex(b).sum() / n.reindex(b).sum() for b in (rng.choice(s, len(s)) for _ in range(B))]
    return diff.sum() / n.sum(), *np.percentile(bs, [2.5, 97.5])


def policy_summary(B: int = 10_000) -> dict:
    out = {}
    for name, f in POLICIES.items():
        r = main(B, f, quiet=True, write=False)
        out[name] = {
            "beat_weakest": sum(x["weakest"]["lo"] > 0 for x in r),
            "indistinguishable_from_strongest": sum(x["strongest"]["lo"] <= 0 <= x["strongest"]["hi"] for x in r),
            "exceed_strongest": sum(x["strongest"]["lo"] > 0 for x in r),
            "H7_minus_H2": same_models(f, B),
        }
        print(name, out[name])
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "policies":
        policy_summary()
    else:
        main()
