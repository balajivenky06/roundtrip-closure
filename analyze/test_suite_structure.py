"""
R1.5 — mechanism of the phi-4 test-stage failure, from the released sweep logs.

The sweep log records, for every Path-1 row, the test-filter outcome
    "[<cell> s=<sample> p1] step3 test_filter -> K/N tests kept (a/b chars)"
where N is the number of `def test_*` functions the test-stage model wrote,
K the number that pass on the reference code, and b the length of the raw
test suite. (The TSV note "filter dropped all tests" is written whenever
K = 0 for any reason, so on its own it cannot distinguish an empty output
from a suite whose tests all failed; the log can.)

For every cell whose log contains exactly one Path-1 run (one line per
sample; H3, H8 and H10 are excluded because their logs contain partial or
repeated runs), the script reports:
    * agreement between the log (K = 0) and the TSV all-dropped flag;
    * all-dropped rows split into "no test functions / empty output"
      (N = 0 or b = 0) and "N >= 1 tests written, all failed";
    * the share of suites written as a single test function (N = 1);
    * the all-dropped rate for single-function vs multi-function suites.

Run locally (no LLM, no cache):  python3 analyze/test_suite_structure.py
Outputs: results/test_suite_structure.json, tables/tab_test_suite_structure.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"\[(\w+) s=(\d+) p1\] step3 test_filter -> (\d+)/(\d+) tests kept \((\d+)/(\d+) chars\)")
TEST_MODEL = {  # model at L_test, for the table
    "M1": "llama3.2", "M2": "phi-4", "M3": "qwen3.6", "M4": "gemma-4", "M5": "mistral-s.",
    "M6": "qwen-coder", "H1": "qwen-coder", "H2": "phi-4", "H4": "qwen-coder",
    "H5": "qwen-coder", "H6": "qwen3.6", "H7": "qwen3.6", "H9": "qwen3.6", "H11": "gemma-4",
}
ORDER = ["M2", "H2", "M1", "M3", "M4", "M5", "M6", "H1", "H4", "H5", "H6", "H7", "H9", "H11"]


def parse_log(cell: str) -> pd.DataFrame | None:
    lines = [PAT.search(l) for l in open(ROOT / f"logs/cell_{cell}_core.log", errors="ignore")]
    rec = [m.groups() for m in lines if m and m.group(1) == cell]
    df = pd.DataFrame(rec, columns=["cell", "s", "kept", "n", "fch", "tch"]).astype(
        {"s": int, "kept": int, "n": int, "fch": int, "tch": int})
    if df.s.duplicated().any():
        return None  # repeated runs: cannot align log lines with TSV rows
    return df.set_index("s")


def main() -> None:
    d = pd.read_csv(ROOT / "results/results_roundtrip.tsv", sep="\t")
    d = d[d.path == 1]
    out = []
    for c in ORDER:
        L = parse_log(c)
        if L is None:
            print(f"skip {c}: repeated runs in log")
            continue
        x = d[d.cell_id == c].set_index("sample_idx")
        x["ad"] = x.notes.astype(str).str.contains("dropped all")
        j = x.join(L, how="inner")
        agree = int(((j.kept == 0) == j.ad).sum())
        a = j[j.ad]
        single, multi = j[j.n == 1], j[j.n >= 2]
        r = {
            "cell": c, "test_model": TEST_MODEL[c], "rows": len(x), "rows_logged": len(j),
            "log_tsv_agreement": agree,
            "all_dropped": int(x.ad.sum()),
            "ad_empty_or_no_tests": int(((a.n == 0) | (a.tch == 0)).sum()),
            "ad_tests_written_all_failed": int(((a.n >= 1) & (a.tch > 0)).sum()),
            "share_single_function": float((j.n == 1).mean()),
            "n_single": len(single), "n_multi": len(multi),
            "ad_rate_single": float(single.ad.mean()) if len(single) else None,
            "ad_rate_multi": float(multi.ad.mean()) if len(multi) else None,
            "median_tests_per_suite": float(j.n.median()),
        }
        out.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/test_suite_structure.json").write_text(json.dumps(out, indent=1))

    def pct(v):
        return "---" if v is None else f"{100*v:.0f}\\%"
    rows = []
    for r in out:
        rows.append(
            f"    {r['cell']} & {r['test_model']} & {r['all_dropped']}/{r['rows']} & "
            f"{r['ad_tests_written_all_failed']} / {r['ad_empty_or_no_tests']} & "
            f"{pct(r['share_single_function'])} & {pct(r['ad_rate_single'])} ({r['n_single']}) & "
            f"{pct(r['ad_rate_multi'])} ({r['n_multi']}) \\\\")
    tex = r"""\begin{table}[!htbp]
  \centering
  \small
  \caption{Structure of the generated test suites on Path~1, from the sweep
    logs. ``All-dropped'' rows are those in which no generated test passes
    on the reference code; they are split into rows where at least one test
    function was written and all failed, and rows with no test function or
    an empty output. ``Single fn.'' is the share of suites written as one
    test function. The last two columns give the all-dropped rate for
    single-function and multi-function suites (number of suites in
    parentheses). Cells whose logs contain partial or repeated runs (H3,
    H8, H10) are omitted.}
  \label{tab:test_suite_structure}
  \begin{tabular}{@{}llrrrrr@{}}
    \toprule
    Cell & $L_{\textsf{test}}$ & All-dropped & Failed / empty & Single fn. & All-dropped if single & All-dropped if multi \\
    \midrule
""" + "\n".join(rows[:2]) + "\n    \\midrule\n" + "\n".join(rows[2:]) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables/tab_test_suite_structure.tex").write_text(tex)
    print("Wrote results/test_suite_structure.json and tables/tab_test_suite_structure.tex")


if __name__ == "__main__":
    main()
