"""
R1.5 — why do phi-4's generated tests fail on the reference code?

RUN ON COLAB with the Drive LLM cache mounted (same setup as
scripts/regen_per_operator.py). Nothing here calls an LLM: every test suite
is recovered from the cache and re-executed locally with pytest.

For each Path-1 row whose generated tests were ALL dropped by the test filter
("filter dropped all tests" in the TSV notes), the script:
  1. recovers D' = L_spec(code) and T' = L_test(D') from the cache, using
     the exact prompt/param reconstruction of scripts/regen_per_operator.py;
  2. splits T' into individual test functions (same prefix rule as the
     filter, closure_metrics._split_test_functions);
  3. runs each test against the ORIGINAL reference code with the same
     harness preparation as mutation_testing.run_tests_against_code
     (bare-assert wrapping, removal of LLM re-definitions of the function),
     capturing pytest's failure summary;
  4. classifies every failing test into one category:
        assertion      AssertionError (wrong expected value)
        type_error     TypeError (wrong arity / argument types)
        name_error     NameError (undefined helper or wrong function name)
        runtime_other  other exception raised inside the code or test
                       (ValueError, IndexError, KeyError, AttributeError, ...)
        syntax         SyntaxError / IndentationError / collection error
        import_error   ImportError / ModuleNotFoundError
        timeout        exceeded the per-test timeout (10 s, as in the sweep)
        no_tests       file collected no tests
        other          anything not matched above

Default cells: phi-4 at L_test (M2, H2, H10 re-run) plus two control cells
whose test generator is not phi-4 (M3, M4). Add closed-weight cells with
--cells (their cache keys use the same scheme).

Outputs
  results/phi4_failure_types_long.csv      one row per test function
  results/phi4_failure_types_summary.json  per-cell counts, coverage
  tables/tab_phi4_failure_types.tex        LaTeX table for the paper

Usage (Colab):
  python3 analyze/phi4_failure_types.py
  python3 analyze/phi4_failure_types.py --cells M2 H2 H10 M3 M4 M5_closed M7_gpt M9_gemini
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import mutation_testing as mt
from closure_metrics import _split_test_functions
from doe import CELLS_BY_ID
from scripts.regen_per_operator import (
    cache_lookup_docstring,
    cache_lookup_tests,
    load_samples,
)

TIMEOUT = mt.TIMEOUT_PER_TEST
CATS = ["assertion", "type_error", "name_error", "runtime_other",
        "syntax", "import_error", "timeout", "no_tests", "other"]

TSV_FOR_CELL = {
    # H10's original sweep is contaminated by Ollama HTTP-400 rows (Table 24);
    # the clean re-run is the right source for the failure analysis.
    "H10": Path("results_rerun/results_h10_rerun.tsv"),
    "M5_closed": Path("results/results_rq4_closed_weight.tsv"),
    "H8_closed": Path("results/results_rq4_closed_weight.tsv"),
    "M7_gpt": Path("results/results_rq4_closed_weight.tsv"),
    "M9_gemini": Path("results/results_rq4_closed_weight.tsv"),
}
DEFAULT_TSV = Path("results/results_roundtrip.tsv")

_FAILED = re.compile(r"^(FAILED|ERROR)\s+\S+(?:\s+-\s+(.*))?$", re.M)
_EXC = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Exit|Interrupt))\b")


def classify(output: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if "no tests ran" in output or "collected 0 items" in output:
        return "no_tests"
    text = output
    m = _FAILED.findall(output)
    if m:
        text = " ".join(f"{kind} {msg or ''}" for kind, msg in m) + "\n" + output
    if re.search(r"SyntaxError|IndentationError|TabError", text):
        return "syntax"
    if re.search(r"ModuleNotFoundError|ImportError", text):
        return "import_error"
    first = None
    for kind, msg in m:
        msg = (msg or "").strip()
        if msg.startswith("assert") or "AssertionError" in msg:
            first = "AssertionError"
            break
        e = _EXC.search(msg)
        if e:
            first = e.group(1)
            break
    if first is None:
        e = _EXC.search(output)
        first = e.group(1) if e else None
    if first is None:
        return "assertion" if re.search(r"^E\s+assert ", output, re.M) else "other"
    if first == "AssertionError":
        return "assertion"
    if first == "TypeError":
        return "type_error"
    if first in ("NameError", "UnboundLocalError"):
        return "name_error"
    if first in ("SyntaxError", "IndentationError"):
        return "syntax"
    return "runtime_other"


def run_one(test_src: str, code: str) -> tuple[str, str]:
    """Run one test function against the reference code; return (status, category)."""
    fn = re.search(r"def\s+(\w+)\s*\(", code)
    fn_name = fn.group(1) if fn else ""
    t = mt._wrap_bare_asserts(test_src)
    if fn_name:
        t = mt._strip_function_redefinition(t, fn_name)
    if "def check(candidate)" in t and fn_name:
        t += f"\n\ndef test_humaneval():\n    check({fn_name})\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "test_mut.py")
        Path(p).write_text(f"{code}\n\n# --- Generated tests ---\n{t}\n")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", p, "-q", "--no-header",
                 "-rfE", "--tb=short", "-p", "no:cacheprovider"],
                capture_output=True, text=True, timeout=TIMEOUT, cwd=tmp)
        except subprocess.TimeoutExpired:
            return "fail", "timeout"
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return "pass", ""
    return "fail", classify(out, False)


def analyse_cell(cell_id: str, samples: dict, rows_out: list) -> dict:
    cell = CELLS_BY_ID[cell_id]
    tsv = TSV_FOR_CELL.get(cell_id, DEFAULT_TSV)
    d = pd.read_csv(tsv, sep="\t")
    d = d[(d.cell_id == cell_id) & (d.path == 1)]
    dropped = d[d.notes.astype(str).str.contains("dropped all")]
    stats = {"cell": cell_id, "tsv": str(tsv), "rows_all_dropped": len(dropped),
             "rows_recovered": 0, "tests": 0, "tests_passing_now": 0,
             "per_test": Counter(), "per_row_majority": Counter()}
    for r in dropped.itertuples():
        s = samples.get(int(r.sample_idx))
        if s is None:
            continue
        d_prime = cache_lookup_docstring(cell.L_spec.ollama_tag, s["code"])
        if d_prime is None:
            continue
        t_prime = cache_lookup_tests(cell.L_test.ollama_tag, d_prime,
                                     s.get("entry_point", ""), s.get("signature", ""))
        if not t_prime:
            continue
        stats["rows_recovered"] += 1
        tests = _split_test_functions(t_prime) or [t_prime]
        row_cats = Counter()
        for k, tsrc in enumerate(tests):
            status, cat = run_one(tsrc, s["code"])
            stats["tests"] += 1
            if status == "pass":
                # Should not happen for an all-dropped row; counted to expose
                # environment drift between the sweep and this re-execution.
                stats["tests_passing_now"] += 1
                cat = "passes_now"
            else:
                stats["per_test"][cat] += 1
                row_cats[cat] += 1
            rows_out.append({"cell_id": cell_id, "sample_idx": int(r.sample_idx),
                             "sample_source": r.sample_source, "test_no": k,
                             "status": status, "category": cat})
        if row_cats:
            stats["per_row_majority"][row_cats.most_common(1)[0][0]] += 1
    return stats


def to_tex(all_stats: list[dict]) -> str:
    head = " & ".join(["Cell", "rows rec.", "tests"] + [c.replace("_", r"\_") for c in CATS[:7]])
    lines = []
    for s in all_stats:
        n = max(1, s["tests"] - s["tests_passing_now"])
        cells = [f"{100*s['per_test'].get(c,0)/n:.0f}\\%" for c in CATS[:7]]
        lines.append(" & ".join([s["cell"].replace("_", r"\_"),
                                 f"{s['rows_recovered']}/{s['rows_all_dropped']}",
                                 str(s["tests"] - s["tests_passing_now"])] + cells) + r" \\")
    return (r"""\begin{table*}[!htbp]
  \centering
  \small
  \caption{Why generated tests fail on the reference code (Path~1 rows in
    which the filter dropped every test). Each failing test function was
    recovered from the LLM cache and re-executed against the original
    function; entries are the share of failing tests per category.
    ``rows rec.'' is the number of all-dropped rows whose test suite was
    recoverable from the cache. phi-4 is the test generator in M2, H2 and
    H10 (clean re-run); M3 and M4 are control cells with a different test
    generator.}
  \label{tab:phi4_failure_types}
  \begin{tabular}{@{}lrr""" + "r" * 7 + r"""@{}}
    \toprule
    """ + head + r""" \\
    \midrule
""" + "\n".join("    " + l for l in lines) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["M2", "H2", "H10", "M3", "M4"])
    a = ap.parse_args()
    samples = load_samples()
    rows: list[dict] = []
    all_stats = []
    for c in a.cells:
        s = analyse_cell(c, samples, rows)
        all_stats.append(s)
        print(f"{c}: recovered {s['rows_recovered']}/{s['rows_all_dropped']} rows, "
              f"{s['tests']} tests, passing-now {s['tests_passing_now']}, "
              f"{dict(s['per_test'])}")
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/phi4_failure_types_long.csv", index=False)
    Path("results/phi4_failure_types_summary.json").write_text(json.dumps(
        [{**{k: v for k, v in s.items() if k not in ("per_test", "per_row_majority")},
          "per_test": dict(s["per_test"]), "per_row_majority": dict(s["per_row_majority"])}
         for s in all_stats], indent=1))
    Path("tables").mkdir(exist_ok=True)
    Path("tables/tab_phi4_failure_types.tex").write_text(to_tex(all_stats))
    print("Wrote results/phi4_failure_types_long.csv, results/phi4_failure_types_summary.json, "
          "tables/tab_phi4_failure_types.tex")


if __name__ == "__main__":
    main()
