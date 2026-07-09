"""
scripts/regen_per_operator.py — regenerate per-mutation-operator kill rates from cache.

G2 gap-closer. RUN THIS ON COLAB (needs the LLM cache on Drive).

For every Path 1 row in the TSV, this script:
    1. Looks up sample by (sample_idx, sample_source) in data/core_sample_150.jsonl
    2. Reconstructs cache keys for L_spec(code) and L_test(docstring, name, sig)
       using the same params closure_paths._call_doc_from_code /
       _call_tests_from_doc use.
    3. Recovers D' and T' from the LLM cache.
    4. Runs closure_metrics.test_filter(T', code) to get filtered_tests.
    5. Runs mutation_testing.evaluate_mutants(code, filtered_tests) — which
       returns a `per_operator` dict {operator: {total, killed}}.
    6. Writes rows to results/tab_per_operator_long.csv.

Cache-miss rows (T' not recoverable) are logged and skipped.

Output:
    results/tab_per_operator_long.csv
        cell_id, sample_idx, sample_source, operator, total, killed, kill_rate

    results/tab_per_operator_summary.csv
        cell_id, operator, mean_kill_rate, sum_total, sum_killed, n_samples

Run as:
    python3 scripts/regen_per_operator.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    TEMPERATURE,
    TOP_P,
    TOP_K,
    NUM_CTX,
    MAX_OUTPUT_TOKENS,
)
from doe import CELLS_BY_ID
import closure_cache
import closure_metrics
import mutation_testing
from closure_paths import (
    _PROMPT_DOC_FROM_CODE,
    _PROMPT_TESTS_FROM_DOC,
    _extract_python,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_samples() -> dict[int, dict]:
    """Return {sample_idx: sample_dict}."""
    path = Path("data/core_sample_150.jsonl")
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["sample_idx"])] = d
    logger.info(f"Loaded {len(out)} samples.")
    return out


def cache_lookup_docstring(model_tag: str, code: str) -> str | None:
    """Try to recover D' = L_spec(code). Returns text or None on miss."""
    prompt = _PROMPT_DOC_FROM_CODE.format(code=code)
    # Try both NUM_CTX values that appeared during the sweep
    for num_ctx in (NUM_CTX, 16384, 8192):
        key = closure_cache.make_key(
            model_tag,
            "L_spec:doc_from_code",
            prompt,
            system_prompt=None,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_ctx=num_ctx,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        cached = closure_cache.get(key)
        if cached is not None and cached.get("text"):
            return cached["text"].strip()
    return None


def cache_lookup_tests(
    model_tag: str, docstring: str, fn_name: str, signature: str
) -> str | None:
    """Try to recover T' = L_test(docstring). Returns extracted code or None."""
    prompt = _PROMPT_TESTS_FROM_DOC.format(
        docstring=docstring,
        fn_name=fn_name or "the function",
        signature=signature or "(unknown)",
    )
    for num_ctx in (NUM_CTX, 16384, 8192):
        key = closure_cache.make_key(
            model_tag,
            "L_test:tests_from_doc",
            prompt,
            system_prompt=None,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_ctx=num_ctx,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        cached = closure_cache.get(key)
        if cached is not None and cached.get("text"):
            return _extract_python(cached["text"])
    return None


def process_row(row, samples: dict[int, dict]) -> list[dict] | None:
    """Return list of per-operator rows for this (cell, sample) or None."""
    cell = CELLS_BY_ID.get(row["cell_id"])
    if cell is None or cell.L_spec is None or cell.L_test is None:
        return None  # ablation cell — skip

    sample = samples.get(int(row["sample_idx"]))
    if sample is None:
        logger.warning(f"sample_idx {row['sample_idx']} not in dataset")
        return None

    code = sample["code"]
    if cell.corrupt_inputs:
        # N1 uses word-shuffled input; too complex to reconstruct; skip
        return None

    # Step 1: recover D'
    d_prime = cache_lookup_docstring(cell.L_spec.ollama_tag, code)
    if d_prime is None:
        return None  # cache miss

    # Step 2: recover T'
    t_prime = cache_lookup_tests(
        cell.L_test.ollama_tag,
        d_prime,
        sample.get("entry_point", ""),
        sample.get("signature", ""),
    )
    if t_prime is None:
        return None

    # Step 3: filter, then per-operator eval
    filtered = closure_metrics.test_filter(t_prime, code)
    if not filtered.strip():
        return None

    breakdown = mutation_testing.evaluate_mutants(code, filtered)
    per_op = breakdown.get("per_operator", {})

    out = []
    for operator, counts in per_op.items():
        total = int(counts.get("total", 0))
        killed = int(counts.get("killed", 0))
        if total == 0:
            continue
        out.append(
            {
                "cell_id": row["cell_id"],
                "sample_idx": int(row["sample_idx"]),
                "sample_source": row["sample_source"],
                "operator": operator,
                "total": total,
                "killed": killed,
                "kill_rate": killed / total,
            }
        )
    return out


def main() -> None:
    tsv = Path("results/results_roundtrip.tsv")
    df = pd.read_csv(tsv, sep="\t")
    path1 = df[df["path"] == 1].copy()
    logger.info(f"Loaded {len(path1)} Path 1 rows across {path1['cell_id'].nunique()} cells.")

    samples = load_samples()

    all_rows: list[dict] = []
    n_hit = 0
    n_miss = 0
    n_ablated = 0

    for i, row in enumerate(path1.itertuples()):
        row_dict = {
            "cell_id": row.cell_id,
            "sample_idx": row.sample_idx,
            "sample_source": row.sample_source,
        }
        try:
            result = process_row(row_dict, samples)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[{row.cell_id} s={row.sample_idx}] crashed: {exc}")
            result = None

        if result is None:
            cell = CELLS_BY_ID.get(row.cell_id)
            if cell is None or cell.L_spec is None or cell.L_test is None or cell.corrupt_inputs:
                n_ablated += 1
            else:
                n_miss += 1
        else:
            n_hit += 1
            all_rows.extend(result)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  processed {i + 1}/{len(path1)} — "
                f"hit={n_hit}, miss={n_miss}, ablated={n_ablated}"
            )

    logger.info(
        f"Done. hit={n_hit}, miss={n_miss}, ablated={n_ablated}, "
        f"total_rows={len(all_rows)}"
    )

    long_df = pd.DataFrame(all_rows)
    long_df.to_csv("results/tab_per_operator_long.csv", index=False)
    logger.info(f"Wrote results/tab_per_operator_long.csv ({len(long_df)} rows)")

    # Aggregate to (cell, operator)
    if not long_df.empty:
        summary = (
            long_df.groupby(["cell_id", "operator"])
            .agg(
                mean_kill_rate=("kill_rate", "mean"),
                sum_total=("total", "sum"),
                sum_killed=("killed", "sum"),
                n_samples=("sample_idx", "nunique"),
            )
            .reset_index()
        )
        summary["weighted_kill_rate"] = summary["sum_killed"] / summary["sum_total"]
        summary.to_csv("results/tab_per_operator_summary.csv", index=False)
        logger.info(f"Wrote results/tab_per_operator_summary.csv ({len(summary)} rows)")

        # Preview
        print("\n=== Weighted kill rate per (cell, operator) ===")
        pivot = summary.pivot(index="cell_id", columns="operator", values="weighted_kill_rate")
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
