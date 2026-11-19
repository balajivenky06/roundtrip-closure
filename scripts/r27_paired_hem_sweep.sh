#!/usr/bin/env bash
# =============================================================================
# R2.7 — paired before/after run on the 25 HEM source functions.
#
# Reviewer 2 asked: "compare the same 25 functions before & after
# transformation". The HEM-25 sweep (results_rerun/results_holdout_
# humaneval_mutated_v2.tsv) ran H1 and M3 on the 25 *transformed* functions.
# This script runs the same two cells, same settings, on the 25 *original*
# HumanEval functions those items were derived from
# (data/humaneval_original_25.jsonl, sample_idx aligned 1:1 with HEM).
#
# Run on Colab with Ollama up and the Drive cache mounted (same session
# setup as scripts/revision_resweep.sh). 6 of the 25 originals are in the
# core sample, so some calls are cache hits. Expected: ~2.5–3 h on A100 (same size as the original HEM-25 re-sweep).
#
# Then, locally or on Colab:
#     python3 analyze/paired_hem_before_after.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results_rerun/results_hem_original_25.tsv}"
mkdir -p results_rerun

for cell in H1 M3; do
    echo "════════════════════════════════════════════════════════════"
    echo "  R2.7 paired: cell=$cell  n=25  dataset=humaneval_original"
    echo "════════════════════════════════════════════════════════════"
    CELL_ID="$cell" \
    DATASET=humaneval_original \
    N_SAMPLES=25 \
    RESULTS_PATH="$OUT" \
        python3 -u train_roundtrip.py
done

echo "Done. Rows written to $OUT"
echo "Next: python3 analyze/paired_hem_before_after.py"
