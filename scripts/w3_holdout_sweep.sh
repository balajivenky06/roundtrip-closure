#!/usr/bin/env bash
#
# W3 contamination-sensitivity sweep — reviewer response
#
# Sweeps the strongest hetero (H1) and strongest mono baselines (M3, M4)
# on the humaneval_mutated_50 dataset (function-rename + docstring-
# paraphrase) and, if available, the livecodebench_25 dataset (post-
# training-cutoff, uncontaminated by construction).
#
# The results TSV is separate from the core sweep TSV so the two sets can
# be compared per-cell per-path.
#
# Run on Colab from repo root:
#   bash scripts/w3_holdout_sweep.sh
#
# Estimated runtime on A100:
#   humaneval_mutated_50:  3 cells × 50 problems × 3 paths ≈ 2 - 3 hours
#   livecodebench_25:      3 cells × 25 problems × 3 paths ≈ 1 - 1.5 hours
#
# Skip livecodebench sweep by exporting SKIP_LCB=1 before running.
# Skip a specific dataset by exporting SKIP_HEM=1 or SKIP_LCB=1.
# Override cells via CELLS_OVERRIDE="H1 M3 M4" if you want to widen scope.
# Override sample count via N_SAMPLES_HEM=50 for a fuller sweep.
set -euo pipefail

# Default: H1 (hetero champion) + M3 (strongest Path-2/3 mono baseline).
# M4 (strongest Path-1 mono) is a nice-to-have but not essential for the
# reviewer's ask — add via CELLS_OVERRIDE="H1 M3 M4".
if [[ -n "${CELLS_OVERRIDE:-}" ]]; then
    read -r -a CELLS <<<"$CELLS_OVERRIDE"
else
    CELLS=("H1" "M3")
fi

N_SAMPLES_HEM="${N_SAMPLES_HEM:-25}"
RESULTS_HEM="results/results_holdout_humaneval_mutated.tsv"
RESULTS_LCB="results/results_holdout_livecodebench.tsv"

echo "W3 sweep config:"
echo "  cells:                 ${CELLS[*]}"
echo "  humaneval_mutated N:   $N_SAMPLES_HEM"
echo "  livecodebench N:       25"
echo "  results HEM:           $RESULTS_HEM"
echo "  results LCB:           $RESULTS_LCB"
echo ""

# ── humaneval_mutated_50 sweep ────────────────────────────────────────
if [[ "${SKIP_HEM:-0}" != "1" ]]; then
    if [[ ! -f data/humaneval_mutated_50.jsonl ]]; then
        echo "! data/humaneval_mutated_50.jsonl not found; run prepare_roundtrip.py first"
        exit 1
    fi
    for cell in "${CELLS[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  W3 sweep: cell=$cell dataset=humaneval_mutated"
        echo "════════════════════════════════════════════════════════════"
        CELL_ID="$cell" \
        DATASET=humaneval_mutated \
        N_SAMPLES="$N_SAMPLES_HEM" \
        RESULTS_PATH="$RESULTS_HEM" \
            python3 train_roundtrip.py
    done
else
    echo "SKIP_HEM=1 — skipping humaneval_mutated sweep"
fi

# ── livecodebench_25 sweep (best-effort) ──────────────────────────────
if [[ "${SKIP_LCB:-0}" != "1" ]]; then
    if [[ ! -f data/livecodebench_25.jsonl ]]; then
        echo ""
        echo "! data/livecodebench_25.jsonl not found — attempting prep …"
        python3 -c "
from prepare_roundtrip import step_4_download_livecodebench
step_4_download_livecodebench()
" || echo "  prep failed; skipping livecodebench sweep"
    fi
    if [[ -f data/livecodebench_25.jsonl ]]; then
        for cell in "${CELLS[@]}"; do
            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo "  W3 sweep: cell=$cell dataset=livecodebench"
            echo "════════════════════════════════════════════════════════════"
            CELL_ID="$cell" \
            DATASET=livecodebench \
            N_SAMPLES=25 \
            RESULTS_PATH="$RESULTS_LCB" \
                python3 train_roundtrip.py
        done
    fi
else
    echo "SKIP_LCB=1 — skipping livecodebench sweep"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  W3 sweep complete"
echo "════════════════════════════════════════════════════════════"
echo "  humaneval_mutated results: $RESULTS_HEM"
echo "  livecodebench results:     $RESULTS_LCB"
echo ""
echo "  Next: python3 analyze/holdout_sensitivity.py"
