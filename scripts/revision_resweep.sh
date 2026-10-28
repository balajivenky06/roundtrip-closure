#!/usr/bin/env bash
#
# IPM Revision resweep — combined sweep for A1 (HEM-25) + A1b (H10 Path 1).
#
# Addresses two reviewer concerns:
#
#   R2 Comment 3 (A1):
#     The submitted Table 12 HEM-25 numbers (79.6%, 32.7%, etc.) imply
#     denominators ≥ 49, inconsistent with the stated n=25 per cell per
#     path. We re-sweep H1 + M3 on the HEM-25 subset cleanly at exactly
#     n=25 so the published numerators/denominators are honest.
#
#   R1 Comment 5 / B2 forensics (A1b):
#     The published H10 Path 1 "55% all_dropped" rate includes ~25
#     rows where phi4:14b returned HTTP 400 "does not support chat"
#     errors (verified in logs/cell_H10_core.log). We re-sweep H10
#     Path 1 cleanly with the current Ollama + phi4 config to produce
#     an infrastructure-free number. If the clean re-run also shows
#     ~44-50% all_dropped, that is the strongest possible answer to
#     R1.5 (independent replication of the finding) and completely
#     defuses the "post-hoc exclusion" criticism.
#
# Outputs go to NEW TSVs so the old-sweep data is preserved for
# cross-reference:
#   results/results_holdout_humaneval_mutated_v2.tsv  (A1)
#   results/results_h10_rerun.tsv                     (A1b)
#
# Expected runtime on Colab Pro+ A100 (both sweeps combined):
#   A1  (HEM-25 × H1+M3 × 3 paths × 25 samples ≈ 150 closures):  2.5 - 3 h
#   A1b (H10 Path 1 × 75 samples ≈ 75 closures):                 1.5 - 2 h
#   Total:                                                       4 - 5 h
#
# Zero API cost — all open-weight via local Ollama (no OpenRouter).
#
# Pre-flight: verify phi4:14b supports chat on the current Ollama.
# Run on Colab (from repo root):
#   bash scripts/revision_resweep.sh
#
# Skip phases:
#   SKIP_A1=1   bash scripts/revision_resweep.sh    # H10 only
#   SKIP_A1B=1  bash scripts/revision_resweep.sh    # HEM only
#
set -euo pipefail

RESULTS_HEM_V2="results/results_holdout_humaneval_mutated_v2.tsv"
RESULTS_H10_RERUN="results/results_h10_rerun.tsv"

echo "IPM revision resweep config:"
echo "  A1  HEM-25 TSV:            $RESULTS_HEM_V2"
echo "  A1b H10 Path 1 TSV:        $RESULTS_H10_RERUN"
echo "  SKIP_A1:                   ${SKIP_A1:-0}"
echo "  SKIP_A1B:                  ${SKIP_A1B:-0}"
echo ""

# ─── Pre-flight: verify phi4:14b supports chat on current Ollama ────
echo "─── Pre-flight: probing phi4:14b /api/chat ─────────────────────"
python3 - <<'EOF'
"""Probe phi4:14b /api/chat to catch the HTTP 400 regression early."""
import sys
try:
    from ollama_client import call_llm
    from config import PHI_4_14B
    out = call_llm(
        PHI_4_14B,
        prompt="Return exactly: OK",
        system="You are a terse test probe.",
    )
    if "OK" in (out.text or ""):
        print("  ✓ phi4:14b chat endpoint OK")
    else:
        print(f"  ⚠ phi4:14b returned unexpected: {out.text[:60]!r}")
except Exception as exc:
    print(f"  ✗ phi4:14b chat FAILED: {exc}")
    print(f"  → H10 re-run (A1b) will not produce clean results until fixed.")
    print(f"  → Possible fixes: update Ollama to latest, pull a different")
    print(f"    phi4 variant (e.g. phi4-mini:3.8b), or set SKIP_A1B=1.")
    sys.exit(1)
EOF
echo ""

# ─── A1: HEM-25 re-sweep on H1 + M3 ─────────────────────────────────
if [[ "${SKIP_A1:-0}" != "1" ]]; then
    if [[ ! -f data/humaneval_mutated_50.jsonl ]]; then
        echo "! data/humaneval_mutated_50.jsonl not found; run prepare_roundtrip.py first"
        exit 1
    fi
    # Fresh TSV — do NOT reuse the submitted one with its dubious denominators.
    rm -f "$RESULTS_HEM_V2"
    for cell in H1 M3; do
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  A1 HEM-25: cell=$cell  n=25  dataset=humaneval_mutated"
        echo "════════════════════════════════════════════════════════════"
        CELL_ID="$cell" \
        DATASET=humaneval_mutated \
        N_SAMPLES=25 \
        RESULTS_PATH="$RESULTS_HEM_V2" \
            python3 -u train_roundtrip.py
    done
else
    echo "SKIP_A1=1 — skipping HEM-25 re-sweep"
fi

# ─── A1b: H10 Path 1 clean re-run (full 75 samples) ──────────────────
if [[ "${SKIP_A1B:-0}" != "1" ]]; then
    rm -f "$RESULTS_H10_RERUN"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  A1b H10 rerun: cell=H10  n=75  dataset=core"
    echo "════════════════════════════════════════════════════════════"
    CELL_ID=H10 \
    DATASET=core \
    N_SAMPLES=75 \
    RESULTS_PATH="$RESULTS_H10_RERUN" \
        python3 -u train_roundtrip.py
else
    echo "SKIP_A1B=1 — skipping H10 re-run"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  revision resweep complete"
echo "════════════════════════════════════════════════════════════"
echo "  A1  HEM-25 results:        $RESULTS_HEM_V2"
echo "  A1b H10 rerun results:     $RESULTS_H10_RERUN"
echo ""
echo "  Next:"
echo "    python3 analyze/holdout_sensitivity.py"
echo "    python3 analyze/phi4_failure_forensics.py  # compare rerun vs original"
echo ""
echo "  Then sync the two new TSVs to the laptop repo and commit."
