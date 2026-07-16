#!/usr/bin/env bash
#
# RQ4 closed-weight sweep — restores the concept-note-pre-registered
# closed-weight cells (M5_closed, H2_closed, H8_closed) that were
# substituted with open-weight variants during the primary sweep, plus
# M7_gpt as a post-pre-registration cross-provider extension.
#
# Pipeline SLM calls route through llm_dispatch:
#   - Claude Sonnet 4.5 / GPT-4o-mini calls go to OpenRouter (needs
#     OPENROUTER_API_KEY in env; account balance ≥ $30 recommended)
#   - Open-weight qwen3.6 / qwen3-coder calls (in H2_closed, H8_closed)
#     go to the local Ollama server as usual
#
# Judge SLM (DeepSeek-R1:14B) is Ollama-local as in the primary sweep.
#
# Estimated cost: ~$25-35 USD API for 4 cells × 60 samples × 3 paths.
# Estimated Colab time: ~10-14 hours (API latency + local Ollama judge +
# local mutation-testing). Sweep is resume-safe; kill any time and rerun.
#
# Run on Colab (from repo root):
#   export OPENROUTER_API_KEY=<fresh-key>
#   bash scripts/rq4_closed_weight_sweep.sh
#
# Skip a specific cell via env: SKIP_M5=1 / SKIP_H2=1 / SKIP_H8=1 / SKIP_GPT=1
# Override sample count: N_SAMPLES=150 bash scripts/rq4_closed_weight_sweep.sh
#
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "! OPENROUTER_API_KEY not set. Aborting."
    exit 1
fi

MODE="${MODE:-pre-registered}"   # fast | pre-registered | full  (default = pre-registered)
if [[ -n "${CELLS_OVERRIDE:-}" ]]; then
    read -r -a CELLS <<<"$CELLS_OVERRIDE"
    N_SAMPLES="${N_SAMPLES:-60}"
elif [[ "$MODE" == "fast" ]]; then
    # Compressed 3-4h variant — 2 monos × 30 samples × 3 paths = 180 closures
    # Answers RQ4 headline (open vs closed ceiling) with dual-provider coverage.
    # Defers H2/H8 stage-decomposition cells to future work.
    CELLS=("M5_closed" "M7_gpt")
    N_SAMPLES="${N_SAMPLES:-30}"
elif [[ "$MODE" == "pre-registered" ]]; then
    # Concept-note-pre-registered 3-cell RQ4 (M5 + H2 + H8) at N=60.
    # Adds M7_gpt as a post-pre-registration extension. ~10-14h.
    CELLS=("M5_closed" "H2_closed" "H8_closed" "M7_gpt")
    N_SAMPLES="${N_SAMPLES:-60}"
elif [[ "$MODE" == "full" ]]; then
    # Full pre-registered N=150. ~22-34h across multiple Colab sessions.
    CELLS=("M5_closed" "H2_closed" "H8_closed" "M7_gpt")
    N_SAMPLES="${N_SAMPLES:-150}"
else
    echo "! Unknown MODE=$MODE. Choose one of: fast | pre-registered | full"
    echo "  Or override with: CELLS_OVERRIDE=\"cell1 cell2\" N_SAMPLES=X"
    exit 1
fi

RESULTS_PATH="results/results_rq4_closed_weight.tsv"

echo "RQ4 closed-weight sweep config (mode=$MODE):"
echo "  cells:      ${CELLS[*]}"
echo "  N_samples:  $N_SAMPLES"
echo "  results:    $RESULTS_PATH"
echo "  API key:    ${OPENROUTER_API_KEY:0:14}…"
echo ""

# Small probe first — verifies key + balance + model slugs
echo "─── OpenRouter probe (verifies key + balance) ─────────────────"
python3 -c "
import llm_dispatch
from config import CLAUDE_SONNET_45, GPT_4O_MINI
for m in (CLAUDE_SONNET_45, GPT_4O_MINI):
    ok = llm_dispatch.ensure_model_available(m)
    print(f'  {m.ollama_tag}: {\"OK\" if ok else \"FAIL\"}')
"
echo ""

for cell in "${CELLS[@]}"; do
    # skip flag: SKIP_M5=1 → skips "M5_closed"; SKIP_GPT=1 → skips "M7_gpt"
    # Skip flag: env var name is SKIP_<CELL_ID_UPPER>
    # e.g. SKIP_M5_CLOSED=1 skips M5_closed only; SKIP_M7_GPT=1 skips M7_gpt only.
    skip_var="SKIP_${cell^^}"
    if [[ "${!skip_var:-0}" == "1" ]]; then
        echo "SKIP $cell (${skip_var}=1)"
        continue
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  RQ4 sweep: cell=$cell dataset=core N=$N_SAMPLES"
    echo "════════════════════════════════════════════════════════════"
    CELL_ID="$cell" \
    DATASET=core \
    N_SAMPLES="$N_SAMPLES" \
    RESULTS_PATH="$RESULTS_PATH" \
        python3 train_roundtrip.py
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  RQ4 closed-weight sweep complete"
echo "════════════════════════════════════════════════════════════"
echo "  results: $RESULTS_PATH"
echo ""
echo "  Next: sync TSV to laptop, then"
echo "    python3 analyze/rq4_closed_vs_open.py"
