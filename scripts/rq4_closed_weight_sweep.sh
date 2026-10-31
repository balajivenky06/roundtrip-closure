#!/usr/bin/env bash
#
# RQ4 closed-weight sweep — COLAB-HARDENED version.
#
# Features (all designed around Colab's ephemeral filesystem + flaky GPUs):
#
#   1. Auto-mounts Google Drive at /content/drive if not already mounted
#   2. Restores TSV from Drive at startup if local copy is missing
#      (e.g., post-reconnect to a fresh runtime)
#   3. Background sync daemon copies TSV + checkpoints to Drive every
#      SYNC_INTERVAL seconds (default 300s / 5 min)
#   4. Status heartbeat prints progress (rows done / expected, elapsed,
#      ETA, cost estimate) alongside each sync
#   5. Exit trap does a final guaranteed sync on normal exit, Ctrl+C,
#      or GPU/kernel kill — zero manual intervention needed
#   6. Live Drive status log at /content/drive/.../rq4_status.log
#      (readable from any other Colab notebook or your laptop via
#      Drive web UI — useful for monitoring from your phone)
#
# Pipeline SLM calls route through llm_dispatch:
#   - Claude Sonnet 5 / GPT-5.4 Mini calls go to OpenRouter (needs
#     OPENROUTER_API_KEY in env; account balance ≥ $40 recommended
#     — 1.5x the $19-31 estimate as a retry buffer)
#   - Open-weight qwen3.6 / qwen3-coder calls (in H2_closed, H8_closed)
#     go to the local Ollama server as usual
#
# Judge SLM (DeepSeek-R1:14B) is Ollama-local as in the primary sweep.
#
# Estimated cost: ~$19-31 USD API for 4 cells × 60 samples × 3 paths at
# Sonnet-5 pricing ($2/$10 per 1M) + GPT-5.4-mini pricing ($0.75/$4.50 per 1M).
# Estimated Colab time: ~10-14 hours. Sweep is resume-safe at the (cell,
# sample, path) level; if the runtime dies, re-run the same command and
# it skips completed closures.
#
# Run on Colab (from repo root):
#   export OPENROUTER_API_KEY=<fresh-key>
#   bash scripts/rq4_closed_weight_sweep.sh
#
# Monitor progress from a second notebook / your phone:
#   open https://drive.google.com/drive/u/0/my-drive
#   navigate to roundtrip_backup/rq4_status.log and refresh
#
# Env flags (optional):
#   SKIP_M5_CLOSED=1   skip Claude Sonnet 5 mono
#   SKIP_H2_CLOSED=1   skip Claude spec + open test/code
#   SKIP_H8_CLOSED=1   skip Claude spec + Claude test + open code
#   SKIP_M7_GPT=1      skip GPT-5.4-mini mono
#   MODE=fast          2 monos × 30 samples instead of 4 × 60 (~3-4h, ~$8-12)
#   MODE=full          4 × 150 samples (~22-34h, ~$45-70)
#   SYNC_INTERVAL=120  shorter backup interval (default 300s)
#
set -euo pipefail

# ─── 1. Config ────────────────────────────────────────────────────────
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "! OPENROUTER_API_KEY not set. Aborting."
    echo "  Get a key from https://openrouter.ai/keys and:"
    echo "    export OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

MODE="${MODE:-pre-registered}"
if [[ -n "${CELLS_OVERRIDE:-}" ]]; then
    read -r -a CELLS <<<"$CELLS_OVERRIDE"
    N_SAMPLES="${N_SAMPLES:-60}"
elif [[ "$MODE" == "fast" ]]; then
    CELLS=("M5_closed" "M7_gpt")
    N_SAMPLES="${N_SAMPLES:-30}"
elif [[ "$MODE" == "pre-registered" ]]; then
    CELLS=("M5_closed" "H2_closed" "H8_closed" "M7_gpt")
    N_SAMPLES="${N_SAMPLES:-60}"
elif [[ "$MODE" == "gemini" ]]; then
    # Google Gemini 2.5 Pro extension — fourth model family (post-pre-registration).
    # Expected cost ~$12-18 at Gemini 2.5 Pro pricing. ~1-2h on Colab.
    # Separate TSV path so this doesn't collide with the pre-registered sweep's TSV.
    CELLS=("M9_gemini" "H9_gemini")
    N_SAMPLES="${N_SAMPLES:-60}"
    RESULTS_PATH="${RESULTS_PATH:-results/results_rq4_gemini.tsv}"
elif [[ "$MODE" == "full" ]]; then
    CELLS=("M5_closed" "H2_closed" "H8_closed" "M7_gpt" "M9_gemini" "H9_gemini")
    N_SAMPLES="${N_SAMPLES:-150}"
else
    echo "! Unknown MODE=$MODE. Choose: fast | pre-registered | gemini | full"
    exit 1
fi

RESULTS_PATH="results/results_rq4_closed_weight.tsv"
DRIVE_BACKUP="${DRIVE_BACKUP:-/content/drive/MyDrive/roundtrip_backup}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
STATUS_LOG="${DRIVE_BACKUP}/rq4_status.log"
EXPECTED_ROWS=$(( ${#CELLS[@]} * N_SAMPLES * 3 ))

# ─── 2. Mount Drive (auto, idempotent) ────────────────────────────────
echo "─── Pre-flight: Google Drive mount ─────────────────────────────"
if [[ ! -d /content/drive/MyDrive ]]; then
    echo "  Drive not mounted — mounting now via google.colab.drive.mount()"
    python3 -c "
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
" || {
        echo "! Drive mount failed. Aborting — would run without a safety net."
        echo "  Mount Drive manually in a Colab cell then re-run this script."
        exit 1
    }
fi
mkdir -p "$DRIVE_BACKUP"
echo "  ✓ Drive mounted; backup dir = $DRIVE_BACKUP"
echo ""

# ─── 3. Restore TSV from Drive if local copy is missing ───────────────
mkdir -p "$(dirname "$RESULTS_PATH")"
DRIVE_TSV="${DRIVE_BACKUP}/$(basename "$RESULTS_PATH")"
if [[ ! -f "$RESULTS_PATH" && -f "$DRIVE_TSV" ]]; then
    echo "─── Restoring TSV from Drive (local copy missing) ──────────────"
    cp "$DRIVE_TSV" "$RESULTS_PATH"
    restored_rows=$(wc -l < "$RESULTS_PATH")
    echo "  ✓ Restored $(( restored_rows - 1 )) completed rows from Drive"
    echo "  Resume will skip these and continue."
    echo ""
fi

# Restore checkpoints tarball too, if present
DRIVE_CKPT="${DRIVE_BACKUP}/checkpoints_rq4_final.tar.gz"
if [[ ! -d checkpoints && -f "$DRIVE_CKPT" ]]; then
    echo "─── Restoring checkpoints cache from Drive ─────────────────────"
    tar xzf "$DRIVE_CKPT"
    echo "  ✓ Restored checkpoints/ ($(du -sh checkpoints | cut -f1))"
    echo ""
fi

# ─── 4. Config summary ────────────────────────────────────────────────
echo "─── RQ4 config summary ────────────────────────────────────────"
echo "  mode:            $MODE"
echo "  cells:           ${CELLS[*]}"
echo "  N_samples/cell:  $N_SAMPLES"
echo "  expected rows:   $EXPECTED_ROWS"
echo "  results TSV:     $RESULTS_PATH"
echo "  Drive backup:    $DRIVE_BACKUP"
echo "  sync interval:   ${SYNC_INTERVAL}s"
echo "  status log:      $STATUS_LOG"
echo "  API key prefix:  ${OPENROUTER_API_KEY:0:14}…"
echo ""

# ─── 5. OpenRouter + model probe ──────────────────────────────────────
export CELLS_PROBE="$(IFS=,; echo "${CELLS[*]}")"
echo "─── Pre-flight: OpenRouter probe (verifies key + model slugs) ──"
python3 -c "
import sys
import llm_dispatch
from config import CLAUDE_SONNET_5, GPT_5_4_MINI, GEMINI_2_5_PRO
# Only probe the models actually used by the selected CELLS.
# Reads CELLS from env (bash-exported via \$CELLS_PROBE)
import os
cells_probe = os.environ.get('CELLS_PROBE', '').split(',')
models_needed = set()
for c in cells_probe:
    c = c.strip()
    if c.startswith('M5') or c.startswith('H2_closed') or c.startswith('H8'):
        models_needed.add('sonnet')
    if c == 'M7_gpt':
        models_needed.add('gpt')
    if c.startswith('M9_gemini') or c.startswith('H9_gemini'):
        models_needed.add('gemini')
probe_list = []
if 'sonnet' in models_needed: probe_list.append(CLAUDE_SONNET_5)
if 'gpt' in models_needed:    probe_list.append(GPT_5_4_MINI)
if 'gemini' in models_needed: probe_list.append(GEMINI_2_5_PRO)
if not probe_list:  # default: probe all (safer when cells list is non-standard)
    probe_list = [CLAUDE_SONNET_5, GPT_5_4_MINI, GEMINI_2_5_PRO]
ok_all = True
for m in probe_list:
    ok = llm_dispatch.ensure_model_available(m)
    print(f'  {m.ollama_tag}: {\"OK\" if ok else \"FAIL\"}')
    if not ok: ok_all = False
if not ok_all:
    print('! One or more OpenRouter models unavailable. Check key balance + slugs.')
    sys.exit(1)
" || exit 1
echo ""

# ─── 6. Background sync daemon ────────────────────────────────────────
# Writes a heartbeat with cost + progress estimate alongside each
# backup. The daemon copies RESULTS_PATH and tarballs checkpoints/.
# Exit cleanup kills the daemon and does one final guaranteed sync.
export START_TIME=$(date +%s)

sync_loop() {
    local prev_rows=0
    while true; do
        sleep "$SYNC_INTERVAL"
        local n_lines=$(wc -l < "$RESULTS_PATH" 2>/dev/null || echo 1)
        local n_rows=$(( n_lines - 1 ))
        local elapsed=$(( $(date +%s) - START_TIME ))
        local new_rows=$(( n_rows - prev_rows ))
        prev_rows=$n_rows

        # Cost estimate: rough per-row scaling from mid-sweep snapshot.
        # Pre-reg has 4 cells × 60 × 3 = 720 rows expected; $19-31 total.
        # Linear scaling: cost ≈ ($19 to $31) × (rows_done / 720)
        local cost_lo=$(awk -v r="$n_rows" 'BEGIN{printf "%.2f", r/720*19}')
        local cost_hi=$(awk -v r="$n_rows" 'BEGIN{printf "%.2f", r/720*31}')

        # ETA
        local eta_s=0
        if (( n_rows > 0 && elapsed > 0 )); then
            local remaining=$(( EXPECTED_ROWS - n_rows ))
            eta_s=$(awk -v r="$remaining" -v d="$n_rows" -v e="$elapsed" 'BEGIN{if(d>0)printf "%d", r*e/d; else print 0}')
        fi

        {
            printf "[%s] " "$(date +'%Y-%m-%d %H:%M:%S')"
            printf "rows=%d/%d  (+%d since last sync)  elapsed=%ds  eta=%ds  cost≈\$%s-\$%s\n" \
                "$n_rows" "$EXPECTED_ROWS" "$new_rows" "$elapsed" "$eta_s" "$cost_lo" "$cost_hi"
            cp "$RESULTS_PATH" "$DRIVE_TSV" 2>/dev/null \
                && echo "  → TSV synced to Drive ($(stat -c%s "$DRIVE_TSV" 2>/dev/null || stat -f%z "$DRIVE_TSV") bytes)" \
                || echo "  → TSV sync failed"
            # Tarball checkpoints only every 4th sync (every ~20 min at default)
            if (( (elapsed / SYNC_INTERVAL) % 4 == 0 )); then
                tar czf "${DRIVE_BACKUP}/checkpoints_rq4_progress.tar.gz" checkpoints/ 2>/dev/null \
                    && echo "  → checkpoints tarball refreshed" \
                    || echo "  → checkpoints tarball failed"
            fi
        } | tee -a "$STATUS_LOG"
    done
}

# ─── 7. Exit trap — guaranteed final sync ─────────────────────────────
# Fires on: normal exit, Ctrl+C (SIGINT), kernel kill (SIGTERM),
# unexpected error (set -e). This is the most important safety net.
cleanup() {
    local exit_code=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ⚠ FINAL SYNC on exit (code=$exit_code)"
    echo "═══════════════════════════════════════════════════════════"
    if [[ -n "${SYNC_PID:-}" ]]; then
        kill "$SYNC_PID" 2>/dev/null || true
    fi
    if [[ -f "$RESULTS_PATH" ]]; then
        cp "$RESULTS_PATH" "$DRIVE_TSV" 2>/dev/null && \
            echo "  ✓ TSV copied to Drive ($(wc -l < "$DRIVE_TSV") lines)"
    fi
    tar czf "${DRIVE_BACKUP}/checkpoints_rq4_final.tar.gz" checkpoints/ 2>/dev/null && \
        echo "  ✓ Checkpoints tarballed to Drive"
    ls -lh "$DRIVE_BACKUP/" 2>/dev/null
    echo "═══════════════════════════════════════════════════════════"
    # Append the final status line
    {
        printf "[%s] FINAL exit_code=%d  rows=%d/%d\n" \
            "$(date +'%Y-%m-%d %H:%M:%S')" "$exit_code" \
            "$(( $(wc -l < "$RESULTS_PATH" 2>/dev/null || echo 1) - 1 ))" "$EXPECTED_ROWS"
    } >> "$STATUS_LOG" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ─── 8. Launch background sync, then run sweep ────────────────────────
echo "─── Launching background sync daemon (PID=\$!) ─────────────────"
sync_loop &
SYNC_PID=$!
echo "  ✓ sync daemon PID=$SYNC_PID; will sync every ${SYNC_INTERVAL}s"
echo "  ✓ follow status: tail -f $STATUS_LOG"
echo ""

echo "─── Starting sweep ───────────────────────────────────────────"
echo ""

for cell in "${CELLS[@]}"; do
    skip_var="SKIP_${cell^^}"
    if [[ "${!skip_var:-0}" == "1" ]]; then
        echo "SKIP $cell (${skip_var}=1)"
        continue
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  cell=$cell  dataset=core  N=$N_SAMPLES  started $(date +%H:%M:%S)"
    echo "════════════════════════════════════════════════════════════"
    CELL_ID="$cell" \
    DATASET=core \
    N_SAMPLES="$N_SAMPLES" \
    RESULTS_PATH="$RESULTS_PATH" \
        python3 -u train_roundtrip.py
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  RQ4 closed-weight sweep complete at $(date +%H:%M:%S)"
echo "════════════════════════════════════════════════════════════"
echo "  rows written: $(( $(wc -l < "$RESULTS_PATH") - 1 )) / $EXPECTED_ROWS"
echo "  local TSV:    $RESULTS_PATH"
echo "  Drive TSV:    $DRIVE_TSV"
echo ""
echo "  Next steps:"
echo "    1. Download $DRIVE_TSV to laptop (into results_rerun/)"
echo "    2. python3 analyze/rq4_closed_vs_open.py"
# The exit trap will fire after this echo, doing a final sync + ls.
