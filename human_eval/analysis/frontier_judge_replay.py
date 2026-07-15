"""
human_eval/analysis/frontier_judge_replay.py

Replay the 60-pair human-evaluation sample against one or more frontier
judges via OpenRouter. Uses the *exact* judge prompt from judge_llm.py
so any observed difference is judge-model, not prompt-drift.

Requires OPENROUTER_API_KEY in the environment.

Run:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python3 human_eval/analysis/frontier_judge_replay.py \\
        --judges openai/gpt-4o-mini anthropic/claude-haiku-4.5 \\
                 meta-llama/llama-3.3-70b-instruct

Writes:
    human_eval/data/frontier_judge_<slug>_ratings.tsv
        Same schema as annotator ratings TSVs, so compute_agreement.py
        can absorb each as a "virtual annotator".
    human_eval/data/frontier_judge_<slug>_raw.jsonl
        Full raw responses + usage for audit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the pipeline's judge prompt + parser verbatim so we're
# comparing judge *models*, not prompts.
from judge_llm import (  # noqa: E402
    _SYSTEM_PROMPT,
    _RUBRIC,
    _USER_TEMPLATE,
    _parse_judge_response,
)

PAIRS_PATH = PROJECT_ROOT / "human_eval" / "data" / "pairs_60_full.jsonl"
OUT_DIR = PROJECT_ROOT / "human_eval" / "data"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS_PATH.read_text().splitlines() if l.strip()]


def slugify(model_id: str) -> str:
    """Turn 'openai/gpt-4o-mini' into 'openai_gpt-4o-mini' for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def call_openrouter(
    api_key: str,
    model: str,
    user_msg: str,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """Return (response_text, usage_dict) or raise RuntimeError on total failure."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,          # frontier judges don't do <think>; 512 ample
    }).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/balajivenky06/roundtrip-closure",
        "X-Title": "roundtrip-closure judge replay",
    }
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            OPENROUTER_URL, data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"]
            return text, resp.get("usage", {})
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 502, 503, 504):
                sleep_s = 2 ** attempt
                print(f"    retry in {sleep_s}s ({last_err})")
                time.sleep(sleep_s)
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"exhausted retries: {last_err}")


def replay_one_judge(api_key: str, model: str, pairs: list[dict]) -> None:
    slug = slugify(model)
    ratings_path = OUT_DIR / f"frontier_judge_{slug}_ratings.tsv"
    raw_path = OUT_DIR / f"frontier_judge_{slug}_raw.jsonl"

    print(f"\n=== judge: {model} → {ratings_path.name} ===")

    total_cost = 0.0
    ratings_rows: list[dict] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for i, pair in enumerate(pairs, 1):
            user_msg = _USER_TEMPLATE.format(
                rubric=_RUBRIC,
                artefact_kind=pair["artefact_kind"],
                artefact_a=pair["artefact_a"],
                artefact_b=pair["artefact_b"],
            )
            t0 = time.perf_counter()
            try:
                text, usage = call_openrouter(api_key, model, user_msg)
            except RuntimeError as e:
                print(f"  {i:>2}/{len(pairs)} {pair['pair_id']}: FAIL {e}")
                ratings_rows.append({
                    "pair_id": pair["pair_id"], "rating": -1,
                    "justification": f"api_error: {e}",
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                    "is_revision": "false",
                })
                continue
            rating, reason = _parse_judge_response(text)
            elapsed = time.perf_counter() - t0
            cost = usage.get("cost", 0.0) or 0.0
            total_cost += cost
            print(f"  {i:>2}/{len(pairs)} {pair['pair_id']} ({pair['artefact_kind']}): "
                  f"rating={rating} ({elapsed:.1f}s, ${cost:.5f})")
            raw_f.write(json.dumps({
                "pair_id": pair["pair_id"],
                "model": model,
                "rating": rating,
                "reason": reason,
                "raw_text": text,
                "usage": usage,
                "elapsed_s": elapsed,
            }) + "\n")
            ratings_rows.append({
                "pair_id": pair["pair_id"], "rating": rating,
                "justification": reason,
                "ts_iso": datetime.now(timezone.utc).isoformat(),
                "is_revision": "false",
            })

    # Write ratings TSV in the annotator-schema format
    with ratings_path.open("w", encoding="utf-8") as f:
        f.write("pair_id\trating\tjustification\tts_iso\tis_revision\n")
        for row in ratings_rows:
            just = row["justification"].replace("\t", " ").replace("\n", " ")
            f.write(f"{row['pair_id']}\t{row['rating']}\t{just}\t"
                    f"{row['ts_iso']}\t{row['is_revision']}\n")

    n_parsed = sum(1 for r in ratings_rows if r["rating"] >= 0)
    print(f"  DONE {model}: {n_parsed}/{len(pairs)} parsed successfully, "
          f"total cost ${total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judges", nargs="+", required=True,
        help="OpenRouter model slugs, e.g. openai/gpt-4o-mini",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("Set OPENROUTER_API_KEY in your shell first.")

    pairs = load_pairs()
    print(f"Loaded {len(pairs)} pairs from {PAIRS_PATH.name}")
    print(f"Replaying against {len(args.judges)} judge(s): {args.judges}")

    for model in args.judges:
        replay_one_judge(api_key, model, pairs)

    print("\nAll judges complete. Next:")
    slugs = " ".join(f"frontier_judge_{slugify(m)}" for m in args.judges)
    print(f"  python3 human_eval/analysis/compute_agreement.py \\")
    print(f"      --annotators BV GS AM DR RR {slugs}")


if __name__ == "__main__":
    main()
