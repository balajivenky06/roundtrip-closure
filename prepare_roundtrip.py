"""
prepare_roundtrip.py — one-time setup script.

Run this once before any experiment. It:
    1. Verifies all 7 Ollama models in the lineup are pulled.
    2. Downloads HumanEval + MBPP via the `datasets` library.
    3. Normalises both into the common sample schema and builds the
       150-function core sample (seed=42).
    4. (Optional) downloads the LiveCodeBench post-cutoff 25-function
       subset — skipped gracefully if unavailable.
    5. Builds the HumanEval-Mutated 50-function subset via decontaminate.py.

Output JSONL files (in data/):
    core_sample_150.jsonl
    livecodebench_25.jsonl       (optional)
    humaneval_mutated_50.jsonl

Each line is a sample dict with the schema:
    {
      "sample_idx": int,
      "source": str,           e.g. "humaneval/HumanEval/0"
      "entry_point": str,
      "signature": str,
      "docstring": str,
      "code": str,
      "tests": str,
    }

Run as:  python3 prepare_roundtrip.py
"""

from __future__ import annotations
import ast
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

from config import (
    PIPELINE_MODELS,
    JUDGE_MODEL,
    DATA_DIR,
    CORE_SAMPLE_SIZE,
    LIVECODEBENCH_SAMPLE_SIZE,
    HUMANEVAL_MUTATED_SAMPLE_SIZE,
    DATASET_SEED,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_roundtrip")


# ──────────────────────────────────────────────────────────────────────
# Step 1 — Verify Ollama models
# ──────────────────────────────────────────────────────────────────────
def step_1_verify_ollama_models() -> tuple[list[str], list[str]]:
    """Return (pulled, missing) lists of model tags."""
    import ollama_client
    try:
        available = ollama_client.list_available_models()
    except Exception as exc:
        logger.error(f"Cannot reach Ollama: {exc}")
        logger.error("Start the server: `ollama serve` in another terminal.")
        return [], [m.ollama_tag for m in PIPELINE_MODELS + (JUDGE_MODEL,)]

    all_required = list(PIPELINE_MODELS) + [JUDGE_MODEL]
    pulled, missing = [], []
    for m in all_required:
        if m.ollama_tag in available or any(
            a.split(":", 1)[0] == m.ollama_tag.split(":", 1)[0] for a in available
        ):
            pulled.append(m.ollama_tag)
        else:
            missing.append(m.ollama_tag)

    logger.info(f"  Pulled  ({len(pulled)}/{len(all_required)}): {pulled}")
    if missing:
        logger.warning(f"  Missing ({len(missing)}): {missing}")
        logger.warning(f"  Run:\n" + "\n".join(f"    ollama pull {m}" for m in missing))
    else:
        logger.info("  ✓ All required models pulled.")
    return pulled, missing


# ──────────────────────────────────────────────────────────────────────
# Step 2 — Download HumanEval + MBPP
# ──────────────────────────────────────────────────────────────────────
def step_2_download_humaneval_mbpp() -> tuple[list[dict], list[dict]]:
    """Return (humaneval_problems, mbpp_problems) — both pre-normalised."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install: pip install datasets") from e

    logger.info("  Downloading HumanEval…")
    he_raw = load_dataset("openai_humaneval", split="test")
    logger.info(f"    {len(he_raw)} problems")

    logger.info("  Downloading MBPP…")
    mbpp_raw = load_dataset("mbpp", split="test")
    logger.info(f"    {len(mbpp_raw)} problems")

    he_norm = [_normalise_humaneval(p) for p in he_raw]
    he_norm = [p for p in he_norm if p is not None]
    mbpp_norm = [_normalise_mbpp(p) for p in mbpp_raw]
    mbpp_norm = [p for p in mbpp_norm if p is not None]

    logger.info(f"  Normalised: {len(he_norm)} HumanEval + {len(mbpp_norm)} MBPP")
    return he_norm, mbpp_norm


# ──────────────────────────────────────────────────────────────────────
# Schema normalisers
# ──────────────────────────────────────────────────────────────────────
def _normalise_humaneval(p: dict) -> Optional[dict]:
    """
    HumanEval shape (as of v0.1.1):
      task_id, prompt, canonical_solution, test, entry_point

    prompt = imports + def + docstring + (optionally examples)
    canonical_solution = function body (indented)
    """
    try:
        full_code = p["prompt"] + p["canonical_solution"]
        docstring = _extract_docstring(full_code)
        signature = _extract_signature(full_code)
        tests = _humaneval_check_to_pytest(p["test"], p["entry_point"])
        return {
            "source": f"humaneval/{p['task_id']}",
            "entry_point": p["entry_point"],
            "signature": signature,
            "docstring": docstring,
            "code": full_code,
            "tests": tests,
        }
    except Exception as exc:                                       # pragma: no cover
        logger.debug(f"HumanEval normalise failed for {p.get('task_id')}: {exc}")
        return None


def _normalise_mbpp(p: dict) -> Optional[dict]:
    """
    MBPP shape:
      task_id, text, code, test_list (list of "assert ..." strings)
    """
    try:
        code = p["code"]
        # Extract entry point from the first `def`
        entry_point = _extract_entry_point(code)
        if not entry_point:
            return None
        # MBPP code is just the function (no docstring inside) — use 'text'
        docstring = p["text"].strip()
        signature = _extract_signature(code)
        tests = _mbpp_assert_list_to_pytest(p["test_list"], entry_point)
        return {
            "source": f"mbpp/{p['task_id']}",
            "entry_point": entry_point,
            "signature": signature,
            "docstring": docstring,
            "code": code,
            "tests": tests,
        }
    except Exception as exc:                                       # pragma: no cover
        logger.debug(f"MBPP normalise failed for {p.get('task_id')}: {exc}")
        return None


def _extract_signature(code: str) -> str:
    for line in code.split("\n"):
        if line.lstrip().startswith("def "):
            return line.rstrip(":").rstrip() + ":"
    return ""


def _extract_entry_point(code: str) -> str:
    m = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else ""


def _extract_docstring(code: str) -> str:
    """Pull the first triple-quoted docstring out of the function body."""
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ds = ast.get_docstring(node)
                if ds:
                    return ds.strip()
        return ""
    except SyntaxError:
        return ""


def _humaneval_check_to_pytest(check_code: str, entry_point: str) -> str:
    """Convert HumanEval's `def check(candidate)` style to pytest."""
    # Strip the check def + METADATA assignments
    lines = check_code.split("\n")
    body_lines: list[str] = []
    in_check = False
    for line in lines:
        if re.match(r"\s*def\s+check\s*\(", line):
            in_check = True
            continue
        if in_check:
            if line.startswith("def ") or (line and not line.startswith((" ", "\t"))):
                in_check = False
            else:
                # Strip one level of indentation
                if line.startswith("    "):
                    body_lines.append(line[4:])
                else:
                    body_lines.append(line)

    body = "\n".join(body_lines).strip()
    # Replace `candidate(` with `<entry_point>(`
    body = re.sub(r"\bcandidate\b", entry_point, body)
    return f"def test_{entry_point}():\n" + "\n".join(
        ("    " + ln) if ln.strip() else "" for ln in body.split("\n")
    )


def _mbpp_assert_list_to_pytest(asserts: list, entry_point: str) -> str:
    """Wrap MBPP's list of assert strings into a single pytest function."""
    return (f"def test_{entry_point}():\n" +
            "\n".join("    " + a for a in asserts))


# ──────────────────────────────────────────────────────────────────────
# Step 3 — Build core 150-function sample
# ──────────────────────────────────────────────────────────────────────
def step_3_build_core_sample(humaneval: list[dict], mbpp: list[dict],
                             seed: int = DATASET_SEED,
                             n: int = CORE_SAMPLE_SIZE) -> Path:
    """
    Stratified random sample of `n` problems from HumanEval ∪ MBPP,
    preserving the Chapter-2 ratio (~70% MBPP / 30% HumanEval per
    seed-42 shuffle). Writes JSONL to data/core_sample_150.jsonl.
    """
    rng = random.Random(seed)
    pool = [("humaneval", p) for p in humaneval] + [("mbpp", p) for p in mbpp]
    rng.shuffle(pool)
    selected = pool[:n]

    out_path = DATA_DIR / "core_sample_150.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (origin, problem) in enumerate(selected):
            sample = {"sample_idx": idx, **problem}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    mbpp_count = sum(1 for o, _ in selected if o == "mbpp")
    he_count = sum(1 for o, _ in selected if o == "humaneval")
    logger.info(f"  Wrote {out_path} — {n} samples ({he_count} HumanEval + {mbpp_count} MBPP)")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Step 4 — LiveCodeBench post-cutoff subset (best effort, skip on fail)
# ──────────────────────────────────────────────────────────────────────
def step_4_download_livecodebench(n: int = LIVECODEBENCH_SAMPLE_SIZE) -> Optional[Path]:
    """
    Try to pull LiveCodeBench problems published after 2024-12-01.
    Returns the output path on success, None on failure.

    LiveCodeBench requires HuggingFace gated access in some versions;
    we fail gracefully if unavailable and instruct the user.
    """
    out_path = DATA_DIR / "livecodebench_25.jsonl"
    try:
        from datasets import load_dataset
        logger.info("  Attempting LiveCodeBench…")
        # This dataset name + schema may shift between releases
        ds = load_dataset("livecodebench/code_generation_lite", split="test",
                          trust_remote_code=True)
    except Exception as exc:
        logger.warning(f"  Skipping LiveCodeBench: {exc}")
        logger.warning(f"    The held-out decontamination defense will still work via")
        logger.warning(f"    the HumanEval-Mutated subset (Step 5).")
        return None

    # Filter by date if a date field exists
    filtered = []
    for p in ds:
        date_str = p.get("contest_date") or p.get("created_at") or ""
        if date_str and date_str >= "2024-12-01":
            filtered.append(p)
        if len(filtered) >= n:
            break

    if not filtered:
        logger.warning("  LiveCodeBench had no post-2024-12-01 samples; skipping.")
        return None

    # Best-effort normalisation
    records: list[dict] = []
    for i, p in enumerate(filtered):
        try:
            code = p.get("solution") or p.get("code") or ""
            entry = _extract_entry_point(code)
            if not entry:
                continue
            records.append({
                "sample_idx": i,
                "source": f"livecodebench/{p.get('question_id', i)}",
                "entry_point": entry,
                "signature": _extract_signature(code),
                "docstring": p.get("question_content", "")[:1000],
                "code": code,
                "tests": p.get("public_test_cases", ""),
            })
        except Exception:
            continue

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {out_path} — {len(records)} samples")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Step 5 — Build HumanEval-Mutated 50-function subset
# ──────────────────────────────────────────────────────────────────────
def step_5_build_humaneval_mutated(humaneval: list[dict],
                                   n: int = HUMANEVAL_MUTATED_SAMPLE_SIZE,
                                   seed: int = DATASET_SEED) -> Path:
    """Run the decontaminate pipeline on n HumanEval problems."""
    import decontaminate
    from config import QWEN_3_6_27B, LLAMA_3_2_3B

    # Use QWEN_3_6_27B if it's pulled, else fall back to LLAMA_3_2_3B
    # (which we know is pulled because the user's been running it)
    paraphraser = LLAMA_3_2_3B  # safe default; user can override
    try:
        import ollama_client
        available = set(ollama_client.list_available_models())
        if QWEN_3_6_27B.ollama_tag in available:
            paraphraser = QWEN_3_6_27B
        logger.info(f"  Paraphraser: {paraphraser.ollama_tag}")
    except Exception:
        pass

    out_path = DATA_DIR / "humaneval_mutated_50.jsonl"
    # Add a sample_idx to each HumanEval problem before passing in
    he_with_idx = [{"sample_idx": i, **p} for i, p in enumerate(humaneval)]
    summary = decontaminate.build_humaneval_mutated_subset(
        he_with_idx, out_path, n=n, seed=seed, paraphraser=paraphraser,
    )
    logger.info(f"  Decontamination summary: {summary}")
    return out_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=== roundtrip-closure: one-time setup ===\n")

    logger.info("[1/5] Verifying Ollama models…")
    pulled, missing = step_1_verify_ollama_models()
    if missing:
        logger.warning(
            f"  Continuing anyway — pipeline cells using missing models will "
            f"return errors gracefully. Pull missing models before running the "
            f"full sweep on Colab."
        )
    print()

    logger.info("[2/5] Downloading HumanEval + MBPP…")
    try:
        humaneval, mbpp = step_2_download_humaneval_mbpp()
    except ImportError as e:
        logger.error(f"  {e}")
        return 1
    print()

    logger.info("[3/5] Building core 150-function sample…")
    step_3_build_core_sample(humaneval, mbpp)
    print()

    logger.info("[4/5] Attempting LiveCodeBench subset…")
    step_4_download_livecodebench()
    print()

    logger.info("[5/5] Building HumanEval-Mutated subset…")
    step_5_build_humaneval_mutated(humaneval)
    print()

    logger.info("✓ Setup complete. You may now run train_roundtrip.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
