"""
prepare_roundtrip.py — one-time setup script.

Run this once before any experiment. It:
    1. Verifies all 7 Ollama models in the lineup are pulled and ready.
    2. Downloads HumanEval + MBPP via the `datasets` library.
    3. Builds the 150-function core sample (seed=42, stratified HE/MBPP).
    4. Downloads the LiveCodeBench post-cutoff 25-function subset.
    5. Builds the HumanEval-Mutated 50-function subset via decontaminate.py.
    6. Writes everything to data/ as JSONL for fast loading.

Stub status: function signatures + flow comments; implementations TBD.
"""

from __future__ import annotations

from config import (
    PIPELINE_MODELS,
    JUDGE_MODEL,
    DATA_DIR,
    CORE_SAMPLE_SIZE,
    LIVECODEBENCH_SAMPLE_SIZE,
    HUMANEVAL_MUTATED_SAMPLE_SIZE,
    DATASET_SEED,
)


def step_1_verify_ollama_models() -> None:
    """
    Walk through PIPELINE_MODELS + JUDGE_MODEL and check each is in
    `ollama list` output. If any are missing, print the `ollama pull`
    command and exit with status 1.
    """
    raise NotImplementedError("Stub — implementation pending.")


def step_2_download_humaneval_mbpp() -> None:
    """
    Download HumanEval (164 problems) + MBPP (974 problems) via
    `datasets.load_dataset`. Cache under data/raw/.
    """
    raise NotImplementedError("Stub — implementation pending.")


def step_3_build_core_sample(seed: int = DATASET_SEED,
                             n: int = CORE_SAMPLE_SIZE) -> None:
    """
    Sample 150 functions from the HumanEval + MBPP union with a fixed
    seed-42 shuffle, preserving the Chapter-2 ratio (~105 MBPP + 45 HE).
    Write to data/core_sample_150.jsonl.
    """
    raise NotImplementedError("Stub — implementation pending.")


def step_4_download_livecodebench(n: int = LIVECODEBENCH_SAMPLE_SIZE) -> None:
    """
    Pull n LiveCodeBench problems with publication date >= 2024-12-01
    (post training-cutoff of all evaluated SLMs). Write to
    data/livecodebench_25.jsonl.
    """
    raise NotImplementedError("Stub — implementation pending.")


def step_5_build_humaneval_mutated(n: int = HUMANEVAL_MUTATED_SAMPLE_SIZE,
                                   seed: int = DATASET_SEED) -> None:
    """
    Apply the decontaminate.py pipeline to n HumanEval problems.
    Write to data/humaneval_mutated_50.jsonl.
    """
    raise NotImplementedError("Stub — implementation pending.")


def main() -> int:
    """Run all five steps in sequence. Idempotent — safe to re-run."""
    print("=== roundtrip-closure: one-time setup ===\n")
    step_1_verify_ollama_models()
    step_2_download_humaneval_mbpp()
    step_3_build_core_sample()
    step_4_download_livecodebench()
    step_5_build_humaneval_mutated()
    print("\n✓ Setup complete. You may now run train_roundtrip.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
