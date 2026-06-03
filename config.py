"""
config.py — central model lineup and experiment-wide constants.

This module is the single source of truth for which Small Language Models
participate in the experiment, what stage they can fill, and what
hyperparameters apply globally. All other modules import from here.

Edit ONLY the dataclass instances below to add/remove models or change
defaults; do not hard-code model strings elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path        = PROJECT_ROOT / "data"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
CACHE_DIR: Path       = CHECKPOINTS_DIR / "cache"
RESULTS_DIR: Path     = PROJECT_ROOT / "results"
LOGS_DIR: Path        = PROJECT_ROOT / "logs"

for d in (DATA_DIR, CHECKPOINTS_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Model lineup (all SLMs, all <30 B parameters)
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """One SLM's identity + properties."""
    ollama_tag: str          # e.g. "qwen3.6:27b" — used in the API call
    short_name: str          # e.g. "qwen3.6" — used in TSV / log columns
    family: str              # e.g. "Alibaba" — used in family-diversity analyses
    size_b: float            # parameter count in billions (total, not active)
    architecture: str        # "dense" | "MoE"
    generation_year: int     # e.g. 2026
    notes: str = ""


# The 6 pipeline SLMs + 1 judge SLM (defaults agreed 2026-06-03):
LLAMA_3_2_3B = ModelSpec(
    ollama_tag="llama3.2:3b",
    short_name="llama3.2",
    family="Meta",
    size_b=3.0,
    architecture="dense",
    generation_year=2024,
    notes="Only Meta SLM under 30 B (Llama 3.3 and Llama 4 exceed the threshold).",
)
PHI_4_14B = ModelSpec(
    ollama_tag="phi4:14b",
    short_name="phi4",
    family="Microsoft",
    size_b=14.0,
    architecture="dense",
    generation_year=2024,
    notes="Reasoning-tuned dense; latest Phi as of mid-2026.",
)
QWEN_3_6_27B = ModelSpec(
    ollama_tag="qwen3.6:27b",
    short_name="qwen3.6",
    family="Alibaba",
    size_b=27.0,
    architecture="dense",
    generation_year=2026,
    notes="April 2026 release — beats Qwen 3.5 397B-A17B on SWE-bench.",
)
GEMMA_4_26B = ModelSpec(
    ollama_tag="gemma4:26b",
    short_name="gemma4",
    family="Google",
    size_b=26.0,
    architecture="MoE",
    generation_year=2026,
    notes="March 2026 release; MoE with ~3.8 B active.",
)
MISTRAL_SMALL_3_2_24B = ModelSpec(
    ollama_tag="mistral-small3.2:24b",
    short_name="mistral-small3.2",
    family="Mistral",
    size_b=24.0,
    architecture="dense",
    generation_year=2025,
    notes="Function-calling-tuned; latest Mistral Small.",
)
QWEN_3_CODER_30B = ModelSpec(
    ollama_tag="qwen3-coder:30b",
    short_name="qwen3-coder",
    family="Alibaba-coder",
    size_b=30.0,
    architecture="MoE",
    generation_year=2025,
    notes="Code-specialised MoE, RL-trained on SWE-bench; 3.3 B active.",
)
DEEPSEEK_R1_14B = ModelSpec(
    ollama_tag="deepseek-r1:14b",
    short_name="deepseek-r1",
    family="DeepSeek",
    size_b=14.0,
    architecture="dense",
    generation_year=2025,
    notes="Reasoning-tuned distill — used ONLY as the external judge LLM.",
)


# Convenience tuples
PIPELINE_MODELS: tuple[ModelSpec, ...] = (
    LLAMA_3_2_3B,
    PHI_4_14B,
    QWEN_3_6_27B,
    GEMMA_4_26B,
    MISTRAL_SMALL_3_2_24B,
    QWEN_3_CODER_30B,
)
JUDGE_MODEL: ModelSpec = DEEPSEEK_R1_14B
ALL_MODELS: tuple[ModelSpec, ...] = PIPELINE_MODELS + (JUDGE_MODEL,)

# Lookup helper
MODELS_BY_SHORT_NAME: dict[str, ModelSpec] = {m.short_name: m for m in ALL_MODELS}
MODELS_BY_OLLAMA_TAG: dict[str, ModelSpec] = {m.ollama_tag: m for m in ALL_MODELS}


# ──────────────────────────────────────────────────────────────────────
# Generation parameters
# ──────────────────────────────────────────────────────────────────────
TEMPERATURE: float    = 0.2
TOP_P: float          = 0.95
TOP_K: int            = 40
REPEAT_PENALTY: float = 1.1
NUM_CTX: int          = 4096
MAX_OUTPUT_TOKENS: int = 2048

TIME_BUDGET_S: int    = 600   # per (cell, function) wall-clock budget


# ──────────────────────────────────────────────────────────────────────
# Dataset configuration
# ──────────────────────────────────────────────────────────────────────
# Per the §4.1.1 decision: 150-function core sweep, 25 LiveCodeBench, 50 HEM.
CORE_SAMPLE_SIZE: int       = 150
LIVECODEBENCH_SAMPLE_SIZE: int = 25
HUMANEVAL_MUTATED_SAMPLE_SIZE: int = 50
DATASET_SEED: int           = 42


# ──────────────────────────────────────────────────────────────────────
# Mutation testing (carried over from Chapter 2)
# ──────────────────────────────────────────────────────────────────────
MAX_MUTANTS_PER_FUNCTION: int = 15
MUTATION_OPERATORS: tuple[str, ...] = (
    "arithmetic",
    "boundary",
    "comparison",
    "negate_bool",
    "return_none",
)


# ──────────────────────────────────────────────────────────────────────
# Output format
# ──────────────────────────────────────────────────────────────────────
RESULTS_TSV: Path = RESULTS_DIR / "results_roundtrip.tsv"
PILOT_RESULTS_TSV: Path = RESULTS_DIR / "pilot_results.tsv"
