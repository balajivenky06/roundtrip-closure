"""
doe.py — pre-registered 20-cell Design-of-Experiments table.

The DOE is the experimental backbone. Every cell is declared here BEFORE
the experiment runs so that the configuration cannot be silently changed
to favourable assignments after results come in.

Strata:
    - Stratum A (Mono):   6 cells, one per pipeline SLM, all 3 stages
                          owned by the same model. Establishes the
                          single-LLM baseline (the null hypothesis for
                          the heterogeneous-LLM claim).
    - Stratum B (Hetero): 11 cells, each is a hypothesis-driven
                          assignment of 3 different SLMs to the 3 stages,
                          with the hypothesis recorded for pre-registration.
    - Stratum C (Null):   3 cells, deliberate ablations / artifact-
                          detectors that protect against trivial signals
                          inflating the closure metric.

Total: 6 + 11 + 3 = 20 cells.

This is a hypothesis-driven *fractional factorial* — full 6^3 = 216 cells
would be intractable; the 20 cells above are pre-registered to test main
effects and key 2-way interactions only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from config import (
    ModelSpec,
    LLAMA_3_2_3B,
    PHI_4_14B,
    QWEN_3_6_27B,
    GEMMA_4_26B,
    MISTRAL_SMALL_3_2_24B,
    QWEN_3_CODER_30B,
)


Stratum = Literal["mono", "hetero", "null"]


@dataclass(frozen=True)
class Cell:
    """One row of the DOE table."""
    cell_id: str
    stratum: Stratum
    L_spec: ModelSpec | None        # None means "skip this stage" (null cell)
    L_test: ModelSpec | None
    L_code: ModelSpec | None
    hypothesis: str
    is_pre_registered: bool = True  # all cells in this file are


# ──────────────────────────────────────────────────────────────────────
# Stratum A — Mono cells (6)
# Each pipeline SLM fills all 3 stages on its own.
# ──────────────────────────────────────────────────────────────────────
M1 = Cell(
    cell_id="M1",
    stratum="mono",
    L_spec=LLAMA_3_2_3B, L_test=LLAMA_3_2_3B, L_code=LLAMA_3_2_3B,
    hypothesis="Small-dense self-consistency floor (3 B Meta).",
)
M2 = Cell(
    cell_id="M2",
    stratum="mono",
    L_spec=PHI_4_14B, L_test=PHI_4_14B, L_code=PHI_4_14B,
    hypothesis="Mid-dense reasoning-tuned mono baseline (14 B Microsoft).",
)
M3 = Cell(
    cell_id="M3",
    stratum="mono",
    L_spec=QWEN_3_6_27B, L_test=QWEN_3_6_27B, L_code=QWEN_3_6_27B,
    hypothesis="Latest 2026 dense mono baseline (27 B Alibaba).",
)
M4 = Cell(
    cell_id="M4",
    stratum="mono",
    L_spec=GEMMA_4_26B, L_test=GEMMA_4_26B, L_code=GEMMA_4_26B,
    hypothesis="Latest 2026 MoE mono baseline (26 B Google).",
)
M5 = Cell(
    cell_id="M5",
    stratum="mono",
    L_spec=MISTRAL_SMALL_3_2_24B, L_test=MISTRAL_SMALL_3_2_24B, L_code=MISTRAL_SMALL_3_2_24B,
    hypothesis="Function-calling-tuned dense mono (24 B Mistral).",
)
M6 = Cell(
    cell_id="M6",
    stratum="mono",
    L_spec=QWEN_3_CODER_30B, L_test=QWEN_3_CODER_30B, L_code=QWEN_3_CODER_30B,
    hypothesis="Code-specialised MoE self-consistency (30 B Alibaba-coder).",
)


# ──────────────────────────────────────────────────────────────────────
# Stratum B — Hetero cells (11)
# Each one assigns three DIFFERENT SLMs to the three stages based on
# their per-stage strengths (informed by Chapter 2 per-operator results
# where applicable).
# ──────────────────────────────────────────────────────────────────────
H1 = Cell(
    cell_id="H1",
    stratum="hetero",
    L_spec=PHI_4_14B, L_test=QWEN_3_CODER_30B, L_code=QWEN_3_CODER_30B,
    hypothesis="Specialise by stage strength: Phi-4 for predicate reasoning "
               "(Ch. 2 §4.4 winner), Qwen-coder for tests + code (Ch. 2 Table 13).",
)
H2 = Cell(
    cell_id="H2",
    stratum="hetero",
    L_spec=QWEN_3_6_27B, L_test=PHI_4_14B, L_code=QWEN_3_CODER_30B,
    hypothesis="Latest-dense for spec, reasoning for tests, coder for synth.",
)
H3 = Cell(
    cell_id="H3",
    stratum="hetero",
    L_spec=GEMMA_4_26B, L_test=MISTRAL_SMALL_3_2_24B, L_code=QWEN_3_CODER_30B,
    hypothesis="Cross-family triple: Google → Mistral → Alibaba-coder.",
)
H4 = Cell(
    cell_id="H4",
    stratum="hetero",
    L_spec=QWEN_3_CODER_30B, L_test=QWEN_3_CODER_30B, L_code=LLAMA_3_2_3B,
    hypothesis="Cheap drafter at synthesis: does a 3 B model suffice once "
               "the spec and tests are nailed by stronger models?",
)
H5 = Cell(
    cell_id="H5",
    stratum="hetero",
    L_spec=LLAMA_3_2_3B, L_test=QWEN_3_CODER_30B, L_code=QWEN_3_CODER_30B,
    hypothesis="Cheap drafter at spec: can a 3 B model produce a workable "
               "docstring that downstream stronger models can use?",
)
H6 = Cell(
    cell_id="H6",
    stratum="hetero",
    L_spec=PHI_4_14B, L_test=QWEN_3_6_27B, L_code=PHI_4_14B,
    hypothesis="Same-family hetero-scale ablation: Phi sandwich with a "
               "different-family middle stage.",
)
H7 = Cell(
    cell_id="H7",
    stratum="hetero",
    L_spec=QWEN_3_CODER_30B, L_test=QWEN_3_6_27B, L_code=PHI_4_14B,
    hypothesis="Reverse-capability gradient: strongest first, weakest last.",
)
H8 = Cell(
    cell_id="H8",
    stratum="hetero",
    L_spec=GEMMA_4_26B, L_test=QWEN_3_CODER_30B, L_code=MISTRAL_SMALL_3_2_24B,
    hypothesis="MoE → MoE → dense: does architecture matching matter?",
)
H9 = Cell(
    cell_id="H9",
    stratum="hetero",
    L_spec=QWEN_3_CODER_30B, L_test=QWEN_3_6_27B, L_code=QWEN_3_CODER_30B,
    hypothesis="Strong-sandwich: MoE bookends dense for synthesis stability.",
)
H10 = Cell(
    cell_id="H10",
    stratum="hetero",
    L_spec=MISTRAL_SMALL_3_2_24B, L_test=PHI_4_14B, L_code=QWEN_3_CODER_30B,
    hypothesis="Best-of-each-family-stage: Mistral spec + Phi tests + Qwen-coder synth.",
)
H11 = Cell(
    cell_id="H11",
    stratum="hetero",
    L_spec=PHI_4_14B, L_test=GEMMA_4_26B, L_code=QWEN_3_CODER_30B,
    hypothesis="Phi spec + Gemma tests + Qwen-coder synth (alt family triple).",
)


# ──────────────────────────────────────────────────────────────────────
# Stratum C — Null cells (3)
# Artifact-detectors that protect against trivial signals.
# ──────────────────────────────────────────────────────────────────────
N1 = Cell(
    cell_id="N1",
    stratum="null",
    L_spec=LLAMA_3_2_3B, L_test=LLAMA_3_2_3B, L_code=LLAMA_3_2_3B,
    hypothesis="Prompt-shuffled control (same as M1 but with stage inputs "
               "corrupted): detects whether the closure metric is fooled "
               "by trivial signals.",
)
N2 = Cell(
    cell_id="N2",
    stratum="null",
    L_spec=None,                        # spec-stage ablation: empty D'
    L_test=QWEN_3_CODER_30B, L_code=QWEN_3_CODER_30B,
    hypothesis="Spec-stage ablation: quantifies how much L_spec contributes.",
)
N3 = Cell(
    cell_id="N3",
    stratum="null",
    L_spec=QWEN_3_CODER_30B,
    L_test=None,                        # test-stage ablation: empty T'
    L_code=QWEN_3_CODER_30B,
    hypothesis="Test-stage ablation: quantifies how much L_test contributes.",
)


# ──────────────────────────────────────────────────────────────────────
# Public registry — the canonical 20-cell DOE
# ──────────────────────────────────────────────────────────────────────
MONO_CELLS:   tuple[Cell, ...] = (M1, M2, M3, M4, M5, M6)
HETERO_CELLS: tuple[Cell, ...] = (H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11)
NULL_CELLS:   tuple[Cell, ...] = (N1, N2, N3)

ALL_CELLS: tuple[Cell, ...] = MONO_CELLS + HETERO_CELLS + NULL_CELLS

CELLS_BY_ID: dict[str, Cell] = {c.cell_id: c for c in ALL_CELLS}


def get_cell(cell_id: str) -> Cell:
    """Look up a cell by its DOE id (e.g. 'M3', 'H1', 'N2')."""
    if cell_id not in CELLS_BY_ID:
        raise KeyError(
            f"Unknown cell id {cell_id!r}. "
            f"Known cells: {sorted(CELLS_BY_ID.keys())}"
        )
    return CELLS_BY_ID[cell_id]


# ──────────────────────────────────────────────────────────────────────
# Pilot subset (Plan C, §6 of the concept note)
# 6 cells chosen to exercise every code path in the system on 30 functions.
# ──────────────────────────────────────────────────────────────────────
PILOT_CELLS: tuple[Cell, ...] = (M1, M3, M6, H1, H4, N2)


if __name__ == "__main__":
    # Quick sanity check
    print(f"DOE has {len(ALL_CELLS)} cells across 3 strata:")
    print(f"  Mono   ({len(MONO_CELLS):2d}): {[c.cell_id for c in MONO_CELLS]}")
    print(f"  Hetero ({len(HETERO_CELLS):2d}): {[c.cell_id for c in HETERO_CELLS]}")
    print(f"  Null   ({len(NULL_CELLS):2d}): {[c.cell_id for c in NULL_CELLS]}")
    print(f"\nPilot subset ({len(PILOT_CELLS)}): {[c.cell_id for c in PILOT_CELLS]}")
