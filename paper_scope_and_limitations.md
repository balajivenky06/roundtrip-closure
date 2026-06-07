# Paper scope decisions & limitations to address before submission

Running notes on design decisions that need explicit acknowledgement in
the Chapter 3 paper's *Limitations* / *Future Work* / *Threats to
Validity* sections. Append new decisions as they come up.

Status legend: **DECIDED** = will not change before submission ·
**OPEN** = still to be resolved · **DRAFT** = paragraph ready, awaiting
final edit pass.

---

## 1. No RAG, no iterative critique  — **DECIDED**

### Question
Are we using any RAG or iterative-critique methodology (the Chapter 1 /
Chapter 2 methods) in the round-trip closure pipeline?

### Current state
**No.** Every LLM call in `closure_paths.py` is single-shot, direct
prompting. Specifically:

- The 5 stage callers (`_call_doc_from_code`, `_call_tests_from_doc`,
  `_call_tests_from_code`, `_call_doc_from_tests`,
  `_call_code_from_doc_tests`) each format ONE prompt → make ONE
  `ollama_client.call_llm` call → return the text.
- No vector store, no nearest-neighbour retrieval, no example bank.
- The judge (`judge_llm.judge_equivalence`) is also one-shot per closure
  check — no critique-and-revise loop on the rating.
- The three round-trip paths (P1: C→D→T, P2: D→T→C, P3: C→T→D) are
  linear pipelines; there is no feedback edge from a downstream stage
  back to an upstream one.

### Rationale
Chapter 1 (ASE, under review) already established the methodology
ranking for docstring generation:
**Iterative Critique RAG > Simple RAG > Plain LLM**. Chapter 2 (SQJ,
under review) carried those methods over to unit-test generation. The
*novelty claim* of Chapter 3 is specifically about **heterogeneous
multi-SLM closure** — i.e., the experimental variable is *which SLM
owns which stage*, not *which methodology each stage uses*.

Adding methodology as a third experimental dimension would inflate the
design:

> 3 methodologies × 20 DOE cells × 150 functions × 3 paths
> = ~27,000 round-trips (≈ 150 GPU-hours on Colab A100)

vs. the current 9,000 round-trips (≈ 50 GPU-hours).

Holding methodology constant at "plain prompting" lets the SLM-assignment
variable be cleanly isolated.

### Paper framing — DRAFT paragraph for §6 (Limitations)
> All stages in this study use single-shot, direct prompting; we do not
> apply the Retrieval-Augmented Generation or Iterative Critique RAG
> methods evaluated in Chapters 1–2 of this thesis. This isolates the
> multi-SLM stage-assignment variable from the methodology variable but
> means the absolute closure rates reported here are **lower bounds**.
> Applying Iterative Critique RAG at each stage would likely improve
> every cell uniformly; we expect it to raise absolute rates without
> reordering the relative ranking of cells, but verifying this is
> non-trivial because the resulting design is a 3-way factorial (model
> × stage × methodology = 60+ cells) that exceeds our GPU budget. We
> leave that factorial as future work.

### Future-work extension (if pursued later)
The cleanest hook would be adding a `methodology` field to `Cell`:

```python
@dataclass(frozen=True)
class Cell:
    ...
    methodology: Literal["plain", "rag", "iter_critique"] = "plain"
```

`closure_paths` would then route each stage through a methodology-aware
caller (e.g. `_call_doc_from_code_iter_critique`). The judge could
remain methodology-agnostic. Effort estimate: ~2 weeks engineering,
~100 GPU-hours additional compute.

---

## 2. Mono baselines — Gemma4 26B MoE is the new best  — **DECIDED**

### Finding (Phase 2 mid-sweep gate, 2026-06-07)
Across all 6 mono cells on the 150-function core sweep, Path 1 NaN rates (mutation kill-rate failed because LLM-generated tests didn't pass the original code) rank:

| Cell | Model | Path 1 NaN | Valid yield | Rank |
|---|---|---|---|---|
| M4 | gemma4:26b (MoE) | **6%** | 441/450 | 🥇 |
| M5 | mistral-small3.2:24b | 16% | 426/450 | 🥈 |
| M6 | qwen3-coder:30b (MoE) | 16% | 426/450 | 🥈 |
| M1 | llama3.2:3b | 18% | 423/450 | 4 |
| M3 | qwen3.6:27b | 22% | 417/450 | 5 |
| M2 | phi4:14b | **44%** | 384/450 | ⚠ |

### Why this matters
Challenges the assumption that the *code-specialised* model (qwen-coder, RL-trained on SWE-bench) should win every code-generation task. Gemma4 26B — a *general-purpose* MoE from Google — produces filter-passing tests **94 % of the time** vs qwen-coder's **84 %**. Hetero cells that use gemma4 as L_test (H3, H8, H11) are now the headline candidates for the §4 Results table.

### Paper framing — DRAFT paragraph for §5 (Discussion)
> Counter to the intuition that domain-specialised models dominate
> their domain, our mono-cell results place the general-purpose
> **Gemma4 26B MoE as the strongest single-model baseline** for
> closure (6 % Path 1 NaN rate), outperforming the code-specialised
> Qwen3-Coder 30B MoE (16 % NaN). This is consistent with prior
> reports that MoE architectures of comparable active-parameter count
> generalise better than dense-trained code specialists when the
> downstream task involves *test generation* (Gemma4 must reason
> about specification coverage; Qwen-Coder is reward-shaped for
> SWE-bench patch generation). We discuss this further in §5.3.

### Future work
- Investigate whether the gemma4 advantage holds at larger N (full sweep across all 20 cells, including hetero cells with gemma4 in any stage).
- Cross-check on a different metric (Path 3 BERTScore) — if gemma4 also leads there, the claim strengthens.

---

## 3. Phi4 14B is a strong L_spec but a weak L_test  — **DECIDED**

### Finding (Phase 1 + Phase 2 mid-sweep gates, 2026-06-07)
Phi4 14B's Path 1 NaN rate **jumps from 13.3 % in H1** (phi4 as L_spec, qwen-coder as L_test+L_code) **to 44 % in M2** (phi4 as L_spec + L_test + L_code).

| Cell | L_spec | L_test | L_code | Path 1 NaN | Source |
|---|---|---|---|---|---|
| H1 | phi4 | qwen-coder | qwen-coder | 13.3 % | pilot |
| M2 | phi4 | phi4 | phi4 | 44 % | full sweep Phase 2 |

The L_spec stage is unchanged across H1 and M2; only the L_test (and L_code) stages differ. **The 3.3× degradation isolates the failure mode to phi4-as-L_test.**

### Why this matters
This is the **central empirical claim of the heterogeneous-SLM thesis**. Per-stage strengths and weaknesses exist; a single mono assignment leaves performance on the table; the right heterogeneous assignment can outperform the best mono baseline. Phi4 + qwen-coder is the cleanest worked example.

### Paper framing — DRAFT paragraph for §4 (Results) or §5.2 (Worked Example)
> The phi4-14B model exhibits a 3.3× degradation in mutation-kill-rate
> yield when assigned to the test-generation stage (M2: 44 % NaN) versus
> when assigned only to the specification stage with qwen-coder taking
> tests (H1: 13.3 % NaN). Because the input to L_spec is the same in
> both cells, this isolates phi4's weakness to test-generation
> specifically. We attribute this to phi4's reasoning-tuning, which
> produces highly-specific test assertions that the original (ground-
> truth) code does not satisfy and are therefore filtered. This is
> the cleanest direct evidence in our data for the per-stage
> heterogeneity hypothesis (H1 of §3.2).

### How to apply
- Treat (M2 NaN, H1 NaN) as a **paired data point** in §5: "mono test-generation weakness with hetero recovery."
- For Figure 4 (mono-vs-hetero), highlight the M2→H1 arrow.
- In Limitations, note that the finding is on phi4 specifically; replication on other reasoning-tuned models (deepseek-r1, qwen3.6) would strengthen the per-stage claim.

---

## Cross-reference

- `concept_note.md` — full Chapter 3 experimental design
- `doe.py` — 20-cell DOE table (mono / hetero / null strata)
- `closure_paths.py` — three-path implementation
- `judge_llm.py` — DeepSeek-R1-as-judge
- Thesis structure: Chapter 1 (ASE, under review), Chapter 2 (SQJ,
  under review), Chapter 3 (this paper, target venue TBD)
