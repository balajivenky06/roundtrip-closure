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

## 2. _<placeholder for next scope decision>_  — **OPEN**

(Empty — to be filled when the next limitation comes up.)

---

## Cross-reference

- `concept_note.md` — full Chapter 3 experimental design
- `doe.py` — 20-cell DOE table (mono / hetero / null strata)
- `closure_paths.py` — three-path implementation
- `judge_llm.py` — DeepSeek-R1-as-judge
- Thesis structure: Chapter 1 (ASE, under review), Chapter 2 (SQJ,
  under review), Chapter 3 (this paper, target venue TBD)
