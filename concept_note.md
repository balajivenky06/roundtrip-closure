# PhD Thesis — Chapter 3 Concept Note
## Heterogeneous Multi-SLM Closure of the Docstring–Test–Code Triangle: A Mutation-Testing Study

**Author:** Balaji Venktesh V
**Supervisor:** Amsaprabhaa M
**Co-author:** Gireesh Sundaram
**Date:** 2026-06-04 (advisor-signed-off 2026-06-03; SLM/all-Ollama pivot 2026-06-03)
**Status:** Concept note approved — engineering scaffold complete (commit `f07b4a6` in `roundtrip-closure/` repo)

---

## 1. Background and motivation

This proposal forms the third paper of a three-paper PhD thesis on **Small Language Models for Software Engineering**:

| Chapter | Paper status | Focus |
|---|---|---|
| Ch. 1 | Submitted to Springer-Nature *Automated Software Engineering*, 2026-06-03 | RAG-based docstring generation across open-weight SLMs |
| Ch. 2 | Submitted to Springer-Nature *Software Quality Journal*, 2026-06-02 | RAG-based unit test generation, evaluated by mutation kill rate |
| **Ch. 3** | **This proposal** | **Heterogeneous multi-SLM round-trip closure across the docstring–test–code triangle** |

The thesis through-line is that **heterogeneous multi-SLM pipelines under software-engineering-relevant metrics reveal interaction effects between method choice and underlying-SLM capability that single-SLM studies cannot surface**. Chapters 1 and 2 establish this claim on individual generation tasks; Chapter 3 generalises it to the cross-task setting.

The empirical problem is concrete. Practitioners deploying SLM-based software-engineering tools — i.e., engineers using open-weight models that run on-premise or on consumer hardware — face the question: **should I use the same SLM for documentation, test generation, and code synthesis, or should I specialise by stage?** Today there is no empirical evidence to answer this with confidence. Existing work either evaluates one stage in isolation (LLM-for-test-gen, LLM-for-doc-gen) or evaluates self-consistency within a single LLM across multiple stages. **No published study has measured cross-stage semantic preservation when stages are owned by different open-weight SLM families, under an SE-relevant metric.**

### 1.1 Operational definition of SLM

Following Microsoft's Phi technical reports (Bubeck et al., 2024) and Google's Gemma reports (DeepMind, 2025–2026), we define a **Small Language Model** (SLM) as any language model with **fewer than 30 billion total parameters**. This admits dense models up to ~28 B and Mixture-of-Experts (MoE) architectures with active-parameter counts in the same range. The 30 B threshold is also the largest size at which inference fits comfortably on a single 40 GB A100 GPU at Q4 quantisation — the standard hardware in modern academic SE replication studies.

---

## 2. Research questions

| RQ | Question |
|---|---|
| RQ1 | When the docstring–test–code triangle is traversed by heterogeneous SLMs (different families per stage), does the closure rate differ significantly from same-family closure, after controlling for sample difficulty? |
| RQ2 | Does mutation kill rate, measured against the original code's ground-truth references, agree with three-annotator human judgments of semantic equivalence? |
| RQ3 | Across our 6-SLM × 3-stage pipeline, which stage is the bottleneck for closure, and does this shift by source benchmark (HumanEval vs MBPP) and by mutation-operator family? |
| RQ4 | Within the open-weight SLM ecosystem, which family combinations dominate the closure-rate-vs-compute Pareto front? (Closed-weight frontier comparison is deferred to future work; see §10 Q1.) |
| RQ5 | What is the false-closure rate — the proportion of pipelines that satisfy the automated closure metric but are judged semantically inequivalent by human annotators? |

---

## 3. Novelty positioning

A literature review identified three pieces of prior art that overlap with the round-trip framing:

| Year | Work | Overlap | Genuine gap |
|---|---|---|---|
| 2024 | DeepMind RTC (Allamanis et al., ICML 2024) — code ↔ NL round-trip | Bidirectional closure idea | Single-LLM, no mutation testing, no 3-way docstring–test–code, closed-weight only |
| 2025 | Coding Triangle (Zhang et al., arXiv 2507.06138) — 3-dimension framework | Triangle framing | Single-LLM self-consistency; explicitly notes "model mixtures can enhance" as future work; competitive-programming benchmarks, no SE-relevant metric |
| 2024 | Mutation-Based Consistency Testing (arXiv 2401.05940) | Uses mutation in LLM-eval | Uses mutation as the *attack* on LLM understanding, not as the closure metric |

We do **not** claim novelty of the round-trip concept. We claim novelty along five orthogonal axes that the prior art leaves explicitly open:

1. **Heterogeneous multi-SLM closure as a research variable** — nobody has empirically tested whether closure rate differs between mono-SLM and hetero-SLM configurations.
2. **Mutation kill rate as the closure metric** — prior work uses surface metrics (BLEU, unit-test pass, self-consistency); we use the SE-validated defect-detection metric carried forward from Chapter 2.
3. **Open-weight-only, cross-family SLM coverage with statistical rigor** — frontier-only closed-weight studies dominate the round-trip-closure literature; our pre-registered fractional-factorial design across 6 open-weight SLM pipeline models drawn from 5 distinct families (Meta, Microsoft, Alibaba, Google, Mistral) plus a 6th family (DeepSeek) as the external judge is unprecedented at this scale.
4. **Per-stage bottleneck decomposition** — leave-one-stage-out reruns to identify which stage is the closure-failure source, decomposed by benchmark and mutation operator.
5. **Human validation of the automated closure metric** — first study to compare an automated closure metric to human equivalence judgments at our scale.

---

## 4. Experimental design

### 4.1 Dataset

- 150 functions sampled (seed = 42) from HumanEval + MBPP — a 50 % stratified sub-sample of the 300-function Chapter 2 set, preserving the same MBPP/HumanEval ratio (≈ 105 MBPP + 45 HumanEval) for direct comparability with Chapter 2 results. The sub-sampling is justified in §4.1.1 below.
- Held-out decontamination subset: 25 LiveCodeBench problems with publication date ≥ 2024-12-01 (post training-cutoff of all evaluated open-weight LLMs)
- HumanEval-Mutated subset: 50 HumanEval problems with function-rename + docstring-paraphrase + parameter-permutation transformations applied

#### 4.1.1 Justification for the 150-function sample

Chapter 2 ran on 300 functions and reported per-cell statistics from the SQJ submission. For Chapter 3 we sub-sample to 150 functions per cell — a deliberate compute-budget decision with three justifications:

1. **Statistical power preserved.** With 20 cells × 150 functions = 3{,}000 per-sample observations, the mixed-effects logistic regression has > 80 % power to detect a 5-percentage-point closure-rate difference between mono and hetero configurations at α = 0.05 (computed via simulation against the Chapter 2 variance estimates). The conventional rule-of-thumb of n ≥ 30 per cell is comfortably exceeded.
2. **Direct cross-paper comparability.** The 150-function sub-sample is drawn from the same seed-42 shuffle as Chapter 2, so any per-function effects observed in Chapter 2 carry directly into Chapter 3 — enabling joint analyses without re-running Chapter 2.
3. **Compute budget feasibility.** The 50 % sample reduction approximately halves GPU-hours from the original 250-hour estimate to ~125 hours, bringing the experiment within a single researcher's Colab Pro+ A100 monthly allowance after caching.

The held-out datasets (LiveCodeBench and HumanEval-Mutated) are also reduced 50 % proportionally to keep the ratio of primary-vs-held-out functions consistent with the original design.

### 4.2 Pipeline (3 stages)

```
              D (docstring)
              ▲   ▲
            α/     \β
            /       \
           C ─────── T
           (code)   (tests)
              γ

Path 1:  C → D → T   close-check: mutation kill rate of T against original C's mutants
Path 2:  D → T → C   close-check: pass rate of ORIGINAL C's reference tests against reconstructed C
Path 3:  C → T → D   close-check: BERTScore + judge-LLM equivalence to original D
```

### 4.3 Models

**Six open-weight SLM pipeline models** + **one open-weight judge SLM**, all served via Ollama. No API calls, no closed-weight models in this design.

| Slot | Ollama tag | Family | Size | Released | Notes |
|---|---|---|---|---|---|
| Small floor | `llama3.2:3b` | Meta | 3 B dense | Oct 2024 | Only Meta SLM under 30 B; widely benchmarked baseline. |
| Mid-dense reasoning | `phi4:14b` | Microsoft | 14 B dense | Dec 2024 | Reasoning-tuned dense; latest Phi as of mid-2026. |
| Latest dense general | `qwen3.6:27b` | Alibaba | 27 B dense | Apr 2026 | April 2026 release — beats Qwen 3.5 397B-A17B on SWE-bench. |
| Latest MoE | `gemma4:26b` | Google | 26 B MoE | Mar 2026 | March 2026 release; ~3.8 B active. |
| Latest Mistral | `mistral-small3.2:24b` | Mistral | 24 B dense | 2025 | Function-calling-tuned; latest Mistral Small. |
| Coder specialist | `qwen3-coder:30b` | Alibaba (coder) | 30 B MoE | 2025 | Code-specialised MoE, RL-trained on SWE-bench; 3.3 B active. |
| **Judge** (external to pipeline) | `deepseek-r1:14b` | DeepSeek | 14 B dense | 2025 | Reasoning-tuned distill, never used in any pipeline cell. |

Five distinct families in the pipeline (Meta, Microsoft, Alibaba × 2, Google, Mistral) plus DeepSeek as the external judge. Total disk footprint ≈ 83 GB at Q4 quantisation; fits easily on Colab persistent storage. Ollama auto-swaps models in and out of GPU memory.

**Cost implication.** Because no API calls are made, the marginal experimental cost is **$0** (Colab Pro+ subscription is already amortised against Chapter 2). Closed-weight reference cells (Claude / GPT-class) have been deferred to a future revision round (§10 Q1).

### 4.4 Pre-registered design-of-experiments table (20 configurations)

| # | Stratum | L_spec (D) | L_test (T) | L_code (C) | Hypothesis |
|---|---|---|---|---|---|
| M1 | Mono | llama3.2 3B | llama3.2 3B | llama3.2 3B | Small-dense self-consistency floor (3 B Meta). |
| M2 | Mono | phi4 14B | phi4 14B | phi4 14B | Mid-dense reasoning-tuned mono baseline (14 B Microsoft). |
| M3 | Mono | qwen3.6 27B | qwen3.6 27B | qwen3.6 27B | Latest 2026 dense mono baseline (27 B Alibaba). |
| M4 | Mono | gemma4 26B | gemma4 26B | gemma4 26B | Latest 2026 MoE mono baseline (26 B Google). |
| M5 | Mono | mistral-small3.2 24B | mistral-small3.2 24B | mistral-small3.2 24B | Function-calling-tuned dense mono (24 B Mistral). |
| M6 | Mono | qwen3-coder 30B | qwen3-coder 30B | qwen3-coder 30B | Code-specialised MoE self-consistency (30 B Alibaba-coder). |
| H1 | Hetero | phi4 14B | qwen3-coder 30B | qwen3-coder 30B | Specialise by stage strength: phi4 for predicate reasoning (Ch. 2 §4.4 winner), qwen3-coder for tests + code (Ch. 2 Table 13). |
| H2 | Hetero | qwen3.6 27B | phi4 14B | qwen3-coder 30B | Latest-dense for spec, reasoning for tests, coder for synth. |
| H3 | Hetero | gemma4 26B | mistral-small3.2 24B | qwen3-coder 30B | Cross-family triple: Google → Mistral → Alibaba-coder. |
| H4 | Hetero | qwen3-coder 30B | qwen3-coder 30B | llama3.2 3B | Cheap drafter at synthesis: does a 3 B model suffice once the spec and tests are nailed by stronger models? |
| H5 | Hetero | llama3.2 3B | qwen3-coder 30B | qwen3-coder 30B | Cheap drafter at spec: can a 3 B model produce a workable docstring that downstream stronger models can use? |
| H6 | Hetero | phi4 14B | qwen3.6 27B | phi4 14B | Same-family hetero-scale ablation: Phi sandwich with a different-family middle stage. |
| H7 | Hetero | qwen3-coder 30B | qwen3.6 27B | phi4 14B | Reverse-capability gradient: strongest first, weakest last. |
| H8 | Hetero | gemma4 26B | qwen3-coder 30B | mistral-small3.2 24B | MoE → MoE → dense: does architecture matching matter? |
| H9 | Hetero | qwen3-coder 30B | qwen3.6 27B | qwen3-coder 30B | Strong-sandwich: MoE bookends dense for synthesis stability. |
| H10 | Hetero | mistral-small3.2 24B | phi4 14B | qwen3-coder 30B | Best-of-each-family-stage: Mistral spec + Phi tests + Qwen-coder synth. |
| H11 | Hetero | phi4 14B | gemma4 26B | qwen3-coder 30B | Phi spec + Gemma tests + Qwen-coder synth (alt family triple). |
| N1 | Null | llama3.2 3B | llama3.2 3B | llama3.2 3B with prompt-shuffled inputs | Prompt-shuffled control: detects whether the closure metric is fooled by trivial signals. |
| N2 | Null | (skip; empty D) | qwen3-coder 30B | qwen3-coder 30B | Spec-stage ablation: quantifies how much L_spec contributes. |
| N3 | Null | qwen3-coder 30B | (skip; empty T) | qwen3-coder 30B | Test-stage ablation: quantifies how much L_test contributes. |

**Total: 20 configurations × 150 functions × 3 closure paths = 9{,}000 round-trips on the core sweep. Adding the held-out subsets (25 LiveCodeBench + 50 HumanEval-Mutated) brings the full sweep to 13{,}500 round-trips. Approximately 110–130 GPU-hours on a single A100 with the caching layer described in §4.7, comfortably inside one month of Colab Pro+ A100 allowance.**

The canonical definition of these 20 cells lives in code at `doe.py` in the project repo (`/Users/balajivenktesh/Desktop/Education/roundtrip-closure/`); the table above is the human-readable mirror of that module.

### 4.5 Statistical methodology

- **Mixed-effects logistic regression:** `closure_success ~ L_spec * L_test * L_code + (1 | function_idx)`, where `closure_success` is binary (strong / not). Random intercept absorbs per-function difficulty. Two-way interactions reveal stage-coupling effects.
- **Type-III ANOVA** on continuous closure rate per (configuration, function) pair.
- **Tukey HSD post-hoc** for pairwise configuration comparisons (Bonferroni-corrected across the 20 × 19 / 2 = 190 pairs at α = 0.05).
- **Per-stage failure decomposition:** leave-one-stage-out reruns for each Stratum-B configuration, identifying which stage owns the closure failure.
- **Per-benchmark and per-mutation-operator slicing** following Chapter 2 methodology.

### 4.6 Human validation (RQ2, RQ5)

- 60 stratified pairs of (original function, round-trip-reconstructed code), **conducted after the full statistical sweep** so that sampling can target the most-informative cases empirically. The **rubric and sampling strategy are pre-registered** before the sweep to prevent post-hoc cherry-picking.
- **Pre-registered sampling strategy:** the 60 pairs are drawn from three buckets — 20 "frontier" cases where the automated metric strongly claims closure but the configuration was deliberately mismatched (e.g., cells H4, H5, N1); 20 "agreement" cases where mono-LLM closure metric and hetero-LLM closure metric agree (sanity check); 20 "disputed" cases where the judge LLM disagrees with the automated metric. This makes the human study a focused investigation into the automated metric's blind spots rather than a generic baseline.
- Three independent annotators rate semantic equivalence on a 0–4 scale (`identical / equivalent / approximately equivalent / clearly different / unrelated`)
- **Calibration session before annotation** — directly addressing the SQJ-reviewer feedback on Chapter 2's rubric calibration. 10 calibration items rated jointly, divergence discussed, anchors re-set, then the 60 production pairs rated individually.
- Target Krippendorff's α ≥ 0.5 (simpler single-axis rubric than Chapter 2's three-axis rubric makes this achievable)
- Validation analysis: false-closure rate = fraction of pipelines where automated metric reports closure but humans + judge LLM disagree

### 4.7 Caching layer (compute-budget optimisation)

A per-stage caching layer reduces redundant LLM calls. Cache key: `SHA256(model_name + stage_role + canonicalised_prompt)`; cache value: the LLM output, stored on disk and reloaded on hit.

The cache eliminates duplicate work in three places:
1. **L_spec stage:** any cell where `L_spec = SLM_X` operating on the same source code produces the identical docstring. Across the 20 cells × 6 unique SLMs at L_spec, we compute each unique (SLM, function) pair once.
2. **Path overlap:** Path 1 (C→D→T) and Path 3 (C→T→D) share the L_spec call on the same code C.
3. **Re-entries during recovery:** if the sweep is interrupted (Colab session timeout, transient network glitch), the cache survives and the sweep resumes from the last completed call rather than restarting.

Realistic cache hit rate measured against Chapter 2's sweep traces: **~28–35 %** of SLM calls become cache hits. Net effect: the naive 20 cells × 150 functions × 3 paths × 2 calls = 18{,}000 SLM calls drops to roughly 11{,}700–13{,}000 actual invocations. This is the source of the 110–130 GPU-hour estimate (vs the ~210 GPU-hours an un-cached run would consume).

The caching layer is implemented in `closure_cache.py` (commit `3ece666`) with a SHA-256 keyed disk store; verified on llama3.2:3b — second call returned in 0.13 ms vs 186 ms first-call (≈1400× speed-up).

---

## 5. Defenses against reviewer blind spots

| Blind spot | Defense |
|---|---|
| **Combinatorial explosion** (6³ = 216 cells is intractable) | Pre-registered fractional-factorial DOE in §4.4 above; 20 cells justified by per-cell hypothesis; computational budget ≈ 110–130 GPU-hours on the 150-function sample with caching (§4.7). |
| **False closure** (error propagation can mimic closure) | Four-layer defense: (a) original-artifact anchoring — closure measured against original code's reference tests, not pipeline-internal tests; (b) triple-path agreement — strong / weak / no closure classification; (c) **external SLM judge** (`deepseek-r1:14b`, DeepSeek family — distinct from every pipeline-cell family) validates equivalence; the judge is a reasoning-tuned distill, which is the recommended profile for equivalence rating, and being from a non-pipeline family avoids "model rating itself"; (d) human study oversamples closure-passing-but-judge-flagged cases. |
| **Data contamination** (HumanEval/MBPP in training data) | Three layers: (a) held-out LiveCodeBench post-2024-12-01 subset; (b) HumanEval-Mutated decontamination transformation (rename + paraphrase + permute); (c) per-SLM training-cutoff disclosure + correlation analysis (contaminated closure − decontaminated closure). |

These defenses materialise as named subsections in the manuscript (§3 Methods has "Pre-registered DOE table", "Decontamination protocol", "Closure validity layers"; §6 Threats has "False closure", "Combinatorial coverage", "Contamination sensitivity").

---

## 6. Expected contributions

1. **First empirical study of heterogeneous-multi-LLM closure** across the docstring–test–code triangle, with mutation kill rate as the closure metric.
2. **Pre-registered design-of-experiments protocol** for multi-stage LLM-pipeline studies — a methodological contribution usable beyond this paper.
3. **Identification of the per-stage closure bottleneck** and how it shifts with benchmark and defect family.
4. **Human-validated automated closure metric** with a documented false-closure rate.
5. **Open replication package** extending the `autoresearch/` codebase, reusing the Chapter 2 infrastructure.

The unifying contribution to the PhD thesis is the empirical demonstration that **heterogeneous multi-LLM pipelines under SE-relevant metrics reveal interaction effects that single-LLM studies cannot surface** — closing the loop on the three-paper arc.

---

## 7. Timeline (single-researcher)

| Months | Activity |
|---|---|
| 1–2 | Engineering: build out the `roundtrip-closure` project (separate repo from `autoresearch`) — 3-stage pipeline + closure-metric module + DeepSeek-R1 judge-SLM integration + decontamination transform. *Scaffold complete 2026-06-03 (commit `f07b4a6`); Batch 1 (closure_cache + ollama_client) complete 2026-06-04 (commit `a51c499`).* |
| 3 | Pilot: 30 functions × 6 cells (M1, M3, M6, H1, H4, N2 — see `doe.PILOT_CELLS`); tune closure-validity thresholds. **Pre-register human-evaluation rubric and sampling strategy now**, before the sweep begins. |
| 4 | Full sweep: 20 configurations × 150 functions + 25 LiveCodeBench + 50 HumanEval-Mutated ≈ 110–130 GPU-hours on A100 (single month of Colab Pro+). |
| 5 | Buffer / spill-over month for sweep re-runs, transient Ollama-server restarts, and additional held-out experiments. |
| 6 | Statistical analysis (mixed-effects, ANOVA, per-stage decomposition); preliminary write-up of §4 Results |
| 7–8 | Human-validation study (60 pairs, 3 annotators, calibration session) — **2-month allocation** because rater-recruitment, calibration, and Krippendorff's-α convergence are historically slow phases |
| 9–10 | Manuscript drafting and revision |
| 11 | Submission to EMSE (primary) or TSE (secondary) |

Total: 11 months from advisor approval to first submission. The two-month buffer on human evaluation is deliberate — Chapter 2's annotation phase taught us that rater coordination is the slowest part of an empirical SE study, and the SQJ reviewer feedback explicitly flagged calibration as a quality bottleneck. We absorb that lesson directly into the plan.

Assumes SQJ reviewer feedback on Chapter 2 has been incorporated by month 4; if a major-revision request arrives during months 4–8, the Chapter 3 sweep can pause and resume without losing intermediate state thanks to the per-configuration checkpoint structure inherited from Chapter 2's `mutation_testing.py` (copied verbatim into `roundtrip-closure/mutation_testing.py`).

---

## 8. Target venues

- **Primary:** *Empirical Software Engineering* (Springer) — strong fit for SE-methodology + multi-LLM empirical contributions
- **Secondary:** *IEEE Transactions on Software Engineering* — accepts longer empirical studies with formal-methods undertones
- **Conference derivative:** ICSE 2028 short paper or ICSE-SEIP if industrial replication partner is found

---

## 9. Reading list (before implementation)

1. Allamanis, Panthaplackel, Yin. *Unsupervised Evaluation of Code LLMs with Round-Trip Correctness.* ICML 2024. arXiv 2402.08699.
2. Zhang et al. *Coding Triangle: How Does Large Language Model Understand Code?* arXiv 2507.06138, 2025.
3. Cohen et al. *VeCoGen: Automating Generation of Formally Verified C Code with LLMs.* arXiv 2411.19275, 2024.
4. Sun et al. *Clover: Closed-Loop Verifiable Code Generation.* Stanford 2024.
5. Tian et al. *Mutation-Based Consistency Testing for Evaluating the Code Understanding Capability of LLMs.* arXiv 2401.05940, 2024.

---

## 10. Decisions log

Decisions accumulated across the design phase (2026-06-02 → 2026-06-04):

| # | Decision | Rationale |
|---|---|---|
| Q1 | **Closed-weight references DEFERRED to revision round.** Originally cells M5, H2, H8 were Claude-based ceilings. After the SLM/all-Ollama pivot (Q6), these cells are dropped from the main design and reintroduced only if a reviewer demands a frontier comparison. | Restricting the design to **all-open-weight SLMs** sharpens the thesis claim (multi-SLM ecosystem) and removes the only variable cost from the experiment. Frontier comparison can be added in revision in a few hundred USD of API spend. |
| Q2 | **External judge SLM = `deepseek-r1:14b`** (DeepSeek family), accessed through the same Ollama interface as the pipeline. | Distinct family from every pipeline cell, so no "model rating itself"; reasoning-tuned distill, the recommended profile for equivalence judgment; fits the SLM thesis framing (judge is also <30 B). |
| Q3 | **Human validation runs AFTER the full statistical sweep**, but rubric and sampling strategy are pre-registered during the pilot (month 3) BEFORE the sweep begins. | After-sweep timing enables strategic sampling of frontier cases (closure-passes-but-mismatched-config, judge-LLM-disagreement). Pre-registration of rubric prevents p-hacking accusations. The study becomes a focused investigation into the automated metric's blind spots rather than a generic baseline. |
| Q4 | **Timeline revised to 11 months** (was 9) with a 2-month allocation for the human-validation phase. | Rater recruitment, calibration session, and Krippendorff's-α convergence are historically the slowest phases of an SE empirical study. Chapter 2 reviewer feedback explicitly flagged calibration as a quality bottleneck; absorbing that lesson directly into the plan. |
| Q5 | **Per-cell sample reduced from 300 → 150 functions** in the core sweep; held-out datasets reduced 50 % proportionally (LiveCodeBench 50 → 25, HumanEval-Mutated 100 → 50). | The original 250-hour GPU budget was infeasible on a single researcher's Colab Pro+ A100 monthly allowance. The 150-function sample preserves > 80 % statistical power for the mixed-effects detection target (5 pp closure-rate difference at α = 0.05) and remains comfortably above the n ≥ 30 per cell ANOVA rule of thumb. Combined with the §4.7 caching layer, total GPU-hours drop to 110–130 — fits one Colab Pro+ month with buffer. |
| Q6 | **Small Language Models (SLM) thesis pivot (2026-06-03).** The chapter is reframed from "multi-LLM" to "multi-SLM"; the entire 6-model pipeline is restricted to <30 B open-weight SLMs; the judge is also <30 B. | Unifies the three-paper thesis (Ch.1, Ch.2, Ch.3 all about SLMs for software engineering); plays directly into the "practitioner deploys on-premise" framing; addresses the SLM-frontier interest in the 2026 community. Cost falls to ~$0 marginal. The closed-weight comparison (originally Q1's purpose) is deferred to a revision round per Q1 above. |
| Q7 | **6-model SLM lineup** finalised: `llama3.2:3b`, `phi4:14b`, `qwen3.6:27b`, `gemma4:26b`, `mistral-small3.2:24b`, `qwen3-coder:30b`. Judge = `deepseek-r1:14b`. All accessed through Ollama. | Five distinct families in the pipeline + DeepSeek for the judge = six families total. All are the latest under-30 B model from their respective family as of June 2026 (Meta only releases Llama-3.3 at 70 B, hence Llama 3.2 3 B is the latest Meta SLM). |
| Q8 | **DOE expanded from 18 → 20 cells** to accommodate the larger model lineup (6 mono instead of 5 + 1 closed; 11 hetero instead of 10 to preserve cross-family triple-coverage; null cells unchanged). | The 5-LLM design had room for one closed-weight ceiling cell; with no closed-weight cells the design needs an extra mono baseline and one extra hetero cell to test the additional family. Statistical power remains > 80 %; Bonferroni count rises from 153 to 190 pairs. |

---

*End of concept note. Ready for advisor sign-off. Source data, infrastructure, and statistical methodology are direct extensions of Chapter 2 (SQJ submission, 2026-06-02).*
