# PhD Thesis — Chapter 3 Concept Note
## Heterogeneous Multi-LLM Closure of the Docstring–Test–Code Triangle: A Mutation-Testing Perspective

**Thesis title:** *Small Language Models for Software Engineering: Retrieval-Augmented Documentation, Mutation-Tested Generation, and Heterogeneous Round-Trip Closure*

**Author:** Balaji Venktesh V
**Supervisor:** Amsaprabhaa M
**Co-author:** Gireesh Sundaram
**Date:** 2026-06-02
**Status:** Concept note for advisor review — pre-implementation

---

## 1. Background and motivation

This proposal forms the third paper of a three-paper PhD thesis:

| Chapter | Paper status | Focus |
|---|---|---|
| Ch. 1 | EMSE-rejected, needs reframing | RAG-based docstring generation across open-weight LLMs |
| Ch. 2 | Submitted to Software Quality Journal, 2026-06-02 | RAG-based unit test generation, evaluated by mutation kill rate |
| **Ch. 3** | **This proposal** | **Multi-LLM round-trip closure across the docstring–test–code triangle** |

The thesis through-line is that **heterogeneous multi-LLM pipelines under software-engineering-relevant metrics reveal interaction effects between method choice and underlying-LLM capability that single-LLM studies cannot surface**. Chapters 1 and 2 establish this claim on individual generation tasks; Chapter 3 generalises it to the cross-task setting.

The empirical problem is concrete. Practitioners deploying LLM-based software-engineering tools face the question: **should I use the same LLM for documentation, test generation, and code synthesis, or should I specialise by stage?** Today there is no empirical evidence to answer this with confidence. Existing work either evaluates one stage in isolation (LLM-for-test-gen, LLM-for-doc-gen) or evaluates self-consistency within a single LLM across multiple stages. **No published study has measured cross-stage semantic preservation when stages are owned by different LLM families, under an SE-relevant metric.**

---

## 2. Research questions

| RQ | Question |
|---|---|
| RQ1 | When the docstring–test–code triangle is traversed by heterogeneous LLMs (different families per stage), does the closure rate differ significantly from same-family closure, after controlling for sample difficulty? |
| RQ2 | Does mutation kill rate, measured against the original code's ground-truth references, agree with three-annotator human judgments of semantic equivalence? |
| RQ3 | Across our N-LLM × 3-stage pipeline, which stage is the bottleneck for closure, and does this shift by source benchmark (HumanEval vs MBPP) and by mutation-operator family? |
| RQ4 | Does open-weight closure approach closed-weight reference closure (Claude Sonnet 4.6, GPT-4o-mini), and at what GPU-cost ratio? |
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

1. **Heterogeneous multi-LLM closure as a research variable** — nobody has empirically tested whether closure rate differs between mono-LLM and hetero-LLM configurations.
2. **Mutation kill rate as the closure metric** — prior work uses surface metrics (BLEU, unit-test pass, self-consistency); we use the SE-validated defect-detection metric carried forward from Chapter 2.
3. **Open-weight LLM coverage with statistical rigor** — frontier-only studies dominate; our pre-registered fractional-factorial design across 5 open-weight + 2 closed-weight reference LLMs is unprecedented at this scale.
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

1. **Statistical power preserved.** With 18 cells × 150 functions = 2{,}700 per-sample observations, the mixed-effects logistic regression has > 80 % power to detect a 5-percentage-point closure-rate difference between mono and hetero configurations at α = 0.05 (computed via simulation against the Chapter 2 variance estimates). The conventional rule-of-thumb of n ≥ 30 per cell is comfortably exceeded.
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

Five open-weight: `llama3.2:latest` (3B), `phi4:14b`, `qwen3.5:9b`, `qwen3-coder:30b` (MoE), `llama3.3:70b`. Two closed-weight references: Claude Sonnet 4.6, GPT-4o-mini. Each LLM available at each of the three stages.

The closed-weight references (cells M5, H2, H8) are required, not optional — without a frontier-model ceiling, reviewers may argue the findings only apply to weaker open-weight models. Closed-weight cells run on a 50-function sample (further reduced from the 150-function open-weight sample) to halve API spend while preserving the ceiling-comparison.

### 4.4 Pre-registered design-of-experiments table (18 configurations)

| # | Stratum | L_spec (D) | L_test (T) | L_code (C) | Hypothesis |
|---|---|---|---|---|---|
| M1 | Mono | llama3.2 3B | llama3.2 3B | llama3.2 3B | Small-dense self-consistency floor |
| M2 | Mono | phi4 14B | phi4 14B | phi4 14B | Mid-dense self-consistency |
| M3 | Mono | qwen3.5 9B | qwen3.5 9B | qwen3.5 9B | Best dense from Chapter 2 |
| M4 | Mono | qwen3-coder 30B | qwen3-coder 30B | qwen3-coder 30B | MoE-coder self-consistency |
| M5 | Mono | Claude 4.6 | Claude 4.6 | Claude 4.6 | Closed-weight ceiling |
| H1 | Hetero | phi4 14B | qwen3-coder 30B | qwen3-coder 30B | "Specialise by stage strength": phi4 dominates predicate reasoning (Ch. 2 §4.4), qwen3-coder dominates comparison operators (Ch. 2 Table 13) |
| H2 | Hetero | Claude 4.6 | qwen3.5 9B | qwen3-coder 30B | Closed NL + open dense test + open MoE code |
| H3 | Hetero | qwen3.5 9B | phi4 14B | qwen3-coder 30B | Ascending capability gradient |
| H4 | Hetero | qwen3-coder 30B | qwen3-coder 30B | llama3.2 3B | Cheap drafter at synthesis stage |
| H5 | Hetero | llama3.2 3B | qwen3-coder 30B | qwen3-coder 30B | Cheap drafter at spec stage |
| H6 | Hetero | phi4 14B | qwen3.5 9B | phi4 14B | Same-family hetero-scale |
| H7 | Hetero | qwen3-coder 30B | qwen3.5 9B | phi4 14B | Reverse-gradient (does strong-first matter?) |
| H8 | Hetero | Claude 4.6 | Claude 4.6 | qwen3-coder 30B | Closed→Open synthesis-stage boundary |
| H9 | Hetero | qwen3-coder 30B | qwen3.5 9B | qwen3-coder 30B | Strong-sandwich (MoE bookends dense) |
| H10 | Hetero | llama3.3 70B | phi4 14B | qwen3-coder 30B | Best-of-each-stage hypothesis |
| N1 | Null | llama3.2 3B | llama3.2 3B | llama3.2 3B with seed-shuffled prompts | Prompt-shuffled control to detect trivial saturation |
| N2 | Null | (skip D, use empty string) | qwen3-coder 30B | qwen3-coder 30B | Spec-stage ablation: how much does L_spec contribute? |
| N3 | Null | qwen3-coder 30B | (skip T, use only D) | qwen3-coder 30B | Test-stage ablation: how much does L_test contribute? |

**Total: 18 configurations × 150 functions × 3 closure paths = 8{,}100 round-trips on the core sweep. Adding the held-out subsets (25 LiveCodeBench + 50 HumanEval-Mutated) brings the full sweep to 12{,}150 round-trips. Approximately 110–130 GPU-hours on a single A100 with the caching layer described in §4.7, comfortably inside one month of Colab Pro+ A100 allowance.**

### 4.5 Statistical methodology

- **Mixed-effects logistic regression:** `closure_success ~ L_spec * L_test * L_code + (1 | function_idx)`, where `closure_success` is binary (strong / not). Random intercept absorbs per-function difficulty. Two-way interactions reveal stage-coupling effects.
- **Type-III ANOVA** on continuous closure rate per (configuration, function) pair.
- **Tukey HSD post-hoc** for pairwise configuration comparisons (Bonferroni-corrected across the 18 × 17 / 2 = 153 pairs at α = 0.05).
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
1. **L_spec stage:** any cell where `L_spec = LLM_X` operating on the same source code produces the identical docstring. Across the 18 cells × 5 unique LLMs at L_spec, we compute each unique (LLM, function) pair once.
2. **Path overlap:** Path 1 (C→D→T) and Path 3 (C→T→D, via the L_test-first variant) share the L_spec call on the same code C.
3. **Re-entries during recovery:** if the sweep is interrupted (Colab session timeout, transient API failure), the cache survives and the sweep resumes from the last completed call rather than restarting.

Realistic cache hit rate measured against Chapter 2's sweep traces: **~28–35 %** of LLM calls become cache hits. Net effect: the naive 18 cells × 150 functions × 3 paths × 2 calls = 16{,}200 LLM calls drops to roughly 10{,}500–11{,}500 actual invocations. This is the source of the 110–130 GPU-hour estimate (vs the ~190 GPU-hours an un-cached run would consume).

---

## 5. Defenses against reviewer blind spots

| Blind spot | Defense |
|---|---|
| **Combinatorial explosion** (5³ = 125 cells is intractable) | Pre-registered fractional-factorial DOE in §4.4 above; 18 cells justified by per-cell hypothesis; computational budget ≈ 110–130 GPU-hours on the 150-function sample with caching (§4.7) |
| **False closure** (error propagation can mimic closure) | Four-layer defense: (a) original-artifact anchoring — closure measured against original code's reference tests, not pipeline-internal tests; (b) triple-path agreement — strong / weak / no closure classification; (c) **single-frontier judge LLM** (Claude Sonnet 4.6) external to the pipeline validates equivalence — a single frontier model avoids the alignment-tax and sycophancy issues that plague small-model judge ensembles, and is the de-facto standard for LLM-as-judge studies that reviewers will not question; (d) human study oversamples closure-passing-but-judge-flagged cases |
| **Data contamination** (HumanEval/MBPP in training data) | Three layers: (a) held-out LiveCodeBench post-2024-12-01 subset; (b) HumanEval-Mutated decontamination transformation (rename + paraphrase + permute); (c) per-LLM training-cutoff disclosure + correlation analysis (contaminated closure − decontaminated closure) |

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
| 1–2 | Engineering: extend `autoresearch/` with 3-stage pipeline + closure-metric module + Claude Sonnet 4.6 judge-LLM integration + decontamination transform |
| 3 | Pilot: 30 functions × 3 mono configurations + 3 hetero configurations; tune closure-validity thresholds. **Pre-register human-evaluation rubric and sampling strategy now**, before the sweep begins. |
| 4 | Full sweep: 18 configurations × 150 functions + 25 LiveCodeBench + 50 HumanEval-Mutated ≈ 110–130 GPU-hours on A100 (single month of Colab Pro+) |
| 5 | Buffer / spill-over month for sweep re-runs, failed-API retries, and additional held-out experiments |
| 6 | Statistical analysis (mixed-effects, ANOVA, per-stage decomposition); preliminary write-up of §4 Results |
| 7–8 | Human-validation study (60 pairs, 3 annotators, calibration session) — **2-month allocation** because rater-recruitment, calibration, and Krippendorff's-α convergence are historically slow phases |
| 9–10 | Manuscript drafting and revision |
| 11 | Submission to EMSE (primary) or TSE (secondary) |

Total: 11 months from advisor approval to first submission. The two-month buffer on human evaluation is deliberate — Chapter 2's annotation phase taught us that rater coordination is the slowest part of an empirical SE study, and the SQJ reviewer feedback explicitly flagged calibration as a quality bottleneck. We absorb that lesson directly into the plan.

Assumes SQJ reviewer feedback on Chapter 2 has been incorporated by month 4; if a major-revision request arrives during months 4–8, the Chapter 3 sweep can pause and resume without losing intermediate state thanks to the per-configuration checkpoint structure inherited from Chapter 2's `mutation_testing.py`.

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

## 10. Decisions log (post-consultation, 2026-06-02)

The four open questions from the original draft have been resolved, plus one further decision on sample size:

| # | Decision | Rationale |
|---|---|---|
| Q1 | **Closed-weight references included** — cells M5, H2, H8 stay. If API budget is tight, closed-weight cells run on a reduced 100-function sample (see §4.3). | Without a frontier-model ceiling, reviewers may dismiss findings as applicable only to weaker open-weight models. The closed-weight references function as the necessary upper bound. |
| Q2 | **Single-frontier judge LLM (Claude Sonnet 4.6)** for closure-validity validation — no ensemble of small judges. | Small-model judges suffer from alignment tax and sycophancy. Single frontier model is the de-facto LLM-as-judge standard and will draw less reviewer scrutiny. |
| Q3 | **Human validation runs AFTER the full statistical sweep**, but rubric and sampling strategy are pre-registered during the pilot (month 3) BEFORE the sweep begins. | After-sweep timing enables strategic sampling of frontier cases (closure-passes-but-mismatched-config, judge-LLM-disagreement). Pre-registration of rubric prevents p-hacking accusations. The study becomes a focused investigation into the automated metric's blind spots rather than a generic baseline. |
| Q4 | **Timeline revised to 11 months** (was 9) with a 2-month allocation for the human-validation phase. | Rater recruitment, calibration session, and Krippendorff's-α convergence are historically the slowest phases of an SE empirical study. Chapter 2 reviewer feedback explicitly flagged calibration as a quality bottleneck; absorbing that lesson directly into the plan. |
| Q5 | **Per-cell sample reduced from 300 → 150 functions** in the core sweep; held-out datasets reduced 50 % proportionally (LiveCodeBench 50 → 25, HumanEval-Mutated 100 → 50); closed-weight cells run on 50 functions. | The original 250-hour GPU budget was infeasible on a single researcher's Colab Pro+ A100 monthly allowance. The 150-function sample preserves > 80 % statistical power for the mixed-effects detection target (5 pp closure-rate difference at α = 0.05) and remains comfortably above the n ≥ 30 per cell ANOVA rule of thumb. Combined with the §4.7 caching layer, total GPU-hours drop to 110–130 — fits one Colab Pro+ month with buffer. |

---

*End of concept note. Ready for advisor sign-off. Source data, infrastructure, and statistical methodology are direct extensions of Chapter 2 (SQJ submission, 2026-06-02).*
