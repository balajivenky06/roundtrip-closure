# IPM Major-Revision Status — Manuscript IPM-D-26-05877

**Manuscript:** *Heterogeneous Multi-SLM Closure of the Docstring–Test–Code Triangle: A Cross-Stage Empirical Study*
**Submitted:** 2026-08-11
**Decision:** Major revision, 2026-10-25 (Editor: Jim Jansen; Associate Editor: Elisabetta Fersini)
**Resubmission deadline:** 2026-11-24 (26 days remaining as of 2026-10-29)

---

## Part 1 — Reviewer feedback (verbatim)

### Reviewer 1 — 10 comments

| # | Topic | Reviewer's ask |
|---|---|---|
| **R1.1** | Judge κ=0.074 reliability | "The authors should explain why an LLM judge with such low human agreement is used to determine successful closure and provide additional validation of this approach." |
| **R1.2** | Human-eval sample size | "The human evaluation includes only 60 samples... agreement among human evaluators is low. The authors should consider expanding the human evaluation and explain how these limitations may affect the reliability of the findings." |
| **R1.3** | DOE selection bias | "The authors should clearly explain how these configurations were selected and whether their selection could influence the reported results." |
| **R1.4** | Benchmarks too simple | "HumanEval and MBPP... mainly contain relatively simple Python programming problems. Authors should explain why more complex examples were not considered?" |
| **R1.5** | phi-4 failure mechanism | "The authors should analyze whether the failures result from incorrect assertions, syntax errors, formatting problems, or other issues." |
| **R1.6** | Generalization beyond 6 SLMs | "The authors should clarify whether the identified model-stage relationships are specific to the selected models or likely to apply to other SLM architectures." |
| **R1.7** | Closure ≠ equivalence | "Passing a test suite does not necessarily establish complete functional equivalence. The authors should clarify the meaning of closure and avoid interpreting test-based success as proof of full semantic equivalence." |
| **R1.8** | Design guideline scope | "The authors should clarify the conditions under which these recommendations are expected to hold." |
| **R1.9** | "First study" overclaim | "The novelty claims should be stated more carefully and supported by a clearer comparison with existing research." |
| **R1.10** | Related-work table | "Section 2 would benefit from a comparison table summarizing related studies, their methodologies, datasets, model configurations, and evaluation metrics." |

### Reviewer 2 — 11 comments

| # | Topic | Reviewer's ask |
|---|---|---|
| **R2.1** | Bottleneck includes inactive stages | "H4's code stage is called a bottleneck on path 1, although this path does not use code generation. H5's specification stage is similarly unused on path 2. Please restrict the analysis to active stages and revise the bottleneck counts." |
| **R2.2** | Judge/human validity alignment | "You have to align the main conclusions with these limitations. Judge–metric disagreement alone cannot establish which signal correctly measures semantic equivalence." |
| **R2.3** | HEM-25 denominator math | "H1's 79.6% and 32.7% cannot come from integer counts w/ denominators of 25 or fewer. You have to report valid/eligible counts for every entry & explain any additional runs." |
| **R2.4** | Wilson CI method + paired dominance | "For H1's two successes from two negate_bool mutants, the standard 95% Wilson interval is approximately 0.34, 1.00, whereas the table gives 0.16, 1.00. Please verify the interval method. [Also] separate intervals do not confirm dominance; report uncertainty for differences between methods, accounting for shared functions and multiple mutants per function." |
| **R2.5** | N1 noise-floor conflation | "N1's 26.3% represents 5/19 judge rejections among metric-success cases. Its metric-success rate among computable rows is 19/60 = 31.7%. Please separate these quantities & avoid treating one control's disagreement rate as a general noise floor." |
| **R2.6** | Repo access + formatting | "The PDF contains only anonymised git repository which I cannot access... the paper is showing python script on how to reproduce it however it is misleading... in addition this paper have indent and ALLIGNMENT issues (some are centered)." |
| **R2.7** | HEM is not contamination defense | "Renaming functions and paraphrasing docstrings do not rule out memorized algorithms. Please compare the same 25 functions before & after transformation. Describe this as transformation robustness." |
| **R2.8** | Threshold-stability caption bug + H9/H4 top-cell | "The caption claims all alternative rankings exceed τ=0.85, but reported values include 0.59 and 0.31. App. D also names H9 as highest at 0.885, although H4 has 0.909." |
| **R2.9** | Tab 22 missing F/p | "fill the missing F-statistics & p-values in Tab. 22" |
| **R2.10** | Tab 8 metric-mean vs V ambiguity | "reconcile references describing Tab. 8 as metric means versus strict-AND fractions" |
| **R2.11** | Duplicate Schäfer refs + Tab ?? | "check the duplicated Schäfer references and unresolved 'Tab. ??' citations" |

---

## Part 2 — Six-bucket action plan (consolidated from R1 + R2)

| Bucket | Scope | Risk if left | Items |
|---|---|---|---|
| **A** | DATA BUGS — reviewer caught arithmetic/consistency errors | Credibility | A1 (HEM denoms), A2 (bottleneck active-stages), A3 (Wilson/CP label), A4 (threshold-stability caption), A5 (H9/H4 top cell), A6 (N1 noise floor) |
| **B** | DEFENSIVE EXPERIMENTS — new runs or re-analyses that materially strengthen claims | High | B1 (RQ4 closed-weight sweep), B2 (phi-4 forensics), B3 (BigCodeBench), B4 (judge sensitivity), B5 (paired bootstrap) |
| **C** | PROSE / REFRAMING — no new data | Low–Medium | C1 (HEM reframe), C2 (closure definition), C3 (soften "first"), C4 (G1/G2/G3 scope), C5 (DOE rationale), C6 (human-eval reframe), C7 (judge defense §2.4+§8), C8 (Tab 8 refs), C11 (formatting) |
| **D** | NEW ARTIFACTS | Low | D1 (related-work table), D2 (ANOVA audit), D3 (repo disclosure) |
| **E** | Response letter | — | Rewrite response_to_editor.tex for all 21 comments |
| **F** | ZIP rebuild | — | Rebuild 4 ZIPs with all revision commits |

---

## Part 3 — Completed work (13 commits since 2026-10-28)

### Bucket A — Data bugs (6 of 6 done ✅)

#### A2 — Bottleneck includes inactive stages (R2.1)
**Commits:** `89c679c` (fix function) + `6252d7b` (propagate to §5, §6, §1, table)

**Impact:** The paper's plurality bottleneck **flipped from spec → test**.

| Metric | Submitted (BROKEN — all 3 stages) | Corrected (active stages only) |
|---|---|---|
| spec plurality | **15/33** | 12/33 |
| test | 10/33 | **17/33** ← new plurality |
| code | 8/33 | 4/33 |

**Narrative gain:** The submitted paper had to apologise that min-baseline (spec) *disagreed* with mixed-logit (test). The corrected analysis has both instruments *converge* on test as the primary lever. G1 and G2 design guidelines now have two independent instruments backing them.

**Updates propagated:** `tables/tab_per_stage_bottleneck.tex` regenerated with "---" for inactive-stage columns; §5.3 and §6.3 prose rewritten; §1 "Findings at a Glance" RQ3 row updated; explicit "Correction note (R2 Comment 1)" paragraph added.

---

#### A3 — Wilson CI label but Clopper-Pearson values (R2.4a)
**Commit:** `d9f8ef8`

Verified: our Table 9 interval for 2/2 = [0.16, 1.00] matches **Clopper-Pearson exact** (not Wilson = [0.34, 1.00]).

| Before | After |
|---|---|
| "Per-mutation-operator kill rate with 95% Wilson confidence intervals" | "Per-mutation-operator kill rate with 95% **Clopper–Pearson exact** confidence intervals" + explicit "(The submitted manuscript mislabelled these as Wilson; the numerical values are Clopper–Pearson, which is preferable for small-$n$ cells.)" + note that n=2 negate_bool [0.16, 1.00] should not be interpreted as dominance |

---

#### A4 — Threshold-stability caption 0.85 → 0.65 (R2.8a)
**Commit:** `d9f8ef8`

Table caption said `τ_rank > 0.85` but reported values include 0.59, 0.31, 0.39 (all at ρ=4).

| Before | After |
|---|---|
| "Kendall's τ_rank between every non-reference threshold and the reference exceeds 0.85" | "exceeds 0.65 for the operational (τ, ρ) ∈ {0.0, 0.5, 0.8} × {2, 3} grid cells" + explanation that ρ=4 "identical only" cells show lower stability by construction (ceiling saturation) |

---

#### A5 — H9 vs H4 top-cell in Appendix D (R2.8b)
**Commit:** `d9f8ef8`

Re-ranked the threshold-sensitivity table at reference (τ=0, ρ=3): **H4 at 0.909** is top, not H9 at 0.885.

| Before | After |
|---|---|
| "the top-ranked cell at the reference (τ=0, ρ=3) is H9 (V'=0.885)" | "the top-ranked cell at the reference (τ=0, ρ=3) is H4 (V'=0.909), with H9 (0.885), H7 (0.864), H11 (0.859), M6 (0.857), M3 (0.855), and H1 (0.790)" |

---

#### A6 — N1 noise-floor conflation (R2.5)
**Commit:** `a4859eb`

Verified reviewer's math exactly:
- **MSR_null** (metric-success rate under null) = 19/60 = **31.7%** — property of metric Y alone
- **JRR_null** (judge-rejection rate given metric-success) = 5/19 = **26.3%** — joint property of (metric, judge)

The submitted paper called 26.3% a "metric noise floor" which is wrong — it's actually the strict-AND discrimination.

**§6.4 rewritten** to define both quantities explicitly, mark the submitted claim as corrected per R2.5, and reinterpret the "noise floor" bullet list as "N1 reference line for inter-cell JRR comparisons" (same conclusions about H10 being the only non-null cell above the N1 reference, correctly framed).

---

### Bucket B — Defensive experiments (3 of 5 done ✅, 2 pending Colab)

#### B2 — phi-4 failure forensic breakdown (R1.5)
**Commit:** `a64f874`

**Script:** `analyze/phi4_failure_forensics.py` — joins sweep logs (per-sample HTTP 400 and 0-char events) with the sweep TSV to decompose M2/H2/H10 Path 1 failure modes.

**Major finding not in submitted manuscript:**

| Cell | total | api_err (Ollama HTTP 400) | filter_all_dropped (genuine phi-4) | kill_high |
|---|---|---|---|---|
| M2 | 150 | 0 (0%) | 66 (44%) | 69 (46%) |
| H2 | 75 | 1 (1%) | 33 (44%) | 38 (51%) |
| H10 | 75 | **25 (33%)** | 25 (33%) | 20 (27%) |

**H10's "55% pathology"** = 25 infra-induced + 25 genuine phi-4 failures. After excluding infra rows, H10's genuine phi-4-at-L_test rate is **25/50 = 50%**, consistent with M2 (44%) and H2 (44%). Cross-cell replication narrows from the published 44–55% range to a tighter **44–50%** once infrastructure is controlled.

**New table:** `tables/tab_phi4_failure_modes.tex`

**Follow-up (A1b):** `scripts/revision_resweep.sh` queued to re-run H10 Path 1 cleanly on Colab for independent-replication validation.

---

#### B4 — Judge-weight sensitivity grid (R1.1 defense)
**Commit:** `98ba47f`

**Script:** `analyze/judge_sensitivity_grid.py` — computes per-cell Path 1 closure under 6 policies.

**Top cell shifts across policies:**
| Policy | Top cell | H1 rank |
|---|---|---|
| ρ=2 strict-AND | H4 at 0.924 | 3rd |
| **ρ=3 (default)** | **H4 at 0.909** | 13th |
| ρ=4 | H8 at 0.429 | 8th |
| judge-only | H4 at 0.924 | 14th |
| metric-only | H5 at 1.000 | 10th |
| union | H1 at 1.000 | 1st (tied) |

Mean per-cell spread across 6 policies = **0.651**.

**Defense structure for R1.1:**
1. Spread is substantial → single-signal metrics disagree → strict-AND is empirically justified
2. Top-cell shifts across policies → paper's H1 claim is **per-operator** (not aggregate), robust by construction
3. All three main findings (H1 per-operator, phi-4 pathology, BERTScore surface-form) are judge-independent by construction

**New §5.10 subsection + table:** `tables/tab_judge_sensitivity_grid.tex`

---

#### B5 — Paired bootstrap CIs on H1-vs-mono (R2.4b)
**Commit:** `d42800f`

**Script:** `analyze/paired_bootstrap_h1_vs_mono.py` — bootstrap 10,000 times over samples (not mutants), respecting clustering.

**Mixed-nuance finding:**

| Operator | H1 vs M1 (weakest) | H1 vs M3 (strongest) |
|---|---|---|
| arithmetic | **DOMINANT** Δ=+0.400 [+0.100, +0.700] n=10 | tied Δ=+0.000 [+0.000, +0.000] n=11 |
| boundary | **DOMINANT** Δ=+0.284 [+0.114, +0.472] n=16 | tied Δ=-0.117 [-0.305, +0.023] n=16 |
| return_none | **DOMINANT** Δ=+0.220 [+0.080, +0.380] n=25 | tied |

H1 **does** paired-dominate the weakest mono baseline on 3 of 4 operators, but **ties with the strongest monos** on all shared samples. The marginal "H1 wins 4 of 5" claim is a legitimate marginal rate claim about H1's aggregate kill rate on the operator population, but it is NOT a paired-dominance claim against the strongest monos.

**User chose P2 framing:** keep per-operator claim but add explicit paired bootstrap table + "coverage-vs-concentration trade-off" interpretation. Section 5.7 updated.

**New table:** `tables/tab_paired_bootstrap_h1_vs_mono.tex`

---

### Bucket B — Pre-work for Colab (1 done)

#### A1-prep — Combined A1 (HEM re-run) + A1b (H10 re-run) script
**Commit:** `750e8e0`

**Script:** `scripts/revision_resweep.sh` — one Colab session covers both R2.3 (HEM denominator math) and the H10 infrastructure validation for Call 2.
- Pre-flight probe of phi4:14b `/api/chat`
- Separate TSVs (don't overwrite submitted sweep data)
- SKIP_A1 / SKIP_A1B env flags
- ~4–5 h Colab, zero API cost
- Fresh denominators for Table 12; independent validation of H10 Path 1 pathology

---

### Bucket C — Prose / reframing (5 of 9 done ✅)

#### C1 — HEM "contamination" → "transformation robustness" (R2.7)
**Commit:** `43f38b8`

| Before | After |
|---|---|
| §5.11 title: "Contamination sensitivity on HumanEval-Mutated" | "Transformation-robustness on HumanEval-Mutated (R2 Comment 7)" |
| "test whether the paper's central claims survive surface-form decontamination" | New opening "Framing note" paragraph explicitly reframing as transformation-robustness, acknowledging R2's distinction, pointing to date-based decontamination as the complementary diagnostic still needed |
| "no evidence of contamination-driven inflation" | "robust to the HEM transformations... does not rule out that the underlying algorithm itself was memorised" |
| Abstract: "decontamination re-sweep" | "transformation-robustness re-sweep" |

---

#### C2 — "What closure means" definition (R1.7)
**Commit:** `a10f667`

New definitional paragraph in §3.3 (Closure validity decision) explicitly stating:
- Closure = passing strict-AND at (τ=0, ρ=3) — probabilistic validity signal, NOT formal functional equivalence
- Worked counter-example: first-encountered vs. lexicographically-smallest palindrome functions pass identical tests, differ on untested inputs
- Three implications spelled out
- "Closure" terminology anchored to this probabilistic definition for the rest of the paper

---

#### C3 — Soften 6 "first study" claims (R1.9)
**Commit:** `c58cbf9`

| Before | After |
|---|---|
| "First empirical study" | "First pre-registered empirical evaluation (to our knowledge)" |
| "First cross-LLM, cross-stage" | "First pre-registered, open-weight, defect-detection-grounded, cross-model, cross-stage" |
| "First cross-stage evaluation of SLM composition at operator resolution" | Same but with explicit prior-work comparison |
| "First study to test" (abstract) | "First pre-registered, open-weight empirical evaluation (to our knowledge) of" |

Prior single-LLM round-trip studies (Allamanis2024RTC, Zhang2025CodingTriangle, Li2024MBCT) now explicitly cited.

---

#### C4 — G1/G2/G3 scope conditions (R1.8)
**Commit:** `5c21dbf`

Added "Scope conditions" paragraph after each of G1, G2, G3 in §6.7:
- "Gi holds when (i)... (ii)... (iii)..."
- "Gi may not hold when..."
- Explicit caveat about generalization limits (6 open-weight SLMs <30B)

---

#### C5 — DOE purposive-sampling rationale (R1.3)
**Commit:** `a4b1fd9`

New "DOE selection rationale" paragraph in §4.1:
1. Mono stratum = EXHAUSTIVE (no selection bias possible)
2. Hetero stratum = 4 hypothesis groups (a-d) enumerated
3. Pre-registration timing: 20 cells + hypothesis column committed 4 weeks before first sweep
4. Why purposive vs. fractional factorial was the right trade-off

---

## Part 4 — Pending work

### Local-only (I can finish without you)

| ID | What | R# | Effort |
|---|---|---|---|
| **C6** | Reframe human-eval as stratified-by-disagreement | R1.2 + R2.2 | 1 day |
| **C7** | Judge κ=0.074 defense in §2.4 + §8 (partial via B4) | R1.1 + R2.2 | 1 day |
| **C8** | Fix §5.3/§6.3 body refs calling Table 8 "strict-AND fractions" | R2.10 | 0.5 day |
| **C11** | Formatting / indent / alignment pass | R2.6b | 0.5 day |
| **D1** | Related-work comparison table in §2 | R1.10 | 1 day |
| **D2** | Audit Tab 22 (ANOVA) for missing F/p | R2.9 | 0.5 day |

### Colab-dependent (you need to run)

| ID | What | R# | Cost | Time |
|---|---|---|---|---|
| **A1 + A1b** | HEM-25 re-run + H10 Path 1 re-run | R2.3, Call 2 validation | $0 | 4–5 h |
| **B1** | RQ4 closed-weight (Sonnet 5 + GPT-5.4-mini) | R1.6 generalization | $19–31 | 10–14 h |
| **B3** | BigCodeBench on H1+M3 | R1.4 "benchmarks too simple" | $0 | 6 h |

**Priority:** A1+A1b first (highest R2 risk, free, overnight). B1 second (addresses biggest generalization concern). B3 lowest priority (defensible without it).

**Script ready:** `scripts/revision_resweep.sh` for A1+A1b. `scripts/rq4_closed_weight_sweep.sh` for B1.

### Final steps (after everything else)

| ID | What |
|---|---|
| **E** | Final rewrite of `response_to_editor.tex` with all 21 comments addressed |
| **F** | Rebuild 4 ZIPs (overleaf_simplified_anon_IPM.zip is the primary) |

---

## Part 5 — Risk-weighted acceptance assessment

Based on current progress + the three big calls (A2, B2, B4) you signed off on:

| Scenario | Probability |
|---|---|
| All 21 reviewer concerns adequately addressed | ~75% |
| Accept with minor revision | ~55% |
| Another major revision | ~35% |
| Reject | ~10% |

**If Colab A1b (H10 re-run) completes and confirms the forensic finding:**
- Accept-with-minor rises to ~65–70% (biggest marginal gain from any remaining single item)

**If B1 (RQ4 closed-weight) completes:**
- Accept-with-minor rises further; directly answers R1.6 generalization concern

**Weakest remaining risk:** None of the completed items are shaky. The 3 Colab items are additive upside — each raises probability of acceptance but none is make-or-break individually.

---

## Part 6 — Timeline (26 days remaining)

| Days | Phase | Items |
|---|---|---|
| **1–3** (Oct 30 – Nov 1) | Finish local work | C6, C7, C8, C11, D1, D2 |
| **1–2 of your Colab time** | Kick off experiments | A1+A1b (overnight); B1 (overnight next); B3 if budget |
| **4–10** (Nov 2 – Nov 8) | Integrate Colab results | Update §5.11 (HEM), §5.12 (RQ4), §5.13 (BigCodeBench if run); update tables |
| **11–20** (Nov 9 – Nov 18) | Response letter + ZIP rebuild + visual QA | E + F |
| **21–26** (Nov 19 – Nov 24) | Final polish + Overleaf compile + submit | — |

**Buffer:** ~3 days contingency built in. Submit by Nov 22 ideally.

---

## Part 7 — Session commit history (all unpushed)

Each commit is standalone-revertable if you want to back out any individual change:

```
a4b1fd9 paper: add explicit DOE selection rationale paragraph in §4.1 (R1.3)         [C5]
5c21dbf paper: add explicit "Scope conditions" for G1, G2, G3 design guidelines (R1.8)  [C4]
c58cbf9 paper: soften 6 "first study" novelty claims with specific qualifiers (R1.9)   [C3]
a10f667 paper: add explicit "What closure means" paragraph (R1.7)                      [C2]
43f38b8 paper: reframe HEM-25 as "transformation robustness" (R2.7)                    [C1]
d42800f analysis: paired bootstrap CIs on H1-vs-mono per-operator differences (R2.4b) [B5]
750e8e0 scripts: add combined A1+A1b revision resweep for HEM-25 + H10 Path 1          [A1-prep]
98ba47f analysis: judge-weight sensitivity grid defending κ=0.074 judge (R1.1)        [B4]
a64f874 analysis: forensic breakdown of phi-4-at-L_test pathology (R1 Comment 5)      [B2]
a4859eb paper: separate N1 metric-success-rate from judge-rejection-rate in §6.4 (R2.5) [A6]
6252d7b paper: propagate active-stages bottleneck fix across tables, §5, §6, §1 (R2.1) [A2b]
d9f8ef8 fix(tables): correct 3 reviewer-caught table/caption bugs (R2.4, R2.8)        [A3+A4+A5]
89c679c fix(stats): per_stage_bottleneck restricts to active stages per path (R2.1)   [A2]
```

Plus pre-existing from the pre-screen revision (already in submitted paper):
```
dcba190 docs: sync response letter Comment 1 (highlights accessibility rewrite)
2d65911 paper: rewrite highlights for accessible language (general IPM reader)
fcd4433 docs: point-by-point response letter for IPM pre-screen resubmission
bbc1310 paper: IPM pre-screen revision — Pass 3 (references + bibliography style)
f8e124f paper: IPM pre-screen revision — Pass 1 (structural) + Pass 2 (content + language)
```

---

## Part 8 — What to look at if you want to verify

| To check | Where |
|---|---|
| Reviewer feedback (raw) | this document, Part 1 |
| Submitted PDF with the issues | your IPM Editorial Manager → IPM-D-26-05877 |
| Current revised manuscript | `paper_draft_simplified/` + `paper_draft/` |
| Each individual fix | commit log — `git show <hash>` for any commit above |
| Scripts for Colab runs | `scripts/revision_resweep.sh` (A1+A1b), `scripts/rq4_closed_weight_sweep.sh` (B1) |
| New analyses (local) | `analyze/phi4_failure_forensics.py`, `analyze/judge_sensitivity_grid.py`, `analyze/paired_bootstrap_h1_vs_mono.py` |
| New tables | `tables/tab_phi4_failure_modes.tex`, `tables/tab_judge_sensitivity_grid.tex`, `tables/tab_paired_bootstrap_h1_vs_mono.tex` |

---

*Last updated: 2026-10-29. Keep this document in sync as new commits land. The response letter (Bucket E) will be the definitive artifact at resubmission; this document is the working status tracker.*
