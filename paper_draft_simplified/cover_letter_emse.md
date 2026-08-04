# Cover letter — EMSE submission

**To:** The Editors, *Empirical Software Engineering*
**From:** Balaji Venktesh V., Amsaprabhaa M., Gireesh Sundaram
Department of Computer Science and Engineering, Shiv Nadar University, Chennai

**Subject:** Manuscript submission — "Heterogeneous Multi-SLM Closure of the
Docstring–Test–Code Triangle: A Mutation-Testing Study"

---

Dear Editors,

We are pleased to submit our manuscript, "Heterogeneous Multi-SLM Closure of
the Docstring–Test–Code Triangle: A Mutation-Testing Study," for
consideration at *Empirical Software Engineering*.

## Contribution and fit

Practitioners deploying multi-stage Small Language Model (SLM) pipelines for
software engineering face a concrete question with no empirically grounded
answer: *should the same SLM own every stage of the pipeline, or should
different SLMs be specialised by stage?* Prior work has evaluated LLM
capability at each artefact-generation task in isolation
(docstring generation, test generation, code synthesis), and a small
literature on single-LLM round-trip consistency exists, but no published
study has measured what happens when the docstring–test–code triangle is
traversed by different LLM families under an SE-validated defect-detection
metric. Our manuscript reports the first cross-LLM × cross-stage evaluation
of round-trip closure on the docstring–test–code triangle, using a
pre-registered 20-cell design of experiments across six open-weight SLMs
on 150 HumanEval and MBPP functions.

The empirical setup fits EMSE's scope directly: a pre-registered DOE (all
20 cells committed to source on 2026-06-03, four weeks before the first
sweep result; verifiable via the public git history of the replication
package), a large sweep (5,890 measurement rows), a completed 60-pair
human-evaluation study with five annotators (three from the author team and
two independent, all attending the shared calibration session), a
frontier-judge replay against three additional LLM judges via OpenRouter,
and a HumanEval-Mutated contamination sensitivity subsweep on the strongest
hetero and mono cells.

## Five findings

The paper reports five findings, ordered by strength of evidence:

1. **Pre-registered composition validated at operator resolution.** The H1
   composition (phi-4 spec + qwen-coder test/code) dominates four of five
   mutation operators as pre-registered, with the largest gain (arithmetic,
   Δ = +0.133) matching a companion mutation-testing study's per-operator
   capability prediction.

2. **A phi-4-in-test-stage pathology invisible in single-stage evaluations.**
   phi-4 in the test-generation stage produces filter-failing tests on
   approximately half of samples across three independent cells (M2, H2,
   H10). The pathology localises to the exact mutation-operator families
   phi-4 mono itself is competitive on — evidence that per-model capability
   on single-artefact tasks does not directly predict per-model capability
   in a downstream stage.

3. **BERTScore is functioning as a surface-form similarity signal on the
   docstring–docstring path.** Path 3 shows near-zero pooled correlation
   (r = 0.02, n = 1,791) between BERTScore F1 and the judge SLM's
   semantic-equivalence rating; per-benchmark correlations are also weak
   (r_HE = 0.15, r_MBPP = 0.21); and the disagreement direction flips
   categorically across benchmarks (χ² = 368, p < 0.0001) — a structural
   signature that a semantic-equivalence signal should not produce.

4. **Closure-path rankings are unstable across paths.** Best cell is
   different for every path (H4/H3/H9); top-3 across all three paths is
   empty; Kendall's τ_rank between path rankings ranges 0.09–0.43. A
   single-path benchmark cannot select a composition for a different path.

5. **Well-composed heterogeneous compositions preserve 78–91% of the
   strongest mono baseline's capability**, but poorly-composed compositions
   collapse (H10 preserves 35% of Path 1 reference capability; H4
   preserves 57% of Path 2). Heterogeneity is not free; the composition
   must match the demanded per-stage strength.

## Empirical rigour and honest limitations

The manuscript is written to be honest about what the experimental data
support and what it does not:

- Section 5.10.3 reports the strict-AND two-signal validity policy against
  the trivial majority-class baseline (86.7% accuracy on the 60-pair
  sample) alongside MCC, balanced accuracy, and cost-sensitive loss. We
  argue for strict-AND on adversarial-robustness and cost-sensitive
  grounds, not on raw-accuracy dominance.
- Section 5.10.4 explicitly scopes the FPR/FNR numbers to the metric-judge
  disagreement region of the stratified 60-pair sample, and identifies a
  uniformly-drawn 60-pair validation as immediate future work.
- A frontier-judge replay (GPT-4o-mini, Claude Haiku 4.5, Llama-3.3-70B)
  shows that weak judge–human agreement (κ_quad ≤ 0.20 against the
  majority human) is a general LLM-as-judge property, not
  DeepSeek-R1-specific. This confirms rather than undermines the paper's
  strict-AND two-signal recommendation.

## Replication package

A complete replication package is publicly available at
[https://github.com/balajivenky06/roundtrip-closure](https://github.com/balajivenky06/roundtrip-closure)
under the MIT licence. It contains the 5,890-row sweep TSV, the
pre-registered DOE (`doe.py`), the closure validity decision as a
tested Python function (`closure_decision.decide_validity` with 43 unit
tests), the test-filter validity gate (8 unit tests), the frontier-judge
replay script, and all 11 LaTeX tables and 10 PNG figures reproducible
from the sweep TSV via a single command.

## Anonymised submission

Author information is stripped from the manuscript per EMSE double-blind
review policy. The submission ZIP contains: `main.pdf` (anonymised),
`main.tex`, all section files, `references.bib`, `figures/*.tex`, and
`tables/*.tex`.

## Suggested reviewers

We list five suggested reviewers below. All have published on empirical
LLM-for-SE evaluation, none has a conflict of interest with the author
team (no co-authorship, no supervisor relationship), and none has
previously reviewed prior drafts of this work.

1. [Reviewer Name] — Affiliation — Expertise fit
2. [Reviewer Name] — Affiliation — Expertise fit
3. [Reviewer Name] — Affiliation — Expertise fit
4. [Reviewer Name] — Affiliation — Expertise fit
5. [Reviewer Name] — Affiliation — Expertise fit

*(This block will be filled in by the corresponding author before
submission; see accompanying draft reviewer-suggestion list.)*

## Non-preferred reviewers

None.

## Conflicts and competing submissions

No conflicts. This manuscript has not been submitted to any other venue.
It is the third of a three-paper thesis on Small Language Models for
software engineering; the first two papers are cited (via companion-paper
references) as under review at Automated Software Engineering
(Springer-Nature) and Information Processing & Management respectively.
Both remain distinct in scope: the first paper covers retrieval-augmented
docstring generation (single-artefact), the second covers unit-test
generation with mutation-testing evaluation (single-artefact), and the
present manuscript is the first cross-stage study of the three papers.

## Contact

**Corresponding author:** Balaji Venktesh V.
Department of Computer Science and Engineering
Shiv Nadar University, Chennai
Tamil Nadu, India
Email: balaji23610040@snuchennai.edu.in

Thank you for your consideration.

Sincerely,

Balaji Venktesh V.
Amsaprabhaa M.
Gireesh Sundaram
