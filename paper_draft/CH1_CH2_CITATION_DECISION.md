# Chapter 1 / Chapter 2 companion-paper citation decision

## The issue

The current manuscript references "the first companion paper" and "the second
companion paper" 15 times across §1, §2, §3, §4, §5, §6, §7 — but there is
no bib entry for either. Reviewers cannot verify claims that depend on
these references (notably: the per-operator capability map that H1's
pre-registration is based on).

## Options

### A. Cite as under-review preprints (recommended if arXiv upload is possible)

Upload both papers to arXiv (or SSRN or another preprint server). Then cite
with a `misc` bib entry using the preprint DOI/arXiv ID:

```bibtex
@misc{VenkteshChapter1,
  author = {Venktesh V., Balaji and Amsaprabhaa M. and Sundaram, Gireesh},
  title = {Retrieval-Augmented Docstring Generation with Small Language Models},
  year = {2026},
  eprint = {arXiv:2607.XXXXX},
  archivePrefix = {arXiv},
  note = {Under review at Automated Software Engineering (Springer-Nature)},
}

@misc{VenkteshChapter2,
  author = {Venktesh V., Balaji and Amsaprabhaa M. and Sundaram, Gireesh},
  title = {Cross-LLM $\times$ Cross-Method Evaluation of Unit-Test Generation with Mutation Testing},
  year = {2026},
  eprint = {arXiv:2607.YYYYY},
  archivePrefix = {arXiv},
  note = {Under review at Information Processing \& Management},
}
```

Then replace "the first companion paper" → `\citet{VenkteshChapter1}` and
"the second companion paper" → `\citet{VenkteshChapter2}` throughout.

**Effort:** ~2h upload + arXiv moderation delay (1-3 days) + 30 min bib
updates + 30 min sed replacements across 15 mentions in 7 files.

**Reviewer benefit:** high — every reviewer will follow the arXiv link.

### B. Cite as under-review without preprint (weaker, faster)

Use a `misc` bib entry citing the target venue but no accessible URL:

```bibtex
@misc{VenkteshChapter2,
  author = {Venktesh V., Balaji and Amsaprabhaa M. and Sundaram, Gireesh},
  title = {Cross-LLM $\times$ Cross-Method Evaluation of Unit-Test Generation with Mutation Testing},
  year = {2026},
  note = {Manuscript under review at Information Processing \& Management, 2026},
}
```

**Effort:** ~30 min bib + 30 min sed replacements.

**Reviewer benefit:** low — reviewers cannot verify the per-operator
capability map. May trigger "please make this citable" in the review.

### C. Drop the companion-paper references entirely (weakest, fastest)

Remove all "companion paper" mentions. For §1 pre-registration
contribution, replace "based on the second companion paper's per-operator
capability map" with "based on prior per-operator capability analysis
documented in `doe.py` commit f07b4a6." For §2 related-work sections,
replace the "series" positioning with a single-paragraph "this paper is
the third in a series..." footnote.

**Effort:** ~1h careful edits.

**Reviewer benefit:** negative — the pre-registered-based-on-external-map
provenance vanishes, weakening the H1 pre-registration story.

## Recommendation

**Option A is worth the delay.** The pre-registration provenance for H1 is
central to Finding #1 (headline finding). Reviewers who cannot verify that
H1's composition was chosen from an independent per-operator capability
analysis (rather than post-hoc reasoning to fit the observed results) will
downgrade the pre-registration claim. arXiv upload is free and cheap
insurance.

**If A is not feasible on the submission timeline**, Option B is acceptable
but sub-optimal. Option C should be avoided unless you're explicitly
retracting the "pre-registered per-operator prediction" language from
§1.

## Concrete next actions if Option A

1. Prepare arXiv PDFs for Chapter 1 and Chapter 2. If they're already at
   PDF stage for their target venues, this is essentially zero-effort
   (just upload the anonymised submission PDF with author info restored).
2. Upload to arXiv under `cs.SE` (Software Engineering) category.
3. Wait 1-3 business days for arXiv moderation.
4. Update `references.bib` with `VenkteshChapter1` and `VenkteshChapter2`
   entries as shown above.
5. Sed-replace the 15 "companion paper" mentions with `\citet{}`.

## Where each reference lives

For your convenience when doing the sed replacements:

| File | Occurrences | Type |
|---|---|---|
| section_1_introduction.tex | 2 | contributions + methodology |
| section_2_related_work.tex | 4 | positioning within series |
| section_3_framework.tex | 1 | mutation-operator protocol |
| section_4_experiment_setup.tex | 1 | DOE oracle |
| section_5_results.tex | 1 | H1 per-operator validation |
| section_6_discussion.tex | 4 | RQ1 + RQ3 mechanism + G1 |
| section_7_conclusion.tex | 2 | position in series |

The bulk of the mentions are in §2 and §6. §1's mention is the most
important for reviewer verifiability.
