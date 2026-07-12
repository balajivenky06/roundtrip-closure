# human_eval/ — 60-pair three-annotator human evaluation

Human evaluation study for Chapter 3 of the PhD thesis on Small Language
Models for software engineering. This directory contains everything three
independent annotators need to rate 60 code-artefact pairs on a
behaviourally-anchored 5-level semantic-equivalence rubric.

The study addresses **RQ2** (does the automated closure metric agree with human
judgment of semantic equivalence?) and produces the numbers reported in the
pending human-evaluation addendum to the manuscript.

---

## Study at a glance

| Item | Value |
|---|---|
| Study population | 3 independent human annotators |
| Instrument | 60 code-artefact pairs, pre-registered stratified sample |
| Instrument stratification | 20 frontier + 20 agreement + 20 disputed (see below) |
| Rubric | 5-level behaviourally-anchored equivalence scale (0–4) |
| Time budget per annotator | ~3–4 hours: 30 min calibration + ~3 min × 60 pairs |
| Blinding | Cell composition, path, judge rating hidden from annotator |
| Order | Randomised per annotator (deterministic seed per annotator ID) |
| Target inter-rater agreement | Krippendorff's $\alpha \geq 0.5$ (ordinal metric) |
| Target judge–human agreement | Cohen's $\kappa \geq 0.4$ (fair-or-better on Landis–Koch) |
| Outputs | 3 per-annotator TSVs → agreement report → §7.3 addendum table |

---

## Directory layout

```
human_eval/
  README.md               ← this file
  rubric.md               ← 5-level rubric with anchors + micro-examples
  calibration_items.md    ← 10 items for the pre-annotation calibration session
  annotator_briefing.md   ← welcome + workflow + FAQ for annotators
  consent_form.md         ← informed-consent form matching manuscript §6

  prep/
    build_worksheet.py    ← fills original_code + reconstructed_code from cache

  app/
    streamlit_app.py      ← the annotation UI
    requirements.txt      ← streamlit + pygments deps

  analysis/
    compute_agreement.py  ← Krippendorff α + Cohen κ + judge–human report

  data/                   ← (created by build_worksheet.py)
    pairs_60_full.jsonl   ← the 60 pairs with code artefacts filled in
    annotator_1_ratings.tsv
    annotator_2_ratings.tsv
    annotator_3_ratings.tsv
    agreement_report.json
    agreement_report.tex
```

---

## Sampling strategy (pre-registered)

The 60 pairs are drawn from the 5,890-row full sweep TSV, deterministically
stratified into three buckets of 20 pairs each. The stratification is
recorded in `results/human_eval_pairs_60.tsv` (already committed to the
repository) and is fixed prior to seeing annotator ratings.

| Bucket | Criterion | Purpose |
|---|---|---|
| **frontier** (20) | Automated metric strongly asserts closure but the cell composition is deliberately mismatched (H4, H5, N1) or exhibits the phi-4-in-test pathology (H10) | Probes the automated metric's false-positive floor under adversarial cell design |
| **agreement** (20) | Mono-cell metric and hetero-cell metric agree, and judge SLM concurs | Sanity check — should produce mostly ratings of 3 or 4 |
| **disputed** (20) | Judge SLM disagrees with the automated metric (either `false_closure_candidate` or `metric_false_negative` from Algorithm 2) | Probes cases where the two automated signals diverge |

---

## Protocol

### Phase 1 — Recruitment (T-4 weeks)

- Three annotators, each with ≥ 3 years Python software-engineering experience.
- No annotator has prior familiarity with this study's cell composition or hypotheses.
- Compensation: regional software-engineering hourly rate.
- Consent form (`consent_form.md`) signed before data access.

### Phase 2 — Calibration session (T-3 weeks, ~1 hour, joint)

- All three annotators + the corresponding author meet on a video call.
- Ten calibration items (`calibration_items.md`) are rated one-by-one:
  1. Each annotator submits a rating independently.
  2. Ratings are revealed.
  3. Any divergence is discussed; rubric anchors are re-affirmed with reference to the item.
  4. Move to the next item.
- The calibration items are **not counted** in the main analysis and are recorded separately for supplementary reporting.
- Post-session, each annotator receives access credentials for the Streamlit annotation app.

### Phase 3 — Production annotation (T-3 to T-0, individual, ~3 hours per annotator)

- Each annotator rates all 60 pairs individually via the Streamlit app.
- Pair order is randomised per annotator (deterministic seed = SHA-256 hash of the annotator ID).
- Ratings are auto-saved after each submission. Sessions are resumable.
- Time-per-pair is logged; no hard time limit but a suggestion of ~3 minutes.
- No communication between annotators during production.
- Cell composition, path, judge rating, and bucket are hidden from the annotator.

### Phase 4 — Agreement analysis (T-0 to T+1 week)

- Three per-annotator TSVs are concatenated by `pair_id`.
- `analysis/compute_agreement.py` produces:
  - Krippendorff's $\alpha$ (ordinal) across the three human annotators;
  - Pairwise Cohen's $\kappa$ for each pair of annotators (binarised at rating $\geq 3$);
  - Cohen's $\kappa$ between the majority human rating and the DeepSeek-R1 judge;
  - Per-bucket agreement rates (frontier / agreement / disputed).
- Results are written to `data/agreement_report.json` and `data/agreement_report.tex`, ready for `\input` into the pending manuscript addendum.

---

## Success criteria

| Criterion | Threshold | Consequence if missed |
|---|---|---|
| Inter-rater α | $\geq 0.5$ | Report as-is with limitation noted; consider re-calibration on a second batch |
| Judge-vs-human κ | $\geq 0.4$ (fair) | Frames the addendum around the judge's reliability |
| Per-annotator completion | 60 / 60 pairs | Missing pairs reported in agreement table; α computed on the intersection |
| Time-per-pair median | 1--6 minutes | Below 1 min: risk of skimming; above 6 min: check for confusing UI/pair |

---

## Reproducing this study

1. Fill in `human_eval/data/pairs_60_full.jsonl` via
   `python3 prep/build_worksheet.py`.
2. Install app dependencies via `pip install -r app/requirements.txt`.
3. Run `streamlit run app/streamlit_app.py`.
4. Each annotator uses a distinct annotator ID (name or opaque token).
5. After all three annotators complete, run
   `python3 analysis/compute_agreement.py`.

The complete pre-registered concept note lives in
`../chapter3_concept_note.md` (§4.6).
