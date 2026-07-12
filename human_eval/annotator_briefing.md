# Annotator briefing

Welcome, and thank you for agreeing to serve as one of the three independent
annotators for this human-evaluation study. This briefing walks you through
everything you need to know before Session 1.

---

## What you will be doing

You will rate 60 pairs of Python artefacts on a 5-level semantic-equivalence
scale. Each pair consists of two artefacts of the same kind — either two
docstrings, two code fragments, or two test suites. Your task is to answer
one question per pair:

> *"To what extent do these two Python artefacts describe or implement the
> same behaviour?"*

The rubric (5 levels: 4 identical, 3 equivalent, 2 approximately equivalent,
1 clearly different, 0 unrelated) is in `rubric.md`. Read it end-to-end
before the calibration session. Rating anchors are grounded in short
worked examples for each level.

---

## What you are being asked to *not* do

- **Do not attempt to identify which language model produced either artefact.**
  This is a blinded study — you are not shown the model. If you infer the
  model from stylistic tells, that is fine, but do not let it influence your
  rating. Rate what is written, not who wrote it.
- **Do not attempt to identify which closure path or cell produced each
  pair.** These are pipeline internals that annotators are not exposed to.
- **Do not communicate with the other two annotators about specific pairs
  during the production phase.** Discussion at the calibration session is
  encouraged; independence during production is required.
- **Do not run the code.** The rubric is a reading-comprehension exercise on
  static artefacts. If you find yourself wanting to run the code, ask
  yourself: *what specific input would produce a divergence?* If you can
  answer that in your head, use rating 2 or 1; if you cannot, use rating 3
  or 4.

---

## Session structure

| Session | When | Duration | Purpose |
|---|---|---|---|
| Consent + kickoff | T-4 weeks | ~15 min | Sign consent form; introductions |
| **Calibration** | T-3 weeks | ~1 hour | Ten items rated jointly with the corresponding author; rubric anchors re-set |
| **Production** | T-3 to T-0 | ~3 hours (self-paced) | Rate all 60 production pairs individually via the Streamlit app |
| Debrief | T-0 + 1 week | ~30 min | Review disagreement patterns; provide qualitative feedback |

Production annotation is self-paced within a ~3-week window. You may split
it across as many sittings as you like. The Streamlit app auto-saves after
each rating and is fully resumable — you can close it mid-session and pick
up where you left off from a different device by re-entering your annotator
ID.

---

## The Streamlit app

You will receive:

- A URL for the annotation app.
- Your annotator ID (a short opaque token — e.g. `ann_alpha`, `ann_beta`,
  `ann_gamma`).
- Login credentials if the app is running behind auth.

Once logged in:

1. The app displays your current progress (X of 60).
2. For each pair:
   - Two panels side by side show Artefact A (labelled "Original") and
     Artefact B (labelled "Reconstructed").
   - Syntax highlighting is applied.
   - The rubric is displayed compactly on the right; the full anchor
     document is one click away.
   - Below the artefacts is a rating radio-button group (0–4) and a
     justification text area (mandatory, one sentence, ≤ 30 words).
3. Click "Submit rating and continue" to advance.
4. You may go back to previous pairs and revise. Revisions are logged with
   timestamps so we can track when you changed your mind.

**Estimated time per pair:** 1–5 minutes. If a specific pair takes you more
than 8 minutes, submit your best guess and note the difficulty in the
justification — do not let one pair block the rest.

---

## Frequently asked questions

**Q: What if I think the "original" is worse than the "reconstructed"?**

A: Rate them for equivalence, not for quality. If they behave identically,
rate 4 regardless of which reads better. If they behave differently, rate
1 or 2 based on how different, regardless of which is "correct".

**Q: What if one of the artefacts is syntactically invalid Python?**

A: Rate for behavioural equivalence based on what a reasonable reader would
infer the artefact was trying to say. If the intent is clear despite the
syntax error, apply the rubric as if the syntax were correct. Note the
syntax issue in your justification. If the intent is genuinely
unrecoverable, rate 0 and note it.

**Q: What if I recognise the docstring from HumanEval or MBPP?**

A: That is fine — the benchmarks are public and expected knowledge. Do not
consult external sources during rating. Your job is to compare the two
artefacts in front of you, not to reason about the benchmark's ground truth.

**Q: What if both artefacts are docstrings and one is much longer than the
other?**

A: Length is not equivalence. A one-line docstring and a ten-line docstring
can both describe the same function. Rate on described behaviour, not on
verbosity.

**Q: What if I want to change my mind about a rating I submitted earlier?**

A: Use the "back" button in the app to return to any previous pair and
revise. The revised rating is what counts. Prior versions are stored for
audit but not used in the analysis.

**Q: What if the app crashes or my ratings get lost?**

A: Ratings are auto-saved to disk after every submission. If the app
crashes, restart it and re-enter your annotator ID; you will resume where
you left off. If you suspect data loss, contact the corresponding author
before continuing.

**Q: I have finished all 60 pairs. What now?**

A: The app displays a completion screen. Click "Confirm submission." Your
ratings are locked. You will receive a debrief-session invitation
approximately one week later.

---

## What happens with your ratings

Your ratings are saved to a per-annotator TSV file. After all three
annotators complete, the three TSVs are combined and analysed for:

1. Inter-rater agreement (Krippendorff's α across the three of you).
2. Pairwise Cohen's κ for each pair of raters.
3. Judge-vs-human agreement (Cohen's κ between the majority human rating
   and the DeepSeek-R1 judge SLM's rating).
4. Per-bucket agreement rates.

You will receive a summary of the aggregate agreement statistics at the
debrief session, but individual ratings are anonymised in all published
outputs — your annotator ID does not appear in any manuscript, replication
package, or public artefact.

---

## Contact

- **Corresponding author:** Balaji Venktesh V.
  (`balaji23610040@snuchennai.edu.in`)
- **Supervisor:** Amsaprabhaa M. (`amsaprabhaam@snuchennai.edu.in`)
- **Study documentation:** This directory (`human_eval/`)

If you have questions about the study, the rubric, the app, or your
compensation, contact the corresponding author. Response within 24 hours on
business days.

Thank you for your time and attention.
