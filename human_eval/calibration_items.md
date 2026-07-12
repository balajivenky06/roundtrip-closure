# Calibration items — 10 pairs for the pre-annotation session

Ten pre-vetted (Artefact A, Artefact B) pairs designed to exercise every
rubric level and every ambiguous boundary. Ratings shown here are the
**consensus expected rating** — annotators submit their rating first, then
the expected rating is revealed and any divergence is discussed.

The items are **not** drawn from the 60-pair production set. Ratings on the
calibration items are recorded separately and are not counted in the main
analysis.

---

## Session structure (~1 hour)

For each of the 10 items in order:

1. Corresponding author displays Artefact A and Artefact B side by side.
2. Each annotator, working silently, writes down their rating (0–4) and a one-sentence justification. **(2 min)**
3. Ratings are revealed simultaneously by the annotators. **(1 min)**
4. Divergences are discussed with reference to the rubric anchor sections. **(3 min)**
5. Move to next item.

**Total time:** 10 items × 6 min ≈ 1 hour.

---

## Item C01 — Rating 4 (Identical, docstring)

**Artefact A (original):**
> Return the sum of two integers.

**Artefact B (reconstructed):**
> Return the sum of two integer values.

**Expected rating:** 4

**Rationale:** Trivial phrasing difference; same behaviour and same conditions.

**Discussion prompt:** Would rating 3 be defensible here? *(No — no implementation
difference is visible; the reworded docstring is not a "different implementation"
because docstrings do not implement anything.)*

---

## Item C02 — Rating 4 (Identical, code)

**Artefact A (original):**
```python
def double(x):
    return x * 2
```

**Artefact B (reconstructed):**
```python
def double(x):
    return 2 * x
```

**Expected rating:** 4

**Rationale:** Multiplication is commutative; the two artefacts produce
identical output on every input. Surface-level difference only.

---

## Item C03 — Rating 3 (Equivalent, code)

**Artefact A (original):**
```python
def contains_vowel(s):
    return any(c in "aeiou" for c in s.lower())
```

**Artefact B (reconstructed):**
```python
def contains_vowel(s):
    for c in s.lower():
        if c in "aeiou":
            return True
    return False
```

**Expected rating:** 3

**Rationale:** Same behaviour on every input; different implementation
strategy (generator + `any` vs. explicit loop + early return).

**Discussion prompt:** Some annotators may argue rating 4 (behaviourally
identical). The distinguishing question: *is there a visible implementation
difference?* Yes — the second is more verbose and uses different Python
primitives. Rating 3.

---

## Item C04 — Rating 3 (Equivalent, docstring)

**Artefact A (original):**
> Given a non-empty list of integers, return the mean of the list rounded to the nearest integer.

**Artefact B (reconstructed):**
> Compute and return the arithmetic average of a list of integers, rounded to the closest whole number. The list must contain at least one element.

**Expected rating:** 3

**Rationale:** Same behaviour, same precondition. Different phrasing style
(concise vs. explanatory) but both would produce identical caller
expectations.

---

## Item C05 — Rating 2 (Approximately equivalent, code)

**Artefact A (original):**
```python
def first_letter(s):
    return s[0]
```

**Artefact B (reconstructed):**
```python
def first_letter(s):
    return s[0] if s else ""
```

**Expected rating:** 2

**Rationale:** Both produce the same output on every non-empty string. On
the empty string, A raises `IndexError` while B returns `""`. A caller
routinely calling on non-empty strings would see identical behaviour; a
caller passing empty strings would see divergence.

**Discussion prompt:** Rating 1 is defensible if the annotator considers
the exception semantics an "observable behaviour" difference on a common
input. Rating 3 is defensible if the annotator considers empty strings a
rare edge case. The rubric anchor for rating 2 is "same behaviour on most
reasonable inputs; differs on rare edge cases" — this fits. Rating 2.

---

## Item C06 — Rating 2 (Approximately equivalent, docstring)

**Artefact A (original):**
> Return the reverse of a string.

**Artefact B (reconstructed):**
> Return the reverse of a string. If the input is None, return the empty string.

**Expected rating:** 2

**Rationale:** B extends A's behaviour to handle a None input. On any
non-None input the two are identical. On None, A raises (implicit contract)
while B returns "".

---

## Item C07 — Rating 1 (Clearly different, code)

**Artefact A (original):**
```python
def is_even(n):
    return n % 2 == 0
```

**Artefact B (reconstructed):**
```python
def is_even(n):
    return n % 2 == 1
```

**Expected rating:** 1

**Rationale:** Inverted logic. Different output on every integer input. A
caller would be actively misled.

---

## Item C08 — Rating 1 (Clearly different, docstring)

**Artefact A (original):**
> Return the maximum of two integers.

**Artefact B (reconstructed):**
> Return the minimum of two integers.

**Expected rating:** 1

**Rationale:** Same problem area (comparing two integers) but opposite
function.

---

## Item C09 — Rating 0 (Unrelated, code)

**Artefact A (original):**
```python
def days_in_month(month, year):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 30
```

**Artefact B (reconstructed):**
```python
def days_in_month(month, year):
    words = month.split()
    return len(words) + year
```

**Expected rating:** 0

**Rationale:** A implements a calendar computation. B does not — it treats
`month` as a string and returns a nonsensical count. The two artefacts
solve different problems.

**Discussion prompt:** Rating 1 might be argued on the grounds that both
take (month, year) parameters. The rubric anchor for rating 0 is "solving
different problems entirely" — B is not attempting the calendar problem
at all. Rating 0.

---

## Item C10 — Rating 0 (Unrelated, docstring)

**Artefact A (original):**
> Compute the checksum of a byte sequence using the CRC-32 algorithm.

**Artefact B (reconstructed):**
> Return True if the input string contains only ASCII characters.

**Expected rating:** 0

**Rationale:** Different problems in every respect — different input types,
different output types, different purposes.

---

## Post-calibration debrief

At the end of the ten items, the corresponding author asks each annotator:

1. Which item did you find the hardest? Why?
2. Where do you feel the rubric anchors are least clear?
3. On production annotation, will you spend more or less than 3 minutes per
   pair on average?

Answers are recorded in a brief calibration-debrief note and archived
alongside the ratings.
