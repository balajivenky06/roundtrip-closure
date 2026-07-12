# Rubric — Semantic Equivalence Rating

**One question. One scale. Five levels.**

For each pair (Artefact~A, Artefact~B), answer:

> *"To what extent do these two Python artefacts describe or implement the same behaviour?"*

Choose exactly one rating from the five levels below.

---

## The scale

| Rating | Label | Definition |
|---:|---|---|
| **4** | Identical | Same output on every valid input. Trivial differences only (whitespace, comment wording, docstring line breaks, variable renaming). |
| **3** | Equivalent | Same observable behaviour on every reasonable input, but a different implementation strategy. Different algorithm, different data structure, different helper decomposition — same result. |
| **2** | Approximately equivalent | Same behaviour on most reasonable inputs, but differs on rare edge cases (e.g. empty input, negative input, unicode). The differences are "would probably not affect a real caller" but are technically visible. |
| **1** | Clearly different | Different output on common inputs. Same broad problem area (e.g. both operate on lists) but different function. Not something a caller could substitute. |
| **0** | Unrelated | The two artefacts describe or implement different problems entirely. A caller expecting one would be materially misled by the other. |

---

## Rating anchors — micro-examples

The examples below are intentionally short. They ground the rubric — refer
back to them whenever a rating decision feels ambiguous.

### Rating 4 — Identical

**Docstring pair example:**

*A:* `Return True if the input string is a palindrome, i.e. reads the same forwards and backwards.`

*B:* `Return True when the given string reads identically in both directions (a palindrome).`

→ Rating 4. Same behaviour, same conditions, no observable difference.

**Code pair example:**

*A:*
```python
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)
```

*B:*
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

→ Rating 4. Formatting only.

---

### Rating 3 — Equivalent

**Code pair example:**

*A:*
```python
def sum_positive(lst):
    return sum(x for x in lst if x > 0)
```

*B:*
```python
def sum_positive(lst):
    total = 0
    for x in lst:
        if x > 0:
            total += x
    return total
```

→ Rating 3. Same behaviour; different implementation (generator vs. explicit loop).

**Docstring pair example:**

*A:* `Return the maximum value in the list, or None if the list is empty.`

*B:* `Compute and return the largest element. When the input list contains no elements, return None instead of raising.`

→ Rating 3. Same observable behaviour, different phrasing and emphasis.

---

### Rating 2 — Approximately equivalent

**Code pair example:**

*A:*
```python
def word_count(s):
    return len(s.split())
```

*B:*
```python
def word_count(s):
    return len(s.strip().split())
```

→ Rating 2. Both work on typical inputs; disagree on leading/trailing whitespace boundary cases where `s = ""` or `s = "   "` (both return 0, so those aren't different). Consider inputs like `"  hello  world  "` — actually both give 2. In practice these behave identically on all inputs; but a strict reader might note the different edge-case handling of pure-whitespace strings. Borderline 2 or 3. When in doubt, choose 3.

**Better example:**

*A:*
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

*B:*
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

→ Rating 2 or 3 depending on interpretation. Both correctly identify primes on all non-negative integers. Rating 3 is defensible (equivalent behaviour, different implementation). Rating 2 is defensible if you interpret "approximately equivalent" as "same output on the intended domain but differs in exception behaviour for negative inputs" (both return False for negatives, so actually they agree there). **When behaviour is identical on the intended domain, choose 3, not 2.** Reserve rating 2 for cases where you can name a specific input on which the two artefacts genuinely diverge.

---

### Rating 1 — Clearly different

**Code pair example:**

*A:*
```python
def average(lst):
    return sum(lst) / len(lst)
```

*B:*
```python
def average(lst):
    return sum(lst) // len(lst)
```

→ Rating 1. `average([1, 2, 4])` returns `2.333...` in A, `2` in B. Different output on common inputs. A caller expecting one would be misled by the other.

**Docstring pair example:**

*A:* `Return the sum of all elements in the list.`

*B:* `Return the product of all elements in the list.`

→ Rating 1. Same problem area (list reduction) but different function.

---

### Rating 0 — Unrelated

**Code pair example:**

*A:* code that computes the factorial of a positive integer.

*B:* code that checks whether a string is a valid email address.

→ Rating 0. Different problems entirely.

**Docstring pair example:**

*A:* `Convert a temperature in Celsius to Fahrenheit.`

*B:* `Compute the shortest path between two nodes in an unweighted graph.`

→ Rating 0.

---

## Deciding between adjacent levels

The most common ambiguous choices, with a decision rule:

- **4 vs. 3:** If you can point to a difference in implementation approach that is visible to a reader (loop vs. recursion, generator vs. explicit loop, different helper decomposition, different formatting of the docstring), it's 3. If the only differences are surface-level (variable names, whitespace, comment wording), it's 4.
- **3 vs. 2:** If you can name a *specific* input on which the two artefacts genuinely give different output, it's 2. If the behaviour is identical on all inputs in the intended domain, it's 3 — even if the implementations differ substantially.
- **2 vs. 1:** If the divergence is only on rare or unusual inputs, it's 2. If a routine caller would produce a divergent result, it's 1.
- **1 vs. 0:** If the artefacts operate on the same problem area (both are list-manipulation functions; both are string-processing docstrings) but produce different results, it's 1. If they solve unrelated problems, it's 0.

---

## Justification field (mandatory)

Every rating is accompanied by a brief one-sentence justification explaining
which rubric level applies and why. The justification is used to:

1. Enable disagreement analysis after annotation completes.
2. Distinguish "the annotator was confident" from "the annotator hedged".
3. Compare against the judge SLM's own justification (which is stored per row).

**Justifications should:**

- Reference the rubric level explicitly ("Rating 3 because …").
- Name a specific behavioural difference or agreement.
- Not exceed 30 words.

**Justifications should not:**

- Reference which SLM produced either artefact (annotators do not have this information).
- Reference which closure path or which cell (annotators do not have this information).
- Speculate about the pipeline that generated the artefact.

---

## Rubric provenance

This rubric is derived from the concept-note §4.6 pre-registration
(committed to source 2026-06-02, prior to any sweep results being observed).
It is a simpler single-axis rubric than Chapter 2's three-axis rubric —
justified by the observation that inter-rater α on the three-axis rubric
was the weakest link in Chapter 2's calibration.
