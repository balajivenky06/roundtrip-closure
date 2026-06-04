"""
closure_paths.py — the three round-trip closure paths.

For each (cell, function) pair, three traversals of the docstring-test-code
triangle are executed:

    Path 1:  C → D → T   close-check: mutation kill rate of T against C
    Path 2:  D → T → C   close-check: pass rate of ORIGINAL C's reference tests
                                       against reconstructed C'
    Path 3:  C → T → D   close-check: BERTScore + judge LLM equivalence(D, D')

Each driver returns a list of ClosureResult records (one per path).

The driver functions are stateless — every LLM call is routed through
ollama_client.call_llm, which transparently uses closure_cache. So if
the Colab session disconnects mid-sweep, on resume every identical
LLM call is a free disk lookup.

Per-result resumability (skip (cell, sample_idx, path) tuples already
written to the TSV) is handled in train_roundtrip.py, not here.
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

from config import ModelSpec, TEMPERATURE, MAX_OUTPUT_TOKENS
from doe import Cell
import ollama_client
import closure_metrics
import judge_llm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public result type
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ClosureResult:
    """One closure measurement, one row in the results TSV."""
    cell_id: str
    sample_idx: int
    sample_source: str       # e.g. "humaneval/12" / "mbpp/77" / "livecodebench/3"
    path: int                # 1, 2, or 3
    metric_name: str         # "mutation_kill_rate" | "reference_pass_rate" | "bertscore"
    metric_value: float      # closure metric in [0, 1] or NaN
    judge_rating: int        # 0-4 from the external judge, or -1 on failure
    judge_justification: str
    valid: bool              # False if filter / empty input dropped this sample
    elapsed_s: float
    cache_hits: int          # how many of n_llm_calls were cache hits
    n_llm_calls: int         # pipeline + judge calls combined
    notes: str = ""

    # Stable column order for the TSV
    TSV_COLUMNS = (
        "cell_id", "sample_idx", "sample_source", "path",
        "metric_name", "metric_value",
        "judge_rating", "judge_justification",
        "valid", "elapsed_s",
        "cache_hits", "n_llm_calls",
        "notes",
    )

    def to_tsv_row(self) -> str:
        """Render as a single tab-separated line (newline-terminated)."""
        d = asdict(self)
        # Escape tabs/newlines in justification + notes
        d["judge_justification"] = _tsv_safe(d["judge_justification"])
        d["notes"] = _tsv_safe(d["notes"])
        return "\t".join(str(d[c]) for c in self.TSV_COLUMNS) + "\n"

    @classmethod
    def tsv_header(cls) -> str:
        return "\t".join(cls.TSV_COLUMNS) + "\n"

    @classmethod
    def key(cls, cell_id: str, sample_idx: int, path: int) -> str:
        """Stable resume-key for (cell, sample, path)."""
        return f"{cell_id}|{sample_idx}|{path}"

    @property
    def resume_key(self) -> str:
        return self.key(self.cell_id, self.sample_idx, self.path)


def _tsv_safe(s: str) -> str:
    """Replace tabs and newlines so they don't break TSV row boundaries."""
    if not s:
        return ""
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


# ──────────────────────────────────────────────────────────────────────
# Prompt templates — one per (source_kind → target_kind) pair
# ──────────────────────────────────────────────────────────────────────
_PROMPT_DOC_FROM_CODE = """\
Write a concise Python docstring (1-3 sentences) that describes what the \
following function does. Respond with ONLY the docstring text — no quotes, \
no code, no extra commentary.

Function:
{code}
"""

_PROMPT_TESTS_FROM_DOC = """\
You are given a Python function's docstring (the function itself is hidden). \
Write a pytest test suite that verifies the behavior described.

Rules:
- Use the function by its name (do not redefine it).
- Cover happy path, edge cases, and any error cases the docstring mentions.
- Respond with ONLY a Python code block inside triple backticks (```python … ```).

Function name: {fn_name}
Function signature: {signature}

Docstring:
{docstring}
"""

_PROMPT_TESTS_FROM_CODE = """\
You are given a Python function. Write a pytest test suite that exercises \
its behavior.

Rules:
- Cover happy path, edge cases, and likely error cases.
- Use the function by name; do not redefine it.
- Respond with ONLY a Python code block inside triple backticks (```python … ```).

Function:
{code}
"""

_PROMPT_DOC_FROM_TESTS = """\
You are given a Python test suite (the function under test is hidden). \
Infer what the function does and write a concise Python docstring \
(1-3 sentences) that summarizes its behavior. Respond with ONLY the \
docstring text — no quotes, no code, no extra commentary.

Tests:
{tests}
"""

_PROMPT_CODE_FROM_DOC_TESTS = """\
You are given a Python function's docstring and a pytest test suite \
for the same function. Write the function body that satisfies the \
docstring AND passes every test.

Rules:
- Respond with ONLY a Python code block inside triple backticks (```python … ```).
- The block must define ONE top-level function and may include necessary imports.

Function name: {fn_name}
Function signature: {signature}

Docstring:
{docstring}

Tests:
{tests}
"""


# ──────────────────────────────────────────────────────────────────────
# Stage callers — every LLM call goes through ollama_client.call_llm
# ──────────────────────────────────────────────────────────────────────
def _call_doc_from_code(model: ModelSpec, code: str) -> tuple[str, bool]:
    """L_spec(code) → docstring. Returns (docstring_text, was_cache_hit)."""
    prompt = _PROMPT_DOC_FROM_CODE.format(code=code)
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_spec:doc_from_code",
        temperature=TEMPERATURE, max_tokens=512,
    )
    return resp.text.strip(), resp.cache_hit


def _call_tests_from_doc(model: ModelSpec, docstring: str,
                         fn_name: str = "", signature: str = ""
                         ) -> tuple[str, bool]:
    """L_test(docstring) → tests."""
    prompt = _PROMPT_TESTS_FROM_DOC.format(
        docstring=docstring, fn_name=fn_name or "the function",
        signature=signature or "(unknown)",
    )
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_test:tests_from_doc",
        temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _extract_python(resp.text), resp.cache_hit


def _call_tests_from_code(model: ModelSpec, code: str) -> tuple[str, bool]:
    """L_test(code) → tests. Used by Path 3."""
    prompt = _PROMPT_TESTS_FROM_CODE.format(code=code)
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_test:tests_from_code",
        temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _extract_python(resp.text), resp.cache_hit


def _call_doc_from_tests(model: ModelSpec, tests: str) -> tuple[str, bool]:
    """L_spec(tests) → docstring. Used by Path 3."""
    prompt = _PROMPT_DOC_FROM_TESTS.format(tests=tests)
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_spec:doc_from_tests",
        temperature=TEMPERATURE, max_tokens=512,
    )
    return resp.text.strip(), resp.cache_hit


def _call_code_from_doc_tests(model: ModelSpec, docstring: str, tests: str,
                              fn_name: str = "", signature: str = ""
                              ) -> tuple[str, bool]:
    """L_code(docstring, tests) → code. Used by Path 2."""
    prompt = _PROMPT_CODE_FROM_DOC_TESTS.format(
        docstring=docstring, tests=tests,
        fn_name=fn_name or "the function",
        signature=signature or "(unknown)",
    )
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_code:code_from_doc_tests",
        temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _extract_python(resp.text), resp.cache_hit


# ──────────────────────────────────────────────────────────────────────
# Code-block extraction from LLM responses
# ──────────────────────────────────────────────────────────────────────
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _extract_python(text: str) -> str:
    """Strip markdown fences if present; else return text as-is."""
    if not text:
        return ""
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Path 1 — Code → Docstring → Tests
# ──────────────────────────────────────────────────────────────────────
def run_path_1(cell: Cell, sample: dict) -> ClosureResult:
    """
    Steps:
        1. D' = L_spec(C)
        2. T' = L_test(D')
        3. Test filter against ORIGINAL C
        4. mutation_kill_rate(T', C)
        5. (optional) judge equivalence(orig_D, D')
    """
    t0 = time.perf_counter()
    code = sample["code"]
    sample_idx = sample["sample_idx"]
    source = sample.get("source", "")
    orig_doc = sample.get("docstring", "")
    fn_name = sample.get("entry_point", "")
    signature = sample.get("signature", "")

    n_calls = 0
    n_hits = 0

    # Step 1: D' from code
    if cell.L_spec is None:                 # N2 ablation
        d_prime = ""
    else:
        d_prime, h = _call_doc_from_code(cell.L_spec, code)
        n_calls += 1
        n_hits += int(h)

    # Step 2: T' from D'
    if cell.L_test is None or not d_prime:
        t_prime = ""
    else:
        t_prime, h = _call_tests_from_doc(cell.L_test, d_prime, fn_name, signature)
        n_calls += 1
        n_hits += int(h)

    # Step 3: filter tests that fail on original C
    filtered_tests = closure_metrics.test_filter(t_prime, code) if t_prime else ""

    if not filtered_tests:
        elapsed = time.perf_counter() - t0
        return ClosureResult(
            cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
            path=1, metric_name="mutation_kill_rate",
            metric_value=float("nan"),
            judge_rating=-1, judge_justification="all_tests_filtered_or_empty",
            valid=False, elapsed_s=elapsed,
            cache_hits=n_hits, n_llm_calls=n_calls,
            notes="filter dropped all tests",
        )

    # Step 4: mutation kill rate
    kill_rate, breakdown = closure_metrics.mutation_kill_rate(filtered_tests, code)

    # Step 5: judge equivalence of D' vs original docstring (if available)
    if orig_doc and d_prime:
        j = judge_llm.judge_equivalence(orig_doc, d_prime, artefact_kind="docstring")
        rating, reason = j.rating, j.justification
        n_calls += 1
        # judge_equivalence goes through ollama_client; cache hits already tracked
        # but we don't easily get the cache_hit flag back from JudgeResult.
        # That's a minor reporting gap; total LLM call count is still accurate.
    else:
        rating, reason = -1, "no_orig_docstring"

    elapsed = time.perf_counter() - t0
    return ClosureResult(
        cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
        path=1, metric_name="mutation_kill_rate",
        metric_value=float(kill_rate),
        judge_rating=rating, judge_justification=reason,
        valid=not _isnan(kill_rate), elapsed_s=elapsed,
        cache_hits=n_hits, n_llm_calls=n_calls,
        notes=f"mutants={breakdown.get('total_mutants', 0)}"
              f",killed={breakdown.get('killed', 0)}",
    )


# ──────────────────────────────────────────────────────────────────────
# Path 2 — Docstring → Tests → Code
# ──────────────────────────────────────────────────────────────────────
def run_path_2(cell: Cell, sample: dict) -> ClosureResult:
    """
    Steps:
        1. T' = L_test(D)            (original D, not generated)
        2. C' = L_code(D, T')
        3. Run ORIGINAL reference tests against C'
        4. Judge equivalence(orig_C, C')
    """
    t0 = time.perf_counter()
    code = sample["code"]
    docstring = sample.get("docstring", "")
    reference_tests = sample.get("tests", "")
    sample_idx = sample["sample_idx"]
    source = sample.get("source", "")
    fn_name = sample.get("entry_point", "")
    signature = sample.get("signature", "")

    if not docstring or not reference_tests:
        elapsed = time.perf_counter() - t0
        return ClosureResult(
            cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
            path=2, metric_name="reference_pass_rate",
            metric_value=0.0,
            judge_rating=-1, judge_justification="missing_docstring_or_tests",
            valid=False, elapsed_s=elapsed,
            cache_hits=0, n_llm_calls=0,
            notes="sample missing required fields",
        )

    n_calls = 0
    n_hits = 0

    # Step 1: T' from D
    if cell.L_test is None:
        t_prime = ""
    else:
        t_prime, h = _call_tests_from_doc(cell.L_test, docstring, fn_name, signature)
        n_calls += 1
        n_hits += int(h)

    # Step 2: C' from D + T'
    if cell.L_code is None or not t_prime:
        c_prime = ""
    else:
        c_prime, h = _call_code_from_doc_tests(cell.L_code, docstring, t_prime, fn_name, signature)
        n_calls += 1
        n_hits += int(h)

    if not c_prime:
        elapsed = time.perf_counter() - t0
        return ClosureResult(
            cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
            path=2, metric_name="reference_pass_rate",
            metric_value=0.0,
            judge_rating=-1, judge_justification="no_reconstructed_code",
            valid=False, elapsed_s=elapsed,
            cache_hits=n_hits, n_llm_calls=n_calls,
            notes="L_code produced empty output",
        )

    # Step 3: original reference tests on C'
    pass_rate = closure_metrics.reference_test_pass_rate(reference_tests, c_prime)

    # Step 4: judge equivalence of C' vs original C
    j = judge_llm.judge_equivalence(code, c_prime, artefact_kind="code")
    n_calls += 1

    elapsed = time.perf_counter() - t0
    return ClosureResult(
        cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
        path=2, metric_name="reference_pass_rate",
        metric_value=float(pass_rate),
        judge_rating=j.rating, judge_justification=j.justification,
        valid=True, elapsed_s=elapsed,
        cache_hits=n_hits, n_llm_calls=n_calls,
        notes="",
    )


# ──────────────────────────────────────────────────────────────────────
# Path 3 — Code → Tests → Docstring
# ──────────────────────────────────────────────────────────────────────
def run_path_3(cell: Cell, sample: dict) -> ClosureResult:
    """
    Steps:
        1. T' = L_test(C)            (tests directly from code)
        2. D' = L_spec(T')           (docstring inferred from tests)
        3. BERTScore(orig_D, D')
        4. Judge equivalence(orig_D, D')
    """
    t0 = time.perf_counter()
    code = sample["code"]
    orig_doc = sample.get("docstring", "")
    sample_idx = sample["sample_idx"]
    source = sample.get("source", "")

    if not orig_doc:
        elapsed = time.perf_counter() - t0
        return ClosureResult(
            cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
            path=3, metric_name="bertscore", metric_value=0.0,
            judge_rating=-1, judge_justification="no_orig_docstring",
            valid=False, elapsed_s=elapsed,
            cache_hits=0, n_llm_calls=0,
            notes="sample missing original docstring",
        )

    n_calls = 0
    n_hits = 0

    # Step 1: T' from C
    if cell.L_test is None:
        t_prime = ""
    else:
        t_prime, h = _call_tests_from_code(cell.L_test, code)
        n_calls += 1
        n_hits += int(h)

    # Step 2: D' from T'
    if cell.L_spec is None or not t_prime:
        d_prime = ""
    else:
        d_prime, h = _call_doc_from_tests(cell.L_spec, t_prime)
        n_calls += 1
        n_hits += int(h)

    if not d_prime:
        elapsed = time.perf_counter() - t0
        return ClosureResult(
            cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
            path=3, metric_name="bertscore", metric_value=0.0,
            judge_rating=-1, judge_justification="no_recovered_docstring",
            valid=False, elapsed_s=elapsed,
            cache_hits=n_hits, n_llm_calls=n_calls,
            notes="pipeline produced empty docstring",
        )

    # Step 3: BERTScore
    try:
        bert = closure_metrics.bert_similarity(orig_doc, d_prime)
    except ImportError:
        bert = 0.0  # bert_score not installed; record 0 and continue

    # Step 4: judge equivalence on the docstrings
    j = judge_llm.judge_equivalence(orig_doc, d_prime, artefact_kind="docstring")
    n_calls += 1

    elapsed = time.perf_counter() - t0
    return ClosureResult(
        cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
        path=3, metric_name="bertscore",
        metric_value=float(bert),
        judge_rating=j.rating, judge_justification=j.justification,
        valid=True, elapsed_s=elapsed,
        cache_hits=n_hits, n_llm_calls=n_calls,
        notes="",
    )


# ──────────────────────────────────────────────────────────────────────
# Convenience — run all three (or a subset) on one sample
# ──────────────────────────────────────────────────────────────────────
_PATH_DRIVERS = {
    1: run_path_1,
    2: run_path_2,
    3: run_path_3,
}


def run_all_paths(cell: Cell, sample: dict,
                  paths_to_run: tuple[int, ...] = (1, 2, 3)
                  ) -> list[ClosureResult]:
    """Run the requested paths in order. Failures in one path do not stop others."""
    results: list[ClosureResult] = []
    for p in paths_to_run:
        if p not in _PATH_DRIVERS:
            logger.warning(f"Unknown path id {p}; skipping.")
            continue
        try:
            results.append(_PATH_DRIVERS[p](cell, sample))
        except Exception as exc:                                  # pragma: no cover
            logger.error(f"Path {p} on cell {cell.cell_id} "
                         f"sample {sample.get('sample_idx', '?')} crashed: {exc}",
                         exc_info=True)
            results.append(ClosureResult(
                cell_id=cell.cell_id,
                sample_idx=sample.get("sample_idx", -1),
                sample_source=sample.get("source", ""),
                path=p, metric_name="error", metric_value=float("nan"),
                judge_rating=-1, judge_justification="path_crashed",
                valid=False, elapsed_s=0.0,
                cache_hits=0, n_llm_calls=0,
                notes=str(exc)[:200],
            ))
    return results


# ──────────────────────────────────────────────────────────────────────
# Tiny helpers
# ──────────────────────────────────────────────────────────────────────
def _isnan(x: float) -> bool:
    return x != x   # NaN is the only value != itself
