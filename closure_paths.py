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
import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

from config import ModelSpec, TEMPERATURE, MAX_OUTPUT_TOKENS, JUDGE_MODEL
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
    """L_spec(code) → docstring. Returns (docstring_text, was_cache_hit).

    max_tokens=MAX_OUTPUT_TOKENS (2048) — small docstrings only need
    ~30-100 tokens, but reasoning-mode SLMs (qwen3.6:27b) consume the
    budget in message.thinking before emitting message.content. With the
    old 512-token cap, qwen3.6 produced empty content. See pilot
    post-mortem (commit d52ede3 / probe 2026-06-06).
    """
    prompt = _PROMPT_DOC_FROM_CODE.format(code=code)
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_spec:doc_from_code",
        temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
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
    """L_spec(tests) → docstring. Used by Path 3.

    max_tokens=MAX_OUTPUT_TOKENS for the same reason as _call_doc_from_code:
    reasoning-mode SLMs need budget headroom for message.thinking before
    message.content gets emitted.
    """
    prompt = _PROMPT_DOC_FROM_TESTS.format(tests=tests)
    resp = ollama_client.call_llm(
        model, prompt,
        role_hint="L_spec:doc_from_tests",
        temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
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
_DEF_TEST_RE = re.compile(r"(?m)^def test_[A-Za-z_0-9]+\s*\(")


def _extract_python(text: str) -> str:
    """Strip markdown fences if present; else return text as-is."""
    if not text:
        return ""
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _word_shuffle(text: str, seed: int) -> str:
    """Deterministically shuffle whitespace-separated tokens within each
    line, preserving line breaks and leading indentation.

    Used by the N1 null cell to feed each path a structurally-valid but
    semantically-mangled first-stage input. Same (text, seed) always
    produces the same output (so the cache key is stable across reruns).
    """
    if not text:
        return text
    rng = random.Random(seed)
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        tokens = stripped.split()
        rng.shuffle(tokens)
        out_lines.append(indent + " ".join(tokens))
    return "\n".join(out_lines)


def _corrupt_seed(cell_id: str, sample_idx: int, stage: str) -> int:
    """Stable per-(cell, sample, stage) seed for word-shuffle.

    Uses SHA-256 (not Python's per-process-salted hash()) so the seed is
    the same on every Colab session and the cache key for the corrupted
    prompt stays stable across reruns.
    """
    h = hashlib.sha256(f"{cell_id}|{sample_idx}|{stage}".encode()).digest()
    return int.from_bytes(h[:4], "big")


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

    tag = f"[{cell.cell_id} s={sample_idx} p1]"
    logger.info(f"{tag} START {source}")

    n_calls = 0
    n_hits = 0

    # N1 control: word-shuffle the code before feeding it to L_spec.
    # Mutation testing still runs against the ORIGINAL code so a sound
    # closure metric should be ~0 on N1.
    spec_input = code
    if cell.corrupt_inputs:
        spec_input = _word_shuffle(code, _corrupt_seed(cell.cell_id, sample_idx, "p1_spec_in"))
        logger.info(f"{tag} corrupt_inputs=True — word-shuffled code "
                    f"({len(code)}→{len(spec_input)} chars) before L_spec")

    # Step 1: D' from code
    if cell.L_spec is None:                 # N2 ablation
        d_prime = ""
        logger.info(f"{tag} step1 L_spec=SKIP (null ablation)")
    else:
        t = time.perf_counter()
        d_prime, h = _call_doc_from_code(cell.L_spec, spec_input)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step1 L_spec={cell.L_spec.short_name} "
                    f"-> D'={len(d_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

    # Step 2: T' from D'
    if cell.L_test is None or not d_prime:
        t_prime = ""
        logger.info(f"{tag} step2 L_test=SKIP (no docstring or null ablation)")
    else:
        t = time.perf_counter()
        t_prime, h = _call_tests_from_doc(cell.L_test, d_prime, fn_name, signature)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step2 L_test={cell.L_test.short_name} "
                    f"-> T'={len(t_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

    # Step 3: filter tests that fail on original C
    if t_prime:
        t = time.perf_counter()
        filtered_tests = closure_metrics.test_filter(t_prime, code)
        n_before = len(_DEF_TEST_RE.findall(t_prime))
        n_after = len(_DEF_TEST_RE.findall(filtered_tests))
        logger.info(f"{tag} step3 test_filter -> "
                    f"{n_after}/{n_before} tests kept "
                    f"({len(filtered_tests)}/{len(t_prime)} chars) "
                    f"+{time.perf_counter()-t:.2f}s")
    else:
        filtered_tests = ""

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
    t = time.perf_counter()
    kill_rate, breakdown = closure_metrics.mutation_kill_rate(filtered_tests, code)
    logger.info(f"{tag} step4 mutation -> kill_rate={kill_rate:.3f} "
                f"({breakdown.get('killed',0)}/{breakdown.get('total_mutants',0)} mutants) "
                f"+{time.perf_counter()-t:.2f}s")

    # Step 5: judge equivalence of D' vs original docstring (if available)
    if orig_doc and d_prime:
        t = time.perf_counter()
        j = judge_llm.judge_equivalence(orig_doc, d_prime, artefact_kind="docstring")
        rating, reason = j.rating, j.justification
        n_calls += 1
        n_hits += int(j.cache_hit)
        logger.info(f"{tag} step5 judge={JUDGE_MODEL.short_name} "
                    f"-> rating={rating} cache={'HIT' if j.cache_hit else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")
    else:
        rating, reason = -1, "no_orig_docstring"
        logger.info(f"{tag} step5 judge=SKIP (no original docstring)")

    elapsed = time.perf_counter() - t0
    logger.info(f"{tag} DONE elapsed={elapsed:.2f}s kill_rate={kill_rate:.3f} "
                f"calls={n_calls} hits={n_hits}")
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

    tag = f"[{cell.cell_id} s={sample_idx} p2]"
    logger.info(f"{tag} START {source}")

    # N1 control: word-shuffle the docstring before feeding it to L_test.
    # Reference tests still run against C' (which is reconstructed from
    # the corrupted docstring) so a sound metric should be ~0 on N1.
    test_input_doc = docstring
    if cell.corrupt_inputs and docstring:
        test_input_doc = _word_shuffle(
            docstring, _corrupt_seed(cell.cell_id, sample_idx, "p2_test_in")
        )
        logger.info(f"{tag} corrupt_inputs=True — word-shuffled docstring "
                    f"({len(docstring)}→{len(test_input_doc)} chars) before L_test")

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
        logger.info(f"{tag} step1 L_test=SKIP")
    else:
        t = time.perf_counter()
        t_prime, h = _call_tests_from_doc(cell.L_test, test_input_doc, fn_name, signature)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step1 L_test={cell.L_test.short_name} "
                    f"-> T'={len(t_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

    # Step 2: C' from D + T'
    if cell.L_code is None or not t_prime:
        c_prime = ""
        logger.info(f"{tag} step2 L_code=SKIP (no tests)")
    else:
        t = time.perf_counter()
        c_prime, h = _call_code_from_doc_tests(cell.L_code, test_input_doc, t_prime, fn_name, signature)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step2 L_code={cell.L_code.short_name} "
                    f"-> C'={len(c_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

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
    t = time.perf_counter()
    pass_rate = closure_metrics.reference_test_pass_rate(reference_tests, c_prime)
    logger.info(f"{tag} step3 ref_tests -> pass_rate={pass_rate:.3f} "
                f"+{time.perf_counter()-t:.2f}s")

    # Step 4: judge equivalence of C' vs original C
    t = time.perf_counter()
    j = judge_llm.judge_equivalence(code, c_prime, artefact_kind="code")
    n_calls += 1
    n_hits += int(j.cache_hit)
    logger.info(f"{tag} step4 judge={JUDGE_MODEL.short_name} -> rating={j.rating} "
                f"cache={'HIT' if j.cache_hit else 'miss'} "
                f"+{time.perf_counter()-t:.2f}s")

    elapsed = time.perf_counter() - t0
    logger.info(f"{tag} DONE elapsed={elapsed:.2f}s pass_rate={pass_rate:.3f} "
                f"calls={n_calls} hits={n_hits}")
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

    tag = f"[{cell.cell_id} s={sample_idx} p3]"
    logger.info(f"{tag} START {source}")

    # N1 control: word-shuffle the code before feeding it to L_test.
    # BERTScore + judge still compare against the ORIGINAL docstring, so
    # a sound metric should be ~0 on N1.
    test_input_code = code
    if cell.corrupt_inputs:
        test_input_code = _word_shuffle(
            code, _corrupt_seed(cell.cell_id, sample_idx, "p3_test_in")
        )
        logger.info(f"{tag} corrupt_inputs=True — word-shuffled code "
                    f"({len(code)}→{len(test_input_code)} chars) before L_test")

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
        logger.info(f"{tag} step1 L_test=SKIP")
    else:
        t = time.perf_counter()
        t_prime, h = _call_tests_from_code(cell.L_test, test_input_code)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step1 L_test={cell.L_test.short_name} "
                    f"-> T'={len(t_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

    # Step 2: D' from T'
    if cell.L_spec is None or not t_prime:
        d_prime = ""
        logger.info(f"{tag} step2 L_spec=SKIP (no tests)")
    else:
        t = time.perf_counter()
        d_prime, h = _call_doc_from_tests(cell.L_spec, t_prime)
        n_calls += 1
        n_hits += int(h)
        logger.info(f"{tag} step2 L_spec={cell.L_spec.short_name} "
                    f"-> D'={len(d_prime)}chars cache={'HIT' if h else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")

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
    t = time.perf_counter()
    bertscore_available = True
    try:
        bert, bert_hit = closure_metrics.bert_similarity(orig_doc, d_prime)
        n_hits += int(bert_hit)
        logger.info(f"{tag} step3 BERTScore -> F1={bert:.3f} "
                    f"cache={'HIT' if bert_hit else 'miss'} "
                    f"+{time.perf_counter()-t:.2f}s")
    except ImportError:
        bert = 0.0
        bertscore_available = False
        logger.warning(f"{tag} step3 BERTScore SKIP (bert-score not installed)")

    # Step 4: judge equivalence on the docstrings
    t = time.perf_counter()
    j = judge_llm.judge_equivalence(orig_doc, d_prime, artefact_kind="docstring")
    n_calls += 1
    n_hits += int(j.cache_hit)
    logger.info(f"{tag} step4 judge={JUDGE_MODEL.short_name} -> rating={j.rating} "
                f"cache={'HIT' if j.cache_hit else 'miss'} "
                f"+{time.perf_counter()-t:.2f}s")

    elapsed = time.perf_counter() - t0
    logger.info(f"{tag} DONE elapsed={elapsed:.2f}s bertscore={bert:.3f} "
                f"valid={bertscore_available} calls={n_calls} hits={n_hits}")
    return ClosureResult(
        cell_id=cell.cell_id, sample_idx=sample_idx, sample_source=source,
        path=3, metric_name="bertscore",
        metric_value=float(bert) if bertscore_available else float("nan"),
        judge_rating=j.rating, judge_justification=j.justification,
        valid=bertscore_available, elapsed_s=elapsed,
        cache_hits=n_hits, n_llm_calls=n_calls,
        notes="" if bertscore_available else "bertscore_unavailable",
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
