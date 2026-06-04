"""
decontaminate.py — build the HumanEval-Mutated subset for §3.X.

Goal: defend against the data-contamination reviewer attack — open-weight
SLMs trained on GitHub may have seen HumanEval/MBPP solutions and be
producing memorised code rather than genuinely reasoning.

Strategy: apply three semantics-preserving transformations to each
HumanEval problem that BREAK surface-level memorisation while keeping
the semantics intact:

    1. Rename function   (AST-based, deterministic)
    2. Permute parameter names  (AST-based, deterministic lookup)
    3. Paraphrase docstring  (one-shot LLM rewrite via Ollama)

Then run a SANITY CHECK: the decontaminated tests should still pass on
the decontaminated code. If they don't, the problem is excluded from
the subset.

Public API:
    decontaminate_problem(problem) -> DecontaminatedProblem
    build_humaneval_mutated_subset(problems, output_path, n=50)
"""

from __future__ import annotations
import ast
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

  # ast.unparse is built-in since Python 3.9 — no external 'astor' dependency

from config import LLAMA_3_2_3B, QWEN_3_6_27B, ModelSpec
import ollama_client
import closure_metrics


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DecontaminatedProblem:
    """One HumanEval problem after decontamination."""
    original_id: str
    decontam_id: str
    original_entry_point: str
    decontam_entry_point: str
    original_code: str
    decontam_code: str
    original_docstring: str
    decontam_docstring: str
    original_tests: str
    decontam_tests: str
    rename_mapping: dict[str, str] = field(default_factory=dict)
    sanity_check_passed: bool = False
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# Parameter rename table — deterministic, semantic-preserving aliases
# ──────────────────────────────────────────────────────────────────────
_PARAM_ALIASES: dict[str, str] = {
    # Strings
    "s": "text", "text": "input_text", "string": "input_string",
    "str_input": "string_value", "name": "label",
    # Numbers
    "n": "value", "num": "number_val", "val": "amount",
    "x": "operand", "y": "second_operand",
    "k": "k_value", "m": "m_value",
    # Containers
    "lst": "items", "l": "elements", "arr": "array_val",
    "li": "list_val", "ls": "values",
    "d": "mapping", "dct": "dict_val", "dict_": "dictionary",
    # Indices / counters
    "i": "idx", "j": "jdx",
    # Ordered pair
    "a": "first", "b": "second", "c": "third",
    # Booleans
    "flag": "is_set", "ok": "succeeded",
}


def _suffix_hash(name: str, length: int = 4) -> str:
    """Deterministic 4-char hex suffix based on the name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:length]


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Rename function (AST)
# ──────────────────────────────────────────────────────────────────────
def rename_function(code: str, new_name: str) -> tuple[str, str]:
    """
    Rename the top-level function (and update every self-recursive
    reference to it) using AST. Returns (rewritten_code, original_name).
    """
    tree = ast.parse(code)

    # Locate the first top-level function definition
    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
    if func_def is None:
        raise ValueError("No top-level function found")
    original_name = func_def.name
    func_def.name = new_name

    # Rename recursive calls inside the function body
    class CallRenamer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == original_name:
                node.func.id = new_name
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == original_name:
                node.id = new_name
            return node

    tree = CallRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), original_name


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Permute parameter names (AST)
# ──────────────────────────────────────────────────────────────────────
def permute_params(code: str, mapping: dict[str, str]) -> str:
    """
    Apply parameter-rename mapping {old: new} throughout the top-level
    function. Conservative: only renames names that appear as the
    declared parameter and references within the function body.
    """
    if not mapping:
        return code

    tree = ast.parse(code)
    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
    if func_def is None:
        return code

    # Capture which actual params we'll rename (only those declared on the function)
    declared = {a.arg for a in func_def.args.args}
    declared |= {a.arg for a in func_def.args.kwonlyargs}
    actual_map = {old: new for old, new in mapping.items() if old in declared}
    if not actual_map:
        return code

    # Rename argument declarations
    for a in list(func_def.args.args) + list(func_def.args.kwonlyargs):
        if a.arg in actual_map:
            a.arg = actual_map[a.arg]

    # Rename body references (only inside this function)
    class ParamRenamer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in actual_map:
                node.id = actual_map[node.id]
            return node

    func_def = ParamRenamer().visit(func_def)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def build_param_mapping(code: str) -> dict[str, str]:
    """Walk the top-level function's parameter list and propose renames
    using _PARAM_ALIASES; falls back to original name if no alias known."""
    tree = ast.parse(code)
    func_def: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
    if func_def is None:
        return {}
    mapping: dict[str, str] = {}
    for a in func_def.args.args:
        if a.arg in _PARAM_ALIASES:
            mapping[a.arg] = _PARAM_ALIASES[a.arg]
    return mapping


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Paraphrase docstring (LLM)
# ──────────────────────────────────────────────────────────────────────
_PARAPHRASE_PROMPT = """\
Rewrite the following Python docstring with DIFFERENT wording but
preserve the EXACT semantic meaning. Keep technical terms accurate;
rephrase sentence structure; do NOT add or remove behaviour.

Respond with ONLY the rewritten docstring text — no quotes, no code,
no explanation.

Original docstring:
{docstring}
"""


def paraphrase_docstring(docstring: str,
                         model: ModelSpec = QWEN_3_6_27B,
                         use_cache: bool = True) -> str:
    """
    Single LLM call to rewrite the docstring. Returns the rewritten
    text. Returns the original docstring on call failure (so the
    pipeline doesn't lose the problem just because the paraphraser
    LLM blipped).
    """
    if not docstring.strip():
        return docstring

    resp = ollama_client.call_llm(
        model,
        _PARAPHRASE_PROMPT.format(docstring=docstring),
        role_hint="decontam:paraphrase",
        temperature=0.0,           # deterministic
        max_tokens=256,
        use_cache=use_cache,
    )
    if resp.finish_reason == "error" or not resp.text.strip():
        logger.warning(f"Paraphrase failed; keeping original docstring. err={resp.error}")
        return docstring
    return resp.text.strip()


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Rewrite tests to use the new function name + params
# ──────────────────────────────────────────────────────────────────────
def rewrite_tests(tests: str, original_name: str, new_name: str,
                  param_mapping: dict[str, str]) -> str:
    """
    Rewrite the tests to call the renamed function. Parameter renames
    in the test arguments are NOT applied (tests pass positional args,
    not keywords), so we only rename function calls.
    """
    if not tests:
        return tests
    # Naive but safe: replace whole-word occurrences of original_name
    import re
    pattern = re.compile(rf"\b{re.escape(original_name)}\b")
    return pattern.sub(new_name, tests)


# ──────────────────────────────────────────────────────────────────────
# Sanity check — decontam tests should pass on decontam code
# ──────────────────────────────────────────────────────────────────────
def sanity_check(decontam_code: str, decontam_tests: str) -> bool:
    """
    Run decontam_tests against decontam_code via pytest. Returns True
    if the suite passes. Reuses closure_metrics under the hood.
    """
    if not decontam_code.strip() or not decontam_tests.strip():
        return False
    rate = closure_metrics.reference_test_pass_rate(decontam_tests, decontam_code)
    return rate == 1.0


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────
def decontaminate_problem(problem: dict,
                          paraphraser: ModelSpec = QWEN_3_6_27B,
                          use_cache: bool = True) -> DecontaminatedProblem:
    """
    Run all three decontamination transformations + sanity check.
    Returns a fully populated DecontaminatedProblem (with
    sanity_check_passed flag).

    Input `problem` dict must have:
        sample_idx, source, entry_point, code, docstring, tests
    """
    original_name = problem["entry_point"]
    original_code = problem["code"]
    original_docstring = problem.get("docstring", "")
    original_tests = problem.get("tests", "")

    # 1. New function name — deterministic suffix on the original
    new_name = f"{original_name}_v{_suffix_hash(original_name, 3)}"

    # 2. Rename function in the code
    try:
        renamed_code, _ = rename_function(original_code, new_name)
    except (SyntaxError, ValueError) as exc:
        return DecontaminatedProblem(
            original_id=problem["source"],
            decontam_id=problem["source"] + "/mutated",
            original_entry_point=original_name,
            decontam_entry_point=original_name,
            original_code=original_code,
            decontam_code=original_code,
            original_docstring=original_docstring,
            decontam_docstring=original_docstring,
            original_tests=original_tests,
            decontam_tests=original_tests,
            rename_mapping={},
            sanity_check_passed=False,
            notes=f"rename_function failed: {exc}",
        )

    # 3. Build & apply parameter rename mapping
    param_mapping = build_param_mapping(renamed_code)
    try:
        renamed_code = permute_params(renamed_code, param_mapping)
    except SyntaxError as exc:
        logger.warning(f"permute_params failed for {original_name}: {exc}")
        param_mapping = {}

    # 4. Paraphrase docstring
    new_docstring = paraphrase_docstring(original_docstring, model=paraphraser,
                                          use_cache=use_cache)

    # 5. Rewrite tests to use the new function name
    new_tests = rewrite_tests(original_tests, original_name, new_name, param_mapping)

    # 6. Sanity check
    passed = sanity_check(renamed_code, new_tests)

    full_mapping = {original_name: new_name, **param_mapping}

    return DecontaminatedProblem(
        original_id=problem["source"],
        decontam_id=problem["source"] + "/mutated",
        original_entry_point=original_name,
        decontam_entry_point=new_name,
        original_code=original_code,
        decontam_code=renamed_code,
        original_docstring=original_docstring,
        decontam_docstring=new_docstring,
        original_tests=original_tests,
        decontam_tests=new_tests,
        rename_mapping=full_mapping,
        sanity_check_passed=passed,
        notes="" if passed else "sanity_check_failed",
    )


# ──────────────────────────────────────────────────────────────────────
# Build the 50-problem HumanEval-Mutated subset
# ──────────────────────────────────────────────────────────────────────
def build_humaneval_mutated_subset(
    humaneval_problems: list[dict],
    output_path: Path,
    n: int = 50,
    seed: int = 42,
    paraphraser: ModelSpec = QWEN_3_6_27B,
    use_cache: bool = True,
) -> dict:
    """
    Sample n HumanEval problems (deterministic with seed), decontaminate
    each, EXCLUDE problems whose sanity check fails, write the resulting
    JSONL to `output_path`.

    Resumability: writes each accepted sample to disk IMMEDIATELY (line-
    by-line JSONL append with fsync), and skips any `original_id` that
    is already present in the output file. Also persists rejected IDs
    in `<output_path>.rejected.jsonl` so they aren't retried either.
    A Colab disconnect mid-build never loses an already-accepted sample.
    """
    import random

    rng = random.Random(seed)
    pool = list(humaneval_problems)
    rng.shuffle(pool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = output_path.with_suffix(output_path.suffix + ".rejected.jsonl")

    # ── Resume: load any accepted IDs already on disk ────────────────
    accepted_ids: set[str] = set()
    existing_records: list[dict] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing_records.append(rec)
                    accepted_ids.add(rec.get("original_id", ""))
                except json.JSONDecodeError:
                    continue
    rejected_ids: set[str] = set()
    rejected_reasons: dict[str, int] = {}
    if rejected_path.exists():
        with rejected_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rejected_ids.add(rec.get("original_id", ""))
                    reason = rec.get("reason", "unknown")
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                except json.JSONDecodeError:
                    continue

    kept = len(accepted_ids)
    rejected = len(rejected_ids)
    if accepted_ids or rejected_ids:
        logger.info(
            f"  Resume: {kept} accepted, {rejected} rejected already on disk; "
            f"continuing from there."
        )

    # ── Iterate, skipping seen IDs ───────────────────────────────────
    for problem in pool:
        if kept >= n:
            break
        problem_id = problem.get("source", "")
        if problem_id in accepted_ids or problem_id in rejected_ids:
            continue

        result = decontaminate_problem(problem, paraphraser=paraphraser,
                                        use_cache=use_cache)

        if result.sanity_check_passed:
            sample = {
                "sample_idx": kept,
                "source": result.decontam_id,
                "entry_point": result.decontam_entry_point,
                "signature": _extract_signature(result.decontam_code),
                "docstring": result.decontam_docstring,
                "code": result.decontam_code,
                "tests": result.decontam_tests,
                "rename_mapping": result.rename_mapping,
                "original_id": result.original_id,
            }
            _append_jsonl_line(output_path, sample)
            accepted_ids.add(problem_id)
            kept += 1
            logger.info(f"  ✓ accepted [{kept}/{n}]: {result.original_id}")
        else:
            reason = result.notes or "unknown"
            _append_jsonl_line(rejected_path, {
                "original_id": problem_id,
                "reason": reason,
            })
            rejected_ids.add(problem_id)
            rejected += 1
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            logger.info(f"  ✗ REJECTED {result.original_id}: {reason}")

    summary = {
        "n_requested": n,
        "n_kept": kept,
        "n_rejected": rejected,
        "rejected_reasons": rejected_reasons,
        "n_total_attempted": kept + rejected,
        "output_path": str(output_path),
        "rejected_path": str(rejected_path),
    }
    logger.info(
        f"\n  HumanEval-Mutated subset: kept={kept}, rejected={rejected}, "
        f"yield={kept / max(kept + rejected, 1):.1%}"
    )
    return summary


def _append_jsonl_line(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file with flush+fsync (durability)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _extract_signature(code: str) -> str:
    """Return the first `def …:` line of the code."""
    for line in code.split("\n"):
        if line.lstrip().startswith("def "):
            return line.rstrip(":") + ":"
    return ""


# ──────────────────────────────────────────────────────────────────────
# Sanity self-test (no Ollama needed for stages 1+2)
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:                                         # pragma: no cover
    print("=== decontaminate self-test ===\n")

    sample_code = (
        "def is_palindrome(s: str) -> bool:\n"
        "    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n"
        "    return s == s[::-1]\n"
    )

    print("1) rename_function…")
    new_code, orig = rename_function(sample_code, "is_palindrome_v2")
    assert "def is_palindrome_v2" in new_code
    assert orig == "is_palindrome"
    print(f"   ✓ renamed: {orig} -> is_palindrome_v2")

    print("\n2) build_param_mapping…")
    mapping = build_param_mapping(sample_code)
    print(f"   mapping = {mapping}")
    assert mapping.get("s") == "text"

    print("\n3) permute_params…")
    permuted = permute_params(sample_code, mapping)
    assert "(text" in permuted, f"Expected param 'text', got:\n{permuted}"
    assert "s[::-1]" not in permuted, "Old param ref still present"
    print(f"   ✓ permuted: s -> text")

    print("\n4) rewrite_tests…")
    tests = "def test_pal():\n    assert is_palindrome('racecar') is True\n"
    new_tests = rewrite_tests(tests, "is_palindrome", "is_palindrome_v2", mapping)
    assert "is_palindrome_v2('racecar')" in new_tests
    assert "is_palindrome('" not in new_tests.replace("is_palindrome_v2", "")
    print(f"   ✓ tests rewritten")

    print("\n5) End-to-end sanity check (no LLM paraphrase, just AST stages)…")
    problem = {
        "sample_idx": 0,
        "source": "smoke/is_palindrome",
        "entry_point": "is_palindrome",
        "code": sample_code,
        "docstring": "Return True if s reads the same forwards and backwards.",
        "tests": tests,
    }
    # Use llama3.2:3b for paraphrasing since it's pulled
    try:
        result = decontaminate_problem(problem, paraphraser=LLAMA_3_2_3B)
    except RuntimeError as exc:
        print(f"   SKIPPED — Ollama unavailable: {exc}")
        return
    print(f"   original name: {result.original_entry_point}")
    print(f"   new name:      {result.decontam_entry_point}")
    print(f"   original doc:  {result.original_docstring[:60]}")
    print(f"   new doc:       {result.decontam_docstring[:60]}")
    print(f"   sanity check:  {'✓ passed' if result.sanity_check_passed else '✗ FAILED'}")
    print(f"   notes:         {result.notes or '(none)'}")
    assert result.sanity_check_passed, "Sanity check should pass on is_palindrome"

    print("\n✓ decontaminate self-test passed.")


if __name__ == "__main__":
    _self_test()
