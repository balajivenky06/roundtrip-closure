"""
mutation_testing.py — Mutation testing analysis for generated unit tests.

Measures the defect-detection capability of LLM-generated tests by running them
against systematically mutated versions of the function under test.

This addresses the EMSE reviewer feedback: "Does the generated test oracle
actually detect software defects?" — a concrete SE-relevant metric beyond
automated proxy scores.

Mutation operators (adapted from mutmut / PIT):
    - Arithmetic: +→-, -→+, *→/, /→*
    - Comparison: ==→!=, !=→==, <→>=, >→<=, <=→>, >=→<
    - Boundary:   +1, -1 on integer literals
    - Return:     return x → return None
    - Negate:     True→False, False→True
    - Remove:     delete one statement from function body

Metrics produced:
    - mutation_kill_rate: fraction of mutants detected (killed) by generated tests
    - mutation_score:     alias for kill rate (standard terminology)
    - survived_mutants:   count of mutants NOT detected (weakness indicator)
    - equivalent_mutants: mutants that pass ground_truth tests too (false positives)

Usage:
    # From Colab (with checkpoints):
    python mutation_testing.py --checkpoints-dir checkpoints/

    # Local re-generation (small subset):
    python mutation_testing.py --regenerate --max-samples 10

    # Analyze existing mutation results:
    python mutation_testing.py --results-only

Output:
    results_mutation.tsv           — per-method/model mutation kill rates
    plots_mutation/                — mutation analysis charts
    plots_mutation/mutation_report.txt — detailed report for thesis
"""

import ast
import copy
import os
import re
import sys
import pickle
import random
import argparse
import time
import tempfile
import subprocess
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Match prepare_unitest.py: Colab uses /content/.cache, elsewhere ~/.cache
_IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
CACHE_DIR = Path(os.environ.get("CACHE_DIR",
    Path("/content/.cache/autoresearch_unitest") if _IN_COLAB
    else Path.home() / ".cache" / "autoresearch_unitest"))
DATASET_CACHE = CACHE_DIR / "eval_dataset_v3.pkl"

OUTPUT_DIR = Path("plots_mutation")
RESULTS_FILE = Path("results_mutation.tsv")
MAIN_RESULTS = Path("results_unitest.tsv")

# Per-sample analysis-resume directory. mutation_testing.py writes here, but
# Drive-synced copies of the same files often land under the no-dot variant.
# _resolve_analysis_dir() prefers the dotted path (Colab runtime default) and
# falls back to the no-dot path if that already has data, so a local re-run
# can pick up where a Colab run left off without manual renaming.
_ANALYSIS_CKPT_DOT    = Path(".checkpoints_mutation_analysis")
_ANALYSIS_CKPT_NO_DOT = Path("checkpoints_mutation_analysis")


def _resolve_analysis_dir() -> Path:
    if _ANALYSIS_CKPT_DOT.is_dir():
        return _ANALYSIS_CKPT_DOT
    if _ANALYSIS_CKPT_NO_DOT.is_dir():
        return _ANALYSIS_CKPT_NO_DOT
    return _ANALYSIS_CKPT_DOT   # will be created on first write


ANALYSIS_CKPT_DIR = _resolve_analysis_dir()

MAX_MUTANTS_PER_FUNCTION = 15   # cap to keep runtime manageable
TIMEOUT_PER_TEST = 10           # seconds per pytest run

# ---------------------------------------------------------------------------
# Mutation operators (AST-based)
# ---------------------------------------------------------------------------

class ArithmeticMutator(ast.NodeTransformer):
    """Replace arithmetic operators: + ↔ -, * ↔ /"""
    SWAPS = {
        ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.FloorDiv: ast.Mult,
        ast.Mod: ast.Add,
    }

    def __init__(self, target_idx=0):
        self.target_idx = target_idx
        self.current_idx = 0
        self.mutated = False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        op_type = type(node.op)
        if op_type in self.SWAPS:
            if self.current_idx == self.target_idx:
                node.op = self.SWAPS[op_type]()
                self.mutated = True
            self.current_idx += 1
        return node


class ComparisonMutator(ast.NodeTransformer):
    """Replace comparison operators: == ↔ !=, < ↔ >=, > ↔ <="""
    SWAPS = {
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Lt: ast.GtE, ast.GtE: ast.Lt,
        ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    }

    def __init__(self, target_idx=0):
        self.target_idx = target_idx
        self.current_idx = 0
        self.mutated = False

    def visit_Compare(self, node):
        self.generic_visit(node)
        new_ops = []
        for op in node.ops:
            op_type = type(op)
            if op_type in self.SWAPS and self.current_idx == self.target_idx:
                new_ops.append(self.SWAPS[op_type]())
                self.mutated = True
            else:
                new_ops.append(op)
            if op_type in self.SWAPS:
                self.current_idx += 1
        node.ops = new_ops
        return node


class BoundaryMutator(ast.NodeTransformer):
    """Modify integer constants by ±1"""

    def __init__(self, target_idx=0):
        self.target_idx = target_idx
        self.current_idx = 0
        self.mutated = False

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self.current_idx == self.target_idx:
                node.value = node.value + 1
                self.mutated = True
            self.current_idx += 1
        return node


class ReturnMutator(ast.NodeTransformer):
    """Replace return value with None"""

    def __init__(self, target_idx=0):
        self.target_idx = target_idx
        self.current_idx = 0
        self.mutated = False

    def visit_Return(self, node):
        self.generic_visit(node)
        if node.value is not None:
            if self.current_idx == self.target_idx:
                node.value = ast.Constant(value=None)
                self.mutated = True
            self.current_idx += 1
        return node


class NegateBoolMutator(ast.NodeTransformer):
    """Swap True ↔ False"""

    def __init__(self, target_idx=0):
        self.target_idx = target_idx
        self.current_idx = 0
        self.mutated = False

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            if self.current_idx == self.target_idx:
                node.value = not node.value
                self.mutated = True
            self.current_idx += 1
        return node


# ---------------------------------------------------------------------------
# Mutant generation
# ---------------------------------------------------------------------------

def _count_targets(mutator_class, tree):
    """Count how many mutation targets exist for a given mutator."""
    counter = mutator_class(target_idx=999999)
    counter.visit(copy.deepcopy(tree))
    return counter.current_idx


def generate_mutants(function_code: str) -> list:
    """
    Generate mutated versions of a function.
    Returns list of (mutant_code: str, mutation_description: str).
    """
    try:
        tree = ast.parse(function_code)
    except SyntaxError:
        return []

    mutators = [
        (ArithmeticMutator, "arithmetic"),
        (ComparisonMutator, "comparison"),
        (BoundaryMutator, "boundary"),
        (ReturnMutator, "return_none"),
        (NegateBoolMutator, "negate_bool"),
    ]

    mutants = []
    for mutator_class, label in mutators:
        n_targets = _count_targets(mutator_class, tree)
        for idx in range(n_targets):
            tree_copy = copy.deepcopy(tree)
            m = mutator_class(target_idx=idx)
            mutated_tree = m.visit(tree_copy)
            if m.mutated:
                try:
                    ast.fix_missing_locations(mutated_tree)
                    code = ast.unparse(mutated_tree)
                    if code.strip() != function_code.strip():
                        mutants.append((code, f"{label}_{idx}"))
                except Exception:
                    continue

    # Cap mutants
    if len(mutants) > MAX_MUTANTS_PER_FUNCTION:
        rng = random.Random(42)
        mutants = rng.sample(mutants, MAX_MUTANTS_PER_FUNCTION)

    return mutants


# ---------------------------------------------------------------------------
# Test execution against mutants
# ---------------------------------------------------------------------------

def _wrap_bare_asserts(test_code: str) -> str:
    """Wrap bare assert statements in a test function for pytest compatibility.

    MBPP/HumanEval ground truth tests are often bare assert lines or wrapped
    in a ``check(candidate)`` function. This normalises them into a pytest-
    compatible ``def test_wrapped():`` function so the subprocess runner can
    execute them uniformly.
    """
    lines = test_code.strip().splitlines()
    # If already has def test_ functions, return as-is
    if any(l.strip().startswith("def test_") for l in lines):
        return test_code

    # HumanEval format: METADATA dict + def check(candidate): ...
    if "def check(candidate)" in test_code:
        # Call check with the function name (extracted at call site)
        return test_code

    # Bare asserts: wrap in a test function
    wrapped_lines = ["def test_wrapped():"]
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("assert") or stripped.startswith("print"):
            wrapped_lines.append(f"    {stripped}")
        elif stripped.startswith("def ") or stripped.startswith("class "):
            # Helper function — keep at module level
            wrapped_lines.insert(0, line)
        else:
            wrapped_lines.append(f"    {stripped}")
    return "\n".join(wrapped_lines)


def _strip_function_redefinition(test_code: str, fn_name: str) -> str:
    """Remove top-level `def <fn_name>(...)` blocks from test_code.

    Some LLMs (notably phi4) prepend a redefinition of the function under test
    to their generated tests. When the harness writes function_code (the mutant)
    followed by test_code, that redefinition shadows the mutant in Python's
    module namespace, so tests call the pristine version and no mutations are
    ever detected. Strip the redefinition so the mutant is the only version
    in scope.
    """
    if not fn_name or not test_code.strip():
        return test_code
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        # Regex fallback: strip "def fn_name(...): ... <until next top-level line>"
        pat = re.compile(
            rf"^def\s+{re.escape(fn_name)}\s*\([^)]*\)[^:]*:.*?(?=^\S|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        return pat.sub("", test_code)

    keep = [n for n in tree.body
            if not (isinstance(n, ast.FunctionDef) and n.name == fn_name)]
    if len(keep) == len(tree.body):
        return test_code

    try:
        return ast.unparse(ast.Module(body=keep, type_ignores=[]))
    except Exception:
        pat = re.compile(
            rf"^def\s+{re.escape(fn_name)}\s*\([^)]*\)[^:]*:.*?(?=^\S|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        return pat.sub("", test_code)


def run_tests_against_code(test_code: str, function_code: str,
                           timeout: int = TIMEOUT_PER_TEST,
                           use_cache: bool = True) -> str:
    """
    Run test_code against function_code in isolated subprocess.
    Returns: "pass" | "fail" | "error" | "timeout"

    When use_cache is True (default), the result is served from
    pytest_cache (SHA-256 keyed on test_code + function_code + timeout).
    This is the single biggest checkpointing improvement for Chapter 3:
    evaluate_mutants runs ~15 pytest subprocesses per function; if Colab
    disconnects mid-evaluation, we now only re-execute the in-flight
    mutant on resume — the rest are free disk lookups.

    Set use_cache=False to force a fresh subprocess (useful for
    diagnosing environment drift).
    """
    if not test_code or not test_code.strip():
        return "error"

    # Cache lookup — done BEFORE any of the parsing/cleanup steps because
    # the cleanup is deterministic from (test_code, function_code).
    if use_cache:
        try:
            import pytest_cache
            cache_key = pytest_cache.make_key(test_code, function_code, timeout)
            cached = pytest_cache.get(cache_key)
            if cached is not None:
                return cached
        except ImportError:
            # pytest_cache module not on path — fall through to live execution
            pytest_cache = None
            cache_key = None
    else:
        pytest_cache = None
        cache_key = None

    # Wrap bare asserts for pytest compatibility
    test_code = _wrap_bare_asserts(test_code)

    # Extract function name and strip any LLM-prepended redefinition that
    # would shadow function_code (the possibly-mutated version we're testing).
    fn_match = re.search(r"def\s+(\w+)\s*\(", function_code)
    fn_name = fn_match.group(1) if fn_match else ""
    if fn_name:
        test_code = _strip_function_redefinition(test_code, fn_name)

    # Handle HumanEval check(candidate) format
    if "def check(candidate)" in test_code and fn_name:
        test_code = test_code + f"\n\ndef test_humaneval():\n    check({fn_name})\n"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_mut.py")
            content = (
                f"# --- Function under test (possibly mutated) ---\n"
                f"{function_code}\n\n"
                f"# --- Generated tests ---\n"
                f"{test_code}\n"
            )
            with open(test_file, "w") as f:
                f.write(content)

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", test_file,
                 "--tb=no", "-q", "--no-header"],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir,
            )
            output = proc.stdout + proc.stderr

            passed = failed = errors = 0
            for match in re.finditer(r"(\d+)\s+passed", output):
                passed = int(match.group(1))
            for match in re.finditer(r"(\d+)\s+failed", output):
                failed = int(match.group(1))
            for match in re.finditer(r"(\d+)\s+error", output):
                errors = int(match.group(1))

            total = passed + failed + errors
            if total == 0:
                result = "error"
            elif failed > 0 or errors > 0:
                result = "fail"
            else:
                result = "pass"

    except subprocess.TimeoutExpired:
        result = "timeout"
    except Exception:
        result = "error"

    # Persist the result before returning so the next Colab session
    # can serve it from disk if the same (test, function) pair recurs.
    if pytest_cache is not None and cache_key is not None:
        try:
            pytest_cache.put(cache_key, result)
        except Exception as exc:                                  # pragma: no cover
            print(f"  [pytest_cache.put failed: {exc}]")

    return result


def _filter_passing_tests(test_code: str, function_code: str,
                          timeout: int = TIMEOUT_PER_TEST) -> str:
    """Filter generated test code to only include tests that pass on original code.

    LLMs often generate a mix of correct and spurious tests (e.g. expecting
    ValueError when the function raises IndexError).  Standard mutation testing
    practice is to discard failing tests before measuring kill rate.

    Returns filtered test code, or empty string if no tests pass.
    """
    lines = test_code.strip().splitlines()

    # Extract individual test functions
    test_blocks = []
    preamble_lines = []   # imports, helpers, etc.
    current_block = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def test_"):
            if current_block is not None:
                test_blocks.append(current_block)
            current_block = [line]
        elif current_block is not None:
            # Continuation of current test (indented or blank)
            if line and not line[0].isspace() and stripped and not stripped.startswith("#"):
                # New top-level definition — end current block
                test_blocks.append(current_block)
                current_block = None
                preamble_lines.append(line)
            else:
                current_block.append(line)
        else:
            preamble_lines.append(line)

    if current_block is not None:
        test_blocks.append(current_block)

    if not test_blocks:
        return test_code  # no test functions found — return as-is

    preamble = "\n".join(preamble_lines)
    passing_blocks = []

    for block in test_blocks:
        block_code = "\n".join(block)
        candidate = f"{preamble}\n\n{block_code}\n"
        result = run_tests_against_code(candidate, function_code, timeout)
        if result == "pass":
            passing_blocks.append(block_code)

    if not passing_blocks:
        return ""

    return preamble + "\n\n" + "\n\n".join(passing_blocks) + "\n"


def evaluate_mutants(function_code: str, test_code: str,
                     ground_truth_tests: str = "") -> dict:
    """
    Run generated tests against all mutants of a function.

    Returns dict with:
        total_mutants, killed, survived, equivalent, kill_rate,
        per_operator: {operator: {total, killed}}
    """
    mutants = generate_mutants(function_code)

    # per_operator is a plain dict (lazily filled via _bump) so the whole
    # result is pickleable for the per-sample analysis-resume checkpoint.
    per_operator: dict = {}

    def _bump(op: str, field: str, by: int = 1) -> None:
        slot = per_operator.setdefault(op, {"total": 0, "killed": 0})
        slot[field] += by

    result = {
        "total_mutants": len(mutants),
        "killed": 0,
        "survived": 0,
        "equivalent": 0,
        "errors": 0,
        "per_operator": per_operator,
    }

    if not mutants:
        result["kill_rate"] = float("nan")
        return result

    # First check: does the test pass on original code?
    original_result = run_tests_against_code(test_code, function_code)
    if original_result != "pass":
        # Filter to only passing tests (standard mutation testing practice)
        test_code = _filter_passing_tests(test_code, function_code)
        if not test_code:
            result["kill_rate"] = float("nan")
            result["original_status"] = "all_tests_fail"
            return result

    for mutant_code, description in mutants:
        operator = description.rsplit("_", 1)[0]
        _bump(operator, "total")

        mutant_result = run_tests_against_code(test_code, mutant_code)

        if mutant_result == "fail":
            # Mutant killed — test detected the defect
            result["killed"] += 1
            _bump(operator, "killed")
        elif mutant_result == "pass":
            # Check if this is an equivalent mutant (ground truth also passes)
            if ground_truth_tests:
                gt_result = run_tests_against_code(ground_truth_tests, mutant_code)
                if gt_result == "pass":
                    result["equivalent"] += 1
                else:
                    result["survived"] += 1
            else:
                result["survived"] += 1
        else:
            result["errors"] += 1

    non_equivalent = result["total_mutants"] - result["equivalent"]
    if non_equivalent > 0:
        result["kill_rate"] = round(result["killed"] / non_equivalent, 6)
    else:
        result["kill_rate"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Checkpoint loading (from Colab Drive)
# ---------------------------------------------------------------------------

def load_checkpoints(checkpoints_dir: str) -> dict:
    """
    Load per-experiment checkpoint files from Colab.

    train_unitest.py checkpoint format:
        {"metrics_list": [...], "step": int, "method": str, "reasoning": str, "model": str}
    Each item in metrics_list has: generated_tests, function_code, ground_truth_tests, source, ...

    Returns: {method_reasoning_model_key: [sample_dicts]}
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        print(f"WARNING: Checkpoints directory not found: {ckpt_dir}")
        return {}

    results = {}
    for pkl_file in sorted(ckpt_dir.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            key = pkl_file.stem

            if isinstance(data, dict) and "metrics_list" in data:
                # train_unitest.py checkpoint format
                ml = data["metrics_list"]
                method = data.get("method", "unknown")
                reasoning = data.get("reasoning", "base")
                model = data.get("model", "unknown")
                key = f"{method}_{reasoning}_{model.replace(':', '_')}"

                # Check if generated_tests are present
                has_tests = any("generated_tests" in m for m in ml)
                if not has_tests:
                    print(f"  Skipping {pkl_file.name}: no generated_tests field "
                          f"(run with updated train_unitest.py)")
                    continue

                samples = []
                for i, m in enumerate(ml):
                    samples.append({
                        "sample_idx": i,
                        "generated_tests": m.get("generated_tests", ""),
                        "function_code": m.get("function_code", ""),
                        "ground_truth_tests": m.get("ground_truth_tests", ""),
                        "source": m.get("source", "unknown"),
                        "method": method,
                        "reasoning": reasoning,
                        "model": model,
                    })
                results[key] = samples
                print(f"  {pkl_file.name}: {len(samples)} samples "
                      f"({method}/{reasoning}, {model})")

            elif isinstance(data, list):
                results[key] = data
            elif isinstance(data, dict) and "samples" in data:
                results[key] = data["samples"]
            else:
                print(f"  Skipping {pkl_file.name}: unexpected format")
        except Exception as e:
            print(f"  Error loading {pkl_file.name}: {e}")

    print(f"Loaded {len(results)} checkpoint files from {ckpt_dir}")
    return results


# ---------------------------------------------------------------------------
# Re-generation mode (for local runs without checkpoints)
# ---------------------------------------------------------------------------

def _ollama_alive(timeout_secs: float = 3.0) -> bool:
    """Return True if the Ollama daemon responds within timeout_secs."""
    try:
        import ollama as _ollama
        # Use a thread to enforce a wall-clock timeout (the ollama client
        # doesn't accept a per-call timeout).
        import threading
        result = {"ok": False}
        def _probe():
            try:
                _ollama.list()
                result["ok"] = True
            except Exception:
                pass
        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join(timeout_secs)
        return result["ok"]
    except Exception:
        return False


def _wait_for_ollama(max_wait_secs: float = 120.0, poll_secs: float = 3.0) -> bool:
    """Block until Ollama is reachable or max_wait_secs elapses. Returns True on success."""
    deadline = time.time() + max_wait_secs
    while time.time() < deadline:
        if _ollama_alive():
            return True
        time.sleep(poll_secs)
    return False


def regenerate_tests(dataset: list, max_samples: int = 10,
                     model: str = None, methods: list = None) -> dict:
    """
    Re-generate tests for mutation testing — lightweight mode.
    Only generates test code, no full evaluation pipeline.

    Uses Ollama to generate tests for each method/reasoning combo.
    Saves results in checkpoint-compatible format for re-use.

    Parameters
    ----------
    dataset : list     — eval dataset samples
    max_samples : int  — max samples per method (default 10)
    model : str        — Ollama model to use (default: first available)
    methods : list     — list of (method, reasoning) tuples to test
    """
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        import train_unitest as tu
    except ImportError:
        print("ERROR: train_unitest.py not found")
        return {}

    from prepare_unitest import build_knowledge_base

    # Determine model
    if model is None:
        try:
            import ollama as _ollama
            available = [m.model for m in _ollama.list().models]
            # Prefer experiment models, fall back to anything available
            preferred = ["llama3.2:latest", "phi4:14b", "qwen3-coder:30b",
                         "qwen3.5:9b", "llama3.2:1b"]
            model = next((m for m in preferred if m in available), None)
            if model is None and available:
                model = available[0]
            if model is None:
                print("ERROR: No Ollama models available")
                return {}
        except Exception:
            model = "llama3.2:latest"
    print(f"  Using model: {model}")

    # Set model in train_unitest module
    tu.GENERATOR_MODEL = model
    tu.HELPER_MODEL = model

    # Build knowledge base for RAG methods
    kb, emb_model = build_knowledge_base()
    tu._kb = kb
    tu._emb_model = emb_model

    if methods is None:
        methods = [
            ("plain_llm", "base"),
            ("random_rag", "base"),
            ("simple_rag", "base"),
            ("iterative_critique", "base"),
        ]

    results = {}
    samples = dataset[:max_samples]
    ckpt_dir = Path(".checkpoints_mutation")
    ckpt_dir.mkdir(exist_ok=True)

    for method, reasoning in methods:
        key = f"{method}_{reasoning}_{model.replace(':', '_')}"
        ckpt_file = ckpt_dir / f"{key}.pkl"

        # Resume from saved checkpoint if exists
        if ckpt_file.exists():
            try:
                with open(ckpt_file, "rb") as f:
                    cached = pickle.load(f)
                if len(cached) >= len(samples):
                    print(f"\n  {key}: loaded {len(cached)} samples from cache")
                    results[key] = cached
                    continue
                else:
                    print(f"\n  {key}: resuming from sample {len(cached)}/{len(samples)}")
                    sample_results = cached
                    start_idx = len(cached)
            except Exception:
                sample_results = []
                start_idx = 0
        else:
            sample_results = []
            start_idx = 0

        # Set method/reasoning in module
        tu.METHOD = method
        tu.REASONING = reasoning

        gen_key = (method, reasoning)
        generator = tu.GENERATORS.get(gen_key)
        if not generator:
            print(f"    Generator not found for {gen_key}")
            continue

        print(f"\n  Generating: {key} ({len(samples) - start_idx} remaining)...")

        for i in range(start_idx, len(samples)):
            sample = samples[i]
            t0 = time.time()
            # Reset per-sample diagnostics
            tu._noise_rate_buf = []
            tu._last_context = ""
            tu._retrieval_secs = 0.0
            tu._llm_secs = 0.0
            tu._tokens_used = 0

            # Per-sample retry: an empty result usually means Ollama hiccupped
            # (the train_unitest._llm wrapper swallows "Failed to connect" and
            # returns ""). Retry up to 3 times with increasing waits, after
            # confirming Ollama is back online.
            generated = ""
            for attempt in range(3):
                try:
                    generated = generator(sample["function_code"])
                except Exception as e:
                    print(f"    Sample {i} attempt {attempt+1} raised: {e}")
                    generated = ""

                if generated and generated.strip():
                    break

                # Empty result — wait for Ollama and retry
                wait_secs = 5 * (attempt + 1)
                print(f"    Sample {i} empty generation (attempt {attempt+1}/3); "
                      f"waiting up to {wait_secs}s for Ollama...", flush=True)
                if not _wait_for_ollama(max_wait_secs=wait_secs):
                    print(f"    Ollama still unreachable after {wait_secs}s")

            dt = time.time() - t0
            sample_results.append({
                "sample_idx": i,
                "task_id": sample.get("task_id", f"sample_{i}"),
                "source": sample.get("source", "unknown"),
                "function_code": sample["function_code"],
                "ground_truth_tests": sample.get("ground_truth_tests", ""),
                "generated_tests": generated,
                "method": method,
                "reasoning": reasoning,
                "model": model,
            })

            print(f"\r    {i+1}/{len(samples)} ({dt:.1f}s)", end="", flush=True)

            # Save after every sample (resume-safe)
            with open(ckpt_file, "wb") as f:
                pickle.dump(sample_results, f)

        print(f"\n    {len(sample_results)} samples generated → {ckpt_file}")
        results[key] = sample_results

    return results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_mutation_analysis(checkpoint_data: dict, dataset: list) -> pd.DataFrame:
    """
    Run mutation testing on all checkpoint data.
    Returns DataFrame with per-method/model mutation kill rates.
    """
    rows = []

    ANALYSIS_CKPT_DIR.mkdir(exist_ok=True)

    for key, samples in checkpoint_data.items():
        print(f"\n  Analyzing: {key} ({len(samples)} samples)...")

        # Load any prior analysis state for this key. Each entry maps
        # sample_idx -> result dict from evaluate_mutants(). On disconnect,
        # the next run picks up from where it left off; we only re-do the
        # in-flight sample (which got partially mutated, never persisted).
        analysis_ckpt = ANALYSIS_CKPT_DIR / f"{key}.pkl"
        if analysis_ckpt.exists():
            try:
                with open(analysis_ckpt, "rb") as f:
                    completed = pickle.load(f)
                if completed:
                    print(f"    Resuming from {len(completed)}/{len(samples)} samples")
            except Exception:
                completed = {}
        else:
            completed = {}

        for i, sample in enumerate(samples):
            if i in completed:
                continue   # already analyzed in a prior run

            generated = sample.get("generated_tests", "")
            func_code = sample.get("function_code", "")
            gt_tests = sample.get("ground_truth_tests", "")

            if not func_code:
                # Try to find function_code from dataset
                idx = sample.get("sample_idx", i)
                if idx < len(dataset):
                    func_code = dataset[idx]["function_code"]
                    gt_tests = dataset[idx].get("ground_truth_tests", "")

            if not generated or not func_code:
                # Mark as skipped so a resume doesn't try this sample again
                completed[i] = {"kill_rate": float("nan"), "killed": 0,
                                "total_mutants": 0, "survived": 0,
                                "equivalent": 0, "per_operator": {}}
            else:
                completed[i] = evaluate_mutants(func_code, generated, gt_tests)

            # Persist after every sample (resume-safe). Tempfile + rename to
            # avoid leaving a half-written pkl if the runtime dies mid-write.
            tmp_path = analysis_ckpt.with_suffix(".pkl.tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(completed, f)
            tmp_path.replace(analysis_ckpt)

            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(samples)} done", end="\r")

        # Aggregate from the full completed map (works the same whether the
        # run finished in one go or resumed from a checkpoint).
        kill_rates = []
        total_killed = 0
        total_mutants = 0
        total_survived = 0
        total_equivalent = 0
        operator_stats = defaultdict(lambda: {"total": 0, "killed": 0})

        for result in completed.values():
            if math.isnan(result.get("kill_rate", float("nan"))):
                continue
            kill_rates.append(result["kill_rate"])
            total_killed += result["killed"]
            total_mutants += result["total_mutants"]
            total_survived += result["survived"]
            total_equivalent += result["equivalent"]
            for op, stats in result.get("per_operator", {}).items():
                operator_stats[op]["total"] += stats["total"]
                operator_stats[op]["killed"] += stats["killed"]

        if kill_rates:
            KNOWN_METHODS = {
                "plain_llm": "Plain LLM",
                "random_rag": "Random RAG",
                "simple_rag": "Simple RAG",
                "iterative_critique": "Iterative Critique",
            }
            # Prefer per-sample metadata (set by load_checkpoints / regenerate_tests).
            # Fall back to parsing the key for backwards compatibility.
            first = samples[0] if samples else {}
            method_raw = first.get("method") or next(
                (m for m in KNOWN_METHODS if key.startswith(m)), key
            )
            reasoning = first.get("reasoning", "")
            model = first.get("model", "")
            method_label = KNOWN_METHODS.get(method_raw, method_raw)

            mean_kill = np.mean(kill_rates)
            std_kill = np.std(kill_rates, ddof=1) if len(kill_rates) > 1 else 0.0

            row = {
                "method": method_label,
                "reasoning": reasoning,
                "model": model,
                "mean_kill_rate": round(mean_kill, 6),
                "std_kill_rate": round(std_kill, 6),
                "median_kill_rate": round(np.median(kill_rates), 6),
                "total_mutants": total_mutants,
                "total_killed": total_killed,
                "total_survived": total_survived,
                "total_equivalent": total_equivalent,
                "n_samples_valid": len(kill_rates),
            }

            # Per-operator kill rates
            for op in sorted(operator_stats.keys()):
                op_total = operator_stats[op]["total"]
                op_killed = operator_stats[op]["killed"]
                row[f"kill_{op}"] = round(op_killed / op_total, 4) if op_total > 0 else float("nan")

            rows.append(row)
            print(f"    {key}: kill_rate={mean_kill:.3f} ± {std_kill:.3f}"
                  f" ({total_killed}/{total_mutants-total_equivalent} mutants, "
                  f"{total_equivalent} equivalent)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_mutation_results(df: pd.DataFrame) -> None:
    """Generate mutation testing charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)

    if df.empty:
        print("  No mutation data to plot.")
        return

    # 1. Kill rate by method
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(df)), 6))
    # Build informative labels: "Method" or "Method (model)" or "Method (model/reasoning)"
    has_model = "model" in df.columns and df["model"].astype(str).str.len().gt(0).any()
    has_reasoning = "reasoning" in df.columns and df["reasoning"].astype(str).str.len().gt(0).any()

    def _label(row):
        parts = [str(row["method"])]
        suffix = []
        if has_model and row.get("model"):
            suffix.append(str(row["model"]))
        if has_reasoning and row.get("reasoning"):
            suffix.append(str(row["reasoning"]))
        if suffix:
            parts.append(f"({' / '.join(suffix)})")
        return " ".join(parts)

    methods = [_label(r) for _, r in df.iterrows()]
    kill_rates = df["mean_kill_rate"].values
    stds = df["std_kill_rate"].values

    method_colors = {
        "Plain LLM": "#4C72B0", "Random RAG": "#8172B2",
        "Simple RAG": "#DD8452", "Iterative Critique": "#55A868",
    }
    bar_colors = [method_colors.get(str(m), "#999999") for m in df["method"].values]

    bars = ax.bar(range(len(methods)), kill_rates, yerr=stds,
                  color=bar_colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mutation Kill Rate", fontsize=12)
    ax.set_title("Mutation Kill Rate by Method\n(Higher = tests detect more injected defects)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend()

    for i, (kr, std) in enumerate(zip(kill_rates, stds)):
        ax.text(i, kr + std + 0.02, f"{kr:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "kill_rate_by_method.png", dpi=150)
    plt.close(fig)
    print("  kill_rate_by_method.png")

    # 2. Per-operator kill rates (if available)
    op_cols = [c for c in df.columns if c.startswith("kill_")]
    if op_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(op_cols))
        width = 0.8 / len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row[c] for c in op_cols]
            ax.bar(x + i * width, vals, width, label=row["method"],
                   color=bar_colors[i] if i < len(bar_colors) else "#999999")

        ax.set_xticks(x + width * len(df) / 2)
        ax.set_xticklabels([c.replace("kill_", "") for c in op_cols],
                           rotation=45, ha="right")
        ax.set_ylabel("Kill Rate")
        ax.set_title("Mutation Kill Rate by Operator Type", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "kill_rate_by_operator.png", dpi=150)
        plt.close(fig)
        print("  kill_rate_by_operator.png")


def write_mutation_report(df: pd.DataFrame) -> None:
    """Write detailed mutation testing report."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append("  MUTATION TESTING REPORT — Unit Test Generation")
    lines.append("  SE-Relevant Metric: Do generated tests detect injected defects?")
    lines.append("=" * 70)

    if df.empty:
        lines.append("\nNo mutation testing data available.")
        report_path = OUTPUT_DIR / "mutation_report.txt"
        report_path.write_text("\n".join(lines))
        return

    lines.append(f"\nExperiments analyzed: {len(df)}")
    lines.append(f"Total mutants generated: {df['total_mutants'].sum()}")
    lines.append(f"Total mutants killed: {df['total_killed'].sum()}")
    lines.append(f"Total equivalent mutants: {df['total_equivalent'].sum()}")

    def _row_label(r):
        parts = [str(r['method'])]
        bits = []
        if 'model' in r and pd.notna(r.get('model', '')) and str(r.get('model', '')):
            bits.append(str(r['model']))
        if 'reasoning' in r and pd.notna(r.get('reasoning', '')) and str(r.get('reasoning', '')):
            bits.append(str(r['reasoning']))
        if bits:
            parts.append(f"({'/'.join(bits)})")
        return ' '.join(parts)

    lines.append(f"\n{'Method (model/reasoning)':<45} {'Kill Rate':>12} {'± Std':>10} {'Killed':>8} {'Total':>8} {'Equiv':>8} {'N':>6}")
    lines.append("-" * 100)
    for _, row in df.iterrows():
        lines.append(
            f"  {_row_label(row):<43} {row['mean_kill_rate']:>10.4f}"
            f" {row['std_kill_rate']:>10.4f}"
            f" {row['total_killed']:>8} {row['total_mutants']:>8}"
            f" {row['total_equivalent']:>8} {row['n_samples_valid']:>6}"
        )

    # Per-operator breakdown
    op_cols = [c for c in df.columns if c.startswith("kill_")]
    if op_cols:
        lines.append(f"\nPer-Operator Kill Rates:")
        lines.append(f"{'Method':<35}" + "".join(f" {c.replace('kill_',''):>12}" for c in op_cols))
        lines.append("-" * (35 + 13 * len(op_cols)))
        for _, row in df.iterrows():
            vals = "".join(f" {row[c]:>12.4f}" if not math.isnan(row[c]) else f" {'N/A':>12}" for c in op_cols)
            lines.append(f"  {_row_label(row):<43}{vals}")

    # Statistical comparison
    if len(df) >= 2:
        lines.append(f"\nStatistical Comparison:")
        from scipy.stats import mannwhitneyu
        best_idx = df["mean_kill_rate"].idxmax()
        best = df.loc[best_idx]
        lines.append(f"  Best method: {best['method']} (kill_rate={best['mean_kill_rate']:.4f})")

    lines.append("\n" + "=" * 70)
    lines.append("  INTERPRETATION FOR THESIS")
    lines.append("=" * 70)
    lines.append("\nMutation kill rate measures the practical defect-detection capability")
    lines.append("of generated test oracles. A kill rate of 0.X means X% of injected")
    lines.append("defects (operator mutations) are caught by the generated tests.")
    lines.append("\nThis directly addresses the SE question: 'Do generated tests")
    lines.append("actually help developers find bugs in software?'")
    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    report_path = OUTPUT_DIR / "mutation_report.txt"
    report_path.write_text(text)
    print(f"  mutation_report.txt")
    print()
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mutation testing for generated unit tests")
    parser.add_argument("--checkpoints-dir", type=str, default=None,
                        help="Path to checkpoints directory (from Colab)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-generate tests locally for mutation testing")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Max samples for re-generation mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model for re-generation (default: auto-detect)")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method/reasoning pairs, e.g. 'plain_llm/base,simple_rag/base'")
    parser.add_argument("--results-only", action="store_true",
                        help="Only analyze existing results_mutation.tsv")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model filter applied to "
                             "--checkpoints-dir loads, e.g. "
                             "'llama3.2:latest,phi4:14b'. Skips any "
                             "checkpoint whose model field is not in the list.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset
    if not DATASET_CACHE.exists():
        print(f"ERROR: Dataset not found at {DATASET_CACHE}")
        print("Run: python prepare_unitest.py")
        sys.exit(1)

    with open(DATASET_CACHE, "rb") as f:
        dataset = pickle.load(f)

    # Shuffle same as train_unitest.py
    random.Random(42).shuffle(dataset)
    print(f"Loaded dataset: {len(dataset)} samples")

    if args.results_only:
        if RESULTS_FILE.exists():
            df = pd.read_csv(RESULTS_FILE, sep="\t")
            plot_mutation_results(df)
            write_mutation_report(df)
        else:
            print(f"ERROR: {RESULTS_FILE} not found")
        return

    # Get checkpoint data
    if args.checkpoints_dir:
        checkpoint_data = load_checkpoints(args.checkpoints_dir)
        # Optional model filter: drop checkpoints whose first-sample model
        # is not in the user-supplied set. Keys store model with ':' → '_',
        # so match against the original colon form recovered from the
        # per-sample 'model' field that load_checkpoints stamps on each entry.
        if args.models:
            wanted = {m.strip() for m in args.models.split(",") if m.strip()}
            kept = {}
            for key, samples in checkpoint_data.items():
                first = samples[0] if samples else {}
                m = first.get("model", "")
                if m in wanted:
                    kept[key] = samples
                else:
                    print(f"  Filter: skipping {key} (model={m!r} "
                          f"not in {sorted(wanted)})")
            checkpoint_data = kept
            print(f"After --models filter: {len(checkpoint_data)} "
                  f"checkpoint(s) remain")
    elif args.regenerate:
        methods_list = None
        if args.methods:
            methods_list = []
            for pair in args.methods.split(","):
                parts = pair.strip().split("/")
                if len(parts) == 2:
                    methods_list.append((parts[0], parts[1]))
                else:
                    print(f"WARNING: invalid method format '{pair}', expected 'method/reasoning'")
        checkpoint_data = regenerate_tests(dataset, args.max_samples,
                                           model=args.model, methods=methods_list)
    else:
        # Try default checkpoint locations
        for candidate in ["checkpoints", "checkpoints/"]:
            if Path(candidate).exists():
                checkpoint_data = load_checkpoints(candidate)
                break
        else:
            print("No checkpoints found. Use --checkpoints-dir or --regenerate")
            print("  --checkpoints-dir PATH  : load from Colab checkpoint files")
            print("  --regenerate            : re-generate tests locally")
            sys.exit(1)

    if not checkpoint_data:
        print("ERROR: No checkpoint data loaded.")
        sys.exit(1)

    # Run mutation analysis
    print(f"\nRunning mutation analysis...")
    df = run_mutation_analysis(checkpoint_data, dataset)

    if not df.empty:
        # Merge with any existing TSV: rows with the same (method, reasoning, model)
        # are replaced by the new run; everything else is preserved. This lets you
        # accumulate per-model results across multiple invocations.
        if RESULTS_FILE.exists():
            try:
                existing = pd.read_csv(RESULTS_FILE, sep="\t")
                key_cols = [c for c in ("method", "reasoning", "model") if c in existing.columns and c in df.columns]
                if key_cols:
                    keys_in_new = df[key_cols].apply(tuple, axis=1).tolist()
                    existing_keys = existing[key_cols].apply(tuple, axis=1)
                    keep = ~existing_keys.isin(keys_in_new)
                    merged = pd.concat([existing[keep], df], ignore_index=True)
                else:
                    merged = df
            except Exception as e:
                print(f"  WARNING: could not merge with existing {RESULTS_FILE}: {e}; overwriting")
                merged = df
        else:
            merged = df

        merged.to_csv(RESULTS_FILE, sep="\t", index=False, float_format="%.6f")
        print(f"\nResults saved → {RESULTS_FILE} ({len(merged)} rows)")

        plot_mutation_results(merged)
        write_mutation_report(merged)
    else:
        print("No valid mutation results produced.")

    print(f"\nDone. Open {OUTPUT_DIR}/ to view outputs.")


if __name__ == "__main__":
    main()
