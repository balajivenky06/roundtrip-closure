"""
human_eval/prep/regenerate_missing.py

For any pair in `human_eval/data/pairs_60_full.jsonl` whose reconstructed
artefact (artefact_b) is still an "[unavailable ...]" placeholder,
re-invoke the exact sweep code path (run_path_1 / run_path_2 / run_path_3)
to produce it — capturing intermediate artefacts by monkey-patching the
internal `_call_*` primitives.

Run on Colab (where the LLM cache resolves via the Drive symlink):

    cd /content/roundtrip-closure
    python3 human_eval/prep/regenerate_missing.py

The script rewrites `human_eval/data/pairs_60_full.jsonl` in place with
the newly-produced artefacts substituted in. Rows that were already real
are left untouched.

Why this exists: `build_worksheet.py` reconstructs cache keys directly and
can miss when the sweep-side call used slightly different generation
parameters (num_ctx variants for the two-step Path 2 lookup, etc.). By
running the actual pipeline entry points we sidestep the cache-key
reconstruction and always get the artefacts the sweep would produce.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60.tsv"
SAMPLES_JSONL = PROJECT_ROOT / "data" / "core_sample_150.jsonl"
JSONL_PATH = PROJECT_ROOT / "human_eval" / "data" / "pairs_60_full.jsonl"

UNAVAILABLE_MARKER = "[unavailable"


# ────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────
def load_pairs_tsv() -> dict[str, dict]:
    with PAIRS_TSV.open("r", encoding="utf-8") as f:
        return {row["pair_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def load_samples() -> dict[int, dict]:
    out = {}
    with SAMPLES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[int(d["sample_idx"])] = d
    return out


def load_current_jsonl() -> list[dict]:
    return [json.loads(l) for l in JSONL_PATH.read_text().splitlines() if l.strip()]


# ────────────────────────────────────────────────────────────────────────
# Capture-by-monkey-patch
# ────────────────────────────────────────────────────────────────────────
def install_capture():
    """Wrap the four artefact-producing calls so their outputs are captured
    into a module-level dict keyed by the *most recent* invocation.

    Returns (capture_dict, restore_fn).
    """
    import closure_paths as cp

    capture: dict[str, str] = {}
    originals = {
        "_call_doc_from_code": cp._call_doc_from_code,
        "_call_tests_from_doc": cp._call_tests_from_doc,
        "_call_tests_from_code": cp._call_tests_from_code,
        "_call_doc_from_tests": cp._call_doc_from_tests,
        "_call_code_from_doc_tests": cp._call_code_from_doc_tests,
    }

    def wrap(name):
        orig = originals[name]
        def wrapped(*args, **kwargs):
            text, hit = orig(*args, **kwargs)
            capture[name] = text
            return text, hit
        return wrapped

    for name in originals:
        setattr(cp, name, wrap(name))

    def restore():
        for name, fn in originals.items():
            setattr(cp, name, fn)

    return capture, restore


# ────────────────────────────────────────────────────────────────────────
# Regeneration
# ────────────────────────────────────────────────────────────────────────
def regenerate_one(cell_id: str, sample_idx: int, path: int,
                   samples: dict[int, dict], capture: dict) -> str | None:
    """Return the reconstructed artefact for this (cell, sample, path), or
    None on failure. Uses the same pipeline code the sweep uses."""
    from doe import get_cell
    import closure_paths as cp

    cell = get_cell(cell_id)
    sample = samples.get(sample_idx)
    if sample is None:
        print(f"  ! sample_idx {sample_idx} not in dataset")
        return None

    capture.clear()
    try:
        if path == 1:
            cp.run_path_1(cell, sample)
            # Path 1 reconstructs docstring D' via L_spec:doc_from_code
            return capture.get("_call_doc_from_code")
        elif path == 2:
            cp.run_path_2(cell, sample)
            # Path 2 reconstructs code C' via L_code:code_from_doc_tests
            return capture.get("_call_code_from_doc_tests")
        elif path == 3:
            cp.run_path_3(cell, sample)
            # Path 3 reconstructs docstring D' via L_spec:doc_from_tests
            return capture.get("_call_doc_from_tests")
    except Exception as e:
        print(f"  ! pipeline error: {type(e).__name__}: {e}")
        return None
    return None


def main() -> None:
    records = load_current_jsonl()
    meta = load_pairs_tsv()
    samples = load_samples()

    missing = [
        r for r in records
        if r["artefact_b"].startswith(UNAVAILABLE_MARKER)
    ]
    print(f"Found {len(missing)} unavailable pair(s) to regenerate")
    if not missing:
        return

    capture, restore = install_capture()
    try:
        n_fixed = 0
        for r in missing:
            pid = r["pair_id"]
            m = meta.get(pid)
            if m is None:
                print(f"  {pid}: no metadata row, skipping")
                continue
            cell_id = m["cell_id"]
            sample_idx = int(m["sample_idx"])
            path = int(m["path"])
            print(f"  {pid}: cell={cell_id} sample={sample_idx} path={path}")
            artefact = regenerate_one(cell_id, sample_idx, path, samples, capture)
            if artefact and artefact.strip():
                r["artefact_b"] = artefact
                n_fixed += 1
                print(f"    -> OK ({len(artefact)} chars)")
            else:
                print("    -> still unavailable")
    finally:
        restore()

    # Rewrite JSONL
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nFixed {n_fixed} / {len(missing)} previously-unavailable pairs")
    print(f"Wrote {JSONL_PATH}")


if __name__ == "__main__":
    main()
