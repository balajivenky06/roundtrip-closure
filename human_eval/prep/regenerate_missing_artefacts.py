"""
human_eval/prep/regenerate_missing_artefacts.py

Fill in reconstructed artefacts that were missing from the Jun 5 Drive-backed
cache snapshot. The uniform-draw 60-pair worksheet had 42/60 pairs whose
reconstructed artefact could not be recovered because the June LLM cache
backed up to Drive is a partial snapshot (~7000 entries vs ~17,000 the full
sweep would have produced).

Regeneration uses identical model + prompt + sampling parameters as the
original sweep, so each regenerated artefact is drawn from the same
conditional distribution as the June output. For the human-eval design
(which rates equivalence to the ground-truth artefact, not reproducibility
of the June reconstruction), this is substitutable.

Idempotent: safe to re-run after interruption. Only touches entries whose
artefact_b currently starts with "[unavailable".

Input (both must exist):
    results/human_eval_pairs_60_uniform.tsv
    human_eval/data/pairs_60_uniform_full.jsonl   (built by build_worksheet.py)

Output: overwrites pairs_60_uniform_full.jsonl in place.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from doe import CELLS_BY_ID
from closure_paths import (
    _call_doc_from_code,
    _call_tests_from_doc,
    _call_tests_from_code,
    _call_doc_from_tests,
    _call_code_from_doc_tests,
    _extract_python,
)

PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60_uniform.tsv"
SAMPLES_JSONL = PROJECT_ROOT / "data" / "core_sample_150.jsonl"
JSONL_PATH = PROJECT_ROOT / "human_eval" / "data" / "pairs_60_uniform_full.jsonl"
PAIR_ID_COL = "uniform_pair_id"


def load_pair_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with PAIRS_TSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            meta[r[PAIR_ID_COL]] = r
    return meta


def load_samples() -> dict[int, dict]:
    out: dict[int, dict] = {}
    with SAMPLES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["sample_idx"])] = d
    return out


def regen_path_1(cell, sample) -> str | None:
    text, _ = _call_doc_from_code(cell.L_spec, sample["code"])
    return text or None


def regen_path_2(cell, sample) -> str | None:
    docstring = sample.get("docstring", "")
    fn_name = sample.get("entry_point", "")
    signature = sample.get("signature", "")
    t_prime, _ = _call_tests_from_doc(cell.L_test, docstring, fn_name, signature)
    if not t_prime:
        return None
    c_prime, _ = _call_code_from_doc_tests(
        cell.L_code, docstring, t_prime, fn_name, signature
    )
    return _extract_python(c_prime) if c_prime else None


def regen_path_3(cell, sample) -> str | None:
    t_prime, _ = _call_tests_from_code(cell.L_test, sample["code"])
    if not t_prime:
        return None
    d_prime, _ = _call_doc_from_tests(cell.L_spec, t_prime)
    return d_prime or None


def main() -> None:
    meta = load_pair_meta()
    samples = load_samples()

    # Load current JSONL (list of dicts, 60 lines)
    pairs: list[dict] = []
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    # Identify pairs needing regen
    needs_regen = [
        p for p in pairs
        if "[unavailable" in (p.get("artefact_b") or "")
    ]
    print(f"Total pairs: {len(pairs)}")
    print(f"Need regen:  {len(needs_regen)}")
    if not needs_regen:
        print("Nothing to do.")
        return

    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for i, pair in enumerate(needs_regen, 1):
        pid = pair["pair_id"]
        row = meta.get(pid)
        if row is None:
            print(f"[{i}/{len(needs_regen)}] {pid}: no TSV row — skip")
            n_fail += 1
            continue
        cell_id = row["cell_id"]
        path = int(row["path"])
        sample_idx = int(row["sample_idx"])
        cell = CELLS_BY_ID.get(cell_id)
        sample = samples.get(sample_idx)
        if cell is None or sample is None:
            print(f"[{i}/{len(needs_regen)}] {pid}: missing cell or sample — skip")
            n_fail += 1
            continue

        t_pair = time.time()
        try:
            if path == 1:
                artefact = regen_path_1(cell, sample)
            elif path == 2:
                artefact = regen_path_2(cell, sample)
            elif path == 3:
                artefact = regen_path_3(cell, sample)
            else:
                artefact = None
        except Exception as e:
            print(f"[{i}/{len(needs_regen)}] {pid} ({cell_id}/path{path}): "
                  f"EXCEPTION {type(e).__name__}: {e}")
            n_fail += 1
            continue

        elapsed = time.time() - t_pair
        if artefact:
            # Mutate the pair in-place
            pair["artefact_b"] = artefact
            n_ok += 1
            status = f"OK ({len(artefact)}c)"
        else:
            n_fail += 1
            status = "EMPTY"
        print(f"[{i}/{len(needs_regen)}] {pid} {cell_id}/path{path} "
              f"sample={sample_idx} ({elapsed:.1f}s) {status}")

        # Persist after each pair so interrupts don't lose work
        with JSONL_PATH.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Regenerated: {n_ok} ok, {n_fail} failed")
    print(f"Total time:  {total_elapsed/60:.1f} min")
    print(f"Output:      {JSONL_PATH}")


if __name__ == "__main__":
    main()
