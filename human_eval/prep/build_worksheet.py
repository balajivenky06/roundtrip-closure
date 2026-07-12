"""
human_eval/prep/build_worksheet.py

Populate the 60-pair worksheet with the original and reconstructed artefacts
so the Streamlit annotation app has something to show.

Input:
    results/human_eval_pairs_60.tsv    — the pre-registered 60 pairs
                                          (pair_id, cell_id, sample_idx,
                                          path, judge_rating, bucket, ...)
    data/core_sample_150.jsonl         — original code + docstring + tests
                                          per sample_idx
    checkpoints/cache/                 — LLM disk cache; if present, the
                                          reconstructed artefact is pulled
                                          from here

Output:
    human_eval/data/pairs_60_full.jsonl
        one line per pair with fields:
            pair_id, artefact_kind, artefact_a, artefact_b,
            (blinded: no cell_id, path, judge_rating, bucket)

The artefact_kind is derived from the path:
    path 1  → docstring (D vs. D')
    path 2  → code       (C vs. C')
    path 3  → docstring (D vs. D')

For Path 2 the "reconstructed" artefact is C' (LLM-produced code); for
Paths 1 and 3 it is the reconstructed docstring D'. When the LLM cache is
not available (e.g. running on a laptop away from the Drive-symlinked
checkpoints directory), the reconstructed artefact is written as an
"[unavailable — regenerate on Colab]" placeholder so the worksheet is still
loadable and reviewable.

Run:
    python3 human_eval/prep/build_worksheet.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Project root so we can import closure_cache + closure_paths for cache lookup
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIRS_TSV = PROJECT_ROOT / "results" / "human_eval_pairs_60.tsv"
SAMPLES_JSONL = PROJECT_ROOT / "data" / "core_sample_150.jsonl"
OUT_JSONL = Path(__file__).resolve().parent.parent / "data" / "pairs_60_full.jsonl"


def load_pairs_tsv() -> list[dict]:
    with PAIRS_TSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


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


def artefact_kind_for(path: int) -> str:
    return "code" if path == 2 else "docstring"


def original_for(sample: dict, path: int) -> str:
    """Return the ground-truth artefact of the same kind as the reconstructed."""
    if path == 2:
        return sample.get("code", "")
    return sample.get("docstring", "")


def reconstructed_for(cell_id: str, sample: dict, path: int) -> str | None:
    """Try to recover the reconstructed artefact from the LLM cache.

    Returns None on cache miss. This is expected on machines without the
    Drive-symlinked checkpoints directory; regenerate on Colab.
    """
    try:
        from doe import CELLS_BY_ID
        from config import TEMPERATURE, TOP_P, TOP_K, NUM_CTX, MAX_OUTPUT_TOKENS
        import closure_cache
        from closure_paths import (
            _PROMPT_DOC_FROM_CODE,
            _PROMPT_TESTS_FROM_DOC,
            _PROMPT_TESTS_FROM_CODE,
            _PROMPT_DOC_FROM_TESTS,
            _PROMPT_CODE_FROM_DOC_TESTS,
            _extract_python,
        )
    except Exception:
        return None

    cell = CELLS_BY_ID.get(cell_id)
    if cell is None or cell.L_spec is None or cell.L_test is None:
        return None

    code = sample["code"]
    fn_name = sample.get("entry_point", "")
    signature = sample.get("signature", "")

    def _lookup(model_tag: str, role: str, prompt: str) -> str | None:
        for num_ctx in (NUM_CTX, 16384, 8192):
            key = closure_cache.make_key(
                model_tag, role, prompt,
                system_prompt=None,
                temperature=TEMPERATURE,
                top_p=TOP_P, top_k=TOP_K,
                num_ctx=num_ctx,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            cached = closure_cache.get(key)
            if cached is not None and cached.get("text"):
                return cached["text"].strip()
        return None

    if path == 1:
        # D' = L_spec(code)
        prompt = _PROMPT_DOC_FROM_CODE.format(code=code)
        return _lookup(cell.L_spec.ollama_tag, "L_spec:doc_from_code", prompt)

    if path == 2:
        # C' = L_code(D, T'). We need T' first.
        docstring = sample.get("docstring", "")
        if cell.L_code is None:
            return None
        t_prompt = _PROMPT_TESTS_FROM_DOC.format(
            docstring=docstring,
            fn_name=fn_name or "the function",
            signature=signature or "(unknown)",
        )
        t_prime = _lookup(cell.L_test.ollama_tag, "L_test:tests_from_doc", t_prompt)
        if not t_prime:
            return None
        t_prime = _extract_python(t_prime)
        c_prompt = _PROMPT_CODE_FROM_DOC_TESTS.format(
            docstring=docstring, tests=t_prime,
            fn_name=fn_name or "the function",
            signature=signature or "(unknown)",
        )
        text = _lookup(
            cell.L_code.ollama_tag, "L_code:code_from_doc_tests", c_prompt
        )
        return _extract_python(text) if text else None

    if path == 3:
        # D' = L_spec(T'). Need T' from code first.
        t_prompt = _PROMPT_TESTS_FROM_CODE.format(code=code)
        t_prime = _lookup(
            cell.L_test.ollama_tag, "L_test:tests_from_code", t_prompt
        )
        if not t_prime:
            return None
        t_prime = _extract_python(t_prime)
        d_prompt = _PROMPT_DOC_FROM_TESTS.format(tests=t_prime)
        return _lookup(cell.L_spec.ollama_tag, "L_spec:doc_from_tests", d_prompt)

    return None


def main() -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs_tsv()
    samples = load_samples()

    n_ok = 0
    n_miss = 0
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in pairs:
            pair_id = row["pair_id"]
            cell_id = row["cell_id"]
            sample_idx = int(row["sample_idx"])
            path = int(row["path"])
            sample = samples.get(sample_idx)
            if sample is None:
                print(f"{pair_id}: sample_idx {sample_idx} not in dataset — skipping")
                continue

            kind = artefact_kind_for(path)
            artefact_a = original_for(sample, path)
            artefact_b = reconstructed_for(cell_id, sample, path)
            if artefact_b is None:
                n_miss += 1
                artefact_b = (
                    "[unavailable — reconstructed artefact could not be "
                    "recovered from the LLM cache on this machine. "
                    "Re-run build_worksheet.py on a machine with the "
                    "Drive-symlinked checkpoints/cache/ directory.]"
                )
            else:
                n_ok += 1

            f.write(json.dumps({
                "pair_id": pair_id,
                "artefact_kind": kind,
                "artefact_a": artefact_a,
                "artefact_b": artefact_b,
            }) + "\n")

    print(f"\nWrote {OUT_JSONL} ({n_ok} pairs with real reconstructions, "
          f"{n_miss} unavailable)")


if __name__ == "__main__":
    main()
