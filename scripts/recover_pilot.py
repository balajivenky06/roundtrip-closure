"""
scripts/recover_pilot.py — clean contaminated state from the pre-think-strip pilot.

Two problems must be undone before re-running cells that hit the M3 bug:

    1. results/pilot_results.tsv has rows whose intermediates were empty
       because ollama_client wasn't stripping <think>...</think>. Those
       rows must be deleted so train_roundtrip.run_cell doesn't skip them
       on resume.

    2. checkpoints/cache/ contains cached LLMResponse entries for the
       affected models whose `text` field is empty (the <think>-stripped
       result was cached). They must be purged so the rerun gets fresh,
       genuinely-correct content.

By default this script targets M3 (qwen3.6:27b) — the cell that the pilot
showed was broken. Other cells/models can be added via CLI.

Run as:
    python3 scripts/recover_pilot.py             # M3 + qwen3.6:27b
    python3 scripts/recover_pilot.py --cells M3,N1 --models qwen3.6:27b,deepseek-r1:14b
    python3 scripts/recover_pilot.py --dry-run   # just report counts, change nothing
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE_DIR, PILOT_RESULTS_TSV, RESULTS_TSV


# ──────────────────────────────────────────────────────────────────────
# TSV cleanup
# ──────────────────────────────────────────────────────────────────────
def purge_tsv_rows(tsv_path: Path, cells_to_drop: set[str],
                   dry_run: bool = False) -> dict:
    """Remove every row whose cell_id is in cells_to_drop. Header stays."""
    if not tsv_path.exists():
        return {"path": str(tsv_path), "kept": 0, "dropped": 0, "skipped": "not_found"}

    with tsv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {"path": str(tsv_path), "kept": 0, "dropped": 0, "skipped": "empty"}

    header, body = lines[0], lines[1:]
    kept, dropped = [], 0
    for line in body:
        cell_id = line.split("\t", 1)[0]
        if cell_id in cells_to_drop:
            dropped += 1
        else:
            kept.append(line)

    if not dry_run:
        backup = tsv_path.with_suffix(tsv_path.suffix + ".bak")
        tsv_path.rename(backup)
        with tsv_path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(kept)
        return {"path": str(tsv_path), "kept": len(kept), "dropped": dropped,
                "backup": str(backup)}
    return {"path": str(tsv_path), "kept": len(kept), "dropped": dropped,
            "dry_run": True}


# ──────────────────────────────────────────────────────────────────────
# Cache cleanup — by model_tag and/or empty-text criterion
# ──────────────────────────────────────────────────────────────────────
def purge_cache_entries(model_tags: set[str], cache_dir: Path,
                        also_purge_empty_text: bool = True,
                        dry_run: bool = False) -> dict:
    """
    Delete cache entries whose JSON payload either:
      - has model_tag in `model_tags`, OR
      - (if also_purge_empty_text) has empty .text — these are the poisoned
        responses that the pre-fix client wrote on <think>-only output.
    """
    if not cache_dir.exists():
        return {"path": str(cache_dir), "scanned": 0, "deleted": 0,
                "skipped": "not_found"}

    scanned = 0
    deleted = 0
    deleted_by_reason: dict[str, int] = {"model_tag": 0, "empty_text": 0}
    errors = 0

    for shard in cache_dir.iterdir():
        if not shard.is_dir():
            continue
        for entry in shard.iterdir():
            if entry.suffix != ".json":
                continue
            scanned += 1
            try:
                with entry.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                errors += 1
                continue

            reason = None
            tag = data.get("model_tag", "")
            if tag in model_tags:
                reason = "model_tag"
            elif also_purge_empty_text and not str(data.get("text", "")).strip():
                reason = "empty_text"

            if reason:
                if not dry_run:
                    try:
                        entry.unlink()
                    except OSError:
                        errors += 1
                        continue
                deleted += 1
                deleted_by_reason[reason] += 1

    return {
        "path": str(cache_dir),
        "scanned": scanned,
        "deleted": deleted,
        "deleted_by_reason": deleted_by_reason,
        "errors": errors,
        "dry_run": dry_run,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", default="M3",
                        help="Comma-separated cell IDs to drop from the TSV "
                             "(default: M3)")
    parser.add_argument("--models", default="qwen3.6:27b",
                        help="Comma-separated Ollama model tags whose cache "
                             "entries should be purged (default: qwen3.6:27b)")
    parser.add_argument("--keep-empty-text-cache", action="store_true",
                        help="Don't purge cache entries with empty .text "
                             "(by default we purge them; they're poisoned)")
    parser.add_argument("--target-tsv", default=str(PILOT_RESULTS_TSV),
                        help="Path to the TSV to clean (default: pilot results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without modifying anything")
    args = parser.parse_args(argv)

    cells = {c.strip() for c in args.cells.split(",") if c.strip()}
    models = {m.strip() for m in args.models.split(",") if m.strip()}

    print(f"Recovery plan {'(DRY RUN)' if args.dry_run else ''}")
    print(f"  cells to drop from TSV: {sorted(cells)}")
    print(f"  models to purge from cache: {sorted(models)}")
    print(f"  also purge empty-text cache entries: "
          f"{not args.keep_empty_text_cache}")
    print()

    print("1) TSV cleanup")
    tsv_result = purge_tsv_rows(
        Path(args.target_tsv), cells, dry_run=args.dry_run
    )
    print(f"   {tsv_result}")
    print()

    print("2) Cache cleanup")
    cache_result = purge_cache_entries(
        models, CACHE_DIR,
        also_purge_empty_text=not args.keep_empty_text_cache,
        dry_run=args.dry_run,
    )
    print(f"   {cache_result}")
    print()

    print("Next: re-run the affected cells, e.g.")
    for c in sorted(cells):
        print(f"   CELL_ID={c} python3 train_roundtrip.py")
    print("Or rerun the whole pilot:")
    print("   python3 -u scripts/run_pilot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
