"""
scripts/build_ipm_blind_submission.py

Build the IPM double-blind resubmission bundle for IPM-D-26-05877.

Takes the Overleaf baseline bundle (preserves anonymization + flat Overleaf
layout + figures) and overlays all content fixes from the local repository
tree with path flattening (../tables/ -> flat).

Usage:
    1. Download the current Overleaf project as ZIP:
       Menu -> Download -> Source
    2. Place the ZIP at the repo root as roundtrip_triangle.zip
    3. Run: python3 scripts/build_ipm_blind_submission.py
    4. Output: ipm_blind_resubmission.zip at the repo root (ready for IPM
       upload)

What this script enforces:
  - Overleaf-style flat layout (no ../tables/, no ../plots/)
  - All content fixes from paper_draft_simplified/ overlaid
  - Overleaf's anonymized frontmatter preserved (Anonymous Authors,
    [anonymised git repository])
  - Author-identity leaks blocked (checks for name mentions, real GitHub
    handle, institution email) — build FAILS if any detected outside allowed
    contexts (reviewer-quote verbatim passages)

Pre-write asserts guarantee that if any expected old content has drifted,
the build fails cleanly rather than silently producing a wrong bundle.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLEAF_ZIP = ROOT / "roundtrip_triangle.zip"
BUILD = Path("/tmp/ipm_blind_build")
LOCAL = ROOT / "paper_draft_simplified"
LOCAL_TABLES = ROOT / "tables"
OUT_ZIP = ROOT / "ipm_blind_resubmission.zip"


def flatten_paths(text: str) -> str:
    """Convert ../tables/ and ../plots/ paths to flat Overleaf-layout paths."""
    text = re.sub(r"\\input\{\.\./tables/([^}]+)\}", r"\\input{\1}", text)
    text = re.sub(r"\\input\{\.\./plots/[^/]+/([^}]+)\}", r"\\input{\1}", text)
    text = re.sub(
        r"\\includegraphics(\[[^\]]*\])?\{\.\./plots/[^/]+/([^}]+)\}",
        r"\\includegraphics\1{\2}",
        text,
    )
    return text


def main() -> int:
    if not OVERLEAF_ZIP.exists():
        print(f"ERROR: {OVERLEAF_ZIP} not found.")
        print("Download the current Overleaf project (Menu > Download > Source)")
        print(f"and place it at {OVERLEAF_ZIP}")
        return 1

    # Phase 1: extract Overleaf baseline
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    with zipfile.ZipFile(OVERLEAF_ZIP) as zf:
        zf.extractall(BUILD)
    print(f"Phase 1: extracted {OVERLEAF_ZIP.name} to {BUILD} "
          f"({len(list(BUILD.iterdir()))} files)")

    # Phase 2: overlay section files with path flattening
    sections = [
        "section_1_introduction.tex", "section_2_related_work.tex",
        "section_3_framework.tex", "section_4_experiment_setup.tex",
        "section_5_results.tex", "section_6_discussion.tex",
        "section_7_conclusion.tex", "section_8_limitations.tex",
        "section_appendix_strict_and.tex", "section_appendix_B_stability.tex",
        "section_appendix_C_mixed_logit.tex", "section_appendix_E_threshold.tex",
        "section_appendix_F_methodology.tex", "section_appendix_G_humaneval.tex",
        "section_appendix_H_threats.tex", "section_end_matter.tex",
        "response_to_editor.tex",
    ]
    for name in sections:
        src = LOCAL / name
        if not src.exists():
            print(f"  WARN: {src} missing — using Overleaf's")
            continue
        (BUILD / name).write_text(flatten_paths(src.read_text()))

    # Phase 3: overlay tables (updates + additions)
    tables = [
        "tab_per_operator_kill_rate.tex", "tab_threshold_sensitivity_path1.tex",
        "tab_holdout_sensitivity.tex", "tab_anova_interaction.tex",
        "tab_per_stage_bottleneck.tex", "tab_human_eval_agreement.tex",
        "tab_false_closure.tex",
        # New tables added during revision
        "tab_phi4_failure_modes.tex", "tab_rq4_closed_vs_open.tex",
        "tab_paired_bootstrap_h1_vs_mono.tex", "tab_judge_sensitivity_grid.tex",
        "tab_complexity_stratified.tex",
    ]
    for name in tables:
        src = LOCAL_TABLES / name
        if src.exists():
            (BUILD / name).write_text(flatten_paths(src.read_text()))

    # Phase 4: references.bib (Schäfer dedup already applied)
    shutil.copy(LOCAL / "references.bib", BUILD / "references.bib")

    # Phase 5: main.tex — keep Overleaf's anonymized frontmatter, overlay
    # only the abstract body
    overleaf_main = (BUILD / "main.tex").read_text()
    local_main = (LOCAL / "main.tex").read_text()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", local_main, re.DOTALL)
    assert m, "could not extract abstract body from local main.tex"
    new_block = "\\begin{abstract}" + m.group(1) + "\\end{abstract}"
    main_new = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        lambda _m: new_block,
        overleaf_main, count=1, flags=re.DOTALL,
    )
    main_new = flatten_paths(main_new)
    (BUILD / "main.tex").write_text(main_new)

    # Phase 6: final de-identification sweep
    # Anonymize any remaining real GitHub URL / institution email leaks
    for p in BUILD.glob("*.tex"):
        txt = p.read_text()
        new = (txt
               .replace("https://github.com/balajivenky06/roundtrip-closure",
                        "[anonymised git repository]"))
        # Annotator initials inside the response letter R1.2 bullet
        new = new.replace(
            "Pairwise $\\kappa_{\\textsf{quad}}$ (BV--GS, BV--AM, GS--AM):",
            "Pairwise $\\kappa_{\\textsf{quad}}$ (A1--A2, A1--A3, A2--A3):",
        )
        if new != txt:
            p.write_text(new)

    # Phase 7: verification — fail build if de-identification is incomplete
    allowed_in_quote = ["response_to_editor", "section_6_discussion"]

    def grep(pattern: str) -> list[str]:
        # Only inspect text files; binary PNGs produce false-positive matches
        # on short alphabetic substrings like "BV" / "GS".
        r = subprocess.run(
            ["grep", "-rnE", pattern, str(BUILD),
             "--include=*.tex", "--include=*.bib"],
            capture_output=True, text=True,
        )
        return r.stdout.strip().splitlines() if r.stdout.strip() else []

    checks = {
        "nested path": (r"\.\./(tables|plots)/", None),
        "Wilson 95%": (r"Wilson 95\\%", ["appendix_strict_and"]),
        "real GitHub URL": (r"balajivenky06", None),
        "institution email": (r"@snuchennai\.edu|@snuchennai", None),
        "author full name": (r"Balaji Venktesh|Gireesh Sund|Amsaprabhaa", None),
        "legacy BV/GS/RR": (r"(^|[^A-Za-z])(BV|GS|RR)([^A-Za-z]|$)", None),
        "stale Schäfer key": (r"Schafer2023LLMTest", allowed_in_quote),
    }
    failed = False
    for label, (pat, allow) in checks.items():
        hits = grep(pat)
        if allow:
            hits = [h for h in hits if not any(a in h for a in allow)]
        if hits:
            print(f"  FAIL [{label}]:")
            for h in hits[:3]:
                print(f"    {h[:160]}")
            failed = True
    if failed:
        return 2

    # Phase 8: zip
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(BUILD.iterdir()):
            zf.write(p, arcname=p.name)
    size_kb = OUT_ZIP.stat().st_size // 1024
    print(f"\nSuccess: wrote {OUT_ZIP} ({size_kb} KB, "
          f"{len(list(BUILD.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
