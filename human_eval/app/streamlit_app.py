"""
human_eval/app/streamlit_app.py

Blinded annotation UI for the 60-pair three-annotator study. Displays each
pair with syntax-highlighted side-by-side artefacts, presents the 5-level
rubric, records rating + justification, auto-saves to per-annotator TSV,
and is fully resumable across sessions.

Run:
    streamlit run human_eval/app/streamlit_app.py

Data files:
    Input:  ../data/pairs_60_full.jsonl (produced by prep/build_worksheet.py)
    Output: ../data/annotator_<ID>_ratings.tsv
    Log:    ../data/annotator_<ID>_events.jsonl  (append-only audit trail)
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


# ────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
PAIRS_PATH = DATA_DIR / "pairs_60_full.jsonl"


def annotator_files(annotator_id: str) -> tuple[Path, Path]:
    """Return (ratings_tsv, events_log_jsonl) paths for this annotator."""
    safe = "".join(c for c in annotator_id if c.isalnum() or c in "_-")
    if not safe:
        safe = "unknown"
    return (
        DATA_DIR / f"annotator_{safe}_ratings.tsv",
        DATA_DIR / f"annotator_{safe}_events.jsonl",
    )


# ────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_pairs() -> list[dict]:
    if not PAIRS_PATH.exists():
        return []
    pairs = []
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def deterministic_order(pairs: list[dict], annotator_id: str) -> list[int]:
    """Return a permutation of pair indices seeded on the annotator ID.
    Different annotators see the same pairs but in different orders."""
    seed = int.from_bytes(
        hashlib.sha256(annotator_id.encode("utf-8")).digest()[:4],
        "big",
    )
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    return indices


def load_existing_ratings(path: Path) -> dict[str, dict]:
    """Load existing ratings as {pair_id -> {rating, justification, ts}}."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            out[row["pair_id"]] = row
    return out


def append_rating_row(
    path: Path,
    pair_id: str,
    rating: int,
    justification: str,
    ts_iso: str,
    revision: bool,
) -> None:
    """Append or overwrite a rating row. TSV is rewritten atomically."""
    existing = load_existing_ratings(path)
    existing[pair_id] = {
        "pair_id": pair_id,
        "rating": str(rating),
        "justification": justification,
        "ts_iso": ts_iso,
        "is_revision": str(revision).lower(),
    }
    # Rewrite the whole TSV so pair_id is a stable primary key
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("pair_id\trating\tjustification\tts_iso\tis_revision\n")
        for row in existing.values():
            just = row["justification"].replace("\t", " ").replace("\n", " ")
            f.write(
                f"{row['pair_id']}\t{row['rating']}\t{just}\t"
                f"{row['ts_iso']}\t{row.get('is_revision', 'false')}\n"
            )
    tmp.replace(path)


def append_event(path: Path, event: dict) -> None:
    """Append-only audit log for debugging + provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────
RATING_LABELS = {
    4: "4 — Identical",
    3: "3 — Equivalent",
    2: "2 — Approximately equivalent",
    1: "1 — Clearly different",
    0: "0 — Unrelated",
}

RUBRIC_COMPACT = """
**4 — Identical** · Same output on every valid input. Trivial differences only.

**3 — Equivalent** · Same observable behaviour, different implementation.

**2 — Approximately equivalent** · Same on most inputs, differs on edge cases.

**1 — Clearly different** · Different output on common inputs.

**0 — Unrelated** · Solving different problems entirely.
"""


def render_consent_screen() -> None:
    st.markdown("# Human evaluation study — consent")
    st.markdown(
        "You are about to enter the annotation UI. Before you begin, please "
        "confirm that:"
    )
    consents = {
        "read_consent": st.checkbox(
            "I have read the consent form (`human_eval/consent_form.md`) "
            "and understand its contents."
        ),
        "read_rubric": st.checkbox(
            "I have read the rubric (`human_eval/rubric.md`) and reviewed "
            "the rating anchors."
        ),
        "attended_calibration": st.checkbox(
            "I have attended the calibration session (or have arranged an "
            "equivalent 1-on-1 briefing with the corresponding author)."
        ),
        "no_communication": st.checkbox(
            "I understand that I must not discuss specific production pairs "
            "with the other annotators during the annotation window."
        ),
    }
    all_consent = all(consents.values())
    annotator_id = st.text_input(
        "Annotator ID (short opaque token, letters/digits/underscore only)",
        placeholder="ann_alpha",
    )
    if st.button("Continue", type="primary", disabled=not (all_consent and annotator_id)):
        st.session_state["annotator_id"] = annotator_id.strip()
        st.session_state["consent_ts"] = datetime.now(timezone.utc).isoformat()
        _, events_path = annotator_files(st.session_state["annotator_id"])
        append_event(events_path, {
            "type": "session_start",
            "ts": st.session_state["consent_ts"],
            "annotator_id": st.session_state["annotator_id"],
            "consents": {k: bool(v) for k, v in consents.items()},
        })
        st.rerun()


def render_pair_screen(pairs: list[dict], order: list[int]) -> None:
    annotator_id = st.session_state["annotator_id"]
    ratings_path, events_path = annotator_files(annotator_id)
    existing = load_existing_ratings(ratings_path)

    # Progress
    n_done = len(existing)
    n_total = len(pairs)
    st.progress(n_done / max(n_total, 1),
                text=f"Progress: {n_done} / {n_total} pairs rated")

    # Current pair index (in the shuffled order)
    if "cur_idx" not in st.session_state:
        # Resume: pick first unrated pair in order
        st.session_state["cur_idx"] = 0
        for i, orig_idx in enumerate(order):
            if pairs[orig_idx]["pair_id"] not in existing:
                st.session_state["cur_idx"] = i
                break

    cur = st.session_state["cur_idx"]
    if cur >= n_total:
        _render_completion_screen(annotator_id, existing)
        return

    pair = pairs[order[cur]]
    pair_id = pair["pair_id"]
    is_revision = pair_id in existing

    # Header + nav
    left_nav, header, right_nav = st.columns([1, 3, 1])
    with left_nav:
        if st.button("← Previous", disabled=(cur == 0)):
            st.session_state["cur_idx"] -= 1
            st.rerun()
    with header:
        artefact_kind = pair.get("artefact_kind", "artefact")
        st.markdown(
            f"### Pair {cur + 1} of {n_total} · "
            f"`{pair_id}` · {artefact_kind}"
            f"{' · (revising)' if is_revision else ''}"
        )
    with right_nav:
        if st.button("Skip →", disabled=(cur == n_total - 1),
                     help="Skip without rating (you can return later)"):
            st.session_state["cur_idx"] += 1
            st.rerun()

    # Side-by-side artefacts
    lang = "python" if pair.get("artefact_kind") in ("code", "tests") else None
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Artefact A (original)**")
        if lang:
            st.code(pair["artefact_a"], language=lang)
        else:
            st.markdown(f"> {pair['artefact_a']}")
    with col_b:
        st.markdown("**Artefact B (reconstructed)**")
        if lang:
            st.code(pair["artefact_b"], language=lang)
        else:
            st.markdown(f"> {pair['artefact_b']}")

    st.divider()

    # Rubric + rating
    r1, r2 = st.columns([3, 2])
    with r1:
        default_rating = int(existing[pair_id]["rating"]) if is_revision else None
        default_just = existing[pair_id]["justification"] if is_revision else ""
        rating = st.radio(
            "**Rating**",
            options=[4, 3, 2, 1, 0],
            format_func=lambda x: RATING_LABELS[x],
            horizontal=False,
            index=[4, 3, 2, 1, 0].index(default_rating) if default_rating is not None else None,
        )
        justification = st.text_area(
            "**Justification** (one sentence, ≤ 30 words). Do NOT reference "
            "which model produced either artefact.",
            value=default_just,
            max_chars=250,
            height=80,
        )
    with r2:
        st.markdown("**Rubric (compact)**")
        st.markdown(RUBRIC_COMPACT)
        with st.expander("Full anchor examples"):
            st.markdown("See `human_eval/rubric.md` for full examples per level.")

    # Submit
    can_submit = rating is not None and len(justification.strip()) >= 5
    if st.button("Submit rating and continue →",
                 type="primary", disabled=not can_submit):
        ts = datetime.now(timezone.utc).isoformat()
        append_rating_row(
            ratings_path, pair_id, rating, justification.strip(), ts,
            revision=is_revision,
        )
        append_event(events_path, {
            "type": "submit_rating",
            "ts": ts, "pair_id": pair_id, "rating": rating,
            "justification_len": len(justification.strip()),
            "revision": is_revision,
        })
        st.session_state["cur_idx"] += 1
        st.rerun()


def _render_completion_screen(annotator_id: str, existing: dict) -> None:
    st.balloons()
    st.markdown(f"# All 60 pairs rated. Thank you, `{annotator_id}`.")
    st.markdown(
        f"Your ratings are saved at "
        f"`{annotator_files(annotator_id)[0]}`. Please confirm submission "
        f"below to lock your ratings."
    )
    if st.button("Confirm submission", type="primary"):
        _, events_path = annotator_files(annotator_id)
        append_event(events_path, {
            "type": "session_complete",
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_ratings": len(existing),
        })
        st.success("Submission confirmed. You may now close the app.")
        st.stop()

    st.markdown("---")
    st.markdown("### Or go back and revise any rating")
    if st.button("← Return to first pair"):
        st.session_state["cur_idx"] = 0
        st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Round-trip closure human eval",
                       page_icon="📝", layout="wide")

    pairs = load_pairs()
    if not pairs:
        st.error(
            "No pairs found. Run `python3 human_eval/prep/build_worksheet.py` "
            "first to produce `human_eval/data/pairs_60_full.jsonl`."
        )
        return

    if "annotator_id" not in st.session_state:
        render_consent_screen()
        return

    annotator_id = st.session_state["annotator_id"]
    order = deterministic_order(pairs, annotator_id)

    with st.sidebar:
        st.markdown(f"**Annotator:** `{annotator_id}`")
        st.markdown(f"**Pairs total:** {len(pairs)}")
        _, events_path = annotator_files(annotator_id)
        st.markdown(f"**Events log:** `{events_path.name}`")
        if st.button("Log out"):
            del st.session_state["annotator_id"]
            if "cur_idx" in st.session_state:
                del st.session_state["cur_idx"]
            st.rerun()
        st.divider()
        st.markdown("### Rubric quick reference")
        st.markdown(RUBRIC_COMPACT)

    render_pair_screen(pairs, order)


if __name__ == "__main__":
    main()
