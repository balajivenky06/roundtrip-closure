"""
Integration tests for Algorithm 2/3 wiring in closure_paths + TSV migration.

Uses only pure ClosureResult construction (no LLM calls) plus a temp-file TSV
migration end-to-end. Verifies:
    1. ClosureResult TSV schema now includes decision_reason + filter_reason.
    2. _result() auto-populates decision_reason via Algorithm 2 when not
       explicitly passed.
    3. ensure_header migrates old-schema TSVs by appending empty columns to
       every existing row.
    4. load_tsv on a legacy 13-column TSV succeeds with decision_reason and
       filter_reason defaulting to "".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from closure_paths import ClosureResult, _result


# ──────────────────────────────────────────────────────────────────────
# 1. Schema
# ──────────────────────────────────────────────────────────────────────
class TestSchema:
    def test_tsv_columns_include_algorithm_columns(self):
        assert "decision_reason" in ClosureResult.TSV_COLUMNS
        assert "filter_reason" in ClosureResult.TSV_COLUMNS

    def test_algorithm_columns_at_end(self):
        # Backward-compat design: new columns appended, not interleaved.
        cols = ClosureResult.TSV_COLUMNS
        assert cols[-2:] == ("decision_reason", "filter_reason")

    def test_tsv_header_matches_schema(self):
        header = ClosureResult.tsv_header().rstrip("\n").split("\t")
        assert header == list(ClosureResult.TSV_COLUMNS)


# ──────────────────────────────────────────────────────────────────────
# 2. _result helper auto-computes decision_reason
# ──────────────────────────────────────────────────────────────────────
class TestResultHelper:
    def _base(self, **overrides):
        base = dict(
            cell_id="X",
            sample_idx=0,
            sample_source="test/source",
            path=1,
            metric_name="mutation_kill_rate",
            metric_value=0.9,
            judge_rating=4,
            judge_justification="ok",
            valid=True,
            elapsed_s=1.0,
            cache_hits=0,
            n_llm_calls=1,
            notes="",
        )
        base.update(overrides)
        return base

    def test_valid_closure_marked_both_agree_valid(self):
        r = _result(**self._base(metric_value=0.9, judge_rating=4, path=1))
        assert r.decision_reason == "both_agree_valid"
        assert r.filter_reason == ""

    def test_false_closure_candidate(self):
        r = _result(**self._base(metric_value=0.9, judge_rating=1, path=1))
        assert r.decision_reason == "false_closure_candidate"

    def test_metric_false_negative(self):
        r = _result(**self._base(metric_value=0.0, judge_rating=4, path=1))
        assert r.decision_reason == "metric_false_negative"

    def test_both_agree_invalid(self):
        r = _result(**self._base(metric_value=0.0, judge_rating=1, path=1))
        assert r.decision_reason == "both_agree_invalid"

    def test_nan_metric_produces_structural_na(self):
        r = _result(**self._base(metric_value=float("nan"), judge_rating=4, path=1))
        assert r.decision_reason == "structural_NA"

    def test_judge_neg1_produces_structural_na(self):
        r = _result(**self._base(metric_value=0.9, judge_rating=-1, path=1))
        assert r.decision_reason == "structural_NA"

    def test_explicit_decision_reason_not_overwritten(self):
        # If caller supplies decision_reason explicitly, keep it (for tests
        # or unusual code paths).
        r = _result(**self._base(
            metric_value=0.9, judge_rating=4, path=1,
            decision_reason="manual_override",
        ))
        assert r.decision_reason == "manual_override"

    def test_filter_reason_passed_through(self):
        r = _result(**self._base(filter_reason="kept_5_of_7"))
        assert r.filter_reason == "kept_5_of_7"


# ──────────────────────────────────────────────────────────────────────
# 3. TSV round-trip
# ──────────────────────────────────────────────────────────────────────
class TestTSVRoundTrip:
    def test_new_row_has_15_tab_separated_fields(self):
        r = _result(
            cell_id="M1", sample_idx=0, sample_source="humaneval/HumanEval/2",
            path=1, metric_name="mutation_kill_rate", metric_value=0.9,
            judge_rating=4, judge_justification="ok",
            valid=True, elapsed_s=1.0, cache_hits=0, n_llm_calls=1,
            notes="mutants=2,killed=2",
            filter_reason="kept_3_of_3",
        )
        row = r.to_tsv_row().rstrip("\n").split("\t")
        assert len(row) == len(ClosureResult.TSV_COLUMNS)
        # Both algorithm columns populated
        col_index = {c: i for i, c in enumerate(ClosureResult.TSV_COLUMNS)}
        assert row[col_index["decision_reason"]] == "both_agree_valid"
        assert row[col_index["filter_reason"]] == "kept_3_of_3"


# ──────────────────────────────────────────────────────────────────────
# 4. ensure_header migration
# ──────────────────────────────────────────────────────────────────────
class TestEnsureHeaderMigration:
    def _write_legacy_tsv(self, tmp_path: Path, n_rows: int = 3) -> Path:
        """Write a fake 13-column TSV mimicking the pre-Algorithm-2 schema."""
        old_cols = [
            "cell_id", "sample_idx", "sample_source", "path",
            "metric_name", "metric_value",
            "judge_rating", "judge_justification",
            "valid", "elapsed_s",
            "cache_hits", "n_llm_calls",
            "notes",
        ]
        p = tmp_path / "legacy.tsv"
        with p.open("w") as f:
            f.write("\t".join(old_cols) + "\n")
            for i in range(n_rows):
                row = [
                    "M1", str(i), "humaneval/HumanEval/2", "1",
                    "mutation_kill_rate", "0.9",
                    "4", "ok",
                    "True", "1.0",
                    "0", "1",
                    "mutants=2,killed=2",
                ]
                f.write("\t".join(row) + "\n")
        return p

    def test_ensure_header_migrates_legacy_tsv(self, tmp_path):
        import train_roundtrip

        p = self._write_legacy_tsv(tmp_path, n_rows=3)
        # Pre-migration: 13 cols
        header_before = p.read_text().splitlines()[0]
        assert len(header_before.split("\t")) == 13

        train_roundtrip.ensure_header(p)

        lines = p.read_text().splitlines()
        header_after = lines[0].split("\t")
        assert len(header_after) == 15  # 13 + 2 new cols
        assert header_after[-2:] == ["decision_reason", "filter_reason"]
        # Every existing row padded with 2 empty fields at the end
        for row in lines[1:]:
            fields = row.split("\t")
            assert len(fields) == 15
            assert fields[-2] == ""
            assert fields[-1] == ""

    def test_ensure_header_noop_on_current_schema(self, tmp_path):
        import train_roundtrip

        # Write a TSV that already has the new schema
        p = tmp_path / "current.tsv"
        with p.open("w") as f:
            f.write(ClosureResult.tsv_header())
            f.write("M1\t0\thumaneval/1\t1\tmutation_kill_rate\t0.9\t4\tok\tTrue\t1.0\t0\t1\tmutants=2\tboth_agree_valid\tkept_3_of_3\n")

        content_before = p.read_text()
        train_roundtrip.ensure_header(p)
        content_after = p.read_text()

        assert content_before == content_after

    def test_ensure_header_creates_new_file(self, tmp_path):
        import train_roundtrip

        p = tmp_path / "new.tsv"
        assert not p.exists()
        train_roundtrip.ensure_header(p)

        assert p.exists()
        header = p.read_text().rstrip("\n").split("\t")
        assert header == list(ClosureResult.TSV_COLUMNS)


# ──────────────────────────────────────────────────────────────────────
# 5. load_tsv tolerance for old-schema TSVs
# ──────────────────────────────────────────────────────────────────────
class TestLoadTSVTolerance:
    def test_load_tsv_defaults_missing_algorithm_columns(self, tmp_path):
        # Legacy 13-column TSV (no decision_reason, no filter_reason)
        p = tmp_path / "legacy.tsv"
        old_cols = list(ClosureResult.TSV_COLUMNS)[:-2]  # drop last two
        with p.open("w") as f:
            f.write("\t".join(old_cols) + "\n")
            f.write(
                "M1\t0\thumaneval/HumanEval/2\t1\tmutation_kill_rate\t0.9\t"
                "4\tok\tTrue\t1.0\t0\t1\tmutants=2\n"
            )

        from analyze.load_results import load_tsv
        df = load_tsv(p)

        assert "decision_reason" in df.columns
        assert "filter_reason" in df.columns
        assert (df["decision_reason"] == "").all()
        assert (df["filter_reason"] == "").all()
