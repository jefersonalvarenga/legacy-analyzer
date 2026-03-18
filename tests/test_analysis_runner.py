"""
test_analysis_runner.py
-----------------------
TDD RED stubs for Phase 9 — Pipeline Integration.

Covers:
  PIPE-01: run_analysis() orchestrates ingest -> analyse -> blueprint -> done/error
  PIPE-02: run_analysis() persists blueprint with clinic_id and infers resources

All 7 tests are STUBS — they assert behaviour that the Phase 7 stub implementation
does NOT yet satisfy. They must be COLLECTED by pytest and FAIL (not skip, not error).

Phase 9 (09-02) will implement the real run_analysis() to make these tests pass (GREEN).
"""

from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Build a Supabase client mock supporting chained .table().*.execute() calls."""
    db = MagicMock()
    # Seed sf_clinics lookup used by run_analysis to fetch clinic name
    db.table("sf_clinics").select().eq().single().execute().data = {
        "id": "clinic-1",
        "name": "Clinica Teste",
    }
    return db


def _make_conversation():
    """Return a single fake conversation with one fake message."""
    conv = MagicMock()
    conv.messages = [MagicMock()]
    return conv


# ---------------------------------------------------------------------------
# TestRunAnalysis — PIPE-01
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Tests for the orchestration contract of run_analysis() — PIPE-01."""

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_full_pipeline_completes(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-01: run_analysis() completes without raising when all steps succeed."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = [_make_conversation()]

        # Must not raise
        run_analysis("job-1", "clinic-1")

        # The Phase 7 stub does NOT call ingest_from_evolution — this assertion will FAIL
        mock_ingest.assert_called_once()

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_job_marked_done(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-01: after successful run_analysis(), job status must be updated to 'done'."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = [_make_conversation()]

        run_analysis("job-1", "clinic-1")

        # Check that at some point the job was updated with status='done'
        update_calls = db.table("la_analysis_jobs").update.call_args_list
        done_calls = [c for c in update_calls if c[0][0].get("status") == "done"]
        # Phase 7 stub never writes status='done' — this assertion will FAIL
        assert done_calls, "Expected job to be marked 'done' after successful pipeline"

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_job_marked_error_on_exception(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-01: when ingest_from_evolution raises, job status must become 'error'."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.side_effect = RuntimeError("Evolution API down")

        run_analysis("job-1", "clinic-1")

        update_calls = db.table("la_analysis_jobs").update.call_args_list
        error_calls = [c for c in update_calls if c[0][0].get("status") == "error"]
        # Phase 7 stub does not call ingest_from_evolution — side_effect never fires via ingest
        # The error path might be hit by another route; assert ingest was attempted to confirm RED
        mock_ingest.assert_called_once()
        assert error_calls, "Expected job to be marked 'error' when ingest_from_evolution raises"

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_empty_conversations_fails_job(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-01: when ingest returns [], job status must become 'error'
        with a non-empty error_message."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = []  # no conversations found

        run_analysis("job-1", "clinic-1")

        # Phase 7 stub never calls ingest — mock_ingest won't return [] in practice
        mock_ingest.assert_called_once()
        update_calls = db.table("la_analysis_jobs").update.call_args_list
        error_calls = [c for c in update_calls if c[0][0].get("status") == "error"]
        assert error_calls, "Expected job to be marked 'error' when conversations list is empty"
        error_msg = error_calls[0][0][0].get("error_message", "")
        assert error_msg, "Expected a non-empty error_message when conversations is empty"


# ---------------------------------------------------------------------------
# TestBlueprintPersistence — PIPE-02
# ---------------------------------------------------------------------------


class TestBlueprintPersistence:
    """Tests for blueprint persistence with clinic_id and resource inference — PIPE-02."""

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_blueprint_saved_with_clinic_id(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-02: run_analysis() must call db.table('la_blueprints').insert()
        with a dict containing key 'clinic_id' equal to the provided clinic_id."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = [_make_conversation()]

        run_analysis("job-1", "clinic-1")

        insert_calls = db.table("la_blueprints").insert.call_args_list
        # Phase 7 stub never writes to la_blueprints — this assertion will FAIL
        assert insert_calls, "Expected db.table('la_blueprints').insert() to be called"
        inserted = insert_calls[0][0][0]
        if isinstance(inserted, list):
            inserted = inserted[0]
        assert inserted.get("clinic_id") == "clinic-1", (
            f"Expected clinic_id='clinic-1' in blueprint insert, got {inserted.get('clinic_id')!r}"
        )

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_resources_persisted_in_same_execution(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-02: infer_and_persist_resources() must be called within the same
        run_analysis() execution that saves the blueprint."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = [_make_conversation()]

        run_analysis("job-1", "clinic-1")

        # Phase 7 stub never calls infer_and_persist_resources — this assertion will FAIL
        mock_infer.assert_called_once()
        # Also ensure blueprint was saved (both in same execution)
        insert_calls = db.table("la_blueprints").insert.call_args_list
        assert insert_calls, "Expected blueprint to be saved alongside resource inference"

    @patch("analyzer.analysis_runner.infer_and_persist_resources", create=True)
    @patch("analyzer.analysis_runner.ingest_from_evolution", create=True)
    @patch("analyzer.analysis_runner.get_db")
    def test_resources_error_does_not_abort_blueprint(self, mock_get_db, mock_ingest, mock_infer):
        """PIPE-02: when infer_and_persist_resources raises, db.table('la_blueprints').insert()
        must still be called — blueprint is saved regardless of resource inference failures."""
        from analyzer.analysis_runner import run_analysis

        db = _make_db()
        mock_get_db.return_value = db
        mock_ingest.return_value = [_make_conversation()]
        mock_infer.side_effect = RuntimeError("DSPy model unavailable")

        run_analysis("job-1", "clinic-1")

        insert_calls = db.table("la_blueprints").insert.call_args_list
        # Phase 7 stub never writes to la_blueprints — this assertion will FAIL
        assert insert_calls, (
            "Expected blueprint insert even when infer_and_persist_resources raises"
        )
