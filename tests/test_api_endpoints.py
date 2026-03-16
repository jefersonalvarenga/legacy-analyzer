"""
tests/test_api_endpoints.py
---------------------------
TDD tests for Phase 7 API endpoints:
  - POST /analyze/{clinic_id}  (API-01, API-02)
  - GET  /jobs/{job_id}        (API-03 — normalized_status)
  - Backward compat: existing endpoints unchanged

Uses FastAPI TestClient + MagicMock to avoid real DB/network calls.
"""

from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient


def _make_db_mock(clinic_exists: bool = True, job_id: str = "job-uuid-001"):
    db = MagicMock()

    # sf_clinics lookup
    clinic_mock = MagicMock()
    clinic_mock.data = {"id": "clinic-uuid-001", "name": "Test Clinic"} if clinic_exists else None

    # la_analysis_jobs insert
    insert_mock = MagicMock()
    insert_mock.data = [{"id": job_id}]

    # la_analysis_jobs select (for GET /jobs/{job_id})
    select_mock = MagicMock()
    select_mock.data = {
        "id": job_id,
        "status": "queued",
        "progress": 10,
        "la_clients": None,
    }

    # Wire: .table() dispatches by name
    def table_side_effect(name):
        t = MagicMock()
        if name == "sf_clinics":
            t.select.return_value.eq.return_value.single.return_value.execute.return_value = clinic_mock
        elif name == "la_analysis_jobs":
            t.insert.return_value.execute.return_value = insert_mock
            t.select.return_value.eq.return_value.single.return_value.execute.return_value = select_mock
        return t

    db.table.side_effect = table_side_effect
    return db


# ================================================================
# TestAnalyzeEndpoint
# ================================================================


class TestAnalyzeEndpoint:
    """Tests for POST /analyze/{clinic_id} (API-01, API-02)."""

    def test_returns_job_id_immediately(self):
        """POST /analyze/{valid_clinic_id} → 202, body has 'job_id'."""
        from main import app

        db_mock = _make_db_mock(clinic_exists=True, job_id="job-uuid-001")

        with patch("main.get_db", return_value=db_mock), \
             patch("main.run_analysis") as mock_runner:
            client = TestClient(app)
            response = client.post("/analyze/clinic-uuid-001")

        assert response.status_code == 202
        body = response.json()
        assert "job_id" in body
        assert body["job_id"] == "job-uuid-001"

    def test_background_task_scheduled(self):
        """POST /analyze/{valid_clinic_id} → run_analysis called once with (job_id, clinic_id)."""
        from main import app

        db_mock = _make_db_mock(clinic_exists=True, job_id="job-uuid-001")

        with patch("main.get_db", return_value=db_mock), \
             patch("main.run_analysis") as mock_runner:
            client = TestClient(app, raise_server_exceptions=False)
            # TestClient runs synchronously, so BackgroundTasks fire inline
            response = client.post("/analyze/clinic-uuid-001")

        assert response.status_code == 202
        mock_runner.assert_called_once_with("job-uuid-001", "clinic-uuid-001")

    def test_returns_404_for_unknown_clinic(self):
        """POST /analyze/{unknown_id} → 404."""
        from main import app

        db_mock = _make_db_mock(clinic_exists=False)

        with patch("main.get_db", return_value=db_mock), \
             patch("main.run_analysis"):
            client = TestClient(app)
            response = client.post("/analyze/nonexistent-clinic")

        assert response.status_code == 404

    def test_no_job_created_on_404(self):
        """db.table('la_analysis_jobs').insert never called when clinic not found."""
        from main import app

        db_mock = _make_db_mock(clinic_exists=False)

        with patch("main.get_db", return_value=db_mock), \
             patch("main.run_analysis"):
            client = TestClient(app)
            response = client.post("/analyze/nonexistent-clinic")

        assert response.status_code == 404

        # Check that no insert was called on la_analysis_jobs
        for call_args in db_mock.table.call_args_list:
            table_name = call_args[0][0]
            assert table_name != "la_analysis_jobs", (
                "la_analysis_jobs.insert() must NOT be called when clinic not found"
            )


# ================================================================
# TestGetJobEndpoint
# ================================================================


class TestGetJobEndpoint:
    """Tests for GET /jobs/{job_id} (API-03 — normalized_status)."""

    def test_returns_status_and_progress(self):
        """GET /jobs/{job_id} → 200, body has 'status' and 'progress'."""
        from main import app

        db_mock = _make_db_mock(job_id="job-uuid-001")

        with patch("main.get_db", return_value=db_mock):
            client = TestClient(app)
            response = client.get("/jobs/job-uuid-001")

        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "progress" in body

    def test_normalized_status_field(self):
        """GET /jobs/{job_id} → body has normalized_status == 'pending' when DB status == 'queued'."""
        from main import app

        db_mock = _make_db_mock(job_id="job-uuid-001")
        # Default mock returns status="queued" — should normalize to "pending"

        with patch("main.get_db", return_value=db_mock):
            client = TestClient(app)
            response = client.get("/jobs/job-uuid-001")

        assert response.status_code == 200
        body = response.json()
        assert "normalized_status" in body
        assert body["normalized_status"] == "pending"

    def test_returns_404_for_unknown_job(self):
        """GET /jobs/{unknown_id} → 404."""
        from main import app

        db = MagicMock()
        not_found = MagicMock()
        not_found.data = None

        def table_side_effect(name):
            t = MagicMock()
            t.select.return_value.eq.return_value.single.return_value.execute.return_value = not_found
            return t

        db.table.side_effect = table_side_effect

        with patch("main.get_db", return_value=db):
            client = TestClient(app)
            response = client.get("/jobs/nonexistent-job-id")

        assert response.status_code == 404


# ================================================================
# TestExistingEndpoints
# ================================================================


class TestExistingEndpoints:
    """Backward compatibility tests — pre-existing routes must remain unchanged."""

    def test_health_unchanged(self):
        """GET /health → 200, body == {'status': 'ok', 'version': '0.1.0'}."""
        from main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.1.0"}

    def test_post_jobs_route_still_exists(self):
        """POST /jobs endpoint still registered — must not return 404 (404 = route missing)."""
        from main import app

        # We just need to verify the route is registered, not that it fully works.
        # Sending invalid data is expected to return 422, not 404.
        client = TestClient(app)
        response = client.post("/jobs", data={})

        # 422 = route exists but request is invalid (missing required fields)
        # 404 = route does NOT exist (failure condition)
        assert response.status_code != 404, (
            "POST /jobs returned 404 — route appears to be missing from main.py"
        )
