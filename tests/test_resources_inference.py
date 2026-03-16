"""
test_resources_inference.py
---------------------------
Unit tests for analyzer.resources_inference.

All tests use unittest.mock (stdlib) — no live Supabase connection or LLM keys required.

Covers:
  RES-01: extract_resources() returns professionals list; persist inserts la_resources row
  RES-02: extract_resources() returns correct schedule_type
  SVC-01: count_service_mentions() returns services list; persist inserts la_services row
  SVC-02: count_service_mentions() returns accurate mention counts sorted descending
  Integration: infer_and_persist_resources() delete-before-insert; empty conversations no-crash

TDD Phase: RED — all tests call the real functions which raise NotImplementedError.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from analyzer.resources_inference import (
    count_service_mentions,
    extract_resources,
    infer_and_persist_resources,
    persist_resources,
)

# ---------------------------------------------------------------------------
# Constants / Fixtures
# ---------------------------------------------------------------------------

CLINIC_ID = "clinic-uuid-001"
JOB_ID = "job-uuid-001"
CLINIC_NAME = "Sorriso Da Gente"


def _make_message(content: str, sender_type: str = "clinic") -> MagicMock:
    """Build a minimal Message mock."""
    msg = MagicMock()
    msg.content = content
    msg.sender_type = sender_type
    return msg


def _make_conversation(clinic_msgs: list[str], patient_msgs: list[str] | None = None) -> MagicMock:
    """Build a minimal Conversation mock with clinic_messages populated."""
    conv = MagicMock()
    conv.clinic_messages = [_make_message(c, "clinic") for c in clinic_msgs]
    patient_msgs = patient_msgs or []
    conv.messages = conv.clinic_messages + [_make_message(p, "patient") for p in patient_msgs]
    return conv


def _make_shadow_dna(local_procedures: list[str] | None = None) -> MagicMock:
    """Build a minimal ShadowDNA mock."""
    dna = MagicMock()
    dna.local_procedures = local_procedures or []
    return dna


def _make_db() -> MagicMock:
    """Build a Supabase client mock that supports .table().delete/insert().eq().execute() chains."""
    db = MagicMock()
    # Each call to db.table() should return a fresh mock that supports chaining
    # MagicMock auto-creates chained attributes by default.
    return db


# ---------------------------------------------------------------------------
# TestExtractResources
# ---------------------------------------------------------------------------


class TestExtractResources:
    """Tests for extract_resources() — RES-01 and RES-02."""

    def test_professionals_extracted(self):
        """RES-01: extract_resources() returns a non-empty professionals list
        when conversations mention a named professional."""
        conversations = [
            _make_conversation(
                clinic_msgs=["Sua consulta com a Dra. Ana foi confirmada para amanha."]
            )
        ]
        # ResourcesModule must be initialized — patch it to return Dra. Ana
        with patch("analyzer.resources_inference._resources_module") as mock_mod:
            mock_result = MagicMock()
            mock_result.professionals = ["Dra. Ana"]
            mock_result.schedule_type = "by_professional"
            mock_mod.forward.return_value = mock_result
            mock_mod.__bool__ = lambda self: True

            result = extract_resources(conversations, CLINIC_NAME)

        assert result.professionals  # non-empty list

    def test_schedule_type_by_professional(self):
        """RES-02: extract_resources() returns 'by_professional' when multiple
        professionals are mentioned across conversations."""
        conversations = [
            _make_conversation(
                clinic_msgs=["Consulta com Dra. Ana confirmada."]
            ),
            _make_conversation(
                clinic_msgs=["Seu horario com Dr. Carlos esta disponivel."]
            ),
        ]
        with patch("analyzer.resources_inference._resources_module") as mock_mod:
            mock_result = MagicMock()
            mock_result.professionals = ["Dra. Ana", "Dr. Carlos"]
            mock_result.schedule_type = "by_professional"
            mock_mod.forward.return_value = mock_result
            mock_mod.__bool__ = lambda self: True

            result = extract_resources(conversations, CLINIC_NAME)

        assert result.schedule_type == "by_professional"

    def test_schedule_type_single(self):
        """RES-02: extract_resources() returns 'single' when no named
        professionals are detected in conversations."""
        conversations = [
            _make_conversation(
                clinic_msgs=["Sua consulta foi confirmada para amanha as 14h."]
            )
        ]
        with patch("analyzer.resources_inference._resources_module") as mock_mod:
            mock_result = MagicMock()
            mock_result.professionals = []
            mock_result.schedule_type = "single"
            mock_mod.forward.return_value = mock_result
            mock_mod.__bool__ = lambda self: True

            result = extract_resources(conversations, CLINIC_NAME)

        assert result.schedule_type == "single"


# ---------------------------------------------------------------------------
# TestCountServiceMentions
# ---------------------------------------------------------------------------


class TestCountServiceMentions:
    """Tests for count_service_mentions() — SVC-01 and SVC-02."""

    def test_service_in_clinic_messages(self):
        """SVC-01: count_service_mentions() returns a list that contains 'implante'
        when clinic messages mention it."""
        conversations = [
            _make_conversation(
                clinic_msgs=[
                    "Realizamos implante dentario com especialistas.",
                    "Temos promocao de implante esse mes.",
                ]
            )
        ]
        service_names = ["implante", "clareamento"]

        result = count_service_mentions(service_names, conversations)

        names = [item["name"] for item in result]
        assert "implante" in names

    def test_mention_count_accuracy(self):
        """SVC-02: count_service_mentions() returns the correct integer mention_count
        for each service (one count per message that contains the term)."""
        conversations = [
            _make_conversation(
                clinic_msgs=[
                    "Fazemos implante com excelencia.",   # 1 mention of implante
                    "Clareamento disponivel agora.",       # 1 mention of clareamento
                    "Novo implante disponivel.",           # 1 more implante
                ]
            )
        ]
        service_names = ["implante", "clareamento"]

        result = count_service_mentions(service_names, conversations)

        counts = {item["name"]: item["mention_count"] for item in result}
        assert counts["implante"] == 2
        assert counts["clareamento"] == 1

    def test_sorted_by_frequency(self):
        """SVC-02: count_service_mentions() returns results sorted by mention_count
        descending (highest frequency first)."""
        conversations = [
            _make_conversation(
                clinic_msgs=[
                    "Implante, implante, implante.",  # 3 messages with implante? No — 1 msg
                    "Implante disponivel.",
                    "Fazemos implante.",
                    "Clareamento disponivel.",         # only 1 mention
                ]
            )
        ]
        service_names = ["implante", "clareamento", "ortodontia"]

        result = count_service_mentions(service_names, conversations)

        # Result must be sorted descending by mention_count
        counts = [item["mention_count"] for item in result]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# TestPersistResources
# ---------------------------------------------------------------------------


class TestPersistResources:
    """Tests for persist_resources() — RES-01 and SVC-01 persistence."""

    def test_professional_inserted(self):
        """RES-01: persist_resources() inserts a row into la_resources
        with name='Dra. Ana' and the correct clinic_id."""
        db = _make_db()

        persist_resources(
            db=db,
            clinic_id=CLINIC_ID,
            job_id=JOB_ID,
            professionals=["Dra. Ana"],
            schedule_type="by_professional",
            services=[],
        )

        # la_resources insert must have been called with a row containing name="Dra. Ana"
        insert_calls = db.table("la_resources").insert.call_args_list
        assert insert_calls, "Expected db.table('la_resources').insert() to be called"
        inserted_data = insert_calls[0][0][0]
        if isinstance(inserted_data, list):
            inserted_data = inserted_data[0]
        assert inserted_data.get("name") == "Dra. Ana"
        assert inserted_data.get("clinic_id") == CLINIC_ID

    def test_service_inserted(self):
        """SVC-01: persist_resources() inserts a row into la_services
        with name='implante'."""
        db = _make_db()

        persist_resources(
            db=db,
            clinic_id=CLINIC_ID,
            job_id=JOB_ID,
            professionals=[],
            schedule_type="single",
            services=[{"name": "implante", "mention_count": 5}],
        )

        insert_calls = db.table("la_services").insert.call_args_list
        assert insert_calls, "Expected db.table('la_services').insert() to be called"
        inserted_data = insert_calls[0][0][0]
        if isinstance(inserted_data, list):
            inserted_data = inserted_data[0]
        assert inserted_data.get("name") == "implante"


# ---------------------------------------------------------------------------
# TestInferAndPersist
# ---------------------------------------------------------------------------


class TestInferAndPersist:
    """Integration-level tests for infer_and_persist_resources()."""

    def test_delete_before_insert(self):
        """RES-01+02: infer_and_persist_resources() calls delete for confirmed=FALSE
        rows before inserting new suggestions."""
        db = _make_db()
        shadow_dna = _make_shadow_dna(local_procedures=["implante"])
        conversations = [
            _make_conversation(
                clinic_msgs=["Consulta com Dra. Ana confirmada."]
            )
        ]

        with patch("analyzer.resources_inference._resources_module") as mock_mod:
            mock_result = MagicMock()
            mock_result.professionals = ["Dra. Ana"]
            mock_result.schedule_type = "by_professional"
            mock_mod.forward.return_value = mock_result
            mock_mod.__bool__ = lambda self: True

            infer_and_persist_resources(
                conversations=conversations,
                clinic_name=CLINIC_NAME,
                clinic_id=CLINIC_ID,
                job_id=JOB_ID,
                shadow_dna=shadow_dna,
                db=db,
            )

        # delete must have been called on both tables
        db.table("la_resources").delete.assert_called()
        db.table("la_services").delete.assert_called()

    def test_empty_conversations_no_crash(self):
        """RES-01+SVC-01: infer_and_persist_resources() with an empty conversations
        list must not raise an exception and must not write to the DB."""
        db = _make_db()
        shadow_dna = _make_shadow_dna(local_procedures=[])

        # Should complete without exception
        infer_and_persist_resources(
            conversations=[],
            clinic_name=CLINIC_NAME,
            clinic_id=CLINIC_ID,
            job_id=JOB_ID,
            shadow_dna=shadow_dna,
            db=db,
        )

        # No DB writes expected for empty conversations
        db.table("la_resources").insert.assert_not_called()
        db.table("la_services").insert.assert_not_called()
