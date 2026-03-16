"""
test_evolution_ingestor.py
--------------------------
Unit tests for analyzer.evolution_ingestor.

All tests use unittest.mock (stdlib) — no live Supabase connection required.

Covers:
  ING-01: Correct instanceId resolution from sf_clinics + Instance tables
  ING-02: Message field mapping (sender_type, timestamp, body, phone)
  ING-03: Group JID exclusion, isolation, and error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from analyzer.parser import Conversation, Message
from analyzer.evolution_ingestor import ingest_from_evolution


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

CLINIC_ID = "clinic-uuid-001"
INSTANCE_NAME = "sgen-instance"
INSTANCE_UUID = "inst-uuid-999"
CLINIC_SENDER = "Sorriso Da Gente"
REMOTE_JID = "5511912345678@s.whatsapp.net"


def _make_message_row(
    remote_jid: str = REMOTE_JID,
    from_me: bool = False,
    timestamp: int = 1700000000,
    message_type: str = "conversation",
    message_body: dict | None = None,
    push_name: str | None = "Maria",
) -> dict:
    """Build a minimal Evolution Message row dict."""
    if message_body is None:
        message_body = {"conversation": "Ola"}
    return {
        "key": {
            "remoteJid": remote_jid,
            "fromMe": from_me,
            "id": "MSG001",
        },
        "pushName": push_name,
        "message": message_body,
        "messageType": message_type,
        "messageTimestamp": timestamp,
    }


def _make_db_mock(
    clinic_row: dict | None,
    instance_row: dict | None,
    message_rows: list[dict],
) -> MagicMock:
    """
    Builds a db mock that returns given rows for sf_clinics, Instance,
    and Message queries.
    """
    db = MagicMock()

    def table_side_effect(table_name: str) -> MagicMock:
        mock_table = MagicMock()
        if table_name == "sf_clinics":
            mock_table.select.return_value.eq.return_value.single.return_value.execute.return_value.data = clinic_row
        elif table_name == "Instance":
            mock_table.select.return_value.eq.return_value.single.return_value.execute.return_value.data = instance_row
        elif table_name == "Message":
            mock_table.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = message_rows
        return mock_table

    db.table.side_effect = table_side_effect
    return db


def _default_db_mock(message_rows: list[dict] | None = None) -> MagicMock:
    """Convenience: returns a db mock with a valid clinic + instance + given messages."""
    return _make_db_mock(
        clinic_row={"evolution_instance_id": INSTANCE_NAME},
        instance_row={"id": INSTANCE_UUID},
        message_rows=message_rows or [_make_message_row()],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueriesByInstanceId:
    """ING-01: The ingestor must resolve instance UUID and query by instanceId."""

    def test_queries_by_instance_id(self):
        """Message query must use the resolved instanceId UUID, not the clinic_id."""
        db_mock = _default_db_mock()

        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)

        # The Message table must have been queried
        message_table_calls = [
            call for call in db_mock.table.call_args_list
            if call.args[0] == "Message"
        ]
        assert len(message_table_calls) >= 1, "Expected at least one call to db.table('Message')"

        # The eq() filter must be called with the resolved instance UUID
        message_table_mock = db_mock.table("Message")
        db_mock.table.reset_mock()

        # Verify via call args on the actual mock chain
        # Re-run to capture the instanceId eq call cleanly
        db_mock2 = _make_db_mock(
            clinic_row={"evolution_instance_id": INSTANCE_NAME},
            instance_row={"id": INSTANCE_UUID},
            message_rows=[_make_message_row()],
        )
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock2):
            ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)

        # Confirm the Message table mock's .eq() was called with instanceId + UUID
        msg_table = db_mock2.table("Message")
        # The select chain: .select(...).eq("instanceId", INSTANCE_UUID)
        eq_call_args = msg_table.select.return_value.eq.call_args
        assert eq_call_args is not None
        assert eq_call_args.args[0] == "instanceId"
        assert eq_call_args.args[1] == INSTANCE_UUID


class TestSenderTypeMapping:
    """ING-02: fromMe=True → sender_type='clinic'; fromMe=False → 'patient'."""

    def test_sender_type_clinic(self):
        rows = [_make_message_row(from_me=True)]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert len(convs) == 1
        assert convs[0].messages[0].sender_type == "clinic"

    def test_sender_type_patient(self):
        rows = [_make_message_row(from_me=False)]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert len(convs) == 1
        assert convs[0].messages[0].sender_type == "patient"


class TestTimestampConversion:
    """ING-02: messageTimestamp (Unix int) must be converted to Python datetime."""

    def test_timestamp_conversion(self):
        ts = 1700000000
        rows = [_make_message_row(timestamp=ts)]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert len(convs) == 1
        expected_dt = datetime.fromtimestamp(ts)
        assert convs[0].messages[0].sent_at == expected_dt


class TestBodyExtraction:
    """ING-02: message body extraction from different JSONB shapes."""

    def test_body_conversation(self):
        rows = [_make_message_row(
            message_type="conversation",
            message_body={"conversation": "Ola, quero agendar"},
        )]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert convs[0].messages[0].content == "Ola, quero agendar"

    def test_body_extended_text_message(self):
        rows = [_make_message_row(
            message_type="extendedTextMessage",
            message_body={"extendedTextMessage": {"text": "Clique aqui: https://exemplo.com"}},
        )]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert convs[0].messages[0].content == "Clique aqui: https://exemplo.com"

    def test_body_media_fallback(self):
        rows = [_make_message_row(
            message_type="audioMessage",
            message_body={"audioMessage": {"url": "...", "seconds": 12}},
        )]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert convs[0].messages[0].content == "[audioMessage]"


class TestOutputTypeCompatibility:
    """ING-02: Returned objects must be instances of Conversation and Message from parser.py."""

    def test_output_type_compatibility(self):
        db_mock = _default_db_mock()
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        assert isinstance(convs, list)
        assert len(convs) >= 1
        conv = convs[0]
        assert isinstance(conv, Conversation), f"Expected Conversation, got {type(conv)}"
        assert len(conv.messages) >= 1
        msg = conv.messages[0]
        assert isinstance(msg, Message), f"Expected Message, got {type(msg)}"


class TestGroupJidExcluded:
    """ING-03: Rows with remoteJid ending in @g.us must NOT appear in output."""

    def test_group_jid_excluded(self):
        rows = [
            _make_message_row(remote_jid="120363XXX123@g.us"),
            _make_message_row(remote_jid="5511912345678@s.whatsapp.net"),
        ]
        db_mock = _default_db_mock(message_rows=rows)
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)
        jids = {c.source_filename for c in convs}
        assert "120363XXX123@g.us" not in jids
        assert "5511912345678@s.whatsapp.net" in jids


class TestIsolationByInstanceId:
    """ING-03: Only rows matching the resolved instanceId appear in output."""

    def test_isolation_by_instance_id(self):
        # The mock only returns rows for the exact instanceId queried.
        # We verify that the eq("instanceId", INSTANCE_UUID) call is made —
        # which is the guard that prevents cross-instance contamination.
        rows = [_make_message_row()]
        db_mock = _make_db_mock(
            clinic_row={"evolution_instance_id": INSTANCE_NAME},
            instance_row={"id": INSTANCE_UUID},
            message_rows=rows,
        )
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            convs = ingest_from_evolution(CLINIC_ID, CLINIC_SENDER)

        # Confirm exactly 1 conversation returned (only the mocked patient)
        assert len(convs) == 1

        # Confirm Message query eq used the correct instance UUID
        msg_table = db_mock.table("Message")
        eq_call_args = msg_table.select.return_value.eq.call_args
        assert eq_call_args.args[0] == "instanceId"
        assert eq_call_args.args[1] == INSTANCE_UUID


class TestResolveInvalidClinicId:
    """ING-03: Unknown clinic_id must raise ValueError with 'not found'."""

    def test_resolve_invalid_clinic_id(self):
        db_mock = _make_db_mock(
            clinic_row=None,  # clinic not found
            instance_row={"id": INSTANCE_UUID},
            message_rows=[],
        )
        with patch("analyzer.evolution_ingestor.get_db", return_value=db_mock):
            with pytest.raises(ValueError, match="not found"):
                ingest_from_evolution("unknown-clinic-id", CLINIC_SENDER)
