"""
evolution_ingestor.py
---------------------
Read-only adapter that queries the Evolution API's ``Message`` table in the
shared Supabase database and produces ``list[Conversation]`` — identical in
type to ``parse_archive()`` output from ``analyzer.parser``.

Purpose: Replace Archive.zip as the message source for online analysis.
         The existing pipeline (metrics, DSPy, outcome detection, Shadow DNA,
         blueprint) consumes the returned list without any modification.

Architecture:
  1. _resolve_instance_id() — two-hop lookup:
       sf_clinics.evolution_instance_id (human-readable name)
       → Instance.id (UUID used as FK on Message rows)
  2. Message query — read-only .select() filtered by instanceId + days_back
  3. _group_messages_by_conversation() — group rows by remoteJid, build
       Conversation + Message objects matching parser.py's dataclasses exactly

Invariants:
  - ZERO writes: only .select() calls — never .insert(), .update(),
    .delete(), or .upsert()
  - Group JIDs (@g.us) are silently excluded
  - Rows with missing / falsy remoteJid are silently skipped
  - fromMe is the sole classifier for sender_type — pushName is NEVER used
    to classify clinic vs patient
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from analyzer.parser import Conversation, Message
from db import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_instance_id(db, clinic_id: str) -> str:
    """
    Two-hop lookup: clinic_id → evolution_instance_id (name) → Instance UUID.

    Raises ValueError with a human-readable message if any hop fails.
    """
    # Hop 1: sf_clinics → evolution_instance_id (name string)
    clinic_result = (
        db.table("sf_clinics")
        .select("evolution_instance_id")
        .eq("id", clinic_id)
        .single()
        .execute()
    )
    clinic_data = clinic_result.data
    if not clinic_data:
        raise ValueError(
            f"Clinic '{clinic_id}' not found in sf_clinics table"
        )
    instance_name: str | None = clinic_data.get("evolution_instance_id")
    if not instance_name:
        raise ValueError(
            f"Clinic '{clinic_id}' has no evolution_instance_id configured"
        )

    # Hop 2: Instance (Evolution table) WHERE name=instance_name → id (UUID)
    instance_result = (
        db.table("Instance")
        .select("id")
        .eq("name", instance_name)
        .single()
        .execute()
    )
    instance_data = instance_result.data
    if not instance_data:
        raise ValueError(
            f"Evolution Instance with name '{instance_name}' not found "
            f"(clinic_id='{clinic_id}')"
        )

    return instance_data["id"]


def _extract_body(message_json: dict | None, message_type: str) -> str:
    """
    Extract the human-readable text from a Message row's ``message`` JSONB.

    Supported shapes:
      {"conversation": "text"}
      {"extendedTextMessage": {"text": "text"}}
      {"audioMessage": {...}}, {"imageMessage": {...}}, etc.

    Falls back to "[{message_type}]" for media / unknown shapes.
    """
    if not message_json:
        return f"[{message_type}]"

    # Direct text message
    conversation_text = message_json.get("conversation")
    if conversation_text is not None:
        return str(conversation_text)

    # Extended text (links, quoted messages, etc.)
    extended = message_json.get("extendedTextMessage")
    if extended and extended.get("text") is not None:
        return str(extended["text"])

    # Media or unknown — return a labelled placeholder
    return f"[{message_type}]"


def _extract_phone(remote_jid: str) -> str:
    """
    Strip WhatsApp suffix to get a bare phone number.

    "5511912345678@s.whatsapp.net"  → "5511912345678"
    "5511912345678@lid"             → "5511912345678"
    "unknown"                       → "unknown"
    """
    if "@" in remote_jid:
        return remote_jid.split("@")[0]
    return remote_jid


def _is_group_jid(remote_jid: str) -> bool:
    """Return True when remoteJid represents a WhatsApp group (@g.us)."""
    return remote_jid.endswith("@g.us")


def _build_sender_type(from_me: bool) -> str:
    """Map Evolution's fromMe flag to parser.py's sender_type."""
    return "clinic" if from_me else "patient"


def _group_messages_by_conversation(
    rows: list[dict],
    clinic_sender_name: str,
) -> list[Conversation]:
    """
    Convert a flat list of Evolution Message rows into grouped Conversation objects.

    - Skips rows with missing / falsy remoteJid
    - Skips group JIDs (@g.us)
    - Groups by remoteJid, sorts messages by sent_at within each group
    - Sets source_filename = remoteJid (conversation identifier)
    - Sets phone = _extract_phone(remoteJid)
    """
    buckets: dict[str, list[Message]] = defaultdict(list)

    for row in rows:
        key: dict = row.get("key") or {}
        remote_jid: str = key.get("remoteJid") or ""

        # Skip missing or group JIDs
        if not remote_jid:
            logger.debug("Skipping row with empty remoteJid")
            continue
        if _is_group_jid(remote_jid):
            logger.debug("Skipping group JID: %s", remote_jid)
            continue

        from_me: bool = bool(key.get("fromMe", False))
        sender_type = _build_sender_type(from_me)

        # Determine sender display name
        if from_me:
            sender = clinic_sender_name
        else:
            push_name: str | None = row.get("pushName")
            sender = push_name or _extract_phone(remote_jid)

        # Convert Unix timestamp → datetime
        raw_ts = row.get("messageTimestamp")
        if raw_ts is not None:
            sent_at = datetime.fromtimestamp(int(raw_ts))
        else:
            sent_at = datetime.min

        message_type: str = row.get("messageType") or "unknown"
        message_json: dict | None = row.get("message")
        content = _extract_body(message_json, message_type)

        msg = Message(
            sent_at=sent_at,
            sender=sender,
            sender_type=sender_type,
            content=content,
            raw_line="",  # no raw text line for Evolution-sourced messages
        )
        buckets[remote_jid].append(msg)

    conversations: list[Conversation] = []
    for remote_jid, messages in buckets.items():
        # Sort messages chronologically within the conversation
        messages.sort(key=lambda m: m.sent_at)
        conv = Conversation(
            source_filename=remote_jid,
            phone=_extract_phone(remote_jid),
            messages=messages,
        )
        conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_from_evolution(
    clinic_id: str,
    clinic_sender_name: str,
    on_progress: Optional[callable] = None,
    days_back: int = 90,
) -> list[Conversation]:
    """
    Query the Evolution API's Message table and return ``list[Conversation]``.

    The returned list is type-compatible with ``parse_archive()`` output and
    can be passed directly to the existing pipeline without modification.

    Args:
        clinic_id:          UUID of the clinic in sf_clinics
        clinic_sender_name: WhatsApp display name used by the clinic
        on_progress:        Optional callback(current: int, total: int) —
                            called after message fetch with (1, 1) for now
        days_back:          Limit messages to the last N days (default 90)
                            — guards against unbounded data volume

    Returns:
        list[Conversation] — one entry per unique remoteJid (individual only)

    Raises:
        ValueError: if clinic_id is not found in sf_clinics, or if the
                    associated Evolution Instance cannot be resolved
    """
    db = get_db()

    # Step 1: Resolve clinic → Evolution instance UUID (fail fast)
    instance_uuid = _resolve_instance_id(db, clinic_id)
    logger.info(
        "Resolved clinic '%s' → instance UUID '%s'", clinic_id, instance_uuid
    )

    # Step 2: Compute the cutoff timestamp for the days_back guard
    cutoff_unix = int(
        (datetime.utcnow() - timedelta(days=days_back)).timestamp()
    )

    # Step 3: Fetch messages — READ ONLY, no writes
    result = (
        db.table("Message")
        .select("id, key, pushName, message, messageType, messageTimestamp")
        .eq("instanceId", instance_uuid)
        .gte("messageTimestamp", cutoff_unix)
        .order("messageTimestamp", desc=False)
        .execute()
    )
    rows: list[dict] = result.data or []
    logger.info(
        "Fetched %d Message rows for instance '%s' (last %d days)",
        len(rows),
        instance_uuid,
        days_back,
    )

    if on_progress:
        on_progress(1, 1)

    # Step 4: Group rows into Conversation objects
    conversations = _group_messages_by_conversation(rows, clinic_sender_name)
    logger.info(
        "Produced %d conversations for clinic '%s'",
        len(conversations),
        clinic_id,
    )

    return conversations
