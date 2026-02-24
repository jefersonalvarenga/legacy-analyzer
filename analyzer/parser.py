"""
parser.py
---------
Extracts and normalises WhatsApp conversation data from:
  - A single .zip containing many inner .zip files (Archive.zip from client)
  - Each inner .zip contains one .txt (the exported WhatsApp chat)

Supports pt-BR WhatsApp export format:
  DD/MM/YYYY HH:MM - Sender: Message text
  DD/MM/YYYY HH:MM - Sender: (continued multi-line)

Returns a list of Conversation dataclass objects, each with a list of Message objects.
"""

import re
import zipfile
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet

logger = logging.getLogger(__name__)

# WhatsApp pt-BR export line pattern
# Examples:
#   03/07/2024 10:23 - Sorriso Da Gente: Olá, bom dia!
#   03/07/2024 10:23 - 55119XXXXXXXX: Tudo bem
WHATSAPP_LINE_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{4})\s(\d{2}:\d{2})\s-\s(.+?):\s(.*)$"
)

# Lines that are WhatsApp system messages (not real messages)
SYSTEM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"mensagens e chamadas são protegidas",
        r"messages and calls are end-to-end encrypted",
        r"mídia omitida",
        r"<mídia omitida>",
        r"áudio omitido",
        r"figurinha omitida",
        r"imagem omitida",
        r"vídeo omitido",
        r"arquivo omitido",
        r"contato omitido",
        r"localização omitida",
        r"você foi adicionado",
        r"you were added",
    ]
]


@dataclass
class Message:
    sent_at: datetime
    sender: str
    sender_type: str          # "clinic" | "patient" | "system"
    content: str
    raw_line: str = field(repr=False, default="")


@dataclass
class Conversation:
    source_filename: str      # original .zip filename (e.g. "chat_5511912345678.zip")
    phone: str                # extracted phone or "unknown"
    messages: list[Message] = field(default_factory=list)

    @property
    def date_start(self) -> Optional[datetime]:
        return self.messages[0].sent_at if self.messages else None

    @property
    def date_end(self) -> Optional[datetime]:
        return self.messages[-1].sent_at if self.messages else None

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def clinic_messages(self) -> list[Message]:
        return [m for m in self.messages if m.sender_type == "clinic"]

    @property
    def patient_messages(self) -> list[Message]:
        return [m for m in self.messages if m.sender_type == "patient"]


def _decode_bytes(raw: bytes) -> str:
    """Decode bytes to string, detecting encoding."""
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def _is_system_message(content: str) -> bool:
    return any(p.search(content) for p in SYSTEM_PATTERNS)


def _classify_sender(sender: str, clinic_sender_name: str) -> str:
    """
    Determine whether a sender is the clinic, a patient, or system.
    - If the sender matches the clinic name → "clinic"
    - If it looks like a phone number → "patient"
    - Otherwise → "patient" (could be a named patient)
    """
    if sender.strip().lower() == clinic_sender_name.strip().lower():
        return "clinic"
    # Phone numbers: digits, spaces, dashes, plus signs
    if re.match(r"^[\d\s\+\-\(\)]+$", sender.strip()):
        return "patient"
    return "patient"


def _extract_phone_from_filename(filename: str) -> str:
    """
    Try to extract a phone number from the zip filename.

    Handles formats exported by WhatsApp pt-BR:
      Conversa do WhatsApp com +55 11 91557-5104.zip
      WhatsApp Chat with 5511912345678.zip
      5511912345678.zip
    """
    # Strip all non-digit characters, then look for a sequence of 8–15 digits
    digits_only = re.sub(r"\D", "", filename)
    if len(digits_only) >= 8:
        return digits_only
    return "unknown"


def _parse_txt_content(
    text: str,
    source_filename: str,
    clinic_sender_name: str,
) -> Conversation:
    """Parse the full text of one WhatsApp export .txt file."""
    phone = _extract_phone_from_filename(source_filename)
    conv = Conversation(source_filename=source_filename, phone=phone)

    current_msg: Optional[Message] = None

    for raw_line in text.splitlines():
        match = WHATSAPP_LINE_RE.match(raw_line)

        if match:
            # Save previous message
            if current_msg is not None:
                if not _is_system_message(current_msg.content):
                    conv.messages.append(current_msg)

            date_str, time_str, sender, content = match.groups()
            try:
                sent_at = datetime.strptime(
                    f"{date_str} {time_str}", "%d/%m/%Y %H:%M"
                )
            except ValueError:
                sent_at = datetime.min

            sender_type = _classify_sender(sender, clinic_sender_name)
            current_msg = Message(
                sent_at=sent_at,
                sender=sender.strip(),
                sender_type=sender_type,
                content=content.strip(),
                raw_line=raw_line,
            )
        else:
            # Continuation of previous message (multi-line)
            if current_msg is not None and raw_line.strip():
                current_msg.content += "\n" + raw_line.strip()

    # Append last message
    if current_msg is not None and not _is_system_message(current_msg.content):
        conv.messages.append(current_msg)

    return conv


def _parse_zip_bytes(
    zip_bytes: bytes,
    zip_name: str,
    clinic_sender_name: str,
) -> Optional[Conversation]:
    """
    Parse one inner .zip file (containing a single .txt export).
    Returns a Conversation or None if no valid content found.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txt_names:
                logger.warning("No .txt found in %s", zip_name)
                return None

            txt_raw = zf.read(txt_names[0])
            txt_content = _decode_bytes(txt_raw)
            conv = _parse_txt_content(txt_content, zip_name, clinic_sender_name)

            if not conv.messages:
                logger.warning("No messages parsed from %s", zip_name)
                return None

            return conv

    except zipfile.BadZipFile:
        logger.error("Bad zip file: %s", zip_name)
        return None
    except Exception as exc:
        logger.error("Error parsing %s: %s", zip_name, exc)
        return None


def parse_archive(
    archive_path: str | Path,
    clinic_sender_name: str,
    on_progress: Optional[callable] = None,
) -> list[Conversation]:
    """
    Main entry point. Reads the outer Archive.zip, iterates over each
    inner .zip, parses the WhatsApp .txt, and returns all conversations.

    Args:
        archive_path:       Path to the outer Archive.zip
        clinic_sender_name: Display name the clinic uses on WhatsApp
        on_progress:        Optional callback(current: int, total: int, filename: str)

    Returns:
        List of Conversation objects (only non-empty ones)
    """
    archive_path = Path(archive_path)
    conversations: list[Conversation] = []

    with zipfile.ZipFile(archive_path) as outer:
        inner_zips = sorted([
            n for n in outer.namelist()
            if n.lower().endswith(".zip")
            and not n.startswith("__MACOSX/")   # ignore macOS metadata entries
            and not Path(n).name.startswith("._")
        ])
        total = len(inner_zips)
        logger.info("Found %d inner zip files in %s", total, archive_path.name)

        for idx, zip_name in enumerate(inner_zips, start=1):
            zip_bytes = outer.read(zip_name)
            short_name = Path(zip_name).name
            conv = _parse_zip_bytes(zip_bytes, short_name, clinic_sender_name)

            if conv:
                conversations.append(conv)

            if on_progress:
                on_progress(idx, total, short_name)

    logger.info(
        "Parsed %d valid conversations out of %d zip files",
        len(conversations),
        total,
    )
    return conversations
