"""
structured_log.py
-----------------
Structured JSON logger for stdout. Reads in Easypanel container viewer.

Usage:
    from analyzer.structured_log import slog

    slog("la.run.started", clinic_id=clinic_id, job_id=job_id)
    slog("la.run.failed", clinic_id=clinic_id, job_id=job_id, error=str(e), level="error")

Conventions:
- event: snake.case domain.verb. Stable name, searchable.
- clinic_id / job_id / request_id: include whenever available.
- level: 'info' (default) | 'warn' | 'error'.
- NEVER include PII (patient name, phone, email content). Pass IDs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Literal

Level = Literal["info", "warn", "error"]
SERVICE = "legacy-analyzer"


def slog(
    event: str,
    *,
    level: Level = "info",
    clinic_id: str | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured JSON log line to stdout (or stderr if level=error)."""
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "level": level,
        "service": SERVICE,
        "event": event,
    }
    if clinic_id is not None:
        payload["clinic_id"] = clinic_id
    if job_id is not None:
        payload["job_id"] = job_id
    if request_id is not None:
        payload["request_id"] = request_id
    payload.update(extra)

    line = json.dumps(payload, default=str, ensure_ascii=False)
    stream = sys.stderr if level == "error" else sys.stdout
    print(line, file=stream, flush=True)
