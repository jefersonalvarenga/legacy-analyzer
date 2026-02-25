"""
notifier.py
-----------
Sends WhatsApp notifications via n8n webhook when analysis jobs complete or fail.

The n8n webhook receives a JSON payload and forwards it to Evolution API.
Configure NOTIFY_WEBHOOK_URL and NOTIFY_PHONE in .env to enable.

n8n workflow needed: la-notify
  Trigger: Webhook POST /webhook/la-notify
  Action:  Send WhatsApp message via Evolution API to payload.phone
"""

import logging
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # seconds


def notify_job_done(
    webhook_url: str,
    phone: str,
    client_name: str,
    job_id: str,
    total_conversations: int,
    knowledge_summary: dict | None = None,
):
    """
    POST job completion notification to n8n webhook.

    Args:
        webhook_url:          NOTIFY_WEBHOOK_URL from config
        phone:                NOTIFY_PHONE — recipient in format 5511999999999
        client_name:          Display name of the analyzed clinic
        job_id:               UUID of the job
        total_conversations:  Number of conversations analyzed
        knowledge_summary:    Optional dict with confirmed_insurances, confirmed_address, etc.
    """
    insurances = []
    address = ""
    if knowledge_summary:
        insurances = knowledge_summary.get("confirmed_insurances", [])
        address = knowledge_summary.get("confirmed_address", "")

    lines = [f"✅ *Análise concluída — {client_name}*"]
    lines.append(f"📊 {total_conversations} conversas processadas")

    if insurances:
        lines.append(f"🏥 Convênios detectados: {', '.join(insurances)}")
    if address:
        lines.append(f"📍 Endereço: {address}")

    lines.append(f"\n🔗 Job: `{job_id[:8]}`")
    lines.append("Blueprint disponível no Supabase (la_blueprints).")

    message = "\n".join(lines)

    _post(webhook_url, {
        "event": "job_done",
        "phone": phone,
        "message": message,
        "job_id": job_id,
        "client_name": client_name,
        "total_conversations": total_conversations,
        "knowledge_summary": knowledge_summary or {},
    })


def notify_job_failed(
    webhook_url: str,
    phone: str,
    client_name: str,
    job_id: str,
    error: str,
):
    """POST job failure notification to n8n webhook."""
    message = (
        f"❌ *Erro na análise — {client_name}*\n"
        f"Job `{job_id[:8]}` falhou.\n"
        f"Erro: {error[:200]}"
    )
    _post(webhook_url, {
        "event": "job_failed",
        "phone": phone,
        "message": message,
        "job_id": job_id,
        "client_name": client_name,
        "error": error[:500],
    })


def _post(webhook_url: str, payload: dict):
    try:
        resp = requests.post(webhook_url, json=payload, timeout=_TIMEOUT)
        if resp.status_code < 300:
            logger.info("[Notifier] Webhook sent OK (%d)", resp.status_code)
        else:
            logger.warning("[Notifier] Webhook returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        # Notification failure must never crash the worker
        logger.warning("[Notifier] Failed to send webhook: %s", e)
