"""
metrics.py
----------
Pure-Python KPI calculation — no LLM calls here.

Takes a Conversation object (from parser.py) and returns a ConversationMetrics
dataclass with all measurable, deterministic KPIs.

KPIs computed:
  - message_count, clinic_message_count, patient_message_count
  - date_start, date_end, duration_days
  - avg_response_time_seconds (clinic response to patient messages)
  - first_response_time_seconds
  - max_response_time_seconds
  - median_response_time_seconds
  - messages_per_day
  - confirmation_rate  (ratio of confirmation keywords in clinic messages)
  - reminders_needed   (how many clinic messages before patient replied)
  - silence_periods    (clinic sent ≥2 messages without patient reply)
  - unanswered_count   (patient questions with no clinic follow-up)
  - busiest_hour       (0–23 UTC)
  - busiest_weekday    (0=Monday … 6=Sunday)
"""

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from analyzer.parser import Conversation, Message

# Keywords that indicate a patient confirmed an appointment
CONFIRMATION_KEYWORDS = [
    "confirmad",   # "confirmado", "confirmada"
    "confirmo",
    "confirm",
    "vou comparecer",
    "estarei lá",
    "estarei la",
    "ok",
    "okay",
    "tudo certo",
    "pode marcar",
    "pode confirmar",
    "sim",
    "claro",
    "combinado",
    "certo",
    "ótimo",
    "otimo",
]

# Keywords for reminder / confirmation-request messages from clinic
REMINDER_KEYWORDS = [
    "lembr",       # "lembrete", "lembrando"
    "confirm",
    "consulta",
    "agendad",
    "agendamento",
    "retorn",
]

# Keywords for cancellation
CANCELLATION_KEYWORDS = [
    "cancel",
    "desmarcar",
    "não poderei",
    "nao poderei",
    "não consigo",
    "nao consigo",
    "remarcar",
    "reagend",
]

# Response time threshold: if gap between messages > this, we consider it a
# new conversation "session" rather than a response (e.g. next-day follow-up)
SESSION_GAP_HOURS = 12
MAX_RESPONSE_SECONDS = SESSION_GAP_HOURS * 3600

# Business hours for off-hours analysis (used by report_builder too)
BUSINESS_HOURS_START = 8    # inclusive  (08:00)
BUSINESS_HOURS_END   = 18   # exclusive  (from 18:00 onwards = off-hours)


def _contains(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


@dataclass
class ConversationMetrics:
    phone: str
    source_filename: str

    # Volume
    message_count: int = 0
    clinic_message_count: int = 0
    patient_message_count: int = 0

    # Timeline
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    duration_days: int = 0

    # Response times (clinic responding to patient) — in seconds
    avg_response_time_seconds: Optional[float] = None
    first_response_time_seconds: Optional[float] = None
    max_response_time_seconds: Optional[float] = None
    median_response_time_seconds: Optional[float] = None

    # Activity
    messages_per_day: float = 0.0
    busiest_hour: Optional[int] = None       # 0–23
    busiest_weekday: Optional[int] = None    # 0=Mon … 6=Sun

    # Appointment flow
    confirmation_rate: float = 0.0           # 0.0 – 1.0
    has_cancellation: bool = False
    reminders_needed: int = 0                # consecutive clinic msgs before patient reply
    silence_periods: int = 0                 # clinic sent ≥2 msgs without patient reply

    # Quality signals
    unanswered_patient_messages: int = 0    # patient msgs never followed by clinic
    avg_patient_message_length: float = 0.0
    avg_clinic_message_length: float = 0.0

    # Raw response times for aggregation
    response_times: list[float] = field(default_factory=list, repr=False)

    # Hour-of-day (0–23) for each patient message — used for hourly heatmap
    patient_message_hours: list[int] = field(default_factory=list, repr=False)


def compute_metrics(conv: Conversation) -> ConversationMetrics:
    msgs = conv.messages
    m = ConversationMetrics(
        phone=conv.phone,
        source_filename=conv.source_filename,
    )

    if not msgs:
        return m

    # -- Basic counts --
    m.message_count = len(msgs)
    m.clinic_message_count = len(conv.clinic_messages)
    m.patient_message_count = len(conv.patient_messages)

    # -- Patient message hours (for hourly contact heatmap) --
    m.patient_message_hours = [
        msg.sent_at.hour for msg in msgs if msg.sender_type == "patient"
    ]

    # -- Timeline --
    m.date_start = conv.date_start
    m.date_end = conv.date_end
    if m.date_start and m.date_end:
        m.duration_days = max(1, (m.date_end - m.date_start).days)
        m.messages_per_day = round(m.message_count / m.duration_days, 2)

    # -- Response times (clinic → patient) --
    response_times: list[float] = []
    for i, msg in enumerate(msgs):
        if msg.sender_type == "patient":
            # Find the next clinic message after this patient message
            for j in range(i + 1, len(msgs)):
                next_msg = msgs[j]
                if next_msg.sender_type == "patient":
                    break  # clinic never replied before patient sent again
                if next_msg.sender_type == "clinic":
                    delta = (next_msg.sent_at - msg.sent_at).total_seconds()
                    if 0 < delta < MAX_RESPONSE_SECONDS:
                        response_times.append(delta)
                    break

    if response_times:
        m.response_times = response_times
        m.avg_response_time_seconds = round(statistics.mean(response_times), 1)
        m.first_response_time_seconds = round(response_times[0], 1)
        m.max_response_time_seconds = round(max(response_times), 1)
        if len(response_times) >= 2:
            m.median_response_time_seconds = round(
                statistics.median(response_times), 1
            )

    # -- Busiest hour and weekday --
    all_hours = [msg.sent_at.hour for msg in msgs]
    all_weekdays = [msg.sent_at.weekday() for msg in msgs]
    if all_hours:
        m.busiest_hour = max(set(all_hours), key=all_hours.count)
        m.busiest_weekday = max(set(all_weekdays), key=all_weekdays.count)

    # -- Appointment flow: confirmation & cancellation --
    clinic_msgs = conv.clinic_messages
    patient_msgs = conv.patient_messages

    reminder_msgs = [
        cm for cm in clinic_msgs if _contains(cm.content, REMINDER_KEYWORDS)
    ]
    confirmed_msgs = [
        pm for pm in patient_msgs if _contains(pm.content, CONFIRMATION_KEYWORDS)
    ]
    cancellation_msgs = [
        pm for pm in patient_msgs if _contains(pm.content, CANCELLATION_KEYWORDS)
    ]

    if reminder_msgs:
        m.confirmation_rate = round(
            min(len(confirmed_msgs), len(reminder_msgs)) / len(reminder_msgs), 2
        )
    m.has_cancellation = len(cancellation_msgs) > 0

    # -- Reminders needed: consecutive clinic messages without patient reply --
    max_consecutive = 0
    current_consecutive = 0
    for msg in msgs:
        if msg.sender_type == "clinic":
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    m.reminders_needed = max_consecutive

    # -- Silence periods (clinic sent ≥2 msgs in a row without patient reply) --
    silence = 0
    consecutive = 0
    for msg in msgs:
        if msg.sender_type == "clinic":
            consecutive += 1
            if consecutive == 2:
                silence += 1
        else:
            consecutive = 0
    m.silence_periods = silence

    # -- Unanswered patient messages --
    unanswered = 0
    for i, msg in enumerate(msgs):
        if msg.sender_type != "patient":
            continue
        # Check if clinic ever replied after this
        followed_up = any(
            m2.sender_type == "clinic"
            for m2 in msgs[i + 1:]
        )
        if not followed_up:
            unanswered += 1
    m.unanswered_patient_messages = unanswered

    # -- Average message lengths --
    if clinic_msgs:
        m.avg_clinic_message_length = round(
            statistics.mean(len(cm.content) for cm in clinic_msgs), 1
        )
    if patient_msgs:
        m.avg_patient_message_length = round(
            statistics.mean(len(pm.content) for pm in patient_msgs), 1
        )

    return m


@dataclass
class AggregatedMetrics:
    """Summary KPIs across all conversations in a job."""
    total_conversations: int = 0
    total_messages: int = 0

    avg_response_time_seconds: Optional[float] = None
    median_response_time_seconds: Optional[float] = None
    p90_response_time_seconds: Optional[float] = None

    avg_confirmation_rate: float = 0.0
    cancellation_count: int = 0
    cancellation_rate: float = 0.0

    avg_quality_score: Optional[float] = None
    avg_sentiment_score: Optional[float] = None
    avg_health_score: Optional[float] = None

    most_common_topics: list[str] = field(default_factory=list)
    most_common_flags: list[str] = field(default_factory=list)

    conversations_with_issues: int = 0   # flags present
    conversations_no_reply: int = 0      # unanswered_patient_messages > 0

    busiest_hour: Optional[int] = None
    busiest_weekday: Optional[int] = None

    # Hourly contact distribution — patient messages per hour of day (index = hour 0–23)
    hourly_contact_distribution: list[int] = field(
        default_factory=lambda: [0] * 24
    )

    # Response time distribution across ALL interactions (not just first response)
    # Keys: "golden" (<5min), "good" (5-10min), "ok" (10-30min), "slow" (30-60min),
    #       "very_slow" (1-4h), "bad" (4-24h), "critical" (>24h)
    response_time_distribution: dict[str, int] = field(default_factory=lambda: {
        "golden":    0,   # < 5 min
        "good":      0,   # 5–10 min
        "ok":        0,   # 10–30 min
        "slow":      0,   # 30–60 min
        "very_slow": 0,   # 1–4 h
        "bad":       0,   # 4–24 h
        "critical":  0,   # > 24 h
    })


def _bucket_response_time(seconds: float) -> str:
    """Return the distribution bucket key for a response time in seconds."""
    if seconds < 300:        return "golden"    # < 5 min
    if seconds < 600:        return "good"      # 5–10 min
    if seconds < 1800:       return "ok"        # 10–30 min
    if seconds < 3600:       return "slow"      # 30–60 min
    if seconds < 14400:      return "very_slow" # 1–4 h
    if seconds < 86400:      return "bad"       # 4–24 h
    return "critical"                           # > 24 h


def aggregate_metrics(metrics_list: list[ConversationMetrics]) -> AggregatedMetrics:
    """Roll up per-conversation metrics into a single summary."""
    agg = AggregatedMetrics()
    agg.total_conversations = len(metrics_list)

    if not metrics_list:
        return agg

    agg.total_messages = sum(m.message_count for m in metrics_list)
    agg.cancellation_count = sum(1 for m in metrics_list if m.has_cancellation)
    agg.conversations_no_reply = sum(
        1 for m in metrics_list if m.unanswered_patient_messages > 0
    )

    if agg.total_conversations:
        agg.cancellation_rate = round(
            agg.cancellation_count / agg.total_conversations, 2
        )

    # Flatten all response times
    all_rt = [rt for m in metrics_list for rt in m.response_times]
    if all_rt:
        all_rt_sorted = sorted(all_rt)
        agg.avg_response_time_seconds = round(statistics.mean(all_rt), 1)
        agg.median_response_time_seconds = round(statistics.median(all_rt), 1)
        p90_idx = int(len(all_rt_sorted) * 0.9)
        agg.p90_response_time_seconds = round(all_rt_sorted[p90_idx], 1)

    # Response time distribution across all interactions (including capped ones removed
    # from the 12h window — here we use the raw per-conversation response_times lists,
    # which are already capped at MAX_RESPONSE_SECONDS; interactions above that cap
    # are counted as "critical" separately by checking unanswered/silence data —
    # for simplicity we bucket everything we have and note the cap in the chart label)
    dist = agg.response_time_distribution
    for rt in all_rt:
        dist[_bucket_response_time(rt)] += 1

    # Hourly contact distribution (patient messages only)
    for m in metrics_list:
        for h in m.patient_message_hours:
            agg.hourly_contact_distribution[h] += 1

    conf_rates = [m.confirmation_rate for m in metrics_list]
    if conf_rates:
        agg.avg_confirmation_rate = round(statistics.mean(conf_rates), 2)

    # Busiest hour/weekday across all conversations
    all_hours = [m.busiest_hour for m in metrics_list if m.busiest_hour is not None]
    all_days = [m.busiest_weekday for m in metrics_list if m.busiest_weekday is not None]
    if all_hours:
        agg.busiest_hour = max(set(all_hours), key=all_hours.count)
    if all_days:
        agg.busiest_weekday = max(set(all_days), key=all_days.count)

    return agg
