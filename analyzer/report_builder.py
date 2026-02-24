"""
report_builder.py
-----------------
Generates a self-contained HTML report from all conversation analyses.

Sections:
  1. Header + Health Score hero
  2. Oportunidade Financeira (opportunity loss — headline for decision maker)
  3. Visão Geral — KPI cards
  4. Desfechos — outcome distribution chart + conversion rate
  5. Análises — sentiment, quality, topics, flags charts
  6. Shadow DNA — tone, emojis, greeting/closing examples
  7. Objeções + Lacunas de Conhecimento
  8. Conversas que precisam de atenção / Melhores conversas
  9. Sugestões de Melhoria
  10. Footer

EasyScale brand: primary #635BFF · secondary #00AFE1 · dark #0F172A
"""

import json
import statistics
from datetime import datetime
from typing import Optional

from analyzer.metrics import AggregatedMetrics, ConversationMetrics, BUSINESS_HOURS_START, BUSINESS_HOURS_END
from analyzer.dspy_pipeline import SemanticAnalysis
from analyzer.outcome_detection import OutcomeSummary
from analyzer.shadow_dna import ShadowDNA
from analyzer.financial_kpis import FinancialKPIs


WEEKDAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}min"
    if minutes > 0:
        return f"{minutes}min {secs}s"
    return f"{secs}s"


def _fmt_brl(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", ".")


def _fmt_pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _health_color(score: float) -> str:
    if score >= 75:
        return "#22C55E"
    if score >= 50:
        return "#F59E0B"
    return "#EF4444"


def _sentiment_emoji(label: str) -> str:
    return {
        "muito_positivo": "😊",
        "positivo": "🙂",
        "neutro": "😐",
        "negativo": "😕",
        "muito_negativo": "😟",
    }.get(label, "😐")


def _outcome_color(outcome: str) -> str:
    return {
        "agendado":      "#22C55E",
        "ghosting":      "#94A3B8",
        "objecao_ativa": "#EF4444",
        "pendente":      "#F59E0B",
        "outro":         "#635BFF",
    }.get(outcome, "#635BFF")


def _li(items: list[str], cls: str = "") -> str:
    if not items:
        return "<li class='muted'>Nenhum identificado</li>"
    return "".join(f"<li class='{cls}'>{i}</li>" for i in items)


# ------------------------------------------------------------------
# Main builder
# ------------------------------------------------------------------

def build_report(
    client_name: str,
    client_slug: str,
    agg: AggregatedMetrics,
    metrics_list: list[ConversationMetrics],
    analyses: list[SemanticAnalysis],
    outcome_summary: Optional[OutcomeSummary] = None,
    shadow_dna: Optional[ShadowDNA] = None,
    financial_kpis: Optional[FinancialKPIs] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    if generated_at is None:
        generated_at = datetime.now()

    # ------------------------------------------------------------------
    # Computed values
    # ------------------------------------------------------------------
    health_scores = [a.health_score for a in analyses]
    avg_health   = round(statistics.mean(health_scores), 1) if health_scores else 0.0
    avg_quality  = round(statistics.mean(a.quality_score for a in analyses), 1) if analyses else 0.0
    avg_sentiment= round(statistics.mean(a.sentiment_score for a in analyses), 2) if analyses else 0.0
    agg.avg_health_score   = avg_health
    agg.avg_quality_score  = avg_quality
    agg.avg_sentiment_score= avg_sentiment
    health_color = _health_color(avg_health)

    busiest_weekday_name = WEEKDAY_NAMES[agg.busiest_weekday] if agg.busiest_weekday is not None else "—"
    busiest_hour_str     = f"{agg.busiest_hour:02d}h" if agg.busiest_hour is not None else "—"

    # Sentiment chart
    sentiment_counts = {"muito_positivo": 0, "positivo": 0, "neutro": 0, "negativo": 0, "muito_negativo": 0}
    for a in analyses:
        lbl = a.sentiment_label if a.sentiment_label in sentiment_counts else "neutro"
        sentiment_counts[lbl] += 1
    sentiment_labels_pt = {
        "muito_positivo": "Muito Positivo", "positivo": "Positivo",
        "neutro": "Neutro", "negativo": "Negativo", "muito_negativo": "Muito Negativo",
    }
    sent_chart_labels = [sentiment_labels_pt[k] for k in sentiment_counts]
    sent_chart_data   = list(sentiment_counts.values())
    sent_chart_colors = ["#22C55E", "#86EFAC", "#94A3B8", "#FCA5A5", "#EF4444"]

    # Topics chart
    topic_freq: dict[str, int] = {}
    for a in analyses:
        for t in a.topics:
            topic_freq[t] = topic_freq.get(t, 0) + 1
    top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Dynamic insight: confirmação vs agendamento topic ranking
    confirmacao_rank = next(
        (i for i, (t, _) in enumerate(top_topics) if "confirm" in t.lower()), None
    )
    agendamento_rank = next(
        (i for i, (t, _) in enumerate(top_topics) if "agend" in t.lower()), None
    )
    if confirmacao_rank is not None and agendamento_rank is not None and confirmacao_rank < agendamento_rank:
        topics_insight_html = (
            f"<div style='margin-top:0.75rem;padding:0.75rem 1rem;"
            f"background:#635BFF18;border-left:3px solid #635BFF;"
            f"border-radius:0 8px 8px 0;font-size:0.82rem;color:#CBD5E1'>"
            f"<strong style='color:#635BFF'>💡 Insight:</strong> "
            f"&ldquo;Confirmação de consulta&rdquo; (#{confirmacao_rank+1}) supera "
            f"&ldquo;Agendamento&rdquo; (#{agendamento_rank+1}) nos tópicos — a maioria dos "
            f"contatos é de pacientes <em>já agendados</em> confirmando presença, não de novos leads. "
            f"Um agente de IA pode absorver este volume integral e 24h."
            f"</div>"
        )
    elif confirmacao_rank is not None and agendamento_rank is None:
        topics_insight_html = (
            f"<div style='margin-top:0.75rem;padding:0.75rem 1rem;"
            f"background:#635BFF18;border-left:3px solid #635BFF;"
            f"border-radius:0 8px 8px 0;font-size:0.82rem;color:#CBD5E1'>"
            f"<strong style='color:#635BFF'>💡 Insight:</strong> "
            f"Confirmações dominam os tópicos — alto potencial para automação de "
            f"lembretes e confirmações 24h com IA."
            f"</div>"
        )
    else:
        topics_insight_html = ""
    topic_chart_labels = [t[0] for t in top_topics]
    topic_chart_data   = [t[1] for t in top_topics]

    # Flags chart
    flag_freq: dict[str, int] = {}
    for a in analyses:
        for f in a.quality_flags:
            flag_freq[f] = flag_freq.get(f, 0) + 1
    top_flags = sorted(flag_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    flag_labels_pt = {
        "sem_resposta": "Sem Resposta", "informacao_incorreta": "Info Incorreta",
        "tom_inadequado": "Tom Inadequado", "demora_excessiva": "Demora Excessiva",
        "pergunta_ignorada": "Pergunta Ignorada", "reclamacao_nao_tratada": "Reclamação Não Tratada",
        "paciente_frustrado": "Paciente Frustrado", "reagendamento_problematico": "Reagendamento Problemático",
    }
    flag_chart_labels = [flag_labels_pt.get(f[0], f[0]) for f in top_flags]
    flag_chart_data   = [f[1] for f in top_flags]

    # Quality buckets
    q_buckets = [0, 0, 0, 0]
    for a in analyses:
        s = a.quality_score
        if s < 4:   q_buckets[0] += 1
        elif s < 6: q_buckets[1] += 1
        elif s < 8: q_buckets[2] += 1
        else:       q_buckets[3] += 1

    # Response time distribution chart
    rt_dist = agg.response_time_distribution
    rt_labels = [
        "< 5 min\n(janela de ouro)",
        "5–10 min",
        "10–30 min",
        "30–60 min",
        "1–4 horas",
        "4–24 horas",
        "> 24 horas",
    ]
    rt_keys   = ["golden", "good", "ok", "slow", "very_slow", "bad", "critical"]
    rt_data   = [rt_dist.get(k, 0) for k in rt_keys]
    rt_total  = sum(rt_data) or 1
    rt_colors = ["#22C55E", "#4ADE80", "#86EFAC", "#F59E0B", "#FB923C", "#EF4444", "#B91C1C"]
    rt_pcts   = [round(v / rt_total * 100, 1) for v in rt_data]
    golden_pct = rt_pcts[0]
    golden_count = rt_data[0]
    _newline = "\n"
    rt_chart_labels = [l.replace("\n", " ") for l in rt_labels]
    rt_sidebar_rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:0.2rem 0;"
        f"border-bottom:1px solid #1E293B'>"
        f"<span style='color:{rt_colors[i]}'>"
        f"{rt_labels[i].replace(_newline, ' ')}"
        f"</span>"
        f"<strong style='color:#F8FAFC'>{rt_data[i]}"
        f" <span style='color:#64748B;font-weight:400'>({rt_pcts[i]}%)</span>"
        f"</strong></div>"
        for i in range(len(rt_keys))
    )

    # Hourly contact distribution (patient messages)
    hourly_data   = agg.hourly_contact_distribution        # list[int], len=24
    hourly_labels = [f"{h:02d}h" for h in range(24)]
    hourly_colors = [
        "#00AFE1" if BUSINESS_HOURS_START <= h < BUSINESS_HOURS_END else "#635BFF"
        for h in range(24)
    ]
    hourly_total      = sum(hourly_data) or 1
    off_hours_count   = sum(
        hourly_data[h] for h in range(24)
        if h < BUSINESS_HOURS_START or h >= BUSINESS_HOURS_END
    )
    off_hours_pct = round(off_hours_count / hourly_total * 100, 1)

    # Outcome chart
    os_ = outcome_summary
    outcome_chart_labels = ["Agendado", "Ghosting", "Objeção Ativa", "Pendente", "Outro"]
    outcome_chart_data   = [os_.agendado, os_.ghosting, os_.objecao_ativa, os_.pendente, os_.outro] if os_ else [0]*5
    outcome_chart_colors = ["#22C55E", "#94A3B8", "#EF4444", "#F59E0B", "#635BFF"]

    # Financial
    fk = financial_kpis
    opp_loss   = _fmt_brl(fk.opportunity_loss_value) if fk else "—"
    recovery   = _fmt_brl(fk.potential_recovery_value) if fk else "—"
    ticket     = _fmt_brl(fk.ticket_medio) if fk else "—"
    ticket_src = "(estimado)" if fk and fk.ticket_medio_source == "llm_estimate" else ""
    leads_lost = fk.leads_lost if fk else 0
    conv_rate  = f"{(os_.conversion_rate * 100):.0f}%" if os_ else "—"

    # Funil de Confirmação
    total_convs_f = agg.total_conversations or 1
    funil_confirmed_count = round(agg.avg_confirmation_rate * total_convs_f)
    funil_cancelled_count = agg.cancellation_count
    funil_sem_resp_count  = (os_.ghosting + os_.pendente) if os_ else 0
    funil_confirmed_pct   = round(agg.avg_confirmation_rate * 100, 1)
    funil_cancelled_pct   = round(agg.cancellation_rate * 100, 1)
    funil_sem_resp_pct    = round(funil_sem_resp_count / total_convs_f * 100, 1)

    # Shadow DNA
    dna = shadow_dna
    tone_badge   = dna.tone_classification if dna else "—"
    greeting_ex  = dna.greeting_example or "—" if dna else "—"
    closing_ex   = dna.closing_example or "—" if dna else "—"
    if dna:
        _tok = dna.average_response_length_tokens
        avg_tok = f"~{round(_tok * 4)} caracteres / ~{round(_tok * 0.75)} palavras"
    else:
        avg_tok = "—"
    top_emojis   = list(dna.emoji_frequency.items())[:8] if dna else []
    emoji_html   = "".join(
        f"<span class='emoji-chip' title='{round(v*100)}% das mensagens'>{e} <small>{round(v*100)}%</small></span>"
        for e, v in top_emojis
    ) if top_emojis else "<span class='muted'>Nenhum emoji detectado</span>"

    objections_li      = _li(os_.common_objections[:8] if os_ else [], "obj-item")
    unresolved_li      = _li(dna.unresolved_queries[:8] if dna else [], "gap-item")
    complaints_li      = _li(dna.common_complaints[:8] if dna else [], "complaint-item")
    procedures_li      = _li(dna.local_procedures[:10] if dna else [])
    payment_li         = _li(dna.local_payment_conditions[:8] if dna else [])
    forbidden_li       = _li(dna.forbidden_terms[:8] if dna else [], "forbidden-item")
    rag_score          = f"{dna.rag_efficiency_score:.0f}%" if dna else "—"

    # Insurance distribution chart
    ins_counts = dna.insurance_mention_counts if dna else {}
    ins_chart_labels = list(ins_counts.keys())[:10]
    ins_chart_data   = list(ins_counts.values())[:10]
    # Add "Particular" estimate: conversations not mentioning any insurance
    ins_total_mentions = sum(ins_chart_data) or 1
    ins_has_data = bool(ins_chart_labels)

    # Attendance flow
    flow_steps = dna.attendance_flow_steps if dna else []
    _flow_rows = []
    for _i, _s in enumerate(flow_steps):
        _sname = str(_s.get("step", f"Passo {_i+1}"))
        _sex   = (str(_s.get("example", "—"))
                  .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        _arrow = ("" if _i == len(flow_steps) - 1
                  else "<div style='text-align:center;color:#635BFF44;font-size:1rem;"
                       "padding:0.1rem 0'>↓</div>")
        _flow_rows.append(
            f"<div style='display:grid;grid-template-columns:110px 1fr;gap:0.75rem;"
            f"align-items:start;padding:0.6rem;background:#0F172A;border-radius:8px;"
            f"border:1px solid #334155;margin-bottom:0.25rem'>"
            f"<div style='display:flex;flex-direction:column;align-items:center;"
            f"justify-content:center;background:#635BFF18;border-radius:6px;"
            f"padding:0.4rem 0.25rem;text-align:center'>"
            f"<div style='width:22px;height:22px;border-radius:50%;background:#635BFF;"
            f"color:#fff;font-size:0.65rem;font-weight:700;display:flex;"
            f"align-items:center;justify-content:center;margin-bottom:0.3rem'>{_i+1}</div>"
            f"<div style='font-size:0.7rem;font-weight:600;color:#635BFF;line-height:1.3'>{_sname}</div>"
            f"</div>"
            f"<div class='quote-block' style='margin-top:0;font-size:0.82rem'>{_sex}</div>"
            f"</div>{_arrow}"
        )
    flow_html = "".join(_flow_rows) if _flow_rows else "<p class='muted' style='padding:1rem'>Fluxo não identificado nesta amostra — rode com mais conversas.</p>"

    # SLA adherence
    sla_pct = f"{dna.response_time_metrics.get('sla_adherence_percentage', 0):.0f}%" if dna else "—"

    # Improvement tips (filter generic/tautological suggestions)
    _GENERIC_TIPS_BLOCKLIST = [
        "manter um tom", "manter o tom", "mantenha um tom", "mantenha o tom",
        "mais profissional", "mais cordial", "mais empático",
        "melhorar a comunicação", "melhorar o atendimento",
        "ser mais claro", "seja mais claro",
        "responder mais rápido", "responda mais rápido",
        "atenção ao cliente", "qualidade do atendimento",
        "comunicação mais eficaz", "comunicação eficaz",
    ]
    all_tips: list[str] = []
    seen: set[str] = set()
    for a in analyses:
        for tip in a.quality_tips:
            t = tip.strip()
            if not t or t in seen:
                continue
            t_lower = t.lower()
            if any(block in t_lower for block in _GENERIC_TIPS_BLOCKLIST):
                continue
            all_tips.append(t)
            seen.add(t)
    tips_html = "".join(f"<li>{t}</li>" for t in all_tips[:8])

    # Best / worst conversations
    combined = list(zip(metrics_list, analyses))
    worst = sorted(combined, key=lambda x: x[1].health_score)[:5]
    best  = sorted(combined, key=lambda x: x[1].health_score, reverse=True)[:5]

    def _conv_row(m: ConversationMetrics, a: SemanticAnalysis) -> str:
        topics_str = ", ".join(a.topics[:3]) if a.topics else "—"
        flags_str  = ", ".join(flag_labels_pt.get(f, f) for f in a.quality_flags[:2]) if a.quality_flags else "—"
        return f"""<tr>
          <td class="mono">{m.phone[:7]}***</td>
          <td>{_fmt_seconds(m.avg_response_time_seconds)}</td>
          <td>{_sentiment_emoji(a.sentiment_label)} {sentiment_labels_pt.get(a.sentiment_label, a.sentiment_label)}</td>
          <td><span class="score-badge" style="background:{_health_color(a.health_score)}">{a.health_score:.0f}</span></td>
          <td class="muted">{topics_str}</td>
          <td class="muted flags">{flags_str}</td>
        </tr>"""

    worst_rows = "".join(_conv_row(m, a) for m, a in worst)
    best_rows  = "".join(_conv_row(m, a) for m, a in best)

    # Pre-compute insurance chart JS (avoids nested f-string issues)
    if ins_has_data:
        _insurance_chart_js = (
            "new Chart(document.getElementById('insuranceChart'), {\n"
            "  type: 'bar',\n"
            "  data: {\n"
            "    labels: " + json.dumps(ins_chart_labels) + ",\n"
            "    datasets: [{ data: " + json.dumps(ins_chart_data) + ", backgroundColor: '#00AFE1', borderRadius: 4 }]\n"
            "  },\n"
            "  options: {\n"
            "    indexAxis: 'y',\n"
            "    responsive: true,\n"
            "    maintainAspectRatio: false,\n"
            "    plugins: {\n"
            "      legend: { display: false },\n"
            "      tooltip: {\n"
            "        callbacks: {\n"
            "          label: function(ctx) {\n"
            "            const pct = Math.round(ctx.raw / " + str(ins_total_mentions) + " * 100);\n"
            "            return ` ${ctx.raw} menções (${pct}%)`;\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "    },\n"
            "    scales: {\n"
            "      x: { grid: { color: '#334155' }, ticks: { precision: 0 } },\n"
            "      y: { grid: { display: false } }\n"
            "    }\n"
            "  }\n"
            "});\n"
        )
    else:
        _insurance_chart_js = ""

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Relatório de Atendimento — {client_name}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0F172A;color:#F8FAFC;line-height:1.6}}
    a{{color:#635BFF;text-decoration:none}}

    .container{{max-width:1200px;margin:0 auto;padding:2rem}}
    .header{{display:flex;align-items:center;justify-content:space-between;padding:1.5rem 2rem;background:#1E293B;border-bottom:1px solid #334155}}
    .header h1{{font-size:1.25rem;font-weight:700}}
    .header .meta{{color:#94A3B8;font-size:0.85rem;text-align:right}}
    .badge{{display:inline-block;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.75rem;font-weight:600;background:#635BFF22;color:#635BFF}}

    /* Hero */
    .hero-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:2rem 0}}
    .health-ring{{display:flex;flex-direction:column;align-items:center;justify-content:center;width:140px;height:140px;border-radius:50%;border:6px solid {health_color};margin:0 auto 0.5rem}}
    .health-ring .score{{font-size:2.5rem;font-weight:800;color:{health_color}}}
    .health-ring .label{{font-size:0.7rem;color:#94A3B8}}
    .opp-card{{background:linear-gradient(135deg,#1E293B,#0F2040);border:1px solid #635BFF44;border-radius:16px;padding:2rem;display:flex;flex-direction:column;justify-content:center}}
    .opp-loss{{font-size:2.8rem;font-weight:800;color:#EF4444;line-height:1}}
    .opp-recovery{{font-size:1.6rem;font-weight:700;color:#22C55E;margin-top:0.75rem}}
    .opp-label{{font-size:0.8rem;color:#94A3B8;margin-top:0.25rem}}
    .opp-meta{{font-size:0.75rem;color:#64748B;margin-top:1rem;border-top:1px solid #334155;padding-top:0.75rem}}

    /* KPI cards */
    .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:2rem 0}}
    .kpi-card{{background:#1E293B;border-radius:12px;padding:1.25rem;border:1px solid #334155}}
    .kpi-card .value{{font-size:1.6rem;font-weight:800;background:linear-gradient(135deg,#635BFF,#00AFE1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .kpi-card .label{{font-size:0.8rem;color:#94A3B8;margin-top:0.25rem}}
    .kpi-card .sub{{font-size:0.75rem;color:#64748B;margin-top:0.5rem}}

    /* Section */
    .section{{margin:2.5rem 0}}
    .section-title{{font-size:1rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#94A3B8;margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid #334155}}

    /* Charts */
    .charts-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.5rem}}
    .chart-card{{background:#1E293B;border-radius:12px;padding:1.5rem;border:1px solid #334155}}
    .chart-card h3{{font-size:0.9rem;color:#94A3B8;margin-bottom:1rem}}
    .chart-wrap{{position:relative;height:220px}}

    /* Outcome pills */
    .outcome-pills{{display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:1rem}}
    .outcome-pill{{display:flex;align-items:center;gap:0.5rem;background:#0F172A;border-radius:10px;padding:0.6rem 1rem;border:1px solid #334155}}
    .outcome-pill .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
    .outcome-pill .val{{font-size:1.25rem;font-weight:700}}
    .outcome-pill .lbl{{font-size:0.75rem;color:#94A3B8}}

    /* Tables */
    table{{width:100%;border-collapse:collapse;font-size:0.875rem}}
    thead th{{text-align:left;padding:0.5rem 0.75rem;color:#94A3B8;font-size:0.75rem;text-transform:uppercase;border-bottom:1px solid #334155}}
    tbody tr{{border-bottom:1px solid #0F172A}}
    tbody tr:hover{{background:#0F172A}}
    tbody td{{padding:0.6rem 0.75rem;vertical-align:middle}}
    .mono{{font-family:monospace;color:#64748B}}
    .muted{{color:#94A3B8}}
    .flags{{color:#FCA5A5;font-size:0.8rem}}
    .score-badge{{display:inline-block;padding:0.15rem 0.5rem;border-radius:6px;font-weight:700;font-size:0.8rem;color:#0F172A}}

    /* Shadow DNA */
    .dna-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}
    .dna-card{{background:#1E293B;border-radius:12px;padding:1.25rem;border:1px solid #334155}}
    .dna-card h4{{font-size:0.8rem;text-transform:uppercase;color:#94A3B8;margin-bottom:0.75rem;letter-spacing:0.04em}}
    .dna-card .big{{font-size:1.4rem;font-weight:700;color:#635BFF}}
    .quote-block{{background:#0F172A;border-left:3px solid #635BFF;padding:0.75rem 1rem;border-radius:0 8px 8px 0;font-size:0.875rem;color:#CBD5E1;font-style:italic;margin-top:0.5rem}}
    .emoji-chip{{display:inline-flex;align-items:center;gap:0.3rem;background:#0F172A;border:1px solid #334155;border-radius:8px;padding:0.3rem 0.6rem;margin:0.2rem;font-size:1rem}}
    .emoji-chip small{{font-size:0.7rem;color:#94A3B8}}

    /* Lists */
    .item-list{{list-style:none;padding:0}}
    .item-list li{{padding:0.4rem 0;border-bottom:1px solid #1E293B;font-size:0.875rem;color:#CBD5E1;display:flex;align-items:flex-start;gap:0.5rem}}
    .item-list li:last-child{{border-bottom:none}}
    .item-list li::before{{content:'›';color:#635BFF;font-weight:700;flex-shrink:0}}
    .obj-item::before{{content:'⚠';color:#F59E0B!important}}
    .gap-item::before{{content:'?';color:#EF4444!important;font-weight:900}}
    .forbidden-item::before{{content:'✕';color:#EF4444!important}}
    .complaint-item::before{{content:'!';color:#F97316!important;font-weight:900}}
    .three-col{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem}}
    @media(max-width:900px){{.three-col{{grid-template-columns:1fr}}}}

    /* Tips */
    .tips-list{{background:#1E293B;border-radius:12px;padding:1.5rem;border:1px solid #334155}}
    .tips-list li{{padding:0.4rem 0;border-bottom:1px solid #334155;color:#CBD5E1;font-size:0.9rem}}
    .tips-list li:last-child{{border-bottom:none}}
    .tips-list li::marker{{color:#635BFF}}

    /* Two col */
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}

    /* Response time distribution */
    .rt-chart-wrap{{position:relative;height:260px}}
    .golden-badge{{display:inline-flex;align-items:center;gap:0.5rem;background:#22C55E18;border:1px solid #22C55E44;border-radius:10px;padding:0.5rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#22C55E;font-weight:600}}

    /* Hourly contact distribution */
    .hourly-chart-wrap{{position:relative;height:520px}}
    .off-hours-badge{{display:inline-flex;align-items:center;gap:0.5rem;background:#635BFF18;border:1px solid #635BFF44;border-radius:10px;padding:0.5rem 1rem;margin-bottom:0.75rem;font-size:0.85rem;color:#635BFF;font-weight:600}}

    /* Footer */
    .footer{{margin-top:4rem;padding:2rem;border-top:1px solid #334155;text-align:center;color:#475569;font-size:0.8rem}}
    .footer strong{{color:#635BFF}}

    @media(max-width:768px){{
      .hero-grid,.two-col{{grid-template-columns:1fr}}
      .charts-grid{{grid-template-columns:1fr}}
      .kpi-grid{{grid-template-columns:repeat(2,1fr)}}
    }}
  </style>
</head>
<body>

<div class="header">
  <div>
    <div style="color:#635BFF;font-weight:800;font-size:1.1rem;letter-spacing:-0.02em;">
      EasyScale <span style="color:#00AFE1">Legacy Analyzer</span>
    </div>
    <h1 style="margin-top:0.25rem">{client_name}</h1>
    <span class="badge">{client_slug}</span>
  </div>
  <div class="meta">
    Relatório gerado em<br/>
    <strong>{generated_at.strftime("%d/%m/%Y às %H:%M")}</strong>
  </div>
</div>

<div class="container">

  <!-- Hero: Health Score + Oportunidade Financeira -->
  <div class="hero-grid" style="margin-top:2rem">

    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;background:#1E293B;border-radius:16px;padding:2rem;border:1px solid #334155">
      <div class="health-ring">
        <div class="score">{avg_health:.0f}</div>
        <div class="label">SAÚDE GERAL</div>
      </div>
      <div style="color:#94A3B8;margin-top:0.75rem;font-size:0.8rem;text-align:center">
        Baseado em qualidade ({avg_quality:.1f}/10) e sentimento ({avg_sentiment:+.2f})
      </div>
      <div style="margin-top:1rem;display:flex;gap:1rem;font-size:0.8rem">
        <span>📅 {busiest_weekday_name}</span>
        <span>🕐 Pico: {busiest_hour_str}</span>
        <span>📩 SLA: {sla_pct}</span>
      </div>
    </div>

    <div class="opp-card">
      <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:#EF4444;font-weight:700;margin-bottom:0.5rem">
        ⚠ Oportunidade Perdida
      </div>
      <div class="opp-loss">{opp_loss}</div>
      <div class="opp-label">{leads_lost} leads não convertidos × ticket {ticket} {ticket_src}</div>
      <div class="opp-recovery">{recovery}</div>
      <div class="opp-label">Recuperável com IA (estimativa conservadora de 30%)</div>
      <div class="opp-meta">
        Taxa de conversão atual: <strong style="color:#F8FAFC">{conv_rate}</strong> ·
        Ghosting: <strong style="color:#94A3B8">{os_.ghosting if os_ else 0}</strong> ·
        Objeção: <strong style="color:#EF4444">{os_.objecao_ativa if os_ else 0}</strong>
      </div>
    </div>
  </div>

  <!-- KPI Cards -->
  <div class="section">
    <div class="section-title">Visão Geral</div>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="value">{agg.total_conversations}</div>
        <div class="label">Conversas Analisadas</div>
        <div class="sub">{agg.total_messages:,} mensagens no total</div>
      </div>
      <div class="kpi-card">
        <div class="value">{_fmt_seconds(agg.avg_response_time_seconds)}</div>
        <div class="label">Tempo Médio de Resposta</div>
        <div class="sub">Mediana: {_fmt_seconds(agg.median_response_time_seconds)} · P90: {_fmt_seconds(agg.p90_response_time_seconds)}</div>
      </div>
      <div class="kpi-card">
        <div class="value">{_fmt_pct(agg.avg_confirmation_rate)}</div>
        <div class="label">Taxa de Confirmação</div>
        <div class="sub">Cancelamentos: {agg.cancellation_count} ({_fmt_pct(agg.cancellation_rate)})</div>
      </div>
      <div class="kpi-card">
        <div class="value">{avg_quality:.1f}<span style="font-size:1rem;color:#94A3B8">/10</span></div>
        <div class="label">Qualidade Média</div>
        <div class="sub">Sentimento: {avg_sentiment:+.2f}</div>
      </div>
      <div class="kpi-card">
        <div class="value">{agg.conversations_no_reply}</div>
        <div class="label">Sem Resposta</div>
        <div class="sub">{_fmt_pct(agg.conversations_no_reply / max(agg.total_conversations, 1))} do total</div>
      </div>

    </div>
  </div>

  <!-- Distribuição de Tempo de Resposta -->
  <div class="section">
    <div class="section-title">Distribuição do Tempo de Resposta</div>
    <div class="charts-grid" style="grid-template-columns:2fr 1fr">
      <div class="chart-card">
        <h3>Todas as interações (clínica → paciente)</h3>
        <div class="rt-chart-wrap"><canvas id="rtDistChart"></canvas></div>
      </div>
      <div class="chart-card" style="display:flex;flex-direction:column;justify-content:center;gap:0.75rem">
        <div class="golden-badge">
          ⚡ {golden_count} respostas na janela de ouro ({golden_pct}%)
        </div>
        <div style="font-size:0.82rem;color:#94A3B8;line-height:1.7">
          {rt_sidebar_rows}
        </div>
        <div style="margin-top:0.5rem;font-size:0.75rem;color:#475569">
          Total de interações: {rt_total}
        </div>
      </div>
    </div>
  </div>

  <!-- Horários de Contato -->
  <div class="section">
    <div class="section-title">Horários de Contato dos Pacientes</div>
    <div class="chart-card">
      <div class="off-hours-badge">
        🌙 {off_hours_count} mensagens fora do expediente ({off_hours_pct}%) — potencial para IA 24h
      </div>
      <div style="font-size:0.78rem;color:#64748B;margin-bottom:1.25rem">
        <span style="display:inline-flex;align-items:center;gap:0.3rem">
          <span style="width:10px;height:10px;border-radius:3px;background:#00AFE1;display:inline-block"></span>
          Dentro do expediente ({BUSINESS_HOURS_START:02d}h–{BUSINESS_HOURS_END:02d}h)
        </span>
        &nbsp;&nbsp;
        <span style="display:inline-flex;align-items:center;gap:0.3rem">
          <span style="width:10px;height:10px;border-radius:3px;background:#635BFF;display:inline-block"></span>
          Fora do expediente
        </span>
      </div>
      <div class="hourly-chart-wrap">
        <canvas id="hourlyChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Desfechos -->
  <div class="section">
    <div class="section-title">Desfechos das Conversas</div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Distribuição de Desfechos</h3>
        <div class="chart-wrap"><canvas id="outcomeChart"></canvas></div>
      </div>
      <div class="chart-card" style="display:flex;flex-direction:column;justify-content:center">
        <h3>Breakdown</h3>
        <div class="outcome-pills">
          <div class="outcome-pill">
            <div class="dot" style="background:#22C55E"></div>
            <div><div class="val">{os_.agendado if os_ else 0}</div><div class="lbl">Agendado</div></div>
          </div>
          <div class="outcome-pill">
            <div class="dot" style="background:#94A3B8"></div>
            <div><div class="val">{os_.ghosting if os_ else 0}</div><div class="lbl">Ghosting</div></div>
          </div>
          <div class="outcome-pill">
            <div class="dot" style="background:#EF4444"></div>
            <div><div class="val">{os_.objecao_ativa if os_ else 0}</div><div class="lbl">Objeção Ativa</div></div>
          </div>
          <div class="outcome-pill">
            <div class="dot" style="background:#F59E0B"></div>
            <div><div class="val">{os_.pendente if os_ else 0}</div><div class="lbl">Pendente</div></div>
          </div>
          <div class="outcome-pill">
            <div class="dot" style="background:#635BFF"></div>
            <div><div class="val">{os_.outro if os_ else 0}</div><div class="lbl">Outro</div></div>
          </div>
        </div>
        <div style="margin-top:1.5rem;font-size:0.85rem;color:#94A3B8">
          Taxa de conversão: <strong style="color:#22C55E;font-size:1.1rem">{conv_rate}</strong>
        </div>
      </div>
    </div>
  </div>

  <!-- Funil de Confirmação -->
  <div class="section">
    <div class="section-title">Funil de Confirmação</div>
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr)">
      <div class="kpi-card" style="border-left:3px solid #22C55E">
        <div class="value" style="color:#22C55E">{funil_confirmed_pct}%</div>
        <div class="label">✅ Confirmações</div>
        <div class="sub">{funil_confirmed_count} conversas com confirmação</div>
      </div>
      <div class="kpi-card" style="border-left:3px solid #EF4444">
        <div class="value" style="color:#EF4444">{funil_cancelled_pct}%</div>
        <div class="label">❌ Cancelamentos</div>
        <div class="sub">{funil_cancelled_count} conversas canceladas</div>
      </div>
      <div class="kpi-card" style="border-left:3px solid #94A3B8">
        <div class="value" style="color:#94A3B8">{funil_sem_resp_pct}%</div>
        <div class="label">👻 Ghosting + Pendente</div>
        <div class="sub">{funil_sem_resp_count} sem desfecho claro</div>
      </div>
    </div>
  </div>

  <!-- Análise de Qualidade e Sentimento -->
  <div class="section">
    <div class="section-title">Qualidade &amp; Sentimento</div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Sentimento dos Pacientes</h3>
        <div class="chart-wrap"><canvas id="sentimentChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Qualidade do Atendimento (distribuição)</h3>
        <div class="chart-wrap"><canvas id="qualityChart"></canvas></div>
      </div>
      <div class="chart-card" style="grid-column:1/-1">
        <h3>Tópicos Mais Frequentes</h3>
        <div class="chart-wrap" style="height:280px"><canvas id="topicsChart"></canvas></div>
        {topics_insight_html}
      </div>
      <div class="chart-card" style="grid-column:1/-1">
        <h3>Problemas Detectados</h3>
        <div class="chart-wrap" style="height:240px"><canvas id="flagsChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Shadow Mode: Fluxo de Atendimento -->
  <div class="section">
    <div class="section-title">Shadow Mode — Fluxo Típico de Atendimento</div>
    <div class="chart-card">
      <div style="margin-bottom:1rem">
        <div style="font-size:0.9rem;font-weight:600;color:#F8FAFC">Como esta clínica atende — passo a passo</div>
        <div style="font-size:0.8rem;color:#64748B;margin-top:0.25rem">
          Extraído das conversas reais · Um agente treinado neste perfil replica exatamente este fluxo
        </div>
      </div>
      {flow_html}
    </div>
  </div>

  <!-- Shadow DNA -->
  <div class="section">
    <div class="section-title">Shadow DNA — Perfil de Comunicação</div>
    <div class="dna-grid">

      <div class="dna-card">
        <h4>Tom Dominante</h4>
        <div class="big">{tone_badge}</div>
        <div style="margin-top:0.5rem;font-size:0.8rem;color:#94A3B8">
          Média de resposta: {avg_tok}<br/>
          Atendimentos &lt; 30min: {sla_pct}
        </div>
      </div>

      <div class="dna-card">
        <h4>Uso de Emojis</h4>
        <div style="margin-top:0.25rem">{emoji_html}</div>
      </div>

      <div class="dna-card">
        <h4>Exemplo de Saudação</h4>
        <div class="quote-block">{greeting_ex}</div>
      </div>

      <div class="dna-card">
        <h4>Exemplo de Encerramento</h4>
        <div class="quote-block">{closing_ex}</div>
      </div>

    </div>
  </div>

  <!-- Objeções + Reclamações + Lacunas -->
  <div class="section">
    <div class="section-title">Objeções, Reclamações &amp; Lacunas</div>
    <div class="three-col">
      <div class="chart-card">
        <h3>⚠ Objeções Comerciais</h3>
        <ul class="item-list">{objections_li}</ul>
      </div>
      <div class="chart-card">
        <h3>! Reclamações de Pacientes</h3>
        <ul class="item-list">{complaints_li}</ul>
      </div>
      <div class="chart-card">
        <h3>? Lacunas de Conhecimento <span style="font-size:0.75rem;color:#EF4444">(perguntas sem resposta)</span></h3>
        <ul class="item-list">{unresolved_li}</ul>
      </div>
    </div>
  </div>

  <!-- Base de Conhecimento + Termos Proibidos -->
  <div class="section">
    <div class="section-title">Base de Conhecimento Detectada</div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Procedimentos &amp; Condições de Pagamento</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem">
          <div>
            <div style="font-size:0.72rem;text-transform:uppercase;color:#94A3B8;letter-spacing:0.05em;margin-bottom:0.5rem">Procedimentos</div>
            <ul class="item-list">{procedures_li}</ul>
          </div>
          <div>
            <div style="font-size:0.72rem;text-transform:uppercase;color:#94A3B8;letter-spacing:0.05em;margin-bottom:0.5rem">Condições de Pagamento</div>
            <ul class="item-list">{payment_li}</ul>
          </div>
        </div>
      </div>
      <div class="chart-card">
        <h3>Termos a Evitar <span style="font-size:0.75rem;color:#94A3B8">(detectados em conversas problemáticas)</span></h3>
        <ul class="item-list">{forbidden_li}</ul>
      </div>
    </div>
    <div class="chart-card" style="margin-top:1.5rem">
      <h3>Convênios — Frequência de Menções</h3>
      {"<div class='chart-wrap' style='height:240px'><canvas id='insuranceChart'></canvas></div>" if ins_has_data else "<p class='muted' style='padding:1rem'>Nenhum convênio detectado nas conversas analisadas.</p>"}
    </div>
  </div>

  <!-- Best / Worst conversations -->
  <div class="section">
    <div class="section-title">Conversas que Precisam de Atenção</div>
    <div class="chart-card">
      <table>
        <thead><tr><th>Paciente</th><th>T. Resposta</th><th>Sentimento</th><th>Score</th><th>Tópicos</th><th>Problemas</th></tr></thead>
        <tbody>{worst_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Melhores Conversas</div>
    <div class="chart-card">
      <table>
        <thead><tr><th>Paciente</th><th>T. Resposta</th><th>Sentimento</th><th>Score</th><th>Tópicos</th><th>Problemas</th></tr></thead>
        <tbody>{best_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Sugestões -->
  <div class="section">
    <div class="section-title">Sugestões de Melhoria</div>
    <ul class="tips-list">
      {tips_html if tips_html else "<li>Nenhuma sugestão identificada — excelente atendimento!</li>"}
    </ul>
  </div>

</div>

<div class="footer">
  Gerado por <strong>EasyScale Legacy Analyzer</strong> ·
  {generated_at.strftime("%d/%m/%Y %H:%M")} ·
  {agg.total_conversations} conversas analisadas
</div>

<script>
Chart.defaults.color = '#94A3B8';
Chart.defaults.borderColor = '#334155';

new Chart(document.getElementById('outcomeChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(outcome_chart_labels)},
    datasets: [{{ data: {json.dumps(outcome_chart_data)}, backgroundColor: {json.dumps(outcome_chart_colors)}, borderWidth: 2, borderColor: '#1E293B' }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'right', labels: {{ boxWidth: 12 }} }} }} }}
}});

new Chart(document.getElementById('sentimentChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(sent_chart_labels)},
    datasets: [{{ data: {json.dumps(sent_chart_data)}, backgroundColor: {json.dumps(sent_chart_colors)}, borderWidth: 2, borderColor: '#1E293B' }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'right', labels: {{ boxWidth: 12 }} }} }} }}
}});

new Chart(document.getElementById('qualityChart'), {{
  type: 'bar',
  data: {{
    labels: ['0–4 (Ruim)', '4–6 (Regular)', '6–8 (Bom)', '8–10 (Ótimo)'],
    datasets: [{{ data: {json.dumps(q_buckets)}, backgroundColor: ['#EF4444','#F59E0B','#22C55E','#635BFF'], borderRadius: 6 }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ grid: {{ color: '#334155' }} }}, x: {{ grid: {{ display: false }} }} }} }}
}});

new Chart(document.getElementById('topicsChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(topic_chart_labels)},
    datasets: [{{ data: {json.dumps(topic_chart_data)}, backgroundColor: '#635BFF', borderRadius: 4 }}]
  }},
  options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ color: '#334155' }} }}, y: {{ grid: {{ display: false }} }} }} }}
}});

new Chart(document.getElementById('flagsChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(flag_chart_labels)},
    datasets: [{{ data: {json.dumps(flag_chart_data)}, backgroundColor: '#EF4444', borderRadius: 4 }}]
  }},
  options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ color: '#334155' }} }}, y: {{ grid: {{ display: false }} }} }} }}
}});

new Chart(document.getElementById('hourlyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(hourly_labels)},
    datasets: [{{
      data: {json.dumps(hourly_data)},
      backgroundColor: {json.dumps(hourly_colors)},
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            const pct = Math.round(ctx.raw / {hourly_total} * 1000) / 10;
            const h = ctx.dataIndex;
            const isOff = h < {BUSINESS_HOURS_START} || h >= {BUSINESS_HOURS_END};
            return ` ${{ctx.raw}} msgs (${{pct}}%)${{isOff ? ' — fora do expediente' : ''}}`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: '#334155' }}, ticks: {{ precision: 0 }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

{_insurance_chart_js}
new Chart(document.getElementById('rtDistChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(rt_chart_labels)},
    datasets: [{{
      data: {json.dumps(rt_data)},
      backgroundColor: {json.dumps(rt_colors)},
      borderRadius: 6,
      borderSkipped: false,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            const pcts = {json.dumps(rt_pcts)};
            return ` ${{ctx.raw}} interações (${{pcts[ctx.dataIndex]}}%)`;
          }}
        }}
      }}
    }},
    scales: {{
      y: {{
        grid: {{ color: '#334155' }},
        ticks: {{ precision: 0 }}
      }},
      x: {{
        grid: {{ display: false }},
        ticks: {{ font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>

</body>
</html>"""

    return html
