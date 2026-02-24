#!/usr/bin/env python3
"""
shadow_cli.py
-------------
Shadow Mode — Simulação ao vivo do atendimento de uma clínica.

Carrega o blueprint JSON mais recente de ./reports/<slug>_blueprint_*.json,
extrai o perfil Shadow DNA e entra em modo de chat interativo simulando
o atendente da clínica, replicando seu tom, fluxo e estilo de comunicação.

Uso:
    python shadow_cli.py --client-slug sgen
    python shadow_cli.py                    # usa o blueprint mais recente

Comandos durante o chat:
    sair / quit / exit   — encerra a sessão
    /perfil              — exibe o perfil carregado
    /novo                — reinicia a conversa (limpa o histórico)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
PURPLE  = "\033[35m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
GREY    = "\033[90m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def _banner(clinic_name: str, agent_name: str, tone: str) -> None:
    print()
    print(_c("━" * 60, PURPLE))
    print(_c("  EasyScale  ", BOLD, PURPLE) + _c("Shadow Mode", BOLD, CYAN))
    print(_c("━" * 60, PURPLE))
    print(f"  {_c('Clínica:', GREY)}  {_c(clinic_name, BOLD)}")
    print(f"  {_c('Agente:', GREY)}   {_c(agent_name or 'Atendente', BOLD, CYAN)}")
    print(f"  {_c('Tom:', GREY)}      {tone}")
    print(_c("━" * 60, PURPLE))
    print(f"  {_c('Digite sua mensagem como se fosse um paciente.', GREY)}")
    print(f"  {_c('Comandos:', GREY)} {_c('/perfil', CYAN)} · {_c('/novo', CYAN)} · {_c('sair', CYAN)}")
    print(_c("━" * 60, PURPLE))
    print()


# ── Blueprint loader ───────────────────────────────────────────────────────────

def _find_blueprint(reports_dir: Path, client_slug: str | None) -> Path:
    """Return path to the most recent blueprint JSON for the given slug."""
    pattern = f"{client_slug}_blueprint_*.json" if client_slug else "*_blueprint_*.json"
    candidates = sorted(reports_dir.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum blueprint encontrado em '{reports_dir}' "
            f"(padrão: {pattern}).\n"
            "Execute primeiro: python run_local.py --client-slug ... --limit 10"
        )
    return candidates[0]


def _load_profile(blueprint_path: Path) -> dict:
    """Load and return the shadow_dna_profile from the blueprint JSON."""
    with open(blueprint_path, encoding="utf-8") as f:
        bp = json.load(f)

    meta    = bp.get("metadata", {})
    profile = bp.get("shadow_dna_profile", {})
    kb      = bp.get("knowledge_base_mapping", {})

    return {
        "clinic_name":           meta.get("client_name", "Clínica"),
        "client_slug":           meta.get("client_slug", ""),
        "agent_name":            profile.get("agent_identity", {}).get("suggested_name", "Atendente"),
        "tone":                  profile.get("communication_style", {}).get("tone", "Empático"),
        "personality_traits":    profile.get("communication_style", {}).get("personality_traits", []),
        "greeting_example":      profile.get("communication_style", {}).get("greeting_example", ""),
        "closing_example":       profile.get("communication_style", {}).get("closing_example", ""),
        "forbidden_terms":       profile.get("communication_style", {}).get("forbidden_terms", []),
        "handoff_keywords":      profile.get("escalation_rules", {}).get("trigger_keywords", []),
        "handoff_situations":    profile.get("escalation_rules", {}).get("trigger_situations", []),
        "local_procedures":      kb.get("confirmed_procedures", []),
        "local_insurances":      kb.get("accepted_insurances", []),
        "local_payment":         kb.get("payment_conditions", []),
        "local_neighborhoods":   kb.get("covered_neighborhoods", []),
        "attendance_flow_steps": profile.get("attendance_flow_steps", []),
        "blueprint_path":        str(blueprint_path),
        "generated_at":          meta.get("generated_at", ""),
    }


# ── System prompt builder ─────────────────────────────────────────────────────

def _build_system_prompt(p: dict) -> str:
    def _fmt_list(items: list) -> str:
        if not items:
            return "Não identificado"
        return "; ".join(str(i) for i in items[:15])

    def _fmt_flow(steps: list) -> str:
        if not steps:
            return "Não identificado"
        return " → ".join(
            f"{i+1}. {s.get('step','?')}"
            for i, s in enumerate(steps)
        )

    flow_detail = ""
    if p["attendance_flow_steps"]:
        lines = []
        for i, s in enumerate(p["attendance_flow_steps"]):
            lines.append(
                f"  {i+1}. {s.get('step','?')}: \"{s.get('example','')}\""
            )
        flow_detail = "\n" + "\n".join(lines)

    return f"""Você é {p['agent_name']}, atendente da {p['clinic_name']}.

## Seu Perfil de Comunicação
- Tom dominante: {p['tone']}
- Traços de personalidade: {_fmt_list(p['personality_traits'])}
- Exemplo de saudação típica: "{p['greeting_example'] or 'Olá! Como posso ajudar?'}"
- Exemplo de encerramento típico: "{p['closing_example'] or 'Qualquer dúvida estou à disposição!'}"

## Conhecimento da Clínica
- Procedimentos oferecidos: {_fmt_list(p['local_procedures'])}
- Convênios aceitos: {_fmt_list(p['local_insurances'])}
- Condições de pagamento: {_fmt_list(p['local_payment'])}
- Regiões atendidas: {_fmt_list(p['local_neighborhoods'])}

## Fluxo Típico de Atendimento
{_fmt_flow(p['attendance_flow_steps'])}{flow_detail}

## Regras
- Responda SOMENTE como este atendente responderia, no mesmo tom e estilo.
- Use linguagem natural em português brasileiro, informal-profissional.
- NUNCA use os termos proibidos: {_fmt_list(p['forbidden_terms'])}
- Se surgir: {_fmt_list(p['handoff_situations'])}, diga que vai transferir para um colega.
- Resposta curta e direta (como WhatsApp) — máximo 3–4 linhas por mensagem.
- Não invente procedimentos ou convênios além dos listados.
- Se não souber algo, diga "Vou verificar e te retorno em breve 😊".
"""


# ── Chat loop ─────────────────────────────────────────────────────────────────

def _chat(profile: dict, openai_api_key: str) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print(_c("Erro: openai não instalado. Execute: pip install openai", RED))
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)
    system_prompt = _build_system_prompt(profile)
    history: list[dict] = [{"role": "system", "content": system_prompt}]

    _banner(profile["clinic_name"], profile["agent_name"], profile["tone"])

    while True:
        try:
            user_input = input(_c("Você  › ", BOLD, GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("sair", "quit", "exit"):
            print(_c("\n  Encerrando Shadow Mode. Até logo! 👋\n", GREY))
            break

        if user_input.lower() == "/perfil":
            print()
            print(_c("  ── Perfil carregado ──────────────────────", GREY))
            print(_c(f"  Blueprint:  {profile['blueprint_path']}", GREY))
            print(_c(f"  Gerado em:  {profile['generated_at']}", GREY))
            print(_c(f"  Tom:        {profile['tone']}", GREY))
            if profile["attendance_flow_steps"]:
                steps = " → ".join(s.get("step", "?") for s in profile["attendance_flow_steps"])
                print(_c(f"  Fluxo:      {steps}", GREY))
            print()
            continue

        if user_input.lower() == "/novo":
            history = [{"role": "system", "content": system_prompt}]
            print(_c("\n  Conversa reiniciada. 🔄\n", YELLOW))
            continue

        # LLM call
        history.append({"role": "user", "content": user_input})
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history,
                temperature=0.4,
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": reply})

            # Pretty print agent response
            agent_label = _c(f"{profile['agent_name']}  › ", BOLD, CYAN)
            print(f"\n{agent_label}{reply}\n")

        except Exception as e:
            print(_c(f"\n  Erro na chamada à API: {e}\n", RED))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shadow Mode — simula o atendimento de uma clínica com IA"
    )
    parser.add_argument(
        "--client-slug", "-s",
        help="Slug do cliente (ex: sgen). Se omitido, usa o blueprint mais recente."
    )
    parser.add_argument(
        "--reports-dir", "-r",
        default=os.environ.get("REPORTS_OUTPUT_DIR", "./reports"),
        help="Diretório onde estão os blueprints (padrão: ./reports)"
    )
    args = parser.parse_args()

    # Load .env if present
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print(_c("Erro: OPENAI_API_KEY não configurada.", RED))
        print(_c("Defina em .env ou como variável de ambiente.", GREY))
        sys.exit(1)

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(_c(f"Erro: diretório de relatórios não encontrado: {reports_dir}", RED))
        print(_c("Execute run_local.py primeiro para gerar um blueprint.", GREY))
        sys.exit(1)

    try:
        blueprint_path = _find_blueprint(reports_dir, args.client_slug)
    except FileNotFoundError as e:
        print(_c(f"Erro: {e}", RED))
        sys.exit(1)

    print(_c(f"\n  Carregando blueprint: {blueprint_path.name}", GREY))
    profile = _load_profile(blueprint_path)
    _chat(profile, openai_api_key)


if __name__ == "__main__":
    main()
