"""
scorer.py — LLM-basiertes Scoring für gefilterte Stellen.

Benutzt OpenAI gpt-4o-mini (günstig, schnell, gut genug für Stellenmatching).
Kein LiteLLM, kein extra Layer — direkt openai SDK.

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."   # oder in .env Datei

Kosten: ~$0.001 pro Stelle (gpt-4o-mini). Bei 50 Matches = ~5 Cent pro Run.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kandidaten-Profil (hardcoded für maximale Zuverlässigkeit)
# ---------------------------------------------------------------------------

CANDIDATE_PROFILE = """
Name: Pavlos Musenidis
Aktuelle Rolle: AI Engineer bei Ippen Digital (Sep 2023 – heute, ~2.5 Jahre)
Vorige Rollen: Working Student Software Engineering (AX Semantics, 6 Monate), Data Science Intern (Porsche AG, 6 Monate)
Gesamte relevante Berufserfahrung: ~2.5 Jahre als AI Engineer

Kern-Stack (täglich produktiv):
- Python (primäre Sprache, Hauptkriterium)
- LangChain, LangGraph, LiteLLM
- FastAPI (mehrere Production Backends)
- RAG Pipelines, Agentic Systems, Multi-Provider LLM Integration
- OpenAI, Anthropic, Google (Modell-APIs)
- PostgreSQL, Milvus (Vector DB)
- Kubernetes, AWS (S3, SageMaker, Secrets Manager), GCP
- Docker, CI/CD, Grafana

Produktionserfahrungen:
- Editorial Automation Platform: 500+ aktive User, 600+ registrierte Editoren
- Election Content Automation: 1.713 automatisierte Artikel, mission-critical deadline
- Central AI Workflow Engine: Microservices-Migration, Multi-Stage Validation Pipelines
- Enterprise Chatbot Prototype: LangGraph ReAct Agents, Milvus, GCP

Ausbildung: M.Sc. Computational Linguistics (Uni Stuttgart, 1.8), B.A. Linguistics
Zertifikate: AWS Cloud Practitioner, Linux Essentials

Gehaltsvorstellung:
- Minimum: 85.000 EUR (unter diesem Wert → kein Interesse)
- Ziel: 95.000 EUR
- Dream: 100.000+ EUR

Schwächen (offen kommunizieren):
- Kein öffentliches GitHub / Portfolio
- Architektur-Erfahrung eher als Contributor, nicht alleiniger Architekt
- Kein reines ML/Training-Hintergrund (eher LLM Application Engineering)

Nicht-Python-Programmiersprachen: Kann ich nicht produktiv. Wenn die Stelle
primär Java, Go, TypeScript, Rust etc. verlangt → schlechter Match.
"""

# ---------------------------------------------------------------------------
# Score-Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredJob:
    """Ein bewertetes Match-Ergebnis vom LLM."""
    score: int              # 1–10
    recommendation: str     # "apply" | "maybe" | "skip"
    years_required: str     # z.B. "3-5", "5+", "nicht angegeben"
    python_required: bool
    python_signal: str      # Zitat oder "nicht erwähnt"
    salary_assessment: str  # "passt", "zu niedrig", "unklar", "kein Gehalt angegeben"
    strengths: list[str]    # 2-4 Punkte warum gut
    concerns: list[str]     # 1-3 Punkte warum nicht ideal
    summary: str            # 1-2 Sätze für den Report

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""Du bist ein präziser Job-Match-Analyst. Du bewertest Stellenanzeigen
für einen spezifischen Kandidaten und gibst ehrliche, prägnante Einschätzungen.

Kandidatenprofil:
{CANDIDATE_PROFILE}

Bewertungsregeln (in dieser Priorität — HARTE REGELN vor weichen):

╔══ HARTE AUSSCHLUSSREGELN (KEIN "apply" oder "maybe") ══╗

H1. 5+ JAHRE ERFAHRUNG gefordert → IMMER: score ≤ 3, recommendation = "skip".
    Keine Ausnahmen. Auch wenn der Stack perfekt passt. Der Kandidat hat 2.5 Jahre —
    Stellen mit "5+ years", "mindestens 5 Jahre", "5 Jahre oder mehr" sind
    grundsätzlich uninteressant.

H2. NICHT-PYTHON ALS PRIMÄRSPRACHE: Wenn die Stelle primär eine andere Sprache als
    Python fordert (Java, Go, TypeScript, Kotlin, Rust als Hauptsprache) →
    score ≤ 3, recommendation = "skip". Keine Ausnahmen.

H3. PYTHON FEHLT KOMPLETT: Wenn Python weder explizit noch implizit gefordert wird →
    score ≤ 4, recommendation = "skip". Keine Ausnahmen.
    "Implizit" = Rolle baut klar auf Python-Ökosystem (LangChain, FastAPI,
    ML-Frameworks) — Python ist dann erwartet auch ohne explizite Nennung.


H4. AUßERHALB VON DEUTSCHLAND/EU: Wenn aus der Beschreibung ersichtlich, dass die Stelle
    nicht in Deutschland ist sind das große Minusunkte. recommendation = "maybe" or "skip".
    wenn außerhalb von EU dann immer recommendation = "skip".

H5. LEAD / MANAGEMENT / STRATEGY-FIRST: Wenn die Rolle primär Teamführung, Line Management,
    Headcount-Verantwortung, Roadmap-only ohne hands-on Code, oder "Engineering Manager"
    mit wenig IC-Anteil ist → score ≤ 4, recommendation = "skip".
    Ausnahme: reine IC-Titel wie "Lead AI Engineer" mit klar hands-on LLM/Python-Stack
    und ohne People-Management-Pflicht → darf bewertet werden, aber bei echter Lead-
    Verantwortung trotzdem "skip".

H6. REINE INFRA / PLATFORM / DEVOPS-SCHWERPUNKT: Wenn Kernaufgabe Kubernetes, Docker,
    CI/CD, Cluster-Betrieb, SRE, Netzwerk, Observability — und Python/LLM/Application-
    Engineering nur Randerscheinung oder gar nicht genannt → score ≤ 3,
    recommendation = "skip". Der Kandidat macht zwar K8s/Docker, will aber keinen
    Job als primärer Infra-Engineer ohne Python/LLM-Fokus.

H7. FEHL AM PLATZ / UTECHNISCH: Stellen die durch den Titelfilter gerutscht sind, aber
    inhaltlich Sales, Success, Consulting ohne Engineering, reines Prompting ohne Code,
    Content-Moderation, Trainingsdaten-Annotation, oder völlig andere Skill-Anforderungen
    als LLM/Python-Application-Engineering → score ≤ 3, recommendation = "skip".
    Lieber zu streng skippen als false positive "apply".

HINWEIS: Der Titelfilter matcht auch "GenAI Engineer", "AI Software Engineer", "LLM Engineer".
    Prüfe IMMER die Stellenbeschreibung: Passt sie wirklich zu einem Python-treibenden
    LLM/GenAI Application Engineer? Wenn nein → H7.

╚════════════════════════════════════════════════════════╝

4. ERFAHRUNGSJAHRE vs. KANDIDAT (2.5 Jahre als AI Engineer) — nur wenn KEIN Hard-Skip:
   - Stelle fordert 0-2 Jahre → zu junior, score -1
   - Stelle fordert 2-4 Jahre → perfect match, score +2
   - Stelle fordert 3-5 Jahre → grenzwertig aber bewerbbar, score +1
   - Nicht angegeben → neutral

5. GEHALT:
   - Gehaltsrahmen ≥ 95.000 EUR → "passt" (Jackpot), score +2
   - Gehaltsrahmen 85.000–94.999 EUR → "passt" (akzeptabel), score +1
   - Gehaltsrahmen < 85.000 EUR → "zu niedrig", score -3, recommendation maximal "maybe"
   - Kein Gehalt angegeben → "kein Gehalt angegeben", neutral

6. STACK-MATCH (Nebenpunkte, je +0.5 für Treffer):
   LangChain, LangGraph, RAG, Agentic Systems, FastAPI, LiteLLM, Milvus/Qdrant,
   Kubernetes, AWS, LLM-Integration, Multi-Provider

7. DOMAIN-ABZÜGE (nur wenn nicht schon H6/H7):
   - Stelle ist primär ML/Training/Research (kein LLM Application Engineering) → score -1
   - Infrastruktur-schwer ohne klaren LLM-Bezug → score -1 bis -2

8. EMPFEHLUNG vs. SCORE (Konsistenz):
   - recommendation = "apply" nur bei score ≥ 7 und klar passendem Stack + Python + LLM/GenAI.
   - recommendation = "maybe" bei score 5–6 oder unscharfer Beschreibung.
   - recommendation = "skip" bei score ≤ 4 oder wenn eine HARTE REGEL greift.

Antworte IMMER als valides JSON-Objekt, KEIN Markdown, keine Erklärungen darum.
Schema:
{{
  "score": <int 1-10>,
  "recommendation": "<apply|maybe|skip>",
  "years_required": "<string>",
  "python_required": <true|false>,
  "python_signal": "<zitat oder 'nicht erwähnt'>",
  "salary_assessment": "<passt|zu niedrig|unklar|kein Gehalt angegeben>",
  "strengths": ["<string>", ...],
  "concerns": ["<string>", ...],
  "summary": "<1-2 Sätze>"
}}
"""

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def score_job(
    client,           # openai.AsyncOpenAI instance
    title: str,
    company: str,
    location: str,
    description: str,
    salary_signals: list,
    model: str = "gpt-4o-mini",
) -> ScoredJob | None:
    """Score a single job. Returns None on API error."""

    # First ~2600 chars: responsibilities often after intro; helps H5–H7
    desc_limit = 2600
    desc_truncated = description[:desc_limit].rstrip()
    if len(description) > desc_limit:
        desc_truncated += "\n[...]"

    # Format salary signals if present
    salary_str = ""
    if salary_signals:
        parts = []
        for sig in salary_signals:
            lo, hi = sig[0], sig[1] if len(sig) > 1 else None
            if hi:
                parts.append(f"{lo:,}–{hi:,} EUR")
            else:
                parts.append(f"{lo:,}+ EUR")
        salary_str = f"\nGehaltssignale aus der Anzeige: {'; '.join(parts)}"

    user_msg = f"""Stelle: {title}
Unternehmen: {company}
Standort: {location}{salary_str}

Stellenbeschreibung:
{desc_truncated}"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        score_val = int(data.get("score", 5))
        score_val = max(1, min(10, score_val))
        rec = str(data.get("recommendation", "maybe")).lower().strip()
        if rec not in ("apply", "maybe", "skip"):
            rec = "maybe"
        # Enforce consistency: very low scores must not be "apply"
        if score_val <= 3:
            rec = "skip"
        elif score_val <= 4 and rec == "apply":
            rec = "maybe"

        return ScoredJob(
            score=score_val,
            recommendation=rec,
            years_required=str(data.get("years_required", "nicht angegeben")),
            python_required=bool(data.get("python_required", False)),
            python_signal=str(data.get("python_signal", "nicht erwähnt")),
            salary_assessment=str(data.get("salary_assessment", "kein Gehalt angegeben")),
            strengths=data.get("strengths", []),
            concerns=data.get("concerns", []),
            summary=str(data.get("summary", "")),
        )

    except Exception as e:
        log.warning("scoring failed for '%s' @ %s: %s", title, company, e)
        return None


async def score_all(
    matches: list,          # list of MatchedJob from filters.py
    api_key: str,
    model: str = "gpt-4o-mini",
    concurrency: int = 5,   # parallel API calls; don't hammer too hard
    cache=None,             # ScoreCache | None  (optional, avoids circular import)
) -> list[tuple]:           # list of (MatchedJob, ScoredJob | None)
    """Score all matches concurrently, with optional score cache."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        return [(m, None) for m in matches]

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def one(m) -> tuple:
        async with sem:
            j = m.job
            url = getattr(j, "url", "") or ""

            if cache and url:
                cached = cache.get(url)
                if cached is not None:
                    log.debug("cache hit: %s @ %s", j.title, j.company_display)
                    scored = ScoredJob(
                        score=int(cached.get("score", 5)),
                        recommendation=str(cached.get("recommendation", "maybe")),
                        years_required=str(cached.get("years_required", "nicht angegeben")),
                        python_required=bool(cached.get("python_required", False)),
                        python_signal=str(cached.get("python_signal", "nicht erwähnt")),
                        salary_assessment=str(
                            cached.get("salary_assessment", "kein Gehalt angegeben")
                        ),
                        strengths=list(cached.get("strengths", [])),
                        concerns=list(cached.get("concerns", [])),
                        summary=str(cached.get("summary", "")),
                    )
                    return (m, scored)

            scored = await score_job(
                client=client,
                title=j.title,
                company=j.company_display,
                location=j.location,
                description=j.description_text,
                salary_signals=m.salary_signals,
                model=model,
            )

            if cache and scored and url:
                cache.set(url, scored.as_dict())

            return (m, scored)

    results = await asyncio.gather(*[one(m) for m in matches], return_exceptions=True)

    if cache:
        cache.save()

    out = []
    for m, res in zip(matches, results):
        if isinstance(res, Exception):
            log.warning("gather error: %s", res)
            out.append((m, None))
        else:
            out.append(res)
    return out
