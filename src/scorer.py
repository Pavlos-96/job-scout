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
import re
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kandidaten-Profil (hardcoded für maximale Zuverlässigkeit)
# ---------------------------------------------------------------------------

CANDIDATE_PROFILE = """
Name: Pavlos Musenidis
Aktuelle Rolle: AI Engineer bei Ippen Digital (Sep 2023 – heute, ~2.5 Jahre)
Vorige Rollen: Working Student Software Engineering (AX Semantics, 6 Monate),
               Data Science Intern (Porsche AG, 6 Monate)
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
- Coding-Assistenten täglich: Cursor, Claude Code, GitHub Copilot

Produktionserfahrungen:
- Editorial Automation Platform: 500+ aktive User, 600+ registrierte Editoren
- Election Content Automation: 1.713 automatisierte Artikel, mission-critical deadline
- Central AI Workflow Engine: Microservices-Migration, Multi-Stage Validation Pipelines
- Enterprise Chatbot Prototype: LangGraph ReAct Agents, Milvus, GCP

Ausbildung: M.Sc. Computational Linguistics (Uni Stuttgart, 1.8), B.A. Linguistics
Zertifikate: AWS Cloud Practitioner, Linux Essentials

Standortpräferenzen:
- Stuttgart: jede Arbeitsweise OK (onsite / hybrid / remote — kann hinfahren)
- Sonst Deutschland (jede Stadt, jedes Bundesland):
  - vollständig remote: ideal
  - hybrid: NUR akzeptabel wenn maximal ~1× pro Monat Office.
    "Hybrid" wird von Firmen sehr unterschiedlich definiert:
    "3 Tage Office / 2 remote" = nicht akzeptabel (zu viel Office)
    "1 Tag pro Woche im Office" = nicht akzeptabel
    "Monatliche Team-Days" = akzeptabel
    "Mostly remote, occasional travel" = akzeptabel
    "Flexible, remote-first hybrid" = akzeptabel
  - pure onsite (kein Remote-Anteil): nicht akzeptabel
- Schweiz (jede Stadt): wie oben (remote ideal, leichtes Hybrid OK)
- Sonst EU/EMEA: nur wenn vollständig remote
- USA/UK-onsite / non-EU: nicht akzeptabel

Gehaltsvorstellung:
- Minimum: 85.000 EUR brutto/Jahr (unter diesem Wert → "zu niedrig")
- Üblich angegeben: 85.000, 88.000 oder 90.000 EUR
- Dream: 90.000 EUR (mehr ist immer besser)
- Schweiz: entsprechend höher in CHF (≥ 105.000 CHF als grobe Faustregel)

Schwächen (nicht verschweigen, aber realistisch):
- Kein öffentliches GitHub / Portfolio
- Architektur-Erfahrung eher als Contributor, nicht alleiniger Architekt
- Kein reines ML/Training-Hintergrund (Fokus: LLM Application Engineering)

Nicht-Python-Programmiersprachen: kann ich nicht produktiv. Wenn die Stelle
PRIMÄR Java, Go, TypeScript, Kotlin, Rust, C++, C# verlangt → schlechter Match.
WICHTIG: "Python OR Java" oder "Python und/oder Java" zählt als Python-OK,
weil ich Python erfülle.

Positive Bonus-Signale (kein K.O.-Kriterium wenn fehlend):
- Arbeit mit Coding-Assistenten (Cursor, Claude Code, Copilot, Cline, Aider, Continue)
- Aktuelle Modelle (Claude 4/5 Sonnet, GPT-4o/5, Gemini 2/3, Llama 3/4)
- LangChain, LangGraph, LlamaIndex, DSPy
- Agentic Systems, Tool-Calling, MCP
- Vector DBs (Milvus, Qdrant, Weaviate, pgvector)
- RAG, Multi-Provider LLM, Eval-Frameworks
"""

# ---------------------------------------------------------------------------
# Score-Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredJob:
    """Ein bewertetes Match-Ergebnis vom LLM."""
    score: int              # 1–10
    recommendation: str     # "apply" | "maybe" | "skip"
    years_required: str     # wortwörtliches Zitat ODER "nicht angegeben"
    python_required: bool
    python_signal: str      # wortwörtliches Zitat ODER "nicht erwähnt"
    salary_assessment: str  # "passt" | "zu niedrig" | "unklar" | "kein Gehalt angegeben"
    salary_quote: str = "nicht angegeben"  # wortwörtliches Zitat
    coding_assistants_mentioned: bool = False
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""Du bist ein präziser Job-Match-Analyst. Du bewertest Stellenanzeigen
für einen spezifischen Kandidaten und gibst ehrliche, prägnante Einschätzungen.

Kandidatenprofil:
{CANDIDATE_PROFILE}

══════════════════════════════════════════════════════════════════
ABSOLUT KRITISCH — ANTI-HALLUZINATIONS-REGELN
══════════════════════════════════════════════════════════════════

A1. NUR FAKTEN AUS DEM TEXT. Du darfst NICHTS behaupten, was nicht
    wortwörtlich in der Stellenbeschreibung steht. Keine Vermutungen,
    keine Branchenstandard-Annahmen, kein "üblicherweise".

A2. JEDE KONKRETE ZAHL MUSS BELEGT SEIN durch ein exaktes Zitat:
    - Jahre Erfahrung
    - Gehaltsangabe
    - Sprachanforderungen
    - Seniority-Level
    Wenn die Zahl nicht im Text steht → schreibe "nicht angegeben" und
    KEINE Mutmaßung. Erfinde keine "5 Jahre Erfahrung" wenn der Text
    keine konkrete Jahreszahl nennt.

A3. python_signal und years_required MÜSSEN entweder ein wortwörtliches
    Zitat aus der Anzeige sein (1–15 Wörter, in Anführungszeichen) ODER
    exakt der String "nicht angegeben". Nichts dazwischen.

A4. Wenn die Anzeige nur sagt "experienced" / "erfahren" ohne Jahreszahl,
    ist years_required = "nicht angegeben" — NICHT "5+ Jahre".

A5. "Senior" im Titel allein begründet KEINEN Jahres-Skip. Senior ist
    ein Level-Label, kein Erfahrungs-Mindestmaß. Nur wenn der Text
    explizit "X+ Jahre" / "minimum X years" sagt, zählt das.

A6. SENIOR-FALLE — diese Signale sind KEINE Jahres-Anforderung:
    - "(Senior)" oder "Senior" im Titel
    - "Mentoring", "Mentor", "technischer Mentor"
    - "Architekt", "Architecture", "Engineering Excellence"
    - "strategischer Partner", "Pre-Sales", "Sparring"
    - "Lead", "Principal" im Titel
    Wenn der Text NUR solche Signale enthält und keine konkrete Zahl,
    ist years_required = "nicht angegeben". Niemals daraus "5+ Jahre"
    konstruieren. Wenn dir die Rolle zu senior wirkt, schreibe das im
    concern: "Rolle wirkt Senior-lastig (Mentoring, Architekt), Kandidat
    hat 2.5 Jahre Erfahrung" — das ist ehrlich. "5+ Jahre gefordert"
    ist erfunden.

A7. JEDES ZITAT MUSS WORTWÖRTLICH IM TEXT STEHEN. years_required,
    python_signal und salary_quote werden nach deiner Antwort automatisch
    gegen den Anzeigentext geprüft. Ein nicht-belegbares Zitat wird auf
    "nicht angegeben" zurückgesetzt und die zugehörigen concerns gelöscht.
    Spare dir und uns die Korrektur: zitiere nur was wirklich dasteht.

══════════════════════════════════════════════════════════════════
HARTE AUSSCHLUSSREGELN — nur greifen wenn klar belegt
══════════════════════════════════════════════════════════════════

H1. ZU VIEL ERFAHRUNG VERLANGT (nur bei explizitem Zitat im Text):
    Wenn die Anzeige WÖRTLICH ≥ 6 Jahre verlangt
    ("6+ years", "mindestens 6 Jahre", "at least 7 years" etc.)
    → score ≤ 3, recommendation = "skip".
    Bei "5+ years" / "5 Jahre" → score 4, recommendation = "maybe"
    (Kandidat hat 2.5 Jahre; bei perfektem Stack noch grenzwertig bewerbbar).
    Bei "3–5 Jahre" oder weniger → KEIN Skip aus diesem Grund.
    Bei keinem Zitat → years_required = "nicht angegeben", KEIN Skip.

H2. NICHT-PYTHON ALS HAUPTSPRACHE: Wenn die Anzeige wörtlich eine andere
    Sprache als PRIMÄR-/HAUPT-/CORE-Sprache nennt (z.B. "Java is our
    primary language", "Go is the backbone"), und Python nirgends genannt
    wird → score ≤ 3, recommendation = "skip".
    WICHTIG: "Python oder Java", "Python and/or TypeScript", "Python,
    Java oder Go" — das ist KEIN Hard-Skip, weil Kandidat Python erfüllt.

H3. PYTHON KOMPLETT ABWESEND: Wenn Python weder explizit noch durch das
    Ökosystem (LangChain, FastAPI, ML-Frameworks, Django, Flask, Pandas,
    PyTorch, TensorFlow) impliziert wird → score ≤ 4, "skip".

H4. NICHT-DACH ohne vollständigen Remote: Nur skippen wenn die Anzeige
    KLAR sagt, dass der Ort außerhalb DE/CH ist UND kein Remote möglich
    ist. "Remote — EU" / "Remote — Europe" → akzeptabel.
    USA-only / India-only / "must reside in X" → "skip".

H5. PEOPLE-MANAGEMENT-PFLICHT: Nur skippen wenn die Anzeige WÖRTLICH
    Personalverantwortung verlangt ("manage a team of X engineers",
    "headcount responsibility", "hire and develop direct reports").
    "Lead Engineer" ohne People-Management = darf bewertet werden.

H6. REINE INFRA/PLATFORM/DEVOPS-ROLLE: Nur skippen wenn Kernaufgaben
    KLAR Cluster-Betrieb / SRE / Netzwerk sind UND Python/LLM nirgends
    in Aufgaben oder Stack genannt werden → score ≤ 3, "skip".

H7. NICHT-ENGINEERING: Sales, Customer Success, Consulting ohne Coding,
    Prompt-only ohne Code, Annotation, Content-Moderation → "skip".

══════════════════════════════════════════════════════════════════
WEICHE BEWERTUNGSSIGNALE
══════════════════════════════════════════════════════════════════

S1. ERFAHRUNGSJAHRE (nur wenn explizit zitiert):
    - 0–2 Jahre verlangt → -1 (zu junior für Kandidat)
    - 2–4 Jahre → +2 (perfect match)
    - 3–5 Jahre → +1 (grenzwertig OK)
    - "nicht angegeben" → 0 (neutral, kein Abzug)

S2. GEHALT:
    - Range ≥ 90.000 EUR (oder ≥ 110.000 CHF) → "passt", +2
    - Range 85.000–89.999 EUR (oder 105.000–109.999 CHF) → "passt", +1
    - Range < 85.000 EUR (oder < 105.000 CHF) → "zu niedrig", -3,
      recommendation maximal "maybe"
    - Keine Angabe → "kein Gehalt angegeben", neutral
    Salary_assessment NUR auf "zu niedrig" setzen wenn eine Zahl < 85K
    EUR im Text steht. Sonst "kein Gehalt angegeben" oder "unklar".

S3. STACK-MATCH (je +0.5 für genannten Treffer):
    LangChain, LangGraph, LlamaIndex, DSPy, RAG, Agentic, MCP, FastAPI,
    LiteLLM, Milvus, Qdrant, Weaviate, pgvector, Kubernetes, AWS,
    Multi-Provider LLM, Eval-Frameworks (Braintrust, Langfuse, Weights & Biases).

S4. CODING-ASSISTENTEN-BONUS (+1 wenn explizit erwähnt):
    Cursor, Claude Code, GitHub Copilot, Cline, Aider, Continue, Codeium,
    Windsurf, Devin. Der Kandidat liebt diese Tools.

S5. AKTUELLE MODELLE-BONUS (+0.5 wenn erwähnt):
    Claude Sonnet 4/5, Claude Opus 4/5, GPT-4o/5, Gemini 2/3, Llama 3/4.

S6. STANDORT-BONUS (lies die Beschreibung GENAU nach Office-Frequenz):
    - Stuttgart (jede Form) → +2
    - Deutschland-Remote / Schweiz-Remote → +1
    - Hybrid mit ≤1× Monat Office (außerhalb Stuttgart) → 0 (akzeptabel)
    - Hybrid mit 1× Woche Office (außerhalb Stuttgart) → -1
    - Hybrid mit 2-3 Tagen/Woche Office (außerhalb Stuttgart) → -2,
      recommendation maximal "maybe"
    - Hybrid mit 4+ Tagen/Woche Office (außerhalb Stuttgart) → -3,
      recommendation "skip"
    - Wenn Office-Frequenz nicht aus dem Text ablesbar → neutral (0),
      Hinweis in "concerns" geben
    - Schweiz hybrid: oft de facto remote — wenn Text "remote-friendly" /
      "flexible" sagt, behandle wie Remote (+1)

S7. DOMAIN-ABZÜGE:
    - Stelle ist primär ML/Training/Research → -1
    - Infrastruktur-schwer ohne klaren LLM-Bezug → -1 bis -2

══════════════════════════════════════════════════════════════════
KONSISTENZ-REGELN
══════════════════════════════════════════════════════════════════

K1. recommendation = "apply" → score ≥ 7 UND Python-Bezug klar UND
    LLM/GenAI/AI-Application-Fokus erkennbar.
K2. recommendation = "maybe" → score 5–6 oder Beschreibung zu vage.
K3. recommendation = "skip" → score ≤ 4 ODER eine harte Regel greift.
K4. Wenn du unsicher bist: lieber "maybe" als ein falsches "apply" oder
    "skip". Falsche apply-Empfehlungen kosten den Kandidaten Zeit.

══════════════════════════════════════════════════════════════════
ANTWORT-FORMAT
══════════════════════════════════════════════════════════════════

Antworte IMMER als valides JSON-Objekt, KEIN Markdown, keine Erklärungen
außerhalb des JSON. Schema:

{{
  "score": <int 1-10>,
  "recommendation": "<apply|maybe|skip>",
  "years_required": "<wortwörtliches Zitat ODER 'nicht angegeben'>",
  "python_required": <true|false>,
  "python_signal": "<wortwörtliches Zitat ODER 'nicht erwähnt'>",
  "salary_assessment": "<passt|zu niedrig|unklar|kein Gehalt angegeben>",
  "salary_quote": "<wortwörtliches Zitat ODER 'nicht angegeben'>",
  "coding_assistants_mentioned": <true|false>,
  "strengths": ["<2-4 prägnante Punkte>"],
  "concerns": ["<1-3 prägnante Punkte>"],
  "summary": "<1-2 Sätze, faktenbasiert, ohne Mutmaßungen>"
}}
"""

# ---------------------------------------------------------------------------
# Post-LLM Halluzinations-Filter
# ---------------------------------------------------------------------------

_YEAR_IN_TEXT_RE = re.compile(
    r"\b\d+\s*\+?\s*(?:-\s*\d+\s*)?(?:jahr|year)", re.IGNORECASE
)
_PLACEHOLDERS = {
    "nicht angegeben", "nicht erwähnt", "nicht erwaehnt",
    "n/a", "na", "none", "-", "",
}


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for substring matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _quote_supported(quote: str, text_norm: str) -> bool:
    """True if `quote` appears verbatim (case- and whitespace-insensitive)."""
    needle = _normalize(quote.strip("\"' "))
    if needle in _PLACEHOLDERS:
        return True
    return needle in text_norm


def _verify_quotes(scored: ScoredJob, description: str) -> ScoredJob:
    """Strip unsupported quotes and the concerns/strengths derived from them.

    The LLM occasionally fabricates a years-of-experience requirement from
    title labels like '(Senior)' or duties like 'Mentoring'. We cannot let
    that bleed into the UI as a hard concern. Strategy:

    1. For each quote field (years_required, python_signal, salary_quote):
       if the value is not a placeholder and not a verbatim substring of the
       posting, reset it to the placeholder.
    2. If years_required ended up "nicht angegeben", drop any concern or
       strength that mentions a year count — those are derived from the
       hallucinated quote.
    """
    text_norm = _normalize(description)
    fixes: list[str] = []

    if not _quote_supported(scored.years_required, text_norm):
        fixes.append(f"years_required={scored.years_required!r}")
        scored.years_required = "nicht angegeben"

    if not _quote_supported(scored.python_signal, text_norm):
        fixes.append(f"python_signal={scored.python_signal!r}")
        scored.python_signal = "nicht erwähnt"

    if not _quote_supported(scored.salary_quote, text_norm):
        fixes.append(f"salary_quote={scored.salary_quote!r}")
        scored.salary_quote = "nicht angegeben"
        if scored.salary_assessment == "zu niedrig":
            scored.salary_assessment = "kein Gehalt angegeben"

    if scored.years_required == "nicht angegeben":
        scored.concerns = [
            c for c in scored.concerns if not _YEAR_IN_TEXT_RE.search(c)
        ]
        scored.strengths = [
            s for s in scored.strengths if not _YEAR_IN_TEXT_RE.search(s)
        ]
        if _YEAR_IN_TEXT_RE.search(scored.summary):
            fixes.append("summary mentions years not in posting")

    if fixes:
        log.warning("hallucination filter triggered: %s", "; ".join(fixes))

    return scored


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.4"


async def score_job(
    client,           # openai.AsyncOpenAI instance
    title: str,
    company: str,
    location: str,
    description: str,
    salary_signals: list,
    model: str = DEFAULT_MODEL,
) -> ScoredJob | None:
    """Score a single job. Returns None on API error."""

    # Requirements and tech signals appear throughout the posting — use enough
    # context to reliably detect Python, experience years, and role fit.
    # gpt-5.4 has 1M context, but we keep the budget tight for cost/latency.
    desc_limit = 8000
    desc_truncated = description[:desc_limit].rstrip()
    if len(description) > desc_limit:
        desc_truncated += "\n[...]"

    salary_str = ""
    if salary_signals:
        parts = []
        for sig in salary_signals:
            lo, hi = sig[0], sig[1] if len(sig) > 1 else None
            if hi:
                parts.append(f"{lo:,}–{hi:,} EUR")
            else:
                parts.append(f"{lo:,}+ EUR")
        salary_str = (
            f"\nGehaltssignale aus dem Pre-Filter (nur Hinweis, dein "
            f"salary_quote MUSS aus dem Text unten kommen): "
            f"{'; '.join(parts)}"
        )

    user_msg = f"""Stelle: {title}
Unternehmen: {company}
Standort: {location}{salary_str}

Stellenbeschreibung (NUR auf dieser Basis bewerten, nichts dazu erfinden):
\"\"\"
{desc_truncated}
\"\"\"
"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        score_val = int(data.get("score", 5))
        score_val = max(1, min(10, score_val))
        rec = str(data.get("recommendation", "maybe")).lower().strip()
        if rec not in ("apply", "maybe", "skip"):
            rec = "maybe"
        if score_val <= 3:
            rec = "skip"
        elif score_val <= 4 and rec == "apply":
            rec = "maybe"
        elif score_val >= 8 and rec == "skip":
            rec = "maybe"

        scored = ScoredJob(
            score=score_val,
            recommendation=rec,
            years_required=str(data.get("years_required", "nicht angegeben")),
            python_required=bool(data.get("python_required", False)),
            python_signal=str(data.get("python_signal", "nicht erwähnt")),
            salary_assessment=str(
                data.get("salary_assessment", "kein Gehalt angegeben")
            ),
            salary_quote=str(data.get("salary_quote", "nicht angegeben")),
            coding_assistants_mentioned=bool(
                data.get("coding_assistants_mentioned", False)
            ),
            strengths=list(data.get("strengths", []) or []),
            concerns=list(data.get("concerns", []) or []),
            summary=str(data.get("summary", "")),
        )

        return _verify_quotes(scored, description)

    except Exception as e:
        log.warning("scoring failed for '%s' @ %s: %s", title, company, e)
        return None


async def score_all(
    matches: list,          # list of MatchedJob from filters.py
    api_key: str,
    model: str = DEFAULT_MODEL,
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
                        salary_quote=str(cached.get("salary_quote", "nicht angegeben")),
                        coding_assistants_mentioned=bool(
                            cached.get("coding_assistants_mentioned", False)
                        ),
                        strengths=list(cached.get("strengths", [])),
                        concerns=list(cached.get("concerns", [])),
                        summary=str(cached.get("summary", "")),
                    )
                    scored = _verify_quotes(scored, j.description_text)
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
