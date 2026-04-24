#!/usr/bin/env python3
"""
write_cover_letter.py — Generates a cover letter from a job posting URL.

Workflow:
  1. Take a job URL from the report
  2. Re-fetch the full job description (no DB needed)
  3. Detect du/Sie from the posting language
  4. Generate cover letter via Anthropic Claude (style-engineered prompt)
  5. Output as plain text + rendered HTML + PDF

Usage:
  python write_cover_letter.py <url>
  python write_cover_letter.py <url> --no-pdf
  python write_cover_letter.py <url> --out cover_letters/

Requires:
  - ANTHROPIC_API_KEY in .env
  - wkhtmltopdf installed (for PDF) — optional, skip with --no-pdf
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

log = logging.getLogger("cover_letter")


# ---------------------------------------------------------------------------
# Re-fetch job from URL — auto-detect ATS
# ---------------------------------------------------------------------------

async def fetch_job_from_url(url: str) -> dict | None:
    """
    Detects which ATS the URL belongs to and fetches the job details.
    Returns dict with title, company, location, description.
    """
    from fetchers import (fetch_greenhouse, fetch_lever, fetch_personio,
                          fetch_ashby, strip_html)

    # Pattern matching to determine ATS + extract token + job_id
    patterns = [
        (r"https?://(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/([^/]+)/jobs/(\d+)", "greenhouse"),
        (r"https?://([^.]+)\.jobs\.personio\.(?:de|com)(?:/job/(\d+))?", "personio"),
        (r"https?://jobs\.lever\.co/([^/]+)/([^/?]+)", "lever"),
        (r"https?://jobs\.ashbyhq\.com/([^/?]+)(?:/([^/?]+))?", "ashby"),
        (r"https?://careers\.datadoghq\.com/detail/(\d+)", "greenhouse_special"),
    ]

    ats = None
    token = None
    job_id = None
    for pat, ats_name in patterns:
        m = re.search(pat, url, re.IGNORECASE)
        if m:
            ats = ats_name
            token = m.group(1) if m.lastindex >= 1 else None
            job_id = m.group(2) if m.lastindex >= 2 else None
            break

    if not ats:
        log.error("Konnte ATS aus URL nicht erkennen: %s", url)
        return None

    log.info("ATS erkannt: %s, Token: %s, Job-ID: %s", ats, token, job_id)

    headers = {"User-Agent": "job-scout/0.1 (+personal-use)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        if ats == "greenhouse" or ats == "greenhouse_special":
            # For greenhouse_special (datadog custom domain), we still go to API with token=datadog
            board_token = token if ats == "greenhouse" else "datadog"
            jobs = await fetch_greenhouse(client, board_token)
        elif ats == "personio":
            jobs = await fetch_personio(client, token)
        elif ats == "lever":
            jobs = await fetch_lever(client, token)
        elif ats == "ashby":
            jobs = await fetch_ashby(client, token)
        else:
            return None

    # Find the specific job by URL match (URLs are unique within a board)
    matched = None
    for j in jobs:
        if j.url == url or (job_id and job_id in j.url):
            matched = j
            break

    if not matched:
        # Fallback: if there's only one job, use it
        if len(jobs) == 1:
            matched = jobs[0]
        else:
            log.error("Stelle nicht gefunden in den %d Stellen von %s", len(jobs), token)
            return None

    return {
        "title": matched.title,
        "company": matched.company_display,
        "location": matched.location,
        "description": matched.description_text,
        "url": matched.url,
    }


# ---------------------------------------------------------------------------
# du/Sie detection
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Style-engineered system prompt
# ---------------------------------------------------------------------------

PROFILE = """
Pavlos Musenidis — AI Engineer

Aktuelle Rolle: AI Engineer bei Ippen Digital (seit September 2023, über 2,5 Jahre).
Ippen Digital ist eines der größten lokalen Journalismus-Netzwerke in Deutschland (über 250 Portale).

Vorige Rollen:
- Working Student Software Engineering bei AX Semantics (Jun-Dez 2022, 6 Monate)
- Data Science Intern bei Porsche AG (Mai-Nov 2021, 6 Monate)

Ausbildung:
- M.Sc. Computational Linguistics, Universität Stuttgart (2020-2023, Note 1,8)
  Schwerpunkte: ML/DL, NLP mit Transformer-Architekturen, Sequence Modeling.
  Theoretisches Fundament hinter LLMs (Attention, Tokenization, Fine-Tuning-Konzepte).
- B.A. Linguistics, Universität Stuttgart (2015-2020, Note 2,0)

Zertifikate: AWS Cloud Practitioner Essentials, Linux Essentials.

Stack (täglich produktiv):
- Python (primär), FastAPI, Async, Pytest, Pydantic, Poetry
- LangChain, LangGraph, LiteLLM, OpenAI, Anthropic, Google
- PostgreSQL, Milvus (Vector DB), Redis
- AWS (S3, SageMaker, Secrets Manager), GCP, Kubernetes, Docker
- Grafana, Kibana, Elasticsearch, CI/CD
- Cursor, Claude Code (strategisch eingesetzt, nicht als Ersatz für Architekturverständnis)

Verfügbarkeit: kurzfristig (etwa 2-3 Wochen).
Wohnort: Stuttgart. Bevorzugt 100% remote oder hybrid mit Pendeln nach München oder Stuttgart.

=== PROJEKTE — NUR AUSWÄHLEN WAS ZUR STELLE PASST (max. 2) ===

STANDARD (immer verwendbar, wenn nichts Spezifischeres passt):
- Pilot (Editorial Automation Platform): Streamlit-Plattform, über 500 täglich aktive User,
  600+ registrierte Redakteure. Use-Cases direkt mit Redakteuren entwickelt, iteriert,
  getestet. Mehrere FastAPI-Backends.
- Wahlautomatisierung: Multi-Stage-Pipeline mit Quality Gates (Fact-Checking,
  Grammatik-Verifikation, Style-Compliance). Zwei Projekte: Kommunalwahlen (über 1.700
  automatisierte Artikel) und Bundestagswahl 2025 (über 7.000 Artikel in drei Phasen,
  mission-critical Deadlines). WICHTIG: Nur nennen welche Wahl zur Stelle/Firma passt,
  oder einfach "Wahlautomatisierung" ohne spezifische Wahl wenn unklar.

NUR WENN RAG / Chatbots / Vector Search / Knowledge-Base in der Stellenbeschreibung:
- OpenOlat Enterprise Chatbot: LangGraph ReAct Agents, Milvus Vector DB, GCP-Integration.
  Knowledge-Base-Suche für Lernmanagement-System (Prototype/PoC).

NUR WENN Architektur / Workflow-Engine / Backend / API-Integrationen im Fokus:
- Tintin (AI Workflow Engine): Komplett selbst entwickeltes Workflow-Orchestrierungssystem,
  konzeptionell ähnlich n8n. Modulare, hoch erweiterbare Architektur, eigene Workflow-Engine,
  API-Integrationen zu Dutzenden externen Diensten, Migration von Monolith zu Microservices.

NUR WENN Frontend / UX / Tool-Entwicklung mit Endnutzern / Consulting explizit gefragt:
- Pilot (Details): Use-Cases wurden direkt mit Redakteuren entwickelt (Requirements-Gespräche,
  Testing-Sessions, Feedback-Loops). Frontend mittels Streamlit, auf die Bedürfnisse der
  Redaktion zugeschnitten. Pavlos kannte die Nutzer persönlich.

NUR WENN Docker / Kubernetes / DevOps / Infrastructure relevant:
- Kubernetes/AWS durchgehend seit Beginn bei Ippen Digital: Logging-Stack (Grafana, Kibana,
  Elasticsearch), CI/CD, Ressourcen-Tuning, Secrets Management, Image-Builds,
  Deployment-Automatisierung. Kein Ops-Spezialist, aber produktiv und self-sufficient.

NUR WENN ML-Training / Research / Modellentwicklung / Data Science gefragt:
- M.Sc. Computational Linguistics mit ML/DL-Schwerpunkt. Versteht das theoretische Fundament
  (Attention-Mechanismen, Transformer-Training, Fine-Tuning-Konzepte). Bei Ippen auf
  Applikationsebene, aber das Fundament ist vorhanden.

Motivation für sinnstiftende Stellen: ECHT. Gesundheit, Bildung, Energiewende,
Barrierefreiheit, öffentlicher Sektor — das soll im ersten Absatz als echte Motivation
auftauchen, nicht aufgesetzt.

=== TONALITÄT (Meta) ===

Pavlos schreibt trocken, beobachtend, leicht analytisch. Er bewertet nicht ("X ist hart",
"das ist beeindruckend"), er beobachtet ("Bei X liegt Y zwischen Z"). Er lobt nicht
("eure Mission begeistert mich"), er zeigt Interesse durch Konkretheit. Er ist nie
performativ. Wenn ein Satz so klingt als könnte ihn jeder zweite Bewerber schreiben,
ist er falsch.
"""

REFERENCE_LETTER = """
Hallo zusammen,

Im Stellentext ist mir hängen geblieben, dass das LLM-Engineering bei euch direkt am
Produkt sitzt und nicht in einer separaten Innovation-Spur. Genau das ist mein
Hintergrund: über 2,5 Jahre LLM-Pipelines im Produktionsbetrieb, nicht als PoC.

Seit über 2,5 Jahren arbeite ich als AI Engineer bei Ippen Digital, einem der größten
lokalen Journalismus-Netzwerke in Deutschland mit über 250 Portalen. Mein täglicher
Stack ist Python mit FastAPI, LangChain/LangGraph und LiteLLM, deployt auf Kubernetes
in AWS.

Ein konkretes Projekt von mir war die Artikelautomatisierung zur Bundestagswahl. Über
7.000 Artikel haben wir in drei Phasen ausgeliefert, mit harten Deadlines und einer
mehrstufigen Pipeline für Fact-Checking und Style-Compliance. Falsche Inhalte hätten
juristische Folgen gehabt, also musste das Quality Gating sitzen.

Python ist meine tägliche Arbeitssprache. Mit neueren AI-Tools wie Cursor und Claude
Code arbeite ich gerne und sehe darin einen deutlichen Effizienzgewinn, gerade in
Kombination mit eigener Architekturarbeit.

Ich bin kurzfristig verfügbar. Über ein Kennenlernen würde ich mich freuen.
"""


def build_system_prompt(form: str, lang: str = "de") -> str:
    anrede_form = (
        "Verwende durchgehend die Du-Form (du, dich, deine, euch, ihr für die Firma)."
        if form == "du"
        else "Verwende durchgehend die Sie-Form (Sie, Ihnen, Ihre)."
    )

    if lang == "en":
        opening = (
            "You are writing a cover letter for Pavlos Musenidis. "
            "Write the ENTIRE letter in English — subject line, body, everything. "
            "The instructions below are in German for internal reference only — "
            "the OUTPUT must be in English. "
            "It must sound like him — direct, analytical, no AI smell."
        )
        lang_reminder = (
            "\n\nCRITICAL REMINDER: Write the cover letter in ENGLISH. "
            "Every sentence of the output must be in English. "
            "The BETREFF line must also be in English (use 'RE:' or just the subject)."
        )
    else:
        opening = (
            "Du schreibst ein Anschreiben für Pavlos Musenidis. "
            "Es muss klingen wie er selbst — direkt, analytisch, ohne KI-Geruch."
        )
        lang_reminder = ""

    return f"""{opening}


KANDIDATEN-PROFIL:
{PROFILE}

REFERENZ-ANSCHREIBEN (Stil treffen, NICHT Inhalt kopieren — nur zeigen wie er schreibt):
{REFERENCE_LETTER}

═══ ABSOLUTE STIL-REGELN (jede Verletzung ruiniert das Anschreiben) ═══

S1. KEINE Gedankenstriche oder Halbgeviertstriche (— oder –). Komma, Punkt oder Klammern stattdessen.

S2. KEINE Doppelpunkte zur Aufzählung ("Mein Stack: Python" → "Zu meinem Stack gehören Python").

S3. KEINE Semikolons.

S4. KEINE Bullet Points. Alles Fließtext.

S5. KEINE Superlative, Enthusiasmus-Signale oder Marketing-Sprache: "leidenschaftlich", "begeistert mich zutiefst", "spannend", "innovativ", "zukunftsweisend". Nüchtern und sachlich.

S6. KEINE Floskeln: "Know-how einbringen", "nächste Generation mitgestalten", "robuste Lösungen", "agiles Umfeld". Konkrete Beschreibung statt leere Versprechen.

S7. KEINE Wiederholungen. Jeder Satz bringt neue Information.

S8. KEINE Sternchen-Gendern. Neutrale Begriffe oder ausgeschriebene Doppelform.

S9. ZAHLEN GENAU: über 2,5 Jahre Erfahrung, über 500 Redakteure, über 250 Portale. Zahlen zur Wahlautomatisierung nur aus dem Profil übernehmen, nicht erfinden.

S10. ANREDE: {anrede_form}

S11. ANTI-AUFZÄHLUNG: Kein Satz darf mehr als 4 Stack-Komponenten aufzählen. Stack-Aufzählungen brechen den Lesefluss. Lieber 2-3 Schlüsselelemente nennen die zur Stelle passen, statt alles. Ein Satz wie "Python, FastAPI, LangChain, LangGraph, LiteLLM, Milvus, AWS, Kubernetes" ist verboten.

═══ GLOBAL VERBOTEN (in JEDEM Absatz, nicht nur einem) ═══

- "Asynchrone Zusammenarbeit / Zeitzonen / verteilte Teams / globale Stakeholder" — nicht zutreffend, auch nicht als Negation einbauen ("auch wenn ich keine Zeitzonen-Erfahrung habe...").
- "Domäne X ist neu für mich, aber..." — Disclaimer über fehlendes Domänenwissen nur dann einbauen, wenn die Stelle das EXPLIZIT verlangt. Sonst weglassen.
- "Dass Sie/ihr remote arbeitet, passt mir ebenfalls" — Standort-/Remote-Kommentare nur wenn explizit gefragt. Bewerbung impliziert das.
- "Eure Mission begeistert mich" / "Spannendes Produkt" / "Beeindruckende Technologie" — wertende PR-Sprache.
- "Ich halte für hart" / "Das ist eine andere Qualität von" — bewertende, herablassend wirkende Formulierungen.

═══ INHALT UND STRUKTUR (5 ABSÄTZE) ═══

Insgesamt 200-260 Wörter. Fünf Absätze mit klar getrennter Funktion. Visuell soll der
Brief atmen, nicht in drei Wänden ankommen.

── ABSATZ 1: HOOK (1-2 SÄTZE, HARTE OBERGRENZE) ──

MAXIMAL 2 SÄTZE. Nicht 3. Wenn unklar: 1 Satz.

Funktion: Lies das Posting wie ein Mensch und finde EINE Stelle, die ehrlich anzieht.
Reagiere darauf in Pavlos' Stimme, kurz und persönlich. Dies ist KEINE Firmen-Analyse
und KEIN Beweis dass du verstanden hast, womit die Firma Geld verdient. Es ist eine
ehrliche Reaktion auf das was im Posting steht.

Andockpunkte (in dieser Reihenfolge bevorzugen):
1. Ein Mission/Werte-Satz im Posting der wirklich resoniert (Sinn, Menschen, Bildung,
   Gesundheit, Zugang). Nur nehmen wenn er da steht und ehrlich passt, nicht erfinden.
2. Eine Rollen-Eigenschaft die Pavlos sucht und die explizit dasteht (Ownership,
   hands-on, Schnittmenge Engineering+Produkt, Wachstum, technische Tiefe).
3. Ein konkretes Tech-Setup oder Problem-Profil das gut zu seinem Hintergrund passt
   (LLM im Produkt-Kern statt Add-on, Pipelines mit harten Qualitäts-Anforderungen).
4. Erst wenn 1-3 nichts hergeben: eine sachliche Beobachtung über Domäne/Produkt,
   ohne zu werten.

Ton: persönlich-reaktiv, nicht analytisch-distanziert. Sätze die mit "Auf die Stelle
bin ich gestoßen, weil...", "Im Stellentext ist mir hängen geblieben..." oder "Was mich
an der Rolle zieht..." beginnen, sind oft besser als "Bei [Firma] sitzt...".

Nach dem Andockpunkt: optional ein kurzer zweiter Satz der zeigt, warum Pavlos zu genau
diesem Punkt passt. KONKRET, nicht allgemein. Beispiele: "über 2,5 Jahre LLM-Pipelines
im Produktionsbetrieb, nicht als PoC", "ich habe genau das bei X gebaut", "das ist mein
Alltagskontext seit über 2,5 Jahren". Nur einbauen wenn es sich natürlich fügt.
VERBOTEN: "Ich bin der ideale Kandidat", "Meine Erfahrung macht mich perfekt für..."

KEINE feste Schablone vorgeben, jeder Hook ist neu und auf das konkrete Posting
zugeschnitten.

BAD-Beispiele (so NICHT):
- "Remote People baut Infrastruktur für globale Teams, bei der Compliance und
  Payroll-Automatisierung im Kern sitzen."
  (Backwards-Analyse, klingt nach Wikipedia-Eintrag, nicht nach jemandem der sich
  bewirbt)
- "Eure Mission, Compliance neu zu denken, begeistert mich."
  (PR-Floskel)
- "Was ihr bei X macht, halte ich für ein hartes Problem."
  (bewertend, anmaßend)
- "Bei Bluefish AI geht es um LLM-Anwendungen, eine andere Qualität von
  Produktionsdruck als..."
  (akademisch, herablassend)

── ABSATZ 2: POSITION (2-3 SÄTZE) ──

Wer Pavlos ist, wo er arbeitet, mit welchem Stack. Ippen Digital mit konkreten Zahlen
(über 2,5 Jahre, über 250 Portale). Stack-Elemente die zur Stelle passen — MAXIMAL
4 Komponenten in einem Satz (S11). Kein Projekt-Name in diesem Absatz, nur Rolle und
Stack.

── ABSATZ 3: KONKRETES BEISPIEL (2-3 SÄTZE) ──

EIN konkretes Projekt, das thematisch zur Stelle passt. Auswahl-Logik steht im Profil
unter "NUR WENN ...". Maximal 2 Projekte erlaubt, aber 1 ist meistens besser. Mit
konkreten Zahlen oder einer technischen Eigenheit, die die Komplexität greifbar macht.
NICHT alle Projekte aufzählen.

── ABSATZ 4: MATCH (2-3 SÄTZE) ──

Was Pavlos konkret mitbringt das zur Stelle passt. Kernpunkt: Python ist tägliche
Arbeitssprache, AI-Tools (Cursor, Claude Code) ergänzen den Workflow. Kurz und ohne
Selbstvermarktungs-Vokabular. Formulierungs-Basis (NICHT wörtlich, aber Richtung und
Länge treffen — 2 Sätze reichen meistens):

"Python ist meine tägliche Arbeitssprache. Mit neueren AI-Tools wie Cursor und Claude
Code arbeite ich gerne und sehe darin einen deutlichen Effizienzgewinn, gerade in
Kombination mit eigener Architekturarbeit."

VERBOTEN hier: "verstehe das Fundament darunter", "messbarer Effizienzgewinn",
"echtes Architekturverständnis", "nicht aus Hype" — alles wertend und arrogant.

Wenn ein Domänen-Gap explizit aus der Stelle kommt: kurz und konstruktiv ansprechen.
Sonst weglassen.

── ABSATZ 5: ABSCHLUSS (1-2 SÄTZE) ──

Eigener kurzer Absatz, nicht angehängt an Absatz 4. Verfügbarkeit kurz, dann
Kennenlern-Satz. Beispiele:

Deutsch: "Ich bin kurzfristig verfügbar. Über ein Kennenlernen würde ich mich freuen."
Englisch: "I'm available to start on short notice. I'd love to connect."

VERBOTEN: "Ein Wechsel" (mehrdeutig), "A move" (klingt nach Umzug).
Nicht ausschmücken.

═══ OUTPUT-FORMAT ═══

Erste Zeile: BETREFF: <Betreff>
Dann Leerzeile.
Dann der vollständige Brief-Text in dieser Reihenfolge:

1. ANREDE — genau eine Zeile, dann Leerzeile. Wähle passend zu Ton und Sprache:
   - Formelles Deutsch: "Sehr geehrte Damen und Herren,"
   - Startup/Du-Ton: "Hallo zusammen,"
   - Formelles Englisch: "Dear Hiring Team," oder "Dear [Company Name] Team,"
   - Casual Englisch: "Hi there," oder "Hi [Company] team,"
   Nicht immer "Hallo zusammen," — wenn die Firma konservativ oder groß wirkt, lieber formell.

2. BODY-ABSÄTZE — durch doppelten Zeilenumbruch getrennt. KEIN Name am Ende.

3. GRUSSFORMEL — als letzter Absatz, NUR die Formel, KEIN "Pavlos Musenidis". Passend wählen:
   - Formelles Deutsch: "Mit freundlichen Grüßen"
   - Casual Deutsch: "Viele Grüße"
   - Formelles Englisch: "Best regards,"
   - Casual Englisch: "Kind regards,"

KEINE Unterschrift (Pavlos Musenidis kommt aus dem Template).
{lang_reminder}"""


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _count_sentences(text: str) -> int:
    """Counts sentences ending in . ! ? — abbreviations are rare in cover letters."""
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]
    return len(parts)


def _parse_response(text: str, fallback_title: str) -> tuple[str, str]:
    """Splits raw LLM output into (subject, body)."""
    lines = text.strip().split("\n")
    subject = ""
    body_start = 0
    for i, line in enumerate(lines):
        for prefix in ("BETREFF:", "RE:", "SUBJECT:", "Subject:"):
            if line.startswith(prefix):
                subject = line[len(prefix):].strip()
                body_start = i + 1
                break
        if subject:
            break

    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    body = "\n".join(lines[body_start:]).strip()
    if not subject:
        subject = f"Bewerbung als {fallback_title}"
    return subject, body


def _first_paragraph(body: str) -> str:
    for para in body.split("\n\n"):
        para = para.strip()
        if para:
            return para
    return ""


def call_anthropic(system_prompt: str, job: dict, api_key: str,
                   model: str = "claude-sonnet-4-5-20250929") -> tuple[str, str] | None:
    """Returns (subject, body_text) on success.

    Performs a hard validation of the 2-sentence hook rule on paragraph 1.
    If the first attempt produces > 2 sentences in paragraph 1, retries once
    with an appended hint. After that, the result is returned regardless so
    the user can edit manually.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n✗ anthropic package fehlt. Run: pip install anthropic\n", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    base_user_msg = f"""STELLE:

Titel: {job['title']}
Unternehmen: {job['company']}
Standort: {job['location']}

Stellenbeschreibung:
{job['description'][:4000]}

Schreibe das Anschreiben jetzt."""

    def _generate(user_msg: str) -> tuple[str, str] | None:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
        except Exception as e:
            log.error("Anthropic API failed: %s", e)
            return None
        return _parse_response(text, job["title"])

    result = _generate(base_user_msg)
    if not result:
        return None

    subject, body = result
    hook_sentences = _count_sentences(_first_paragraph(body))
    if hook_sentences > 2:
        log.warning(
            "Hook hat %d Saetze (>2). Retry mit Hinweis.", hook_sentences
        )
        retry_msg = (
            base_user_msg
            + "\n\nWICHTIG: Vorheriger Versuch hatte zu viele Saetze in Absatz 1. "
            "Absatz 1 darf MAXIMAL 2 Saetze enthalten. Kuerze entsprechend."
        )
        retry = _generate(retry_msg)
        if retry:
            subject, body = retry
            retry_sentences = _count_sentences(_first_paragraph(body))
            if retry_sentences > 2:
                log.warning(
                    "Retry-Hook hat immer noch %d Saetze. Gebe trotzdem zurueck.",
                    retry_sentences,
                )

    return subject, body


# ---------------------------------------------------------------------------
# Render to HTML + PDF
# ---------------------------------------------------------------------------

def _company_slug(subject: str) -> str:
    """Extracts a short slug from the subject line for use in filenames."""
    s = re.sub(r"(?i)(bewerbung\s*(als|–|-)?|application\s*(as|–|-)?)\s*", "", subject)
    s = re.sub(r"[^\w\s]", "", s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s[:40] or "anschreiben"


def render_html(subject: str, body: str, company: str = "", form: str = "") -> str:
    template_path = ROOT / "anschreiben_template.html"
    if not template_path.exists():
        template_path = Path("/mnt/project/anschreiben_template.html")
    template = template_path.read_text(encoding="utf-8")

    date_str = datetime.now().strftime("%d.%m.%Y")

    # Split body into paragraphs; first = greeting, last = closing
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        anrede = paragraphs[0]
        gruss = paragraphs[-1]
        content = paragraphs[1:-1]
    elif len(paragraphs) == 2:
        anrede = paragraphs[0]
        gruss = ""
        content = paragraphs[1:]
    else:
        anrede = ""
        gruss = ""
        content = paragraphs

    body_html = "\n            ".join(f"<p>{p}</p>" for p in content)

    company_part = re.sub(r"\s+", "_", company.strip().lower())[:30] if company else _company_slug(subject)
    titel = f"pavlos_musenidis_anschreiben_{company_part}"

    return (template
            .replace("{{TITEL}}", titel)
            .replace("{{DATUM}}", date_str)
            .replace("{{BETREFF}}", subject)
            .replace("{{ANREDE}}", anrede)
            .replace("{{GRUSS}}", gruss)
            .replace("{{BODY}}", body_html))


def render_pdf(html_path: Path, pdf_path: Path) -> bool:
    if not shutil.which("wkhtmltopdf"):
        log.warning("wkhtmltopdf nicht installiert, PDF wird nicht erzeugt")
        log.warning("  Install: brew install --cask wkhtmltopdf  (macOS)")
        return False
    try:
        subprocess.run(
            ["wkhtmltopdf", "--enable-local-file-access",
             "--print-media-type", "--quiet",
             str(html_path), str(pdf_path)],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("wkhtmltopdf failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s[:50]


async def main_async(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(
            "\n✗ Anthropic API Key fehlt.\n"
            "  Holen: https://console.anthropic.com/settings/keys\n"
            "  Dann in .env eintragen: ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n→ Hole Stellenbeschreibung von {args.url}")
    job = await fetch_job_from_url(args.url)
    if not job:
        print("✗ Konnte Stelle nicht laden. Ist die URL korrekt?", file=sys.stderr)
        sys.exit(1)

    print(f"  Titel:       {job['title']}")
    print(f"  Unternehmen: {job['company']}")
    print(f"  Standort:    {job['location']}")
    print(f"  Länge:       {len(job['description'])} Zeichen")

    form = args.form or "du"
    lang = args.lang or "de"
    print(f"  Anrede-Form: {form.upper()}, Sprache: {lang.upper()}")

    print(f"\n→ Generiere Anschreiben mit {args.model}...")
    system_prompt = build_system_prompt(form, lang)
    result = call_anthropic(system_prompt, job, api_key, args.model)
    if not result:
        print("✗ LLM-Aufruf fehlgeschlagen", file=sys.stderr)
        sys.exit(1)

    subject, body = result

    # Output dir
    out_dir = Path(args.out) if args.out else ROOT / "cover_letters"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"{ts}_{slugify(job['company'])}"

    # Plain text — body already contains greeting and closing
    txt_path = out_dir / f"{base}.txt"
    txt_path.write_text(
        f"Betreff: {subject}\n\n"
        f"{body}\n\n"
        f"Pavlos Musenidis\n",
        encoding="utf-8"
    )

    # HTML + PDF
    html = render_html(subject, body)
    html_path = out_dir / f"{base}.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = None
    if not args.no_pdf:
        pdf_candidate = out_dir / f"{base}.pdf"
        if render_pdf(html_path, pdf_candidate):
            pdf_path = pdf_candidate

    # Console output
    print("\n" + "=" * 70)
    print(f"BETREFF: {subject}")
    print("=" * 70)
    print(f"\n{body}\n")
    print("Pavlos Musenidis")
    print("=" * 70)

    print(f"\n✓ Text:  {txt_path}")
    print(f"✓ HTML:  {html_path}")
    if pdf_path:
        print(f"✓ PDF:   {pdf_path}")
    print()


def main():
    p = argparse.ArgumentParser(description="Anschreiben-Generator für eine Job-URL")
    p.add_argument("url", help="Job-URL aus dem Report (Greenhouse, Lever, Personio, Ashby)")
    p.add_argument("--out", default=None, help="Output-Verzeichnis (default: ./cover_letters)")
    p.add_argument("--form", choices=["du", "sie"], default=None,
                   help="Du oder Sie (default: du)")
    p.add_argument("--lang", choices=["de", "en"], default=None,
                   help="Briefsprache: de oder en (default: de)")
    p.add_argument("--model", default="claude-sonnet-4-5-20250929",
                   help="Anthropic Modell (default: claude-sonnet-4-5)")
    p.add_argument("--api-key", default=None, help="Anthropic API Key (oder in .env)")
    p.add_argument("--no-pdf", action="store_true", help="Keine PDF erzeugen")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
