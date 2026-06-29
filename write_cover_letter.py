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
# CV loader — pulls the live HTML CV into a plain-text block the LLM can use
#
# The HTML CV is the single source of truth for projects, stack, education and
# certifications. By reading it on every run we don't need to keep a parallel
# project list in code.
# ---------------------------------------------------------------------------

CV_PATH = Path(
    os.environ.get(
        "CV_HTML_PATH",
        "/Users/pmusenidis/Documents/Musenidis_Lebenslauf/pavlos_cv_merged.html",
    )
)

_CV_TAG_RE = re.compile(r"<[^>]+>")
_CV_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_CV_WS_RE = re.compile(r"[ \t]+")
_CV_NEWLINES_RE = re.compile(r"\n{3,}")
_cv_text_cache: str | None = None


def _html_to_text(html_src: str) -> str:
    """Best-effort HTML → plain text, preserves block-level line breaks."""
    src = _CV_STYLE_RE.sub(" ", html_src)
    block_tags = (
        "p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "br",
    )
    for tag in block_tags:
        src = re.sub(
            rf"<{tag}[^>]*>", "\n", src, flags=re.IGNORECASE,
        )
        src = re.sub(rf"</{tag}>", "\n", src, flags=re.IGNORECASE)
    src = _CV_TAG_RE.sub(" ", src)
    src = (src
           .replace("&nbsp;", " ")
           .replace("&amp;", "&")
           .replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&quot;", "\"")
           .replace("&#39;", "'"))
    lines = [_CV_WS_RE.sub(" ", line).strip() for line in src.split("\n")]
    lines = [line for line in lines if line]
    out = "\n".join(lines)
    return _CV_NEWLINES_RE.sub("\n\n", out)


def load_cv_text(force_reload: bool = False) -> str:
    """Read the CV HTML once per process, return cleaned plain text.

    Returns an empty string if the file is missing, so the prompt builder can
    fall back to the inline PROFILE constant.
    """
    global _cv_text_cache
    if _cv_text_cache is not None and not force_reload:
        return _cv_text_cache
    try:
        raw = CV_PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        log.warning("CV file not readable at %s: %s", CV_PATH, exc)
        _cv_text_cache = ""
        return ""
    _cv_text_cache = _html_to_text(raw)
    log.info("CV loaded from %s (%d chars)", CV_PATH, len(_cv_text_cache))
    return _cv_text_cache


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

PROFILE_PREFERENCES = """
Pavlos Musenidis arbeitet aktuell als AI Engineer bei Ippen Digital, Deutschlands
größtem lokalen Journalismus-Netzwerk mit über 250 Portalen. Seit September 2023,
also über 2,5 Jahre. Alle Projekte, Stack-Komponenten, Ausbildungs- und
Zertifikatsangaben kommen aus dem CV-Block unten. Erfinde nichts darüber hinaus.

Verfügbarkeit: kurzfristig, mindestens 2 Wochen.
Wohnort: Stuttgart. Präferenz 100% remote oder hybrid mit Pendeln nach München
oder Stuttgart.
Sprachen: Deutsch Muttersprache, Englisch fließend.
"""

FAKTENTREUE = """
Anti-Halluzinations-Regeln. Diese gelten kompromisslos, weil das LLM hier
historisch konsistent danebenliegt.

1. Wahlautomatisierung 2025: drei Wahlzyklen in NRW, Bayern und Hessen.
   NICHT als Bundestagswahl benennen. NRW war Co-Lead-Rolle (zweiter Entwickler),
   Bayern und Hessen war Pavlos alleiniger Owner. Gesamt über 7.000 Artikel über
   2.600+ Kommunen, über 1 Million Pageviews. Wenn unklar welche Wahl im Brief
   passt, neutral von "Wahlautomatisierung 2025" sprechen.

2. Central AI Workflow Engine: Pavlos war CO-DEVELOPER, NICHT alleiniger
   Architekt. Auch nicht "ich habe konzipiert", auch nicht "von mir gebaut".
   Der ehrliche Ausdruck ist "mitentwickelt" oder "Co-Developer".

3. Enterprise Chatbot Prototype: KEINEN spezifischen Kunden namentlich nennen
   (kein "OpenOlat" oder ähnliches). Bleibt bei "Enterprise Chatbot Prototype".

4. Keine Erfindung von Zahlen, Projekten, Frameworks oder Erfahrungen, die nicht
   im CV stehen. Wenn etwas im Posting verlangt wird und im CV nicht steht,
   weglassen statt erfinden. Lieber ein kürzerer Brief als eine Lüge.
"""

MOTIVATION = """
Pavlos legt Wert darauf zu wissen, wofür er baut. Wenn die Stelle erkennbar
sinnstiftend ist (Gesundheit, Bildung, Energie, Barrierefreiheit, öffentlicher
Sektor, echter Nutzer-Impact), darf das Motiv im Hook auftauchen, aber nüchtern
und ohne PR-Wording. Wenn das nicht zur Stelle passt, weglassen.
"""

TONALITAET = """
Pavlos schreibt sachlich, beobachtend, leicht analytisch. Er bewertet nicht
("X ist beeindruckend", "Y ist hart"), er beobachtet ("Bei X liegt Y zwischen
Z"). Er lobt keine Firmen ("eure Mission begeistert mich"), er zeigt Interesse
durch Konkretheit. Er ist nie performativ. Wenn ein Satz so klingt als könnte
ihn jeder zweite Bewerber schreiben, ist er falsch.

"Nicht performativ" heißt konkret: keine Bewerber-Standardfloskeln, keine
Selbstbewertung, keine Begeisterungssignale ohne Substanz. Pavlos formuliert
Aussagen über seine Arbeit oder Beobachtungen über die Stelle, keine
Anpreisungen seiner Person.
"""

# ---------------------------------------------------------------------------
# Optional prompt blocks — only injected when the user fills them in.
# Keeps the base prompt lean for the default case (AI Engineer, no special
# remote/salary/location signal) and lets the user steer for non-default
# applications (consulting, ML focus, far-from-Stuttgart locations etc).
# ---------------------------------------------------------------------------

ROLE_FOCUS_BLOCKS = {
    "ai_engineer": """
ROLLEN-FOKUS  AI Engineer (Pavlos' Tagesgeschäft).

Kein Umschwung zu erklären, das ist exakt sein Feld. Wenn die Stelle ein
erkennbar besseres Setup bietet (mehr Ownership, breitere Themen,
stärkeres Team, klareres Produkt, größerer Wachstumsraum), darf das im
Hook oder Match-Absatz andocken. Die sinngemäße Linie ist, dass Pavlos
mag was er tut, aber eine neue Arbeitsumgebung sucht in der er weiter
wachsen kann. Den aktuellen Arbeitgeber nicht abwerten.
""",
    "ai_consultant": """
ROLLEN-FOKUS  AI Consultant / Consulting AI Engineer.

Pavlos bewirbt sich gezielt auf eine Consulting-Rolle, weil ihm die
Konzept- und MVP-Phase besonders liegt und er die Abwechslung zwischen
verschiedenen Projekten und Kunden schätzt. Dieses Motiv darf sachlich
im Hook oder Match-Absatz auftauchen, mit Beleg aus seinem bisherigen
Tun. Belegmaterial findest du im CV (Aufbau der AI-Pipeline von Null
bei Ippen Digital, mehrere Use-Cases im Pilot eigenständig konzipiert,
direkte Stakeholder-Arbeit mit den Redaktionen).

Stärken die hier zählen sind End-to-End-Ownership, schnell von
Anforderung zur Architektur, MVP- und Prototyp-Mentalität, direkter
Umgang mit nicht-technischen Stakeholdern.

Verboten sind Floskeln wie "spannende Vielfalt", "abwechslungsreiche
Projekte" oder "Begeisterung für Beratung". Das Motiv soll als nüchterne
Präferenz formuliert sein, nicht als Marketing-Aussage.
""",
    "ml_engineer": """
ROLLEN-FOKUS  ML Engineer / ML Specialist.

Pavlos kommt aus der LLM-Anwendungsseite und bewirbt sich auf eine
ML-zentrische Rolle. Sein Umschwung-Motiv hat zwei Teile.

Fachlich begründet, M.Sc. Computational Linguistics mit Schwerpunkt
Deep Learning für NLP, Master-Thesis zu T5-Fine-Tuning, Erfahrung mit
Fine-Tuning von Open-Source-Modellen deployt via AWS SageMaker bei
Ippen Digital. Das Fundament ist vorhanden.

Beobachtungsbasiert, für viele konkrete Aufgaben sind spezialisierte
Modelle (kleiner, gezielt trainiert, deterministischer) die bessere
Lösung als generische LLMs. Genau in dieser Richtung will Pavlos sich
jetzt vertiefen.

Dieses Doppelmotiv darf sachlich im Hook oder Match-Absatz auftauchen,
am besten ein Satz pro Teil. Stärken die hier zählen sind das Studium,
Fine-Tuning-Erfahrung und die produktionsseitige Sicht auf wo LLM
aufhört und spezialisierte Modelle anfangen.

Verboten sind Selbstüberhöhung im ML-Theorie-Bereich,
"leidenschaftlicher ML-Forscher"-Sprache und Überverkaufen des
akademischen Hintergrunds.
""",
}

REMOTE_PREF_BLOCKS = {
    "remote_with_visits": """
SETUP-WUNSCH  Vollremote mit gelegentlichen Vor-Ort-Besuchen (etwa
einmal im Monat bis alle zwei Monate).

Wenn die Stelle dieses Setup nicht von vornherein anbietet ODER wenn
der Standort weit von Stuttgart entfernt ist, das einmal kurz und
sachlich im Brief klarstellen. Sinnvoller Platz ist der Schluss-Absatz
nahe der Verfügbarkeits-Zeile. Wenn die Stelle sowieso vollremote
ausgeschrieben ist und Standort egal ist, weglassen.
""",
    "hybrid_weekly": """
SETUP-WUNSCH  Hybrid mit etwa einem Vor-Ort-Tag pro Woche.

Das passt typischerweise bei Standorten im Pendel-Radius von Stuttgart
(etwa eine Stunde oder weniger, zum Beispiel Heidelberg, Karlsruhe,
Tübingen, Mannheim). Wenn die Stelle eine höhere Vor-Ort-Frequenz
erwartet (drei Tage, vier Tage, fünf Tage pro Woche), einen Satz dazu
der Pavlos' realistische Frequenz benennt. Wenn die Stelle sowieso
flexibel hybrid ist, weglassen.
""",
    "remote_only": """
SETUP-WUNSCH  Vollständig remote, ohne regelmäßige Vor-Ort-Termine.

Wenn die Stelle vor-Ort-Anwesenheit verlangt oder kein klares
Remote-Setup beschreibt, das einmal kurz und sachlich im Brief
klarstellen, am besten im Schluss-Absatz. Wenn die Stelle sowieso
vollremote ausgeschrieben ist, weglassen.
""",
}


def _salary_block(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return f"""
GEHALTSVORSTELLUNG  {value}

Pavlos möchte das Gehaltsband im Brief erwähnen. Sachlich, in einem
einzelnen Satz, am besten im Schluss-Absatz vor der Verfügbarkeit. Kein
Verhandlungs-Wording, kein "im Rahmen von", einfach die genannte Angabe
übernehmen. Wenn die Angabe ein Bereich ist (zum Beispiel "85-90k"), als
Bereich übernehmen.
"""


def _location_pref_block(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return f"""
STANDORT-PRÄFERENZ  {value}

Die Stelle ist offenbar für mehrere Standorte ausgeschrieben. Pavlos
bevorzugt den oben genannten. Das einmal kurz im Brief klarmachen,
sinnvoller Platz ist der Schluss-Absatz oder bei der Verfügbarkeit.
Sachlich, ohne lange Begründung.
"""


# Short voice samples to calibrate tone without giving a copy-paste template.
# Each pair shows ONE typical move Pavlos would make and the LLM-default he
# would NOT make. The model picks up the rhythm and avoidance patterns from
# these without inheriting a full letter structure.
VOICE_SAMPLES = """
So klingt Pavlos. So klingt er nicht.

GUT: "Was in der Stellenbeschreibung steht, beschreibt ziemlich genau, was ich
      seit über 2,5 Jahren bei Ippen Digital tue."
NICHT: "Mit großem Interesse habe ich Ihre Stellenanzeige gelesen und möchte
       mich hiermit auf die ausgeschriebene Position bewerben."

GUT: "Die Pipeline läuft seit über einem Jahr in Produktion, etwa 100 Aufrufe
      pro Tag."
NICHT: "Ich konnte erfolgreich umfangreiche Erfahrungen im Bereich
       Production-LLM-Pipelines sammeln."

GUT: "Bei Ippen Digital baue ich Multi-Step-LLM-Pipelines mit LangChain und
      Multi-Provider-Setup."
NICHT: "Mein umfangreiches Knowhow im Bereich Production-AI bringe ich gerne
       in Ihr dynamisches Team ein."

GUT: "An Engineering reizt mich die Phase, in der der Lösungsweg noch nicht
      feststeht. Genau das habe ich bei Ippen Digital von Null aufgebaut."
NICHT: "Ich bin leidenschaftlich daran interessiert, an spannenden und
       zukunftsweisenden AI-Projekten mitzuwirken."

GUT: "Multi-Provider-LLM-Orchestrierung über OpenAI, Anthropic und Google ist
      seit über 2,5 Jahren mein Tagesgeschäft."
NICHT: "Meine ausgeprägte Expertise in Multi-Provider-LLM-Orchestrierung
       qualifiziert mich ideal für diese spannende Herausforderung."
"""


def build_system_prompt(
    lang: str = "de",
    role_focus: str = "ai_engineer",
    remote_pref: str = "",
    salary: str = "",
    location_pref: str = "",
) -> str:
    """Build the cover-letter system prompt.

    Same prompt is used for fresh generation and for refining an existing
    letter. The caller decides what to put in the user message (job posting
    for generation, existing letter + change request for refinement).

    Optional steering parameters (only inject prompt blocks when set):
      role_focus    one of "ai_engineer" (default), "ai_consultant", "ml_engineer"
      remote_pref   one of "", "remote_with_visits", "hybrid_weekly", "remote_only"
      salary        free text, e.g. "85-90k EUR"
      location_pref free text, e.g. "Heidelberg" for multi-location postings

    The Du/Sie decision is delegated to the model based on the job posting
    style (du-Posting → Du, Sie-Posting or formal corporate → Sie).
    """
    anrede_form = (
        "ANREDE-FORM (Du oder Sie) leitest du aus dem Stellenposting ab. "
        "Wenn die Stelle dich klar duzt (du, deine, euch, ihr) oder erkennbar "
        "Startup/Du-Kultur signalisiert (zum Beispiel Hallo statt Sehr "
        "geehrte, casual Tonalität, Tech-Startup-Branding), antwortest du im "
        "ganzen Brief konsistent auf Du. Wenn die Stelle siezt (Sie, Ihre) "
        "oder erkennbar konservativ/Konzern wirkt, antwortest du konsistent "
        "auf Sie. Wenn das Posting auf Englisch ist, entfällt die "
        "Du/Sie-Frage, dann gelten die englischen Anrede- und Schluss-Regeln "
        "weiter unten. Bei Unklarheit auf Deutsch wählst du Sie, weil das "
        "der konservativere Default ist."
    )

    if lang == "en":
        opening = (
            "You are writing a cover letter for Pavlos Musenidis. "
            "Write the entire letter in English including the subject line. "
            "Internal instructions stay in German but the output must be English. "
            "It must sound like him, direct and analytical, with no AI smell."
        )
        lang_reminder = (
            "\n\nCRITICAL REMINDER. Write the cover letter in English. "
            "Every sentence of the output must be in English. "
            "The BETREFF line must also be in English, use 'RE:' or just the subject."
        )
    else:
        opening = (
            "Du schreibst ein Anschreiben für Pavlos Musenidis. "
            "Es muss klingen wie er selbst, direkt und analytisch, ohne KI-Geruch."
        )
        lang_reminder = ""

    role_block = ROLE_FOCUS_BLOCKS.get(role_focus, ROLE_FOCUS_BLOCKS["ai_engineer"])
    remote_block = REMOTE_PREF_BLOCKS.get(remote_pref, "")
    salary_block_text = _salary_block(salary)
    location_block_text = _location_pref_block(location_pref)

    optional_blocks = []
    if remote_block:
        optional_blocks.append(remote_block.strip())
    if salary_block_text:
        optional_blocks.append(salary_block_text.strip())
    if location_block_text:
        optional_blocks.append(location_block_text.strip())
    optional_section = ""
    if optional_blocks:
        optional_section = (
            "\n═══ STELLEN-SPEZIFISCHE WÜNSCHE (von Pavlos eingegeben) ═══\n\n"
            + "\n\n".join(optional_blocks)
            + "\n"
        )

    cv_text = load_cv_text()
    if cv_text:
        cv_block = (
            "═══ LEBENSLAUF (PRIMÄRE FAKTENQUELLE) ═══\n"
            "Alles was du an konkreten Projekten, Stack-Elementen, Ausbildungs- "
            "oder Zertifikatsangaben nennst, MUSS aus diesem CV-Text kommen. "
            "Erfinde nichts darüber hinaus. Wenn die Stelle nach etwas fragt "
            "das hier nicht steht, dann steht es Pavlos nicht zur Verfügung "
            "und gehört nicht in den Brief.\n\n"
            f"{cv_text}"
        )
    else:
        cv_block = (
            "═══ LEBENSLAUF ═══\n"
            "(CV-Datei nicht lesbar. Halte dich strikt an das Präferenz-Profil "
            "unten und erfinde keine Projekte oder Stack-Details.)"
        )

    return f"""{opening}


═══ KANDIDATEN-KONTEXT (Präferenzen, nicht im CV) ═══
{PROFILE_PREFERENCES}

{cv_block}

═══ FAKTENTREUE ═══
{FAKTENTREUE}

═══ MOTIVATION ═══
{MOTIVATION}

═══ ROLLEN-FOKUS (Pavlos' aktuelle Bewerbungs-Richtung) ═══
{role_block}
{optional_section}
═══ TONALITÄT ═══
{TONALITAET}

═══ MIKRO-STIL-BEISPIELE ═══
{VOICE_SAMPLES}

**═══ HOOK-REGEL ═══**

Die GUT-Beispiele oben zeigen nur den Stil, nie den Inhalt. Übernimm niemals eine dieser Formulierungen direkt oder leicht abgewandelt als ersten Satz.

Der Hook muss aus dem spezifischen Match zwischen Pavlos' Profil und dieser konkreten Stelle entstehen. Finde den interessantesten Andockpunkt, also den Punkt wo sein Hintergrund und die Stelle am unerwartesten oder präzisesten zusammentreffen. Das kann ein Projekt sein, eine Technologie, eine Verantwortungsrolle, eine Domäne oder eine Art zu arbeiten. Nicht der offensichtlichste Punkt, sondern der ehrlichste und konkreteste.

Jeder Brief hat einen anderen ersten Satz. Wenn der Hook für zwei verschiedene Stellen identisch klingen könnte, ist er falsch.

═══ INHALT ═══

Der Brief beantwortet zwei Fragen aus Pavlos' Sicht und in seinem Stil.
Erstens, warum genau diese Stelle. Zweitens, warum er konkret dafür passt.

Lies die Stellenbeschreibung sorgfältig und finde den ehrlichsten Andockpunkt.
Das kann der Stack sein, die Art der Aufgabe, die Domäne, der Ownership-Grad,
die Phase des Produkts oder etwas anderes. Es gibt kein vorgegebenes Raster.
Was du NICHT machen darfst, ist die Stellenbeschreibung paraphrasieren oder
loben. Du darfst nur eine Aussage über Pavlos formulieren, die zufällig zu
dieser Stelle passt.

Konkrete Belege kommen aus dem CV oben. Wähle die Projekte und Stack-Elemente,
die zur Stelle passen, statt einer Aufzählung von allem. Maximal zwei Projekte
nennen, eines ist meistens besser. Bei jedem Projekt einen konkreten Anker
(Volumen, Verantwortungs-Rolle, Tech-Eigenheit), keine generische Beschreibung.

Wenn Pavlos die Anforderungen der Stelle stark abdeckt (Stack-Overlap deutlich,
Kernaufgabe identisch zu seinem Tagesgeschäft, Ownership-Profil passt), darf
das im Brief sichtbar werden. Nicht durch Selbstbewertung ("ich bin der perfekte
Kandidat"), sondern durch eine sachliche Beobachtung mit Beleg ("der Stack
deckt sich mit dem was ich bei Ippen Digital täglich nutze, also Python mit
FastAPI und LangChain in Produktion"). Wenn der Match nicht stark ist, lass es.

Länge insgesamt 200-280 Wörter. Vier bis fünf kurze Absätze, optisch atmend.
Keine fixen Vorgaben für Inhalt je Absatz, aber jeder Absatz braucht eine
eigene Funktion. Doppelte Aussagen oder Wiederholungen streichen.

═══ STIL-REGELN (kompromisslos) ═══

S1. KEINE GEDANKENSTRICHE. Verboten sind die Zeichen — und – als Satzzeichen.
    Auch nicht als Klammer-Ersatz, auch nicht zur Betonung, auch nicht zwischen
    Satzteilen. Ein einziger Gedankenstrich im Brief ist ein Totalausfall.
    Erlaubt bleibt nur der Bindestrich (-) in zusammengesetzten Wörtern wie
    AI-Engineer, Multi-Agent oder Quality-Gates.

    Falsch  "Mein Stack ist Python, FastAPI, LangChain — alles produktiv."
    Richtig "Mein Stack ist Python, FastAPI und LangChain. Alles produktiv."

S2. KEINE DOPPELPUNKTE im Brief-Body. Keine Aufzählungs-Doppelpunkte, keine
    Einleitungs-Doppelpunkte, keine "Mein Stack:"-Konstruktionen. Doppelpunkte
    gehören in Spec-Dokumente, nicht in Anschreiben. Einzige Ausnahme im
    gesamten Output ist die BETREFF-Zeile, weil das ein technisches Prefix ist.

    Falsch  "Mein Stack: Python, FastAPI, LangChain."
    Richtig "Zu meinem Stack gehören Python, FastAPI und LangChain."

S3. KEINE SEMIKOLONS. Nirgendwo. Punkt oder Komma stattdessen.

S4. KEINE Bullet Points. Reiner Fließtext.

S5. KEINE Superlative, Enthusiasmus-Signale oder Marketing-Sprache.
    Wörter wie "leidenschaftlich", "begeistert mich zutiefst", "spannend",
    "innovativ", "zukunftsweisend" sind verboten. Nüchtern und sachlich.

S6. KEINE Floskeln. "Know-how einbringen", "nächste Generation mitgestalten",
    "robuste Lösungen", "agiles Umfeld" und ähnliches haben im Brief nichts
    verloren.

S7. KEINE Selbstbewertung. "Ich bin der perfekte Kandidat", "100% Match",
    "ideale Besetzung" sind verboten. Stattdessen Beobachtungen mit Beleg.

S8. KEINE Stack-Aufzählungen mit mehr als vier Komponenten in einem Satz.
    Lieber zwei oder drei passende Elemente nennen als alles.

S9. KEINE Wiederholungen. Jeder Satz bringt neue Information.

S10. KEINE Sternchen-Gendern. Neutrale Begriffe oder ausgeschriebene Doppelform.

S11. ANREDE-FORM. {anrede_form} Innerhalb des Briefs muss die gewählte Form
    konsistent durchgehalten werden, kein Mix aus Du und Sie.

S12. KEINE Standort-, Remote- oder Zeitzonen-Kommentare wenn die Stelle das
    nicht explizit verlangt. Eine Bewerbung impliziert das.

═══ OUTPUT-FORMAT ═══

SPRACHE-KONSISTENZ. Anrede, Body und Grußformel müssen in derselben Sprache
sein. Body deutsch heißt Anrede deutsch und Grußformel deutsch. Body englisch
heißt Anrede englisch und Grußformel englisch. Der Firmenname ändert daran
nichts. Eine Firma mit englischem Namen (Bluefish AI, Anthropic) bekommt
trotzdem eine deutsche Anrede, wenn der Brief auf Deutsch ist.

Reihenfolge der Ausgabe.

Zeile 1   BETREFF: <Betreff>
Leerzeile
Anrede    Eine Zeile, danach Leerzeile.

   Wenn Body deutsch ist, wähle eine deutsche Variante.
   Formell (große oder konservative Firma)  "Sehr geehrte Damen und Herren,"
   Locker  (Startup oder Du-Ton in der Stelle)  "Hallo zusammen,"

   Wenn Body englisch ist, wähle eine englische Variante.
   Formell "Dear Hiring Team," oder "Dear [Company] Team,"
   Locker  "Hi there," oder "Hi [Company] team,"

Body      Mehrere Absätze, durch doppelten Zeilenumbruch getrennt. Kein Name.

Grußformel Letzter Absatz, nur die Formel, kein "Pavlos Musenidis".

   Wenn Body deutsch ist  "Mit freundlichen Grüßen" oder "Viele Grüße"
   Wenn Body englisch ist "Best regards," oder "Kind regards,"

═══ LETZTE PRÜFUNG VOR DEM ABSENDEN ═══

Vor der Ausgabe drei Dinge selbst prüfen.

1. Sprach-Konsistenz. Anrede, Body, Grußformel in derselben Sprache. Sonst
   neu schreiben. Der Firmenname spielt keine Rolle.

2. Verbotene Zeichen im Body.
     Gedankenstrich  —  oder  –   ersetzen durch Komma oder Punkt
     Doppelpunkt     :              ersetzen (außer BETREFF-Zeile)
     Semikolon       ;              ersetzen durch Komma oder Punkt

3. Faktentreue. Jedes konkrete Detail (Projekt, Zahl, Stack, Zertifikat)
   stammt entweder aus dem CV-Block oder aus dem Präferenz-Block. Keine
   Erfindungen. Wenn unsicher, weglassen.
{lang_reminder}"""


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

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


def call_anthropic(system_prompt: str, job: dict, api_key: str,
                   model: str = "claude-sonnet-4-5-20250929") -> tuple[str, str] | None:
    """Returns (subject, body_text) on success."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n✗ anthropic package fehlt. Run: pip install anthropic\n", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    user_msg = f"""STELLE

Titel        {job['title']}
Unternehmen  {job['company']}
Standort     {job['location']}

Stellenbeschreibung
{job['description'][:4000]}

Schreibe das Anschreiben jetzt."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text
    except Exception as exc:  # noqa: BLE001
        log.error("Anthropic API failed: %s", exc)
        return None
    return _parse_response(text, job["title"])


def call_anthropic_refine(
    system_prompt: str,
    subject: str,
    body: str,
    instruction: str,
    api_key: str,
    job: dict | None = None,
    model: str = "claude-sonnet-4-5-20250929",
) -> tuple[str, str] | None:
    """Refine an existing letter using the SAME system prompt.

    The shared system prompt keeps the style rules, fact rules and language
    consistency rules active during refinement, so the model cannot quietly
    reintroduce em-dashes or change language.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        log.error("anthropic package missing")
        return None

    job_block = ""
    if job:
        job_block = (
            "ZUGEHÖRIGE STELLE (Faktentreue gilt weiter)\n\n"
            f"Titel        {job.get('title', '')}\n"
            f"Unternehmen  {job.get('company', '')}\n"
            f"Standort     {job.get('location', '')}\n\n"
            f"Stellenbeschreibung\n{(job.get('description') or '')[:4000]}\n\n"
        )

    user_msg = (
        "Du überarbeitest ein bestehendes Anschreiben gemäß einer Anweisung. "
        "Alle Stil-, Sprach- und Faktentreue-Regeln aus dem System-Prompt "
        "gelten weiter. Insbesondere keine Gedankenstriche, keine Doppelpunkte "
        "im Body, keine Semikolons, gleiche Sprache in Anrede, Body und "
        "Grußformel.\n\n"
        f"{job_block}"
        f"BESTEHENDES ANSCHREIBEN\n\nBETREFF: {subject}\n\n{body}\n\n"
        f"ANWEISUNG\n{instruction}\n\n"
        "Gib das vollständige überarbeitete Anschreiben im üblichen Format "
        "zurück (BETREFF-Zeile, Anrede, Body-Absätze, Grußformel). KEINE "
        "Meta-Kommentare wie 'Hier ist die überarbeitete Version'."
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0.6,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text
    except Exception as exc:  # noqa: BLE001
        log.error("Anthropic refine failed: %s", exc)
        return None

    return _parse_response(text, subject or "Bewerbung")


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

    lang = args.lang or "de"
    print(f"  Sprache: {lang.upper()} (Du/Sie entscheidet das LLM aus dem Posting)")

    print(f"\n→ Generiere Anschreiben mit {args.model}...")
    system_prompt = build_system_prompt(lang=lang)
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
