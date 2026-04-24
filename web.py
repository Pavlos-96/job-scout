#!/usr/bin/env python3
"""
Job Scout Web UI.

Usage:
    python web.py            # startet auf http://localhost:8000
    python web.py --port 8080

Requires: pip install fastapi uvicorn jinja2 aiosqlite python-multipart
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
import sys
import tempfile
from pathlib import Path
from typing import Any

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

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.db import (
    finish_run,
    get_cover_letter,
    get_job,
    get_jobs,
    get_latest_run,
    get_stats,
    hide_job,
    init_db,
    mark_seen,
    save_cover_letter,
    save_notes,
    start_run,
    toggle_applied,
    upsert_jobs,
)

log = logging.getLogger("web")

# ---------------------------------------------------------------------------
# Active pipeline/discover state (simple in-memory; single-user local tool)
# ---------------------------------------------------------------------------

_state: dict[str, dict[str, Any]] = {
    "pipeline": {"status": "idle", "message": ""},
    "discover": {"status": "idle", "message": ""},
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    await init_db()
    log.info("Job Scout UI started — http://localhost:8000")
    yield


app = FastAPI(title="Job Scout", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def _time_ago(iso_str: str) -> str:
    """Jinja2 filter: ISO timestamp → 'vor X Tagen' string."""
    from datetime import datetime, timezone
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                minutes = delta.seconds // 60
                return "gerade eben" if minutes < 2 else f"vor {minutes} Min."
            return f"vor {hours} Std."
        if days == 1:
            return "gestern"
        if days < 7:
            return f"vor {days} Tagen"
        if days < 30:
            weeks = days // 7
            return f"vor {weeks} Woche{'n' if weeks > 1 else ''}"
        if days < 365:
            months = days // 30
            return f"vor {months} Monat{'en' if months > 1 else ''}"
        years = days // 365
        return f"vor {years} Jahr{'en' if years > 1 else ''}"
    except (ValueError, TypeError):
        return iso_str[:10] if iso_str else ""


templates.env.filters["time_ago"] = _time_ago


# ---------------------------------------------------------------------------
# Jobs list (main view)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def jobs_list(
    request: Request,
    rec: str = "",
    applied: str = "",
    seen: str = "",
):
    filter_rec = rec if rec in ("apply", "maybe", "skip") else None
    filter_applied = True if applied == "1" else (False if applied == "0" else None)
    filter_seen = True if seen == "1" else (False if seen == "0" else None)

    if filter_applied is None:
        # Default view: open jobs on top, applied section pinned to the bottom.
        open_jobs = await get_jobs(
            recommendation=filter_rec,
            applied=False,
            seen=filter_seen,
            hidden=False,
        )
        applied_jobs = await get_jobs(applied=True, hidden=False)
        jobs = open_jobs
    else:
        jobs = await get_jobs(
            recommendation=filter_rec,
            applied=filter_applied,
            seen=filter_seen,
            hidden=False,
        )
        applied_jobs = []

    stats = await get_stats()
    run = await get_latest_run()

    return templates.TemplateResponse(request, "jobs.html", {
        "jobs": jobs,
        "applied_jobs": applied_jobs,
        "stats": stats,
        "run": run,
        "filter_rec": rec,
        "filter_applied": applied,
        "filter_seen": seen,
        "pipeline": _state["pipeline"],
    })


# ---------------------------------------------------------------------------
# Manual job add
# ---------------------------------------------------------------------------

@app.get("/jobs/add-manual", response_class=HTMLResponse)
async def job_add_manual_page(request: Request):
    return templates.TemplateResponse(request, "job_add_manual.html", {})


@app.post("/jobs/add-manual")
async def job_add_manual(
    request: Request,
    company: str = Form(default=""),
    title: str = Form(default=""),
    location: str = Form(default=""),
    url: str = Form(default=""),
    description: str = Form(default=""),
):
    from datetime import timezone
    from src.fetchers import Job
    from src.filters import MatchedJob
    from src.scorer import score_job
    from src.db import upsert_jobs, get_jobs
    from openai import AsyncOpenAI

    if not company.strip() or not title.strip() or not description.strip():
        return templates.TemplateResponse(request, "job_add_manual.html", {
            "error": "Unternehmen, Titel und Stellenbeschreibung sind Pflichtfelder.",
        })

    job_url = url.strip() or (
        f"manual://{re.sub(r'[^a-z0-9]', '-', company.lower().strip()[:30])}"
        f"/{re.sub(r'[^a-z0-9]', '-', title.lower().strip()[:40])}"
        f"/{int(datetime.now(tz=timezone.utc).timestamp())}"
    )

    job = Job(
        source="manual",
        company=company.strip(),
        company_display=company.strip(),
        title=title.strip(),
        location=location.strip() or "Nicht angegeben",
        url=job_url,
        description_text=description.strip(),
        posted_at=None,
        workplace_type=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
    )

    matched = MatchedJob(job=job, reasons=["manuell hinzugefügt"], score_hints=[])

    scored = None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            import asyncio
            client = AsyncOpenAI(api_key=api_key)
            scored = await score_job(
                client,
                title=job.title,
                company=job.company_display,
                location=job.location,
                description=job.description_text,
                salary_signals=[],
            )
        except Exception as exc:
            log.warning("score_job failed for manual job: %s", exc)

    await upsert_jobs([(matched, scored)])

    # Find the newly inserted job's ID to redirect to its detail page
    all_jobs = await get_jobs()
    job_id = next((j["id"] for j in all_jobs if j["job_url"] == job_url), None)
    if job_id:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    from fastapi.responses import RedirectResponse
    return RedirectResponse("/", status_code=303)


# Job detail
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: int):
    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)
    await mark_seen(job_id)
    cover = await get_cover_letter(job["job_url"])
    return templates.TemplateResponse(request, "job_detail.html", {
        "job": job,
        "cover": cover,
    })


@app.post("/jobs/{job_id}/toggle-applied", response_class=HTMLResponse)
async def job_toggle_applied(job_id: int):
    new_val = await toggle_applied(job_id)
    label = "Beworben" if new_val else "Nicht beworben"
    css = "badge-applied" if new_val else "badge-not-applied"
    return HTMLResponse(
        f'<button class="badge {css}" '
        f'hx-post="/jobs/{job_id}/toggle-applied" '
        f'hx-swap="outerHTML">{label}</button>'
    )


@app.post("/jobs/{job_id}/hide", response_class=HTMLResponse)
async def job_hide(job_id: int):
    await hide_job(job_id)
    return HTMLResponse("")


@app.post("/jobs/{job_id}/notes", response_class=HTMLResponse)
async def job_save_notes(job_id: int, notes: str = Form(default="")):
    await save_notes(job_id, notes)
    return HTMLResponse('<span class="save-confirm">Gespeichert</span>')


# ---------------------------------------------------------------------------
# Pipeline (run.py)
# ---------------------------------------------------------------------------

async def _run_pipeline(score: bool):
    _state["pipeline"] = {"status": "running", "message": "Stellen werden geladen..."}
    run_id = await start_run()

    try:
        from src.fetchers import fetch_all
        from src.filters import FilterConfig, filter_jobs
        from src.companies import COMPANIES

        _state["pipeline"]["message"] = f"Fetche Jobs von {len(COMPANIES)} Unternehmen..."
        jobs = await fetch_all(COMPANIES, concurrency=8)
        _state["pipeline"]["message"] = f"{len(jobs)} Jobs gefetcht, filtere..."

        cfg = FilterConfig(
            require_senior_title=False,
            require_germany_or_remote=True,
            prefer_munich=True,
            min_salary=90000,
            exclude_junior=True,
            require_posted_in_current_year=True,
            exclude_unknown_post_date=False,
        )
        matches = filter_jobs(jobs, cfg)
        _state["pipeline"]["message"] = f"{len(matches)} Matches, score mit LLM..."

        scored_pairs = None
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if score and api_key:
            from src.scorer import score_all
            from src.cache import ScoreCache
            cache = ScoreCache(ROOT / "cache" / "scores.json")
            scored_pairs = await score_all(
                matches, api_key=api_key, model="gpt-4o-mini",
                concurrency=5, cache=cache,
            )
            cache.save()

        pairs_for_db = scored_pairs if scored_pairs else [(m, None) for m in matches]
        new_count = await upsert_jobs(pairs_for_db)

        await finish_run(
            run_id,
            jobs_fetched=len(jobs),
            jobs_matched=len(matches),
            jobs_new=new_count,
        )
        _state["pipeline"] = {
            "status": "done",
            "message": (
                f"{len(jobs)} gefetcht, {len(matches)} Matches, "
                f"{new_count} neu in DB"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed")
        await finish_run(run_id, 0, 0, 0, error=str(exc))
        _state["pipeline"] = {"status": "error", "message": str(exc)}


@app.post("/run/start", response_class=HTMLResponse)
async def run_start(background_tasks: BackgroundTasks, score: str = Form(default="0")):
    if _state["pipeline"].get("status") == "running":
        return HTMLResponse('<span class="run-status running">Bereits am Laufen...</span>')
    background_tasks.add_task(_run_pipeline, score == "1")
    return HTMLResponse(
        '<span class="run-status running" '
        'hx-get="/run/status" hx-trigger="every 2s" hx-swap="outerHTML">'
        'Gestartet...</span>'
    )


@app.get("/run/status", response_class=HTMLResponse)
async def run_status():
    s = _state["pipeline"]
    status = s.get("status", "idle")
    msg = s.get("message", "")

    if status == "running":
        return HTMLResponse(
            f'<span class="run-status running" '
            f'hx-get="/run/status" hx-trigger="every 2s" hx-swap="outerHTML">'
            f'{msg}</span>'
        )
    if status == "done":
        return HTMLResponse(
            f'<span class="run-status done">{msg} '
            f'<a href="/">Aktualisieren</a></span>'
        )
    if status == "error":
        return HTMLResponse(
            f'<span class="run-status error">Fehler: {msg}</span>'
        )
    return HTMLResponse('<span class="run-status idle"></span>')


# ---------------------------------------------------------------------------
# Cover letter
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/cover-letter", response_class=HTMLResponse)
async def cover_letter_page(request: Request, job_id: int):
    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)
    cover = await get_cover_letter(job["job_url"])
    return templates.TemplateResponse(request, "cover_letter.html", {
        "job": job,
        "cover": cover,
    })


@app.post("/jobs/{job_id}/cover-letter/generate", response_class=HTMLResponse)
async def cover_letter_generate(
    request: Request,
    job_id: int,
    lang: str = Form(default="de"),
    form: str = Form(default="du"),
):
    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return HTMLResponse(
            '<p class="error">ANTHROPIC_API_KEY fehlt in .env</p>'
        )

    try:
        import importlib
        import write_cover_letter as _cl
        importlib.reload(_cl)
        build_system_prompt = _cl.build_system_prompt
        call_anthropic = _cl.call_anthropic

        job_data = {
            "title": job["title"],
            "company": job["company_display"],
            "location": job["location"],
            "description": job["description_text"] or "",
            "url": job["job_url"],
        }
        log.info("cover_letter_generate: lang=%r form=%r", lang, form)
        system_prompt = build_system_prompt(form, lang)
        log.info("prompt_start: %r", system_prompt[:120])
        result = call_anthropic(system_prompt, job_data, api_key)

        if not result:
            return HTMLResponse('<p class="error">LLM-Aufruf fehlgeschlagen.</p>')

        subject, body = result
        await save_cover_letter(job["job_url"], subject, body)

        return templates.TemplateResponse(request, "_cover_letter_form.html", {
            "job": job,
            "cover": {"subject": subject, "content": body},
        })
    except Exception as exc:  # noqa: BLE001
        log.exception("Cover letter generation failed")
        return HTMLResponse(f'<p class="error">Fehler: {exc}</p>')


@app.post("/jobs/{job_id}/cover-letter/save", response_class=HTMLResponse)
async def cover_letter_save(
    job_id: int,
    subject: str = Form(default=""),
    content: str = Form(default=""),
):
    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)
    await save_cover_letter(job["job_url"], subject, content)
    return HTMLResponse('<span class="save-confirm">Gespeichert</span>')


@app.get("/jobs/{job_id}/cover-letter/pdf")
async def cover_letter_pdf(job_id: int):
    import shutil
    import subprocess

    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)

    cover = await get_cover_letter(job["job_url"])
    if not cover or not cover.get("content"):
        return HTMLResponse(
            "<p>Kein Anschreiben vorhanden. Bitte zuerst generieren.</p>",
            status_code=400,
        )

    try:
        from write_cover_letter import render_html
        subject = cover.get("subject", "")
        body = cover.get("content", "")
        company = job["company_display"]
        html_content = render_html(subject, body, company=company)
    except (OSError, KeyError, ValueError) as exc:
        return HTMLResponse(f"<p>HTML-Rendering fehlgeschlagen: {exc}</p>", status_code=500)

    company_slug = re.sub(r"\s+", "_", company.strip().lower())[:30]
    filename = f"pavlos_musenidis_anschreiben_{company_slug}.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "cover.html"
        pdf_path = Path(tmpdir) / "cover.pdf"
        html_path.write_text(html_content, encoding="utf-8")

        if not shutil.which("wkhtmltopdf"):
            # wkhtmltopdf not installed — serve HTML for browser print-to-PDF
            return Response(
                content=html_content.encode(),
                media_type="text/html",
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )

        try:
            subprocess.run(
                ["wkhtmltopdf", "--enable-local-file-access",
                 "--print-media-type", "--quiet",
                 str(html_path), str(pdf_path)],
                check=True,
            )
            pdf_bytes = pdf_path.read_bytes()
        except subprocess.CalledProcessError as exc:
            return HTMLResponse(f"<p>PDF-Erstellung fehlgeschlagen: {exc}</p>", status_code=500)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Cover letter refinement (shared helper + two endpoints)
# ---------------------------------------------------------------------------

def _call_anthropic_refine(
    subject: str,
    content: str,
    instruction: str,
    api_key: str,
) -> tuple[str, str] | None:
    """
    Refine an existing cover letter via a user instruction.
    Returns (subject, body) — only the text, no meta-commentary.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    system = (
        "Du überarbeitest ein Anschreiben nach einer spezifischen Anweisung.\n\n"
        "ABSOLUT WICHTIG:\n"
        "- Gib NUR den überarbeiteten Text zurück.\n"
        "- KEINE Einleitung wie 'Hier ist...' oder 'Ich habe...'.\n"
        "- KEINE Erklärungen was du geändert hast.\n"
        "- KEINE Kommentare vor oder nach dem Text.\n"
        "- KEIN Markdown.\n\n"
        "Format der Ausgabe (exakt so):\n"
        "BETREFF: <Betreff>\n\n"
        "<Body ohne Anrede, ohne Grußformel, ohne Unterschrift>"
    )
    user = (
        f"Aktuelles Anschreiben:\n"
        f"BETREFF: {subject}\n\n"
        f"{content}\n\n"
        f"Anweisung: {instruction}"
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001
        log.error("Anthropic refine failed: %s", exc)
        return None

    lines = text.split("\n")
    new_subject = subject
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("BETREFF:"):
            new_subject = line[len("BETREFF:"):].strip()
            body_start = i + 1
            break

    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    body = "\n".join(lines[body_start:]).strip()
    return new_subject, body


@app.post("/jobs/{job_id}/cover-letter/refine", response_class=HTMLResponse)
async def cover_letter_refine_job(
    request: Request,
    job_id: int,
    subject: str = Form(default=""),
    content: str = Form(default=""),
    instruction: str = Form(default=""),
):
    if not instruction.strip():
        return HTMLResponse('<p class="error">Bitte eine Überarbeitungs-Anweisung eingeben.</p>')

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return HTMLResponse('<p class="error">ANTHROPIC_API_KEY fehlt in .env</p>')

    job = await get_job(job_id)
    if not job:
        return HTMLResponse("<p>Stelle nicht gefunden.</p>", status_code=404)

    result = _call_anthropic_refine(subject, content, instruction, api_key)
    if not result:
        return HTMLResponse('<p class="error">Überarbeitung fehlgeschlagen.</p>')

    new_subject, new_body = result
    await save_cover_letter(job["job_url"], new_subject, new_body)

    return templates.TemplateResponse(request, "_cover_letter_form.html", {
        "job": job,
        "cover": {"subject": new_subject, "content": new_body},
    })


@app.post("/cover-letter/refine", response_class=HTMLResponse)
async def cover_letter_refine_standalone(
    request: Request,
    subject: str = Form(default=""),
    content: str = Form(default=""),
    instruction: str = Form(default=""),
    company: str = Form(default=""),
):
    if not instruction.strip():
        return HTMLResponse('<p class="error">Bitte eine Überarbeitungs-Anweisung eingeben.</p>')

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return HTMLResponse('<p class="error">ANTHROPIC_API_KEY fehlt in .env</p>')

    result = _call_anthropic_refine(subject, content, instruction, api_key)
    if not result:
        return HTMLResponse('<p class="error">Überarbeitung fehlgeschlagen.</p>')

    new_subject, new_body = result
    return templates.TemplateResponse(request, "_cover_letter_standalone_result.html", {
        "subject": new_subject,
        "body": new_body,
        "company": company,
    })


# ---------------------------------------------------------------------------
# Standalone cover letter (no DB job required)
# ---------------------------------------------------------------------------

@app.get("/cover-letter", response_class=HTMLResponse)
async def standalone_cover_letter_page(request: Request):
    return templates.TemplateResponse(request, "cover_letter_standalone.html", {})


@app.post("/cover-letter/generate", response_class=HTMLResponse)
async def standalone_cover_letter_generate(
    request: Request,
    company: str = Form(default=""),
    title: str = Form(default=""),
    location: str = Form(default=""),
    description: str = Form(default=""),
    lang: str = Form(default="de"),
    form: str = Form(default="du"),
):
    if not description.strip():
        return HTMLResponse('<p class="error">Bitte Stellenbeschreibung einfügen.</p>')

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return HTMLResponse('<p class="error">ANTHROPIC_API_KEY fehlt in .env</p>')

    try:
        import importlib
        import write_cover_letter as _cl
        importlib.reload(_cl)
        build_system_prompt = _cl.build_system_prompt
        call_anthropic = _cl.call_anthropic

        job_data = {
            "title": title or "Stelle",
            "company": company or "Unternehmen",
            "location": location or "",
            "description": description,
            "url": "",
        }
        system_prompt = build_system_prompt(form, lang)
        result = call_anthropic(system_prompt, job_data, api_key)

        if not result:
            return HTMLResponse('<p class="error">LLM-Aufruf fehlgeschlagen.</p>')

        subject, body = result
        return templates.TemplateResponse(request, "_cover_letter_standalone_result.html", {
            "subject": subject,
            "body": body,
            "company": company,
        })
    except Exception as exc:  # noqa: BLE001
        log.exception("Standalone cover letter failed")
        return HTMLResponse(f'<p class="error">Fehler: {exc}</p>')


@app.post("/cover-letter/pdf-download")
async def standalone_cover_letter_pdf(
    subject: str = Form(default=""),
    content: str = Form(default=""),
    company: str = Form(default="anschreiben"),
):
    import shutil
    import subprocess

    if not content.strip():
        return HTMLResponse("<p>Kein Inhalt.</p>", status_code=400)

    try:
        from write_cover_letter import render_html
        html_content = render_html(subject, content, company=company)
    except (OSError, KeyError, ValueError) as exc:
        return HTMLResponse(f"<p>Rendering fehlgeschlagen: {exc}</p>", status_code=500)

    company_slug = re.sub(r"\s+", "_", company.strip().lower())[:30] or "anschreiben"
    filename = f"pavlos_musenidis_anschreiben_{company_slug}.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "cover.html"
        pdf_path = Path(tmpdir) / "cover.pdf"
        html_path.write_text(html_content, encoding="utf-8")

        if not shutil.which("wkhtmltopdf"):
            return Response(
                content=html_content.encode(),
                media_type="text/html",
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )

        try:
            subprocess.run(
                ["wkhtmltopdf", "--enable-local-file-access",
                 "--print-media-type", "--quiet",
                 str(html_path), str(pdf_path)],
                check=True,
            )
            pdf_bytes = pdf_path.read_bytes()
        except subprocess.CalledProcessError as exc:
            return HTMLResponse(f"<p>PDF-Erstellung fehlgeschlagen: {exc}</p>", status_code=500)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

@app.get("/companies", response_class=HTMLResponse)
async def companies_page(request: Request):
    from src.companies import COMPANIES
    return templates.TemplateResponse(request, "companies.html", {
        "companies": COMPANIES,
    })


@app.post("/companies/add", response_class=HTMLResponse)
async def companies_add(names: str = Form(default="")):
    name_list = [n.strip() for n in names.splitlines() if n.strip()]
    if not name_list:
        return HTMLResponse('<p class="error">Keine Namen angegeben.</p>')

    try:
        import re
        from discover import discover_all, append_to_companies, load_existing_names

        existing_names = load_existing_names()

        def _slugify(s: str) -> str:
            slug = re.sub(r"[^\w\s-]", "", s.lower())
            return re.sub(r"[\s_]+", "-", slug).strip("-")

        unknown = [
            n for n in name_list
            if n.lower().strip() not in existing_names
            and _slugify(n) not in existing_names
        ]
        already_known = [n for n in name_list if n not in unknown]

        if not unknown:
            known_str = ", ".join(already_known)
            return HTMLResponse(
                f'<p class="info">Alle bereits bekannt: {known_str}</p>'
            )

        found, not_found = await discover_all(unknown, concurrency=4)
        added = append_to_companies(found)

        lines = []
        if already_known:
            lines.append(f"Bereits bekannt: {', '.join(already_known)}")
        if found:
            names_found = [e["name"] for e in found]
            lines.append(f"Gefunden und eingetragen ({added}): {', '.join(names_found)}")
        if not_found:
            lines.append(f"Nicht gefunden: {', '.join(not_found)}")

        return HTMLResponse(
            "<br>".join(f'<p class="info">{line}</p>' for line in lines)
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("companies_add failed")
        return HTMLResponse(f'<p class="error">Fehler: {exc}</p>')


# ---------------------------------------------------------------------------
# Serper Discovery
# ---------------------------------------------------------------------------

async def _run_discovery():
    _state["discover"] = {"status": "running", "message": "Suche läuft..."}
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "serper_discover.py"), "--update"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        output = result.stdout.strip() or result.stderr.strip() or "Fertig."
        _state["discover"] = {"status": "done", "message": output}
    except OSError as exc:
        _state["discover"] = {"status": "error", "message": str(exc)}


@app.get("/discover", response_class=HTMLResponse)
async def discover_page(request: Request):
    return templates.TemplateResponse(request, "discover.html", {
        "discover": _state["discover"],
    })


@app.post("/discover/start", response_class=HTMLResponse)
async def discover_start(background_tasks: BackgroundTasks):
    if _state["discover"].get("status") == "running":
        return HTMLResponse('<span class="run-status running">Bereits am Laufen...</span>')
    background_tasks.add_task(_run_discovery)
    return HTMLResponse(
        '<span class="run-status running" '
        'hx-get="/discover/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        'Suche gestartet...</span>'
    )


@app.get("/discover/status", response_class=HTMLResponse)
async def discover_status():
    s = _state["discover"]
    status = s.get("status", "idle")
    msg = s.get("message", "").replace("\n", "<br>")

    if status == "running":
        return HTMLResponse(
            f'<span class="run-status running" '
            f'hx-get="/discover/status" hx-trigger="every 3s" hx-swap="outerHTML">'
            f'{msg}</span>'
        )
    if status == "done":
        return HTMLResponse(
            f'<div class="run-status done"><pre>{msg}</pre>'
            f'<a href="/companies">Unternehmen ansehen</a></div>'
        )
    if status == "error":
        return HTMLResponse(f'<span class="run-status error">Fehler: {msg}</span>')
    return HTMLResponse('<span class="run-status idle"></span>')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="Job Scout Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    uvicorn.run(
        "web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
