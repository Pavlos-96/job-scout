"""
Database module for Job Scout.

Single SQLite file (jobs.db) stores all job data, LLM scores,
cover letters, and pipeline run history.

Usage:
    from src.db import init_db, upsert_jobs, get_jobs, get_job

    await init_db()
    await upsert_jobs(scored_pairs)
    jobs = await get_jobs(recommendation="apply")
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "jobs.db"

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url          TEXT    UNIQUE NOT NULL,
    source           TEXT,
    company          TEXT,
    company_display  TEXT,
    title            TEXT,
    location         TEXT,
    workplace_type   TEXT,
    posted_at        TEXT,
    salary_min       INTEGER,
    salary_max       INTEGER,
    salary_currency  TEXT,
    departments      TEXT,
    description_text TEXT,
    -- filter metadata (JSON strings)
    reasons          TEXT,
    location_flags   TEXT,
    salary_signals   TEXT,
    salary_ok        INTEGER,
    score_hints      TEXT,
    -- LLM score
    llm_score        INTEGER,
    llm_recommendation TEXT,
    llm_summary      TEXT,
    llm_years_required TEXT,
    llm_python_required INTEGER,
    llm_salary_assessment TEXT,
    llm_strengths    TEXT,
    llm_concerns     TEXT,
    -- user state
    applied          INTEGER DEFAULT 0,
    seen             INTEGER DEFAULT 0,
    hidden           INTEGER DEFAULT 0,
    notes            TEXT    DEFAULT '',
    -- timestamps
    first_seen_at    TEXT    NOT NULL,
    last_seen_at     TEXT    NOT NULL
)
"""

_CREATE_COVER_LETTERS = """
CREATE TABLE IF NOT EXISTS cover_letters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url    TEXT    NOT NULL,
    subject    TEXT    NOT NULL DEFAULT '',
    content    TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
)
"""

_CREATE_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT    NOT NULL,
    finished_at  TEXT,
    status       TEXT    NOT NULL DEFAULT 'running',
    jobs_fetched INTEGER,
    jobs_matched INTEGER,
    jobs_new     INTEGER,
    error        TEXT
)
"""


async def _migrate(db: aiosqlite.Connection) -> None:
    """Apply incremental schema migrations (idempotent)."""
    migrations = [
        "ALTER TABLE jobs ADD COLUMN hidden INTEGER DEFAULT 0",
    ]
    for sql in migrations:
        try:
            await db.execute(sql)
        except Exception:
            pass  # column already exists


async def init_db(path: Path = DB_PATH) -> None:
    """Create tables if they don't exist and run migrations."""
    async with aiosqlite.connect(path) as db:
        await db.execute(_CREATE_JOBS)
        await db.execute(_CREATE_COVER_LETTERS)
        await db.execute(_CREATE_PIPELINE_RUNS)
        await _migrate(db)
        await db.commit()
    log.debug("DB initialised: %s", path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def upsert_jobs(
    scored_pairs: list[tuple[Any, Any | None]],
    path: Path = DB_PATH,
) -> int:
    """
    Upsert a list of (MatchedJob, ScoredJob | None) pairs into the DB.
    Returns the number of newly inserted rows.
    """
    now = _now()
    new_count = 0

    async with aiosqlite.connect(path) as db:
        for matched, scored in scored_pairs:
            j = matched.job
            url = j.url or ""
            if not url:
                continue

            # Check if already exists
            async with db.execute(
                "SELECT id FROM jobs WHERE job_url = ?", (url,)
            ) as cur:
                existing = await cur.fetchone()

            base = {
                "job_url": url,
                "source": j.source,
                "company": j.company,
                "company_display": j.company_display,
                "title": j.title,
                "location": j.location,
                "workplace_type": j.workplace_type,
                "posted_at": j.posted_at,
                "salary_min": j.salary_min,
                "salary_max": j.salary_max,
                "salary_currency": j.salary_currency,
                "departments": json.dumps(j.departments or []),
                "description_text": j.description_text,
                "reasons": json.dumps(matched.reasons),
                "location_flags": json.dumps(matched.location_flags),
                "salary_signals": json.dumps(
                    [[a, b] for a, b in (matched.salary_signals or [])]
                ),
                "salary_ok": (
                    1 if matched.salary_ok is True
                    else 0 if matched.salary_ok is False
                    else None
                ),
                "score_hints": json.dumps(matched.score_hints),
                "last_seen_at": now,
            }

            if scored:
                base.update({
                    "llm_score": scored.score,
                    "llm_recommendation": scored.recommendation,
                    "llm_summary": scored.summary,
                    "llm_years_required": scored.years_required,
                    "llm_python_required": 1 if scored.python_required else 0,
                    "llm_salary_assessment": scored.salary_assessment,
                    "llm_strengths": json.dumps(scored.strengths),
                    "llm_concerns": json.dumps(scored.concerns),
                })

            if existing:
                # Update job data + score, but preserve user state (applied, seen, notes)
                set_clause = ", ".join(
                    f"{k} = :{k}" for k in base if k != "job_url"
                )
                await db.execute(
                    f"UPDATE jobs SET {set_clause} WHERE job_url = :job_url",
                    base,
                )
            else:
                base["first_seen_at"] = now
                cols = ", ".join(base.keys())
                placeholders = ", ".join(f":{k}" for k in base)
                await db.execute(
                    f"INSERT INTO jobs ({cols}) VALUES ({placeholders})",
                    base,
                )
                new_count += 1

        await db.commit()

    log.info("upsert_jobs: %d new, %d total", new_count, len(scored_pairs))
    return new_count


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: aiosqlite.Row) -> dict:
    d = dict(row)
    for field in ("departments", "reasons", "location_flags",
                  "salary_signals", "score_hints", "llm_strengths", "llm_concerns"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (ValueError, TypeError):
                d[field] = []
    return d


async def get_jobs(
    recommendation: str | None = None,
    applied: bool | None = None,
    seen: bool | None = None,
    hidden: bool | None = False,
    filter_current_year: bool = True,
    limit: int = 500,
    path: Path = DB_PATH,
) -> list[dict]:
    """Return jobs ordered by llm_score DESC, first_seen_at DESC.

    hidden=False (default): exclude hidden jobs.
    hidden=True: only hidden jobs.
    hidden=None: no filter on hidden column.

    filter_current_year=True (default): only show jobs posted in the
    current calendar year. Manual jobs (source='manual') and jobs with
    no posted_at date are always included regardless.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if recommendation:
        conditions.append("llm_recommendation = ?")
        params.append(recommendation)
    if applied is not None:
        conditions.append("applied = ?")
        params.append(1 if applied else 0)
    if seen is not None:
        conditions.append("seen = ?")
        params.append(1 if seen else 0)
    if hidden is not None:
        conditions.append("hidden = ?")
        params.append(1 if hidden else 0)
    if filter_current_year:
        # Keep manual jobs and undated jobs; drop jobs older than 365 days.
        # date(posted_at) extracts YYYY-MM-DD from ISO8601; works with timezone
        # suffixes like +00:00 because SQLite reads the leading date portion.
        conditions.append(
            "(source = 'manual' OR posted_at IS NULL OR posted_at = '' "
            "OR date(posted_at) >= date('now', '-365 days'))"
        )

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT * FROM jobs {where} "
        "ORDER BY llm_score DESC NULLS LAST, first_seen_at DESC "
        "LIMIT ?"
    )
    params.append(limit)

    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def get_job(job_id: int, path: Path = DB_PATH) -> dict | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def mark_seen(job_id: int, path: Path = DB_PATH) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute("UPDATE jobs SET seen = 1 WHERE id = ?", (job_id,))
        await db.commit()


async def hide_job(job_id: int, path: Path = DB_PATH) -> None:
    """Soft-delete: sets hidden=1. Job stays in DB but won't show in main list."""
    async with aiosqlite.connect(path) as db:
        await db.execute("UPDATE jobs SET hidden = 1 WHERE id = ?", (job_id,))
        await db.commit()


async def toggle_applied(job_id: int, path: Path = DB_PATH) -> bool:
    """Toggle applied flag. Returns new value."""
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT applied FROM jobs WHERE id = ?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        new_val = 0 if row[0] else 1
        await db.execute(
            "UPDATE jobs SET applied = ? WHERE id = ?", (new_val, job_id)
        )
        await db.commit()
    return bool(new_val)


async def save_notes(job_id: int, notes: str, path: Path = DB_PATH) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE jobs SET notes = ? WHERE id = ?", (notes, job_id)
        )
        await db.commit()


async def get_stats(path: Path = DB_PATH) -> dict:
    """Return summary counts for the stats banner."""
    async with aiosqlite.connect(path) as db:
        async def count(sql: str, params: tuple = ()) -> int:
            async with db.execute(sql, params) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

        total = await count("SELECT COUNT(*) FROM jobs")
        new = await count("SELECT COUNT(*) FROM jobs WHERE seen = 0")
        to_apply = await count(
            "SELECT COUNT(*) FROM jobs WHERE llm_recommendation = 'apply' AND applied = 0"
        )
        applied = await count("SELECT COUNT(*) FROM jobs WHERE applied = 1")

    return {"total": total, "new": new, "to_apply": to_apply, "applied": applied}


# ---------------------------------------------------------------------------
# Cover letters
# ---------------------------------------------------------------------------

async def get_cover_letter(job_url: str, path: Path = DB_PATH) -> dict | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM cover_letters WHERE job_url = ? ORDER BY updated_at DESC LIMIT 1",
            (job_url,),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def save_cover_letter(
    job_url: str,
    subject: str,
    content: str,
    path: Path = DB_PATH,
) -> None:
    now = _now()
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT id FROM cover_letters WHERE job_url = ?", (job_url,)
        ) as cur:
            existing = await cur.fetchone()
        if existing:
            await db.execute(
                "UPDATE cover_letters SET subject = ?, content = ?, updated_at = ? "
                "WHERE job_url = ?",
                (subject, content, now, job_url),
            )
        else:
            await db.execute(
                "INSERT INTO cover_letters (job_url, subject, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_url, subject, content, now, now),
            )
        await db.commit()


# ---------------------------------------------------------------------------
# Pipeline run tracking
# ---------------------------------------------------------------------------

async def start_run(path: Path = DB_PATH) -> int:
    """Insert a new pipeline_runs row and return its id."""
    async with aiosqlite.connect(path) as db:
        cur = await db.execute(
            "INSERT INTO pipeline_runs (started_at, status) VALUES (?, 'running')",
            (_now(),),
        )
        run_id = cur.lastrowid
        await db.commit()
    return run_id


async def finish_run(
    run_id: int,
    jobs_fetched: int,
    jobs_matched: int,
    jobs_new: int,
    error: str | None = None,
    path: Path = DB_PATH,
) -> None:
    status = "error" if error else "done"
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE pipeline_runs SET finished_at=?, status=?, "
            "jobs_fetched=?, jobs_matched=?, jobs_new=?, error=? WHERE id=?",
            (_now(), status, jobs_fetched, jobs_matched, jobs_new, error, run_id),
        )
        await db.commit()


async def get_latest_run(path: Path = DB_PATH) -> dict | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None
