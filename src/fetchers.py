"""
ATS Fetchers: Greenhouse, Lever, Personio.

Each fetcher returns a list of normalized Job dicts.
No LLM, no external dependencies beyond httpx + stdlib.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

# --- Normalized schema --------------------------------------------------------

@dataclass
class Job:
    """Unified job representation across ATS providers."""
    source: str                    # "greenhouse" | "lever" | "personio"
    company: str                   # the board_token / company slug
    company_display: str           # human-readable (may equal company)
    title: str
    location: str                  # free-form; we normalize later
    url: str
    description_text: str          # plain text, no HTML
    posted_at: str | None          # ISO8601 if known
    workplace_type: str | None     # "remote" | "hybrid" | "onsite" | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    departments: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)
    commitment: str | None = None  # "Full-time", etc.
    raw_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# --- HTML/text helpers --------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def strip_html(s: str | None) -> str:
    if not s:
        return ""
    # Remove script/style blocks entirely
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


# --- Fetcher: Greenhouse ------------------------------------------------------
# Public endpoint, no auth, no rate limit (per their own docs).
# https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true

async def fetch_greenhouse(
    client: httpx.AsyncClient,
    board_token: str,
    display_name: str | None = None,
) -> list[Job]:
    # Some companies (e.g. JetBrains) use the EU datacenter.
    # Try US first, fall back to EU on 404.
    endpoints = [
        f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true",
        f"https://boards-api.eu.greenhouse.io/v1/boards/{board_token}/jobs?content=true",
    ]
    data = None
    for url in endpoints:
        try:
            r = await client.get(url, timeout=20.0)
            if r.status_code == 404:
                continue  # try next endpoint
            r.raise_for_status()
            data = r.json()
            break
        except httpx.HTTPStatusError as e:
            log.debug("greenhouse: HTTP %s for %s (%s)", e.response.status_code, board_token, url)
        except Exception as e:
            log.debug("greenhouse fetch failed for %s: %s", board_token, e)

    if data is None:
        log.debug("greenhouse: no data for token '%s'", board_token)
        return []

    jobs: list[Job] = []
    for j in data.get("jobs", []):
        loc = (j.get("location") or {}).get("name") or ""
        # Greenhouse sometimes puts workplace type in metadata
        workplace = None
        for m in j.get("metadata") or []:
            name = (m.get("name") or "").lower()
            if "workplace" in name or "remote" in name:
                val = m.get("value")
                if isinstance(val, str):
                    workplace = val.lower()
        depts = [d.get("name", "") for d in j.get("departments", []) if d.get("name")]

        jobs.append(Job(
            source="greenhouse",
            company=board_token,
            company_display=display_name or board_token,
            title=j.get("title", "").strip(),
            location=loc.strip(),
            url=j.get("absolute_url", ""),
            description_text=strip_html(j.get("content")),
            posted_at=j.get("first_published") or j.get("updated_at"),
            workplace_type=workplace,
            salary_min=None,
            salary_max=None,
            salary_currency=None,
            departments=depts,
            teams=[],
            commitment=None,
            raw_id=str(j.get("id", "")),
        ))
    return jobs


# --- Fetcher: Lever -----------------------------------------------------------
# Public endpoint, no auth.
# https://api.lever.co/v0/postings/{clientname}?mode=json
# Supports query filters: team, department, location, commitment, level, skip, limit

async def fetch_lever(
    client: httpx.AsyncClient,
    board_token: str,
    display_name: str | None = None,
) -> list[Job]:
    url = f"https://api.lever.co/v0/postings/{board_token}?mode=json"
    try:
        r = await client.get(url, timeout=20.0)
        if r.status_code == 404:
            log.debug("lever: 404 for token '%s' — token wrong or company not on Lever", board_token)
            return []
        if r.status_code in (301, 302):
            log.warning("lever: redirect (%s) for token '%s' — check token", r.status_code, board_token)
            return []
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        log.warning("lever: HTTP %s for %s", e.response.status_code, board_token)
        return []
    except Exception as e:
        log.warning("lever fetch failed for %s: %s", board_token, e)
        return []

    jobs: list[Job] = []
    for p in data:
        cats = p.get("categories", {}) or {}
        loc = cats.get("location", "")
        commitment = cats.get("commitment")
        team = cats.get("team")
        dept = cats.get("department")
        workplace = cats.get("allLocations") and None
        workplace_raw = p.get("workplaceType")  # "on-site" | "remote" | "hybrid" | "unspecified"
        if workplace_raw and workplace_raw != "unspecified":
            workplace = workplace_raw

        # Build a plain-text description from lists + descriptionHtml
        parts: list[str] = []
        if p.get("descriptionPlain"):
            parts.append(p["descriptionPlain"])
        elif p.get("description"):
            parts.append(strip_html(p["description"]))
        for lst in p.get("lists", []) or []:
            parts.append(strip_html(lst.get("text", "")) + " " + strip_html(lst.get("content", "")))
        if p.get("additionalPlain"):
            parts.append(p["additionalPlain"])
        elif p.get("additional"):
            parts.append(strip_html(p["additional"]))
        desc_text = _WS_RE.sub(" ", " ".join(parts)).strip()

        # Salary (optional, rarely populated)
        salary = p.get("salaryRange") or {}
        salary_min = salary.get("min")
        salary_max = salary.get("max")
        salary_cur = salary.get("currency")

        # Timestamp: Lever uses millisecond epoch for createdAt
        ts = p.get("createdAt")
        posted_at = None
        if isinstance(ts, (int, float)):
            try:
                posted_at = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            except Exception:
                pass

        jobs.append(Job(
            source="lever",
            company=board_token,
            company_display=display_name or board_token,
            title=p.get("text", "").strip(),
            location=loc.strip() if isinstance(loc, str) else "",
            url=p.get("hostedUrl") or p.get("applyUrl") or "",
            description_text=desc_text,
            posted_at=posted_at,
            workplace_type=workplace,
            salary_min=int(salary_min) if isinstance(salary_min, (int, float, str)) and str(salary_min).replace(".","").isdigit() else None,
            salary_max=int(salary_max) if isinstance(salary_max, (int, float, str)) and str(salary_max).replace(".","").isdigit() else None,
            salary_currency=salary_cur,
            departments=[dept] if dept else [],
            teams=[team] if team else [],
            commitment=commitment,
            raw_id=p.get("id"),
        ))
    return jobs


# --- Fetcher: Personio --------------------------------------------------------
# Public XML feed per company. Some use .de, some .com.
# https://{company}.jobs.personio.de/xml?language=en

async def fetch_personio(
    client: httpx.AsyncClient,
    board_token: str,
    display_name: str | None = None,
    language: str = "en",
) -> list[Job]:
    # Try .de first, fall back to .com.
    # Never follow redirects: a missing subdomain redirects to personio.de,
    # and following that in bulk triggers 429 rate limits.
    candidates = [
        f"https://{board_token}.jobs.personio.de/xml?language={language}",
        f"https://{board_token}.jobs.personio.com/xml?language={language}",
    ]
    body = None
    used_url = None
    for url in candidates:
        try:
            r = await client.get(url, timeout=20.0, follow_redirects=False)
            if r.status_code == 200 and r.content:
                body = r.content
                used_url = url
                break
            if r.status_code in (301, 302):
                log.debug("personio: %s redirect for %s — trying next TLD", r.status_code, board_token)
                continue
            if r.status_code == 429:
                log.warning("personio: 429 rate-limit for %s", board_token)
        except Exception as e:
            log.debug("personio try %s failed: %s", url, e)
    if not body:
        log.debug("personio: no feed found for %s", board_token)
        return []

    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        log.warning("personio: XML parse failed for %s: %s", board_token, e)
        return []

    jobs: list[Job] = []
    # Personio XML format: <workzag-jobs><position>...</position></workzag-jobs>
    # Historically uses element names like id, subcompany, office, department,
    # recruitingCategory, name, employmentType, seniority, schedule, yearsOfExperience,
    # createdAt, jobDescriptions
    for pos in root.findall(".//position"):
        def txt(tag: str) -> str:
            el = pos.find(tag)
            return (el.text or "").strip() if el is not None and el.text else ""

        title = txt("name")
        office = txt("office")
        dept = txt("department")
        emp_type = txt("employmentType")
        schedule = txt("schedule")
        created = txt("createdAt")
        pid = txt("id")

        # Description: jobDescriptions/jobDescription/value
        desc_parts: list[str] = []
        for jd in pos.findall(".//jobDescription"):
            name_el = jd.find("name")
            val_el = jd.find("value")
            if name_el is not None and name_el.text:
                desc_parts.append(name_el.text.strip())
            if val_el is not None and val_el.text:
                desc_parts.append(strip_html(val_el.text))
        description = _WS_RE.sub(" ", " ".join(desc_parts)).strip()

        # Construct URL: Personio job page follows {board}.jobs.personio.{tld}/job/{id}
        base = used_url.split("/xml")[0]
        job_url = f"{base}/job/{pid}" if pid else base

        jobs.append(Job(
            source="personio",
            company=board_token,
            company_display=display_name or board_token,
            title=title,
            location=office,
            url=job_url,
            description_text=description,
            posted_at=created or None,
            workplace_type=None,  # Personio doesn't expose this reliably
            salary_min=None,
            salary_max=None,
            salary_currency=None,
            departments=[dept] if dept else [],
            teams=[],
            commitment=emp_type or schedule or None,
            raw_id=pid or None,
        ))
    return jobs


# --- Fetcher: Ashby -----------------------------------------------------------
# Public endpoint, no auth.
# https://api.ashbyhq.com/posting-api/job-board/{clientname}?includeCompensation=true

async def fetch_ashby(
    client: httpx.AsyncClient,
    board_token: str,
    display_name: str | None = None,
) -> list[Job]:
    url = (f"https://api.ashbyhq.com/posting-api/job-board/{board_token}"
           f"?includeCompensation=true")
    try:
        r = await client.get(url, timeout=20.0)
        if r.status_code == 404:
            log.debug("ashby: 404 for token '%s' — token wrong or company not on Ashby", board_token)
            return []
        if r.status_code in (301, 302):
            log.warning("ashby: redirect (%s) for token '%s' — check token", r.status_code, board_token)
            return []
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        log.warning("ashby: HTTP %s for %s", e.response.status_code, board_token)
        return []
    except Exception as e:
        log.warning("ashby fetch failed for %s: %s", board_token, e)
        return []

    jobs: list[Job] = []
    for p in data.get("jobs", []):
        if p.get("isListed") is False:
            continue
        workplace = (p.get("workplaceType") or "").lower() or None
        if workplace == "unspecified":
            workplace = None

        # Compensation: Ashby exposes a summary string and optionally tiers
        salary_min = salary_max = None
        salary_cur = None
        comp = p.get("compensation") or {}
        # Some Ashby responses nest compensation per-job under `compensationTiers`;
        # others only expose `scrapeableCompensationSalarySummary` like "$81K - $87K"
        summary = (comp.get("scrapeableCompensationSalarySummary")
                   or comp.get("compensationTierSummary") or "")
        if summary:
            # Best-effort parse: numbers with k/K, optional currency
            nums = re.findall(r"(\d+(?:[.,]\d+)?)\s*[kK]", summary)
            parsed = []
            for n in nums:
                try:
                    parsed.append(int(float(n.replace(",", ".")) * 1000))
                except ValueError:
                    pass
            if len(parsed) >= 2:
                salary_min, salary_max = min(parsed), max(parsed)
            elif len(parsed) == 1:
                salary_min = parsed[0]
            if "€" in summary or "EUR" in summary.upper():
                salary_cur = "EUR"
            elif "$" in summary or "USD" in summary.upper():
                salary_cur = "USD"

        loc = p.get("location") or ""
        if not loc and p.get("address"):
            addr = p["address"].get("postalAddress") or {}
            parts = [addr.get("addressLocality"), addr.get("addressCountry")]
            loc = ", ".join([x for x in parts if x])

        jobs.append(Job(
            source="ashby",
            company=board_token,
            company_display=display_name or board_token,
            title=(p.get("title") or "").strip(),
            location=loc,
            url=p.get("jobUrl") or p.get("applyUrl") or "",
            description_text=(p.get("descriptionPlain")
                              or strip_html(p.get("descriptionHtml", ""))),
            posted_at=p.get("publishedAt"),
            workplace_type=workplace,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_cur,
            departments=[p["department"]] if p.get("department") else [],
            teams=[p["team"]] if p.get("team") else [],
            commitment=p.get("employmentType"),
            raw_id=p.get("id"),
        ))
    return jobs


# --- Dispatcher ---------------------------------------------------------------

FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "personio": fetch_personio,
    "ashby": fetch_ashby,
}

async def fetch_company(client: httpx.AsyncClient, ats: str, token: str,
                        display: str | None = None) -> list[Job]:
    f = FETCHERS.get(ats)
    if not f:
        log.error("unknown ATS: %s", ats)
        return []
    return await f(client, token, display)


async def fetch_all(companies: list[dict], concurrency: int = 8) -> list[Job]:
    """
    companies: list of {"ats": "greenhouse"|"lever"|"personio", "token": "...", "name": "..."}
    """
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
    headers = {"User-Agent": "job-scout/0.1 (+personal-use)"}
    all_jobs: list[Job] = []

    async with httpx.AsyncClient(limits=limits, headers=headers,
                                 follow_redirects=True) as client:
        async def one(c: dict) -> list[Job]:
            async with sem:
                return await fetch_company(client, c["ats"], c["token"], c.get("name"))

        tasks = [one(c) for c in companies]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for c, res in zip(companies, results):
        if isinstance(res, Exception):
            log.warning("fetch error for %s/%s: %s", c.get("ats"), c.get("token"), res)
            continue
        all_jobs.extend(res)
    return all_jobs