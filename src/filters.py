"""
Filters and matchers.

Deterministic, no LLM. Takes a list of Job and a FilterConfig,
returns the subset that matches plus some annotations.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

log = logging.getLogger(__name__)

from fetchers import Job


def posted_calendar_year(posted_at: str | None) -> int | None:
    """Calendar year from ATS posted/created date. None if missing."""
    if not posted_at:
        return None
    s = str(posted_at).strip()
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        if 1990 <= y <= 2100:
            return y
    return None


def is_too_old(posted_at: str | None, max_age_days: int = 365) -> bool:
    """
    Returns True if the posting date is older than max_age_days.
    Returns False for missing/unparseable dates (keep the job).
    """
    if not posted_at:
        return False
    date_str = str(posted_at).strip()[:10]
    if len(date_str) < 10:
        return False
    try:
        from datetime import date, timedelta as td
        post_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        cutoff = datetime.now(timezone.utc).date() - td(days=max_age_days)
        return post_date < cutoff
    except (ValueError, TypeError):
        return False


# --- Title matching -----------------------------------------------------------

# Regex for AI / GenAI engineering titles including common variants:
#   "AI Engineer", "Senior AI Engineer", "Generative AI Engineer",
#   "GenAI Engineer" (one word), "Applied AI Engineer",
#   "AI Software Engineer", "LLM Engineer"
# Optional level prefix: senior, staff, lead, principal (lead roles are
# still fetched; the LLM scorer marks most as skip).
# Reject ML-first titles, PM, sales, etc. via TITLE_EXCLUDE.

_TITLE_LEVEL = r"(?:sr\.?|senior|staff|lead|principal|\(senior\))"

TITLE_MATCH = re.compile(
    rf"""
    \b
    (?:
        # Standard: (level)? (generative|applied|gen )? AI Engineer
        (?:
            {_TITLE_LEVEL}\s+
        )?
        (?:generative\s+|applied\s+|gen\s+)?
        ai\s+engineer
        |
        # GenAI as one token: GenAI Engineer, Senior GenAI Engineer
        (?:
            {_TITLE_LEVEL}\s+
        )?
        genai\s+engineer
        |
        # AI Software Engineer (explicit software in title)
        (?:
            {_TITLE_LEVEL}\s+
        )?
        ai\s+software\s+engineer
        |
        # LLM-focused software engineer titles
        (?:
            {_TITLE_LEVEL}\s+
        )?
        llm\s+engineer
        |
        # Agentic / Gen AI phrasing with Engineer
        (?:
            {_TITLE_LEVEL}\s+
        )?
        (?:agentic\s+ai|gen\s+ai)\s+engineer
        |
        # NLP Engineer — directly relevant to CL background
        (?:
            {_TITLE_LEVEL}\s+
        )?
        nlp\s+engineer
        |
        # AI/ML Engineer where AI comes first (implies application focus,
        # not pure ML training) — TITLE_EXCLUDE still blocks ml-only titles
        (?:
            {_TITLE_LEVEL}\s+
        )?
        ai[\s/]+ml\s+engineer
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hard reject: ML/MLOps/Data roles, non-engineering AI titles, sales/HR.
TITLE_EXCLUDE = re.compile(
    r"""
    \b(
        machine\s+learning\s+engineer |
        (?<!ai[\s/])ml\s+engineer |
        ml[-\s]*ops\s+engineer |
        mlops\s+engineer |
        data\s+scientist |
        data\s+engineer |
        research\s+scientist |
        research\s+engineer |
        prompt\s+engineer |
        ai\s+product\s+manager |
        ai\s+researcher |
        ai\s+trainer |
        ai\s+content |
        head\s+of\s+ai |
        director\s+of\s+ai |
        vp\s+of\s+ai |
        chief\s+ai\s+officer |
        ai\s+consultant |
        technical\s+account |
        customer\s+success |
        business\s+development |
        account\s+executive |
        recruiter |
        talent\s+acquisition
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def title_matches(title: str) -> bool:
    if not title:
        return False
    if TITLE_EXCLUDE.search(title):
        return False
    return bool(TITLE_MATCH.search(title))


# --- Seniority detection ------------------------------------------------------

SENIOR_RE = re.compile(r"\b(senior|sr\.?|staff|lead|principal|\(senior\))\b", re.IGNORECASE)
JUNIOR_RE = re.compile(r"\b(junior|jr\.?|intern|working\s+student|werkstudent|praktikum|trainee|apprentice)\b", re.IGNORECASE)

def is_senior_title(title: str) -> bool:
    return bool(SENIOR_RE.search(title or ""))

def is_junior_title(title: str) -> bool:
    return bool(JUNIOR_RE.search(title or ""))


# --- Location matching --------------------------------------------------------

GERMANY_TOKENS = [
    "germany", "deutschland", "de-", " de,", "(de)",
    "berlin", "münchen", "munich", "stuttgart", "hamburg", "köln", "cologne",
    "frankfurt", "düsseldorf", "duesseldorf", "leipzig", "dresden", "nürnberg",
    "nuremberg", "karlsruhe", "mannheim", "bremen", "hannover", "dortmund",
    "essen", "bonn", "heidelberg", "freiburg", "aachen", "ulm", "augsburg",
    "bavaria", "bayern", "baden-württemberg", "nrw",
]
REMOTE_TOKENS = [
    "remote", "remote-first", "fully remote", "fully-remote",
    "100% remote", "distributed", "anywhere", "work from home", "wfh",
]
EMEA_TOKENS = ["emea", "europe", "eu", "european union"]

# Countries whose presence in a location string marks the job as NOT
# accessible from Germany.  Keep entries lower-case; matched via substring.
NON_EU_COUNTRIES = [
    "united states", "u.s.a", "usa",
    "canada",
    "india",
    "indonesia",
    "brazil", "brasil",
    "argentina",
    "colombia",
    "mexico",
    "singapore",
    "south korea", "korea",
    "japan",
    "china",
    "australia",
    "new zealand",
    "south america", "latin america", "latam",
    "apac",
    "asia",
]

# City-level non-EU signals (standalone city names without country).
# Kept short — only cities that rarely appear in EU job posts.
NON_EU_CITIES = [
    "buenos aires",
    "são paulo", "sao paulo",
    "bogotá", "bogota",
    "lima",
    "mexico city", "ciudad de mexico",
    "nairobi",
    "johannesburg",
    "lagos",
    "mumbai", "bangalore", "bengaluru", "hyderabad", "delhi", "pune",
    "toronto", "vancouver", "montreal",
    "sydney", "melbourne",
    "tokyo", "beijing", "shanghai", "shenzhen",
    "los altos", "menlo park", "palo alto", "mountain view", "sunnyvale",
    "new york", "san francisco", "seattle", "austin", "boston",
    "los angeles", "chicago", "denver", "atlanta",
    "new york city", "nyc",
]

# Catches "Remote - US", "Remote – India", etc.
NON_EU_SUFFIX_RE = re.compile(
    r"[-–]\s*(us|usa|united\s+states|india|canada|brazil|singapore|korea|"
    r"japan|china|australia|indonesia|colombia|argentina|mexico|new\s+zealand)\b",
    re.IGNORECASE,
)

# US-only work-authorization phrases in job description text.
# If found, the job is US-only regardless of the location field saying "Remote".
US_WORK_AUTH_RE = re.compile(
    r"authorized?\s+to\s+work\s+in\s+the\s+(?:us|united\s+states)|"
    r"eligible\s+to\s+work\s+in\s+the\s+(?:us|united\s+states)|"
    r"must\s+(?:be\s+)?(?:a\s+)?(?:us\s+citizen|authorized?\s+to\s+work\s+in)|"
    r"(?:candidates?|applicants?)\s+(?:must\s+be\s+)?(?:located|based|residing)\s+in\s+the\s+(?:us|united\s+states|usa)\b|"
    r"open\s+(?:only\s+)?to\s+(?:us|united\s+states)\s+(?:candidates?|residents?|citizens?)|"
    r"this\s+role\s+is\s+(?:only\s+)?(?:open|available)\s+(?:to\s+)?(?:us|us-based|united\s+states)|"
    r"right\s+to\s+work\s+in\s+the\s+(?:us|united\s+states)",
    re.IGNORECASE,
)


def _lower(s: str | None) -> str:
    return (s or "").lower()

def classify_location(job: Job) -> dict:
    """
    Returns a dict with all location flags needed for filtering.

    Accessibility rules:
    - Stuttgart: any work-mode (onsite / hybrid / remote) is OK.
    - Munich: must be at least hybrid (not pure onsite).
    - Any other German city: remote required.
    - Non-Germany: EU/EMEA remote or truly global remote required.
      A job tagged "Remote - US / India / ..." is NOT accessible.

    Strategy: trust the structured location field first; only fall back to
    the description when the location field is completely empty.
    """
    loc = _lower(job.location)
    wt = _lower(job.workplace_type)

    has_munich = "munich" in loc or "münchen" in loc or "munchen" in loc
    has_stuttgart = "stuttgart" in loc
    in_germany = any(t in loc for t in GERMANY_TOKENS)
    is_remote = "remote" in wt or any(t in loc for t in REMOTE_TOKENS)
    is_hybrid = "hybrid" in wt or "hybrid" in loc
    in_emea = any(t in loc for t in EMEA_TOKENS)

    has_non_eu = (
        any(t in loc for t in NON_EU_COUNTRIES)
        or any(t in loc for t in NON_EU_CITIES)
        or bool(NON_EU_SUFFIX_RE.search(loc))
    )

    # Only check description if the location field is completely empty.
    # Avoids false positives where "Germany, Remote" appears somewhere in
    # a multi-location listing whose actual primary location is New York.
    if not loc.strip():
        desc = _lower(job.description_text)[:600]
        if not in_germany and ("germany" in desc or "deutschland" in desc):
            in_germany = True
        if not is_remote and any(t in desc for t in REMOTE_TOKENS):
            is_remote = True
        if not has_munich and ("munich" in desc or "münchen" in desc):
            has_munich = True
        if not has_stuttgart and "stuttgart" in desc:
            has_stuttgart = True

    # Scan first 3000 chars of description for US work-authorization phrases.
    # These signal a US-only role regardless of the location field.
    if not has_non_eu and job.description_text:
        if US_WORK_AUTH_RE.search(job.description_text[:3000]):
            has_non_eu = True

    # A job is EU-accessible when:
    #   - It is in Germany, OR
    #   - It is in EMEA/Europe, OR
    #   - It is remote AND does not specify a clearly non-EU country.
    is_eu_accessible = (
        in_germany
        or in_emea
        or (is_remote and not has_non_eu)
    )

    return {
        "in_germany": in_germany,
        "is_remote": is_remote,
        "is_hybrid": is_hybrid,
        "in_emea": in_emea,
        "has_munich": has_munich,
        "has_stuttgart": has_stuttgart,
        "has_non_eu": has_non_eu,
        "is_eu_accessible": is_eu_accessible,
    }


# --- Exclusions (ANÜ / staffing / defense) ------------------------------------

EXCLUDE_COMPANY_KEYWORDS = re.compile(
    r"\b(arbeitnehmerüberlassung|arbeitnehmerueberlassung|zeitarbeit|"
    r"personaldienstleist|staffing\s+agency|recruiting\s+agency)\b",
    re.IGNORECASE,
)
EXCLUDE_DEFENSE = re.compile(
    r"\b(defense|defence|weapons?|munition|bundeswehr|military\s+contractor|"
    r"lockheed|raytheon|helsing|rheinmetall|hensoldt|diehl\s+defence|"
    r"airbus\s+defence|bae\s+systems|northrop)\b",
    re.IGNORECASE,
)

def is_excluded(job: Job) -> str | None:
    text = f"{job.company_display} {job.title} {job.description_text[:2000]}"
    if EXCLUDE_COMPANY_KEYWORDS.search(text):
        return "staffing/ANÜ"
    if EXCLUDE_DEFENSE.search(text):
        return "defense"
    return None


# --- Salary signal extraction (best-effort, no LLM) ---------------------------
#
# Matches patterns like:
#   "95.000 EUR", "95.000 €", "95,000 EUR", "95k EUR", "95 k€",
#   "85.000 - 115.000 EUR", "between 90,000 and 110,000",
#   "80.000€ to 100.000€", "up to 120k EUR", "ab 90.000 Euro".

_NUM = r"(\d{1,3}(?:[.,]\d{3})+|\d{2,3}(?:[.,]\d{1,2})?k?|\d{5,6})"
SALARY_PATTERNS = [
    re.compile(
        rf"{_NUM}\s*(?:-|–|bis|to|and|und)\s*{_NUM}\s*(?:€|eur(?:o)?|k€?)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:€|eur(?:o)?)\s*{_NUM}\s*(?:-|–|bis|to|and|und)\s*{_NUM}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:ab|from|starting\s+at|up\s+to|bis\s+zu)\s+{_NUM}\s*(?:€|eur(?:o)?|k€?)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_NUM}\s*(?:€|eur(?:o)?|k€?)\s*(?:\+|plus)?(?=\b|$)",
        re.IGNORECASE,
    ),
]

def _parse_num(n: str) -> int | None:
    """Parse '95.000', '95,000', '95k', '95' -> 95000."""
    if not n:
        return None
    s = n.strip().lower().replace(" ", "")
    if s.endswith("k"):
        s = s[:-1]
        try:
            v = float(s.replace(",", "."))
            return int(v * 1000)
        except ValueError:
            return None
    # Handle "95.000" (de) and "95,000" (us): both mean 95000
    s = s.replace(".", "").replace(",", "")
    if s.isdigit():
        v = int(s)
        # Heuristic: plausible annual salaries in EUR are 20k..500k
        if 20000 <= v <= 500000:
            return v
    return None

def extract_salary_signals(job: Job) -> list[tuple[int, int | None]]:
    """
    Returns a list of (min, max) tuples (max may be None for open-ended).
    Empty list if no plausible salary found.
    """
    if job.salary_min is not None:
        # Structured salary already present (Lever)
        return [(int(job.salary_min), int(job.salary_max) if job.salary_max else None)]

    text = job.description_text
    found: list[tuple[int, int | None]] = []
    for pat in SALARY_PATTERNS:
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g]
            nums = [_parse_num(g) for g in groups]
            nums = [n for n in nums if n is not None]
            if not nums:
                continue
            if len(nums) >= 2:
                found.append((min(nums), max(nums)))
            else:
                found.append((nums[0], None))
    return found


# --- Filter config + main filter entry point ---------------------------------

@dataclass
class FilterConfig:
    require_senior_title: bool = False
    allow_no_senior_level: bool = True  # match "AI Engineer" too
    require_germany_or_remote: bool = True
    prefer_munich: bool = True
    min_salary: int | None = 90000      # Only used to mark "salary_ok", not to drop
    exclude_junior: bool = True
    # Drop listings older than max_age_days (rolling window, not calendar year).
    # Jobs with no date are always kept. Set to 0 to disable.
    require_posted_in_current_year: bool = True  # kept for compat; uses max_age_days
    exclude_unknown_post_date: bool = False
    max_age_days: int = 365


@dataclass
class MatchedJob:
    job: Job
    reasons: list[str] = field(default_factory=list)
    location_flags: dict = field(default_factory=dict)
    salary_signals: list[tuple[int, int | None]] = field(default_factory=list)
    salary_ok: bool | None = None   # None = unknown, True = signal >= min, False = below
    score_hints: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "job": self.job.as_dict(),
            "reasons": self.reasons,
            "location_flags": self.location_flags,
            "salary_signals": [[a, b] for (a, b) in self.salary_signals],
            "salary_ok": self.salary_ok,
            "score_hints": self.score_hints,
        }


def filter_jobs(jobs: Iterable[Job], cfg: FilterConfig) -> list[MatchedJob]:
    matched: list[MatchedJob] = []
    skipped_post_year = 0

    for job in jobs:
        if not title_matches(job.title):
            continue
        if cfg.exclude_junior and is_junior_title(job.title):
            continue
        if cfg.require_senior_title and not is_senior_title(job.title):
            continue

        if cfg.require_posted_in_current_year and cfg.max_age_days > 0:
            if is_too_old(job.posted_at, cfg.max_age_days):
                skipped_post_year += 1
                continue
            if job.posted_at is None and cfg.exclude_unknown_post_date:
                skipped_post_year += 1
                continue

        excl = is_excluded(job)
        if excl:
            continue

        loc_flags = classify_location(job)

        # --- Location gate 1: EU accessibility --------------------------------
        # Drops jobs tied to non-EU countries (e.g. "Remote - US / India").
        if not loc_flags["is_eu_accessible"]:
            continue

        # --- Location gate 2: Germany-specific work-mode rules ----------------
        # Rules (in priority order):
        #   Stuttgart  → any work-mode OK (onsite / hybrid / remote)
        #   Munich     → hybrid OR remote required (pure onsite → skip)
        #   Elsewhere  → remote required
        if loc_flags["in_germany"] and not loc_flags["is_remote"]:
            if loc_flags["has_stuttgart"]:
                pass  # Stuttgart: onsite/hybrid/remote all fine
            elif loc_flags["has_munich"] and loc_flags["is_hybrid"]:
                pass  # Munich hybrid: acceptable
            else:
                continue  # Munich onsite or other German city non-remote: skip

        reasons: list[str] = []
        if is_senior_title(job.title):
            reasons.append("senior title")
        if loc_flags["has_munich"]:
            reasons.append("Munich")
        if loc_flags["has_stuttgart"]:
            reasons.append("Stuttgart")
        if loc_flags["is_remote"]:
            reasons.append("remote")
        if loc_flags["is_hybrid"]:
            reasons.append("hybrid")
        if loc_flags["in_germany"]:
            reasons.append("Germany")
        elif loc_flags["in_emea"]:
            reasons.append("EMEA-wide")

        sigs = extract_salary_signals(job)
        salary_ok: bool | None = None
        if sigs and cfg.min_salary:
            # Use the highest max (or min when no max) as the optimistic number
            best = max((b if b else a) for (a, b) in sigs)
            salary_ok = best >= cfg.min_salary

        hints = []
        if salary_ok is True:
            hints.append(f"salary signal >= {cfg.min_salary}")
        elif salary_ok is False:
            hints.append(f"salary signal < {cfg.min_salary}")

        matched.append(MatchedJob(
            job=job, reasons=reasons, location_flags=loc_flags,
            salary_signals=sigs, salary_ok=salary_ok, score_hints=hints,
        ))

    if skipped_post_year:
        log.info(
            "filter: excluded %d job(s) (not posted in calendar year %s)",
            skipped_post_year,
            current_year,
        )

    return matched


# --- Simple ranking -----------------------------------------------------------

def rank_score(m: MatchedJob) -> tuple[int, int, str]:
    """
    Higher = better. We return a tuple so stable-sort works nicely.
    Priority order matches Pavlos' stated preferences:
      1. Munich (hybrid/remote) or Stuttgart
      2. Germany remote
      3. Senior title
      4. Salary signal present and >= 90k
    """
    j = m.job
    score = 0
    if m.location_flags.get("has_munich"):
        score += 40
    if m.location_flags.get("has_stuttgart"):
        score += 35
    if m.location_flags.get("is_remote"):
        score += 30
    if m.location_flags.get("in_germany"):
        score += 15
    if is_senior_title(j.title):
        score += 20
    if m.salary_ok is True:
        score += 25
    elif m.salary_ok is False:
        score -= 15
    # Recency bonus via posted_at isn't reliable cross-source; skip for now.
    return (score, len(m.salary_signals), j.posted_at or "")
