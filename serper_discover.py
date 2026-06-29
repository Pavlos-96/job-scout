"""
serper_discover.py — Neue Firmen automatisch via Google Dorks entdecken.

Durchsucht Google nach Stellen auf Personio/Ashby/Greenhouse/Lever
und extrahiert daraus Firmen-Tokens, die dann direkt in discover.py
geprüft und in companies.py eingetragen werden können.

Benötigt: SERPER_API_KEY in .env (kostenlos: serper.dev → 2.500 Suchen/Monat)

Usage:
    python serper_discover.py               # alle Dorks, fügt neue Firmen hinzu
    python serper_discover.py --dry-run     # zeigt nur was gefunden würde
    python serper_discover.py --update      # automatisch companies.py aktualisieren
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("serper_discover")

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Google Dorks — each targets one ATS, extracts company tokens from URLs
# ---------------------------------------------------------------------------

# Job-title phrases aligned with src/filters.py TITLE_MATCH (Serper/Google).
_ROLE_OR = (
    '("AI Engineer" OR "Senior AI Engineer" OR "(Senior) AI Engineer" '
    'OR "GenAI Engineer" OR "Generative AI Engineer" OR "Applied AI Engineer" '
    'OR "LLM Engineer" OR "AI Software Engineer" OR "Agentic AI Engineer" '
    'OR "Gen AI Engineer")'
)
_LOC_DE_REMOTE = "(Deutschland OR Germany OR Remote OR EU OR Europe)"
_LOC_BROAD = (
    "(Germany OR Deutschland OR Remote OR Europe OR EU OR München OR Munich "
    'OR "Remote Europe")'
)

# Format: (query, ats_name, url_pattern_to_extract_token)
DORKS: list[tuple[str, str, str]] = [
    # Personio: firma.jobs.personio.de → token = firma
    (
        f"site:jobs.personio.de {_ROLE_OR} {_LOC_DE_REMOTE}",
        "personio",
        r"https?://([^.]+)\.jobs\.personio\.(?:de|com)",
    ),
    (
        f"site:jobs.personio.de {_ROLE_OR} {_LOC_BROAD}",
        "personio",
        r"https?://([^.]+)\.jobs\.personio\.(?:de|com)",
    ),
    # Ashby: jobs.ashbyhq.com/Firma → token = Firma
    (
        f"site:jobs.ashbyhq.com {_ROLE_OR} {_LOC_DE_REMOTE}",
        "ashby",
        r"https?://jobs\.ashbyhq\.com/([^/\s?]+)",
    ),
    (
        f"site:jobs.ashbyhq.com {_ROLE_OR} {_LOC_BROAD}",
        "ashby",
        r"https?://jobs\.ashbyhq\.com/([^/\s?]+)",
    ),
    # Greenhouse: boards.greenhouse.io/firma oder job-boards.greenhouse.io/firma
    (
        f"site:greenhouse.io {_ROLE_OR} {_LOC_DE_REMOTE}",
        "greenhouse",
        r"https?://(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io/([^/\s?]+)",
    ),
    (
        f"site:greenhouse.io {_ROLE_OR} {_LOC_BROAD}",
        "greenhouse",
        r"https?://(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io/([^/\s?]+)",
    ),
    # Lever: jobs.lever.co/firma
    (
        f"site:lever.co {_ROLE_OR} {_LOC_DE_REMOTE}",
        "lever",
        r"https?://jobs\.lever\.co/([^/\s?]+)",
    ),
    (
        f"site:lever.co {_ROLE_OR} {_LOC_BROAD}",
        "lever",
        r"https?://jobs\.lever\.co/([^/\s?]+)",
    ),
]

# Tokens to always skip (job IDs, not company names, or known noise)
SKIP_TOKENS = {
    "jobs", "job", "embed", "apply", "careers", "career",
    "search", "all", "de", "en", "api", "v1", "v0",
}

# ---------------------------------------------------------------------------
# Serper API call
# ---------------------------------------------------------------------------

_SERPER_MAX_NUM = 10  # Serper free tier: max 10 results per query


async def serper_search(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    num: int = 10,
) -> list[dict]:
    """Returns list of organic result dicts with 'link', 'title', 'snippet'."""
    try:
        r = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": min(num, _SERPER_MAX_NUM), "gl": "de", "hl": "de"},
            timeout=15.0,
        )
        if r.status_code != 200:
            log.warning(
                "serper %d for query %r: %s",
                r.status_code, query[:60], r.text[:200],
            )
            return []
        data = r.json()
        return data.get("organic", [])
    except Exception as e:
        log.warning("serper search failed for query %r: %s", query[:60], e)
        return []


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_token_from_url(url: str, pattern: str) -> str | None:
    m = re.search(pattern, url, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).strip("/")
    # Strip job IDs (long hex strings or pure numbers)
    if re.match(r"^[0-9a-f]{8,}$", token, re.IGNORECASE):
        return None
    if token.isdigit():
        return None
    if token.lower() in SKIP_TOKENS:
        return None
    return token


def extract_company_name_from_result(result: dict, token: str) -> str:
    """Best-effort: get a human-readable company name from the search result."""
    title = result.get("title", "")
    _snippet = result.get("snippet", "")

    # Greenhouse/Lever titles often look like "Jobs at CompanyName" or "CompanyName Careers"
    for pattern in [
        r"Jobs? at (.+?)(?:\s*[|\-–]|$)",
        r"^(.+?)\s+(?:Careers?|Jobs?)\s*[|\-–|$]",
        r"^(.+?)\s+is hiring",
    ]:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if 2 < len(name) < 60:
                return name

    # Ashby/Personio: "Career at X" or "X Stellenangebote"
    for pattern in [
        r"Karriere(?:\s+bei)?\s+(.+?)(?:\s*[|\-–]|$)",
        r"(.+?)\s+Stellenangebote",
        r"(.+?)\s+Job(?:s|\s+Angebote)",
    ]:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if 2 < len(name) < 60:
                return name

    # Fallback: capitalize the token
    return token.replace("-", " ").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Load existing tokens to avoid re-adding
# ---------------------------------------------------------------------------

def load_existing_tokens() -> set[tuple[str, str]]:
    """Returns set of (ats, token) already in companies.py."""
    companies_path = ROOT / "src" / "companies.py"
    if not companies_path.exists():
        return set()
    import ast
    src = companies_path.read_text(encoding="utf-8")
    tokens: set[tuple[str, str]] = set()
    for line in src.splitlines():
        stripped = line.strip().rstrip(",")
        if stripped.startswith('{"ats":') or stripped.startswith("{'ats':"):
            try:
                entry = ast.literal_eval(stripped)
                tokens.add((entry.get("ats", ""), entry.get("token", "")))
            except Exception:
                pass
    return tokens


# ---------------------------------------------------------------------------
# Live discovery — city × role targeted dorks for ALL 4 ATS
# ---------------------------------------------------------------------------
#
# Per-city (NOT paired) because Google delivers significantly better results
# when each query targets exactly one location — empirically validated for
# manual Google searches. Pairing cities (e.g. "Hamburg OR Berlin") halves
# the result quality.
#
# Budget awareness: Serper free tier = 2500 searches/month. Defaults are
# tuned so a typical run consumes ~100–150 searches; `--full` mode burns
# ~500+ for occasional comprehensive scans.

# Top German cities (focused mode picks the first N).
_DE_CITIES_FOCUSED = [
    "Berlin", "München", "Hamburg", "Köln",
    "Frankfurt", "Düsseldorf", "Leipzig",
]
_DE_CITIES_FULL = _DE_CITIES_FOCUSED + [
    "Munich", "Cologne", "Dresden", "Nürnberg", "Hannover", "Bremen",
    "Mannheim", "Heidelberg", "Karlsruhe", "Freiburg", "Bonn",
    "Dortmund", "Essen", "Aachen", "Ulm", "Augsburg", "Wiesbaden",
]
# Stuttgart is handled separately — no remote requirement (can commute).
_STUTTGART = "Stuttgart"

# Swiss cities (focused mode picks the first N).
_CH_CITIES_FOCUSED = ["Zürich", "Basel", "Bern"]
_CH_CITIES_FULL = _CH_CITIES_FOCUSED + [
    "Zurich", "Genf", "Geneva", "Lausanne", "Zug",
    "Luzern", "Lucerne", "Winterthur",
]

# Country-level remote — catches "Remote, Germany" without a city name.
_COUNTRY_REMOTE = ["Deutschland", "Germany", "Schweiz", "Switzerland"]

# Roles aligned with filters.py TITLE_MATCH.
_ROLES_FOCUSED = [
    '"AI Engineer"',
    '"GenAI Engineer"',
    '"LLM Engineer"',
    '"AI Software Engineer"',
    '"AI Developer"',
]
_ROLES_FULL = _ROLES_FOCUSED + [
    '"Senior AI Engineer"',
    '"Senior AI Software Engineer"',
    '"Generative AI Engineer"',
    '"Gen AI Engineer"',
    '"Senior LLM Engineer"',
    '"Applied AI Engineer"',
    '"Agentic AI Engineer"',
    '"ML Engineer"',
    '"Machine Learning Engineer"',
    '"NLP Engineer"',
    '"AI/ML Engineer"',
]

# Greenhouse has multiple subdomains; use just the root domain so Serper
# searches all subdomains (boards.greenhouse.io, job-boards.greenhouse.io,
# job-boards.eu.greenhouse.io) in a single query.
_GREENHOUSE_SITE = "site:greenhouse.io"

# (ats, site_filter, url_pattern)
_ATS_TARGETS: list[tuple[str, str, str]] = [
    ("personio",   "site:jobs.personio.de",
     r"https?://([^.]+)\.jobs\.personio\.(?:de|com)"),
    ("ashby",      "site:jobs.ashbyhq.com",
     r"https?://jobs\.ashbyhq\.com/([^/\s?]+)"),
    ("greenhouse",
     _GREENHOUSE_SITE,
     r"https?://(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io/([^/\s?]+)"),
    ("lever",      "site:jobs.lever.co",
     r"https?://jobs\.lever\.co/([^/\s?]+)"),
    ("recruitee",  "site:recruitee.com",
     r"https?://([\w-]+)\.recruitee\.com/"),
    ("smartrecruiters", "site:jobs.smartrecruiters.com",
     r"https?://jobs\.smartrecruiters\.com/([^/\s?]+)/"),
]


def _build_live_dorks(mode: str = "focused") -> list[tuple[str, str, str]]:
    """
    Generate per-city × role dorks. `mode` controls the API budget.

    focused (~120 queries):
      4 ATS × 5 roles × (1 Stuttgart + 7 DE + 3 CH + 2 country) = ~260
      → trimmed by dedup of country/CH variants
    full (~600 queries):
      4 ATS × 16 roles × (1 + 25 + 11 + 4) ≈ 2600 → use only with care
    """
    roles = _ROLES_FOCUSED if mode == "focused" else _ROLES_FULL
    de_cities = (
        _DE_CITIES_FOCUSED if mode == "focused" else _DE_CITIES_FULL
    )
    ch_cities = (
        _CH_CITIES_FOCUSED if mode == "focused" else _CH_CITIES_FULL
    )
    countries = _COUNTRY_REMOTE if mode == "full" else ["Deutschland", "Schweiz"]

    dorks: list[tuple[str, str, str]] = []
    remote_or = '(remote OR "work from home")'

    for ats, site_filter, pattern in _ATS_TARGETS:
        for role in roles:
            dorks.append((
                f'{site_filter} {role} "{_STUTTGART}"',
                ats, pattern,
            ))
            for city in de_cities:
                dorks.append((
                    f'{site_filter} {role} "{city}" {remote_or}',
                    ats, pattern,
                ))
            for city in ch_cities:
                dorks.append((
                    f'{site_filter} {role} "{city}" {remote_or}',
                    ats, pattern,
                ))
            for country in countries:
                dorks.append((
                    f'{site_filter} {role} "{country}" {remote_or}',
                    ats, pattern,
                ))
    return dorks


def google_query_template(
    role_titles: list[str] | None = None,
    location: str = "München OR Munich",
    require_remote: bool = True,
    include_join_com: bool = True,
) -> str:
    """
    Produce a copy-paste-ready Google search query in Pavlos' preferred style.

    Example:
        (site:personio.de OR site:careers.join.com OR site:jobs.lever.co
         OR site:jobs.ashbyhq.com OR site:greenhouse.io)
        ("AI Engineer" OR "ML Engineer" OR "GenAI Engineer")
        (remote OR "work from home")
        (München OR Munich)
    """
    roles = role_titles or [
        "AI Engineer", "ML Engineer", "GenAI Engineer",
        "AI Software Engineer", "AI Developer", "LLM Engineer",
    ]
    sites = [
        "site:jobs.personio.de",
        "site:jobs.lever.co",
        "site:jobs.ashbyhq.com",
        "site:greenhouse.io",
    ]
    if include_join_com:
        sites.insert(1, "site:careers.join.com")
    site_part = "(" + " OR ".join(sites) + ")"
    role_part = "(" + " OR ".join(f'"{r}"' for r in roles) + ")"
    parts = [site_part, role_part]
    if require_remote:
        parts.append('(remote OR "work from home")')
    if location:
        parts.append(f"({location})")
    return " ".join(parts)


# Backwards-compat alias so old code paths keep working while we add new ones.
async def discover_personio_live(
    api_key: str,
    save_to_companies: bool = False,
) -> list[dict]:
    """Backwards-compat wrapper that now discovers across ALL ATS."""
    return await discover_live(
        api_key,
        save_to_companies=save_to_companies,
        ats_filter=None,
    )


# ---------------------------------------------------------------------------
# Direct job-URL discovery (handles JOIN.com, Workday, Recruitee tenants
# we don't have in companies.py yet, plus standalone career pages)
# ---------------------------------------------------------------------------

# Patterns for extracting a stable job-URL signature from Serper results.
# These are LAST-RESORT importers: when we find a job URL but no ATS we
# already fetch, we store the job directly in the DB with source="serper".
_DIRECT_JOB_PATTERNS: list[tuple[str, str]] = [
    # JOIN.com: careers.join.com/companies/COMPANY/jobs/JOB-SLUG
    ("join", r"https?://careers\.join\.com/companies/[^/\s?]+/jobs/[^\s?]+"),
    # Workday public boards (subdomain varies by company: tenant.wdN.myworkdayjobs.com)
    ("workday", r"https?://[\w-]+\.wd\d+\.myworkdayjobs\.com/[^\s?]+"),
    # SmartRecruiters direct postings
    ("smartrecruiters", r"https?://jobs\.smartrecruiters\.com/[^/\s?]+/\d+[^\s?]*"),
    # Recruitee direct postings
    ("recruitee", r"https?://[\w-]+\.recruitee\.com/o/[^\s?]+"),
    # Wellfound (formerly AngelList) job postings
    ("wellfound", r"https?://wellfound\.com/jobs/\d+[^\s?]*"),
]


def serper_results_to_jobs(results: list[dict]) -> list:
    """
    Convert raw Serper-direct-discovery results into Job dataclass instances
    that can be passed through the same filter / scorer / DB pipeline as
    ATS-fetched jobs.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from fetchers import Job
    jobs = []
    for r in results:
        snippet = r.get("description", "") or ""
        ats = r.get("ats_origin", "unknown")
        # The snippet is short, so be generous and include the query metadata.
        desc = (
            f"{snippet}\n\n"
            f"[Discovered via Serper query: {r.get('query', '')}]\n"
            f"[ATS detected: {ats}]"
        )
        jobs.append(Job(
            source=f"serper-{ats}",
            company=(r.get("company") or "").lower().replace(" ", "-") or "unknown",
            company_display=r.get("company") or "Unknown Company",
            title=r.get("title", "").strip() or "Untitled",
            location=r.get("location", ""),
            url=r.get("url", ""),
            description_text=desc,
            posted_at=None,
            workplace_type="remote" if "remote" in (r.get("location", "") + snippet).lower() else None,
            salary_min=None,
            salary_max=None,
            salary_currency=None,
        ))
    return jobs


async def discover_direct_jobs(
    api_key: str,
    queries: list[str] | None = None,
    max_queries: int | None = None,
    results_per_query: int = 10,
) -> list[dict]:
    """
    Run per-city × role dorks via Serper and return a list of dict
    suitable for `MatchedJob` insertion — each entry IS a job (not a
    company). Catches all ATS types we don't have a dedicated fetcher
    for (JOIN.com, Workday tenants, etc.).

    Returns dicts with keys: title, company, location, url, source,
    description (snippet only — limited!), posted_at.
    """
    if queries is None:
        # Use the per-city dorks WITHOUT site filter so we catch any ATS.
        # We then pattern-match the URL to detect what ATS each result is.
        queries = _build_universal_dorks()
    if max_queries and max_queries > 0:
        queries = queries[:max_queries]

    found: dict[str, dict] = {}  # url -> entry

    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(2)

        async def one(query: str):
            async with sem:
                await asyncio.sleep(0.5)
                results = await serper_search(
                    client, query, api_key, num=results_per_query
                )
                for r in results:
                    url = r.get("link", "")
                    if not url or url in found:
                        continue
                    ats_hit = None
                    for ats_name, pattern in _DIRECT_JOB_PATTERNS:
                        if re.search(pattern, url):
                            ats_hit = ats_name
                            break
                    if not ats_hit:
                        continue
                    title = r.get("title", "").strip()
                    snippet = r.get("snippet", "").strip()
                    found[url] = {
                        "title": _clean_title(title),
                        "company": _infer_company(url, title, ats_hit),
                        "location": _infer_location(snippet),
                        "url": url,
                        "source": f"serper-{ats_hit}",
                        "description": snippet,
                        "ats_origin": ats_hit,
                        "query": query[:80],
                    }

        await asyncio.gather(*[one(q) for q in queries])

    log.info(
        "discover_direct_jobs: %d queries → %d unique job URLs",
        len(queries), len(found),
    )
    return list(found.values())


def _build_universal_dorks() -> list[str]:
    """
    Per-city × role dorks targeting ATS platforms not covered by companies.py:
    Workday (inurl) and JOIN.com (site:). These return direct job URLs that are
    imported as individual jobs, not as company tokens.

    ~60 queries total (cheap budget).
    """
    roles = ['"AI Engineer"', '"GenAI Engineer"', '"LLM Engineer"',
             '"AI Software Engineer"', '"AI Developer"',
             '"Machine Learning Engineer"', '"ML Engineer"']

    # Locations: Stuttgart needs no remote filter; all others require remote/hybrid.
    de_locs = [
        '"Berlin"', '"München"', '"Hamburg"', '"Frankfurt"',
        '"Köln"', '"Düsseldorf"', '"Deutschland"',
    ]
    ch_locs = ['"Zürich"', '"Basel"', '"Schweiz"']
    remote_suffix = '(remote OR hybrid OR "work from home")'

    # ATS site filters for platforms we can't enumerate as tokens.
    # Workday has no consistent site: filter (subdomains vary by company), so
    # we use inurl: which still returns direct job-posting URLs.
    ats_prefixes = [
        'site:careers.join.com',
        'inurl:myworkdayjobs.com',
    ]

    queries = []
    for ats in ats_prefixes:
        for role in roles:
            # Stuttgart: no remote required (can commute)
            queries.append(f'{ats} {role} "Stuttgart"')
            for loc in de_locs:
                queries.append(f'{ats} {role} {loc} {remote_suffix}')
            for loc in ch_locs:
                queries.append(f'{ats} {role} {loc} {remote_suffix}')
    return queries


_TITLE_TAIL_RE = re.compile(
    r"\s+[-|·–]\s+(?:Join|Apply|Bewerben|Careers?|Jobs?|Stellenangebote?).*$",
    re.IGNORECASE,
)
_TITLE_AT_COMPANY_RE = re.compile(r"\s+(?:at|bei|@)\s+([^|\-–]+)", re.IGNORECASE)


def _clean_title(title: str) -> str:
    """Strip noisy suffixes from Google search result titles."""
    out = _TITLE_TAIL_RE.sub("", title)
    return out.strip(" -|·–")


def _infer_company(url: str, title: str, ats: str) -> str:
    """Best-effort: pull a human-readable company name from URL or title."""
    # First try: title patterns like "Title at Company" or "Title | Company"
    m = _TITLE_AT_COMPANY_RE.search(title)
    if m:
        cand = m.group(1).strip(" |·–-")
        if 2 < len(cand) < 60:
            return cand
    # Pipe-separated: "Title | Company"
    if "|" in title:
        parts = [p.strip() for p in title.split("|") if p.strip()]
        if len(parts) >= 2:
            for cand in parts[1:]:
                if 2 < len(cand) < 60 and not cand.lower().startswith(("apply", "join")):
                    return cand
    # URL-based fallbacks per ATS
    if ats == "join":
        m = re.search(r"join\.com/companies/([^/]+)", url)
        if m:
            return m.group(1).replace("-", " ").title()
    if ats == "workday":
        m = re.search(r"https?://([\w-]+)\.wd\d+\.myworkdayjobs\.com", url)
        if m:
            return m.group(1).replace("-", " ").title()
    if ats == "recruitee":
        m = re.search(r"https?://([\w-]+)\.recruitee\.com", url)
        if m:
            return m.group(1).replace("-", " ").title()
    if ats == "smartrecruiters":
        m = re.search(r"smartrecruiters\.com/([^/]+)/", url)
        if m:
            return m.group(1).replace("-", " ").title()
    return ""


_LOC_HINT_RE = re.compile(
    r"\b(Stuttgart|München|Munich|Berlin|Hamburg|Frankfurt|Köln|Cologne|"
    r"Düsseldorf|Leipzig|Dresden|Nürnberg|Hannover|Bremen|Mannheim|"
    r"Heidelberg|Karlsruhe|Freiburg|Zürich|Zurich|Basel|Bern|Geneva|"
    r"Genf|Lausanne|Zug|Remote|Hybrid|Germany|Deutschland|Switzerland|"
    r"Schweiz)\b",
    re.IGNORECASE,
)


def _infer_location(snippet: str) -> str:
    """Pull a location string out of the search snippet, best effort."""
    hits = _LOC_HINT_RE.findall(snippet)
    if not hits:
        return ""
    # Deduplicate preserving order
    seen: list[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return ", ".join(seen[:3])


async def discover_live(
    api_key: str,
    save_to_companies: bool = False,
    ats_filter: str | None = None,
    mode: str = "focused",
    max_queries: int | None = None,
    results_per_query: int = 10,
) -> list[dict]:
    """
    Run per-city × role dorks via Serper for ALL supported ATS and return
    new company dicts (same schema as COMPANIES in src/companies.py).

    Args:
        ats_filter: if set, only run dorks for this ATS (e.g. "personio").
        mode: "focused" (~250 queries, default) or "full" (~2600 queries).
        max_queries: hard cap on total Serper calls (cost control).
        results_per_query: how many search results to consider per dork.
    """
    live_dorks = _build_live_dorks(mode=mode)
    if ats_filter:
        live_dorks = [d for d in live_dorks if d[1] == ats_filter]
    if max_queries and max_queries > 0:
        live_dorks = live_dorks[:max_queries]
    log.info(
        "live-discover: %d queries (mode=%s, ats=%s)",
        len(live_dorks), mode, ats_filter or "all",
    )
    existing = load_existing_tokens()
    found: dict[tuple[str, str], dict] = {}

    # Serper free tier burst limit: keep concurrency low and add a per-request
    # delay to avoid 400 errors from exceeding the API's rate window.
    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(2)

        async def one(query: str, ats: str, pattern: str):
            async with sem:
                await asyncio.sleep(0.5)
                results = await serper_search(
                    client, query, api_key, num=results_per_query
                )
                for result in results:
                    url = result.get("link", "")
                    token = extract_token_from_url(url, pattern)
                    if not token:
                        continue
                    key = (ats, token)
                    if key in existing or key in found:
                        continue
                    name = extract_company_name_from_result(result, token)
                    found[key] = {
                        "ats": ats,
                        "token": token,
                        "name": name,
                        "tier": "scaleup",
                        "verified": False,
                        "notes": f"live-discovered: {query[:80]}",
                    }
                    log.info(
                        "live-discover: new %s company [%s] %s",
                        ats, token, name,
                    )

        await asyncio.gather(*[one(q, a, p) for q, a, p in live_dorks])

    companies = list(found.values())
    log.info(
        "live-discover: %d queries → %d new companies",
        len(live_dorks), len(companies),
    )

    if save_to_companies and companies:
        try:
            added = append_to_companies(companies)
            log.info("live-discover: saved %d new entries to companies.py", added)
        except Exception as e:
            log.warning("live-discover: could not save to companies.py: %s", e)

    return companies


# ---------------------------------------------------------------------------
# Main discovery logic
# ---------------------------------------------------------------------------

async def run_dorks(api_key: str) -> list[dict]:
    """
    Runs all dorks, extracts unique (ats, token) pairs not already in companies.py.
    Returns list of candidate dicts ready for discover.py verification.
    """
    existing = load_existing_tokens()
    found: dict[tuple[str, str], dict] = {}  # (ats, token) -> entry

    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(2)

        async def one_dork(query: str, ats: str, pattern: str):
            async with sem:
                await asyncio.sleep(0.5)
                results = await serper_search(client, query, api_key, num=10)
                for result in results:
                    url = result.get("link", "")
                    token = extract_token_from_url(url, pattern)
                    if not token:
                        continue
                    key = (ats, token)
                    if key in existing:
                        log.debug("already known: %s / %s", ats, token)
                        continue
                    if key not in found:
                        name = extract_company_name_from_result(result, token)
                        found[key] = {
                            "ats": ats,
                            "token": token,
                            "name": name,
                            "tier": "scaleup",
                            "verified": False,
                            "notes": f"serper-discovered via: {query[:60]}",
                            "_source_url": url,
                        }
                        log.info("new candidate: [%s] %s  (%s)", ats, token, name)

        await asyncio.gather(*[one_dork(q, a, p) for q, a, p in DORKS])

    return list(found.values())


# ---------------------------------------------------------------------------
# Verify candidates via ATS APIs (reuse discover.py logic)
# ---------------------------------------------------------------------------

async def verify_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Calls the actual ATS API for each candidate to confirm the token works
    and get the real job count.
    Returns (verified, unverified).
    """
    sem = asyncio.Semaphore(8)
    limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
    headers = {"User-Agent": "job-scout-discover/0.1 (+personal-use)"}

    verified: list[dict] = []
    unverified: list[dict] = []

    async with httpx.AsyncClient(limits=limits, headers=headers,
                                 follow_redirects=True) as client:
        async def one(c: dict):
            # For serper-found candidates we already know the ATS and token,
            # so we just probe that specific combination directly.
            from discover import PROBERS
            prober = PROBERS.get(c["ats"])
            if not prober:
                return c, None
            async with sem:
                n = await prober(client, c["token"])
            return c, n

        results = await asyncio.gather(*[one(c) for c in candidates])

    for c, n in results:
        if n is not None:
            verified.append({**c, "job_count": n, "verified": True,
                             "notes": f"serper+api verified, {n} open jobs"})
        else:
            unverified.append(c)

    return verified, unverified


# ---------------------------------------------------------------------------
# Append to companies.py
# ---------------------------------------------------------------------------

def append_to_companies(verified: list[dict]) -> int:
    """Appends verified entries to src/companies.py. Returns count added."""
    from discover import append_to_companies as _append
    return _append(verified)


# ---------------------------------------------------------------------------
# Targeted job search — Serper → company tokens → full jobs fetched now
#
# This is the primary Serper usage path during a pipeline run.
# Unlike discover_live (which saves tokens to companies.py for future runs),
# this function fetches full jobs IMMEDIATELY and returns them for the
# current run's filter/score pipeline.
#
# Workflow:
#   1. Run ~35 well-crafted dorks with negative filters (-junior, -intern …)
#   2. Extract company tokens from result URLs
#   3. Skip tokens already in companies.py (main pipeline fetches those)
#   4. Fetch ALL jobs from newly found companies via ATS API
#   5. Return Job objects → go directly into filter → LLM scorer
#
# Cost: ~35 Serper credits per call (stays well within 2,500/month free tier
# even if run daily).
# ---------------------------------------------------------------------------

def _build_job_search_dorks() -> list[tuple[str, str, str]]:
    """
    ~35 high-signal dorks for direct job discovery.
    Returns list of (query, ats_name, token_pattern).

    Key differences from live-discover dorks:
    - Negative keyword filters to cut noise at Google level
    - Fewer but broader location groups (fewer credits used per run)
    - Personio queried WITHOUT location (it's a DE-native ATS)
    """
    neg = '-junior -intern -Werkstudent -Praktikum'
    de_ch = (
        '(Germany OR Deutschland OR München OR Munich OR Berlin OR Stuttgart '
        'OR Zürich OR Zurich OR Basel OR remote OR "work from home")'
    )
    eu_broad = '(Europe OR Germany OR Zürich OR remote OR "work from home")'

    roles_core = ['"AI Engineer"', '"LLM Engineer"', '"GenAI Engineer"']
    roles_extra = ['"AI Software Engineer"', '"Machine Learning Engineer"',
                   '"NLP Engineer"', '"AI Developer"']

    # (site_filter, ats_name, token_pattern)
    intl_targets: list[tuple[str, str, str]] = [
        ("site:jobs.ashbyhq.com", "ashby",
         r"https?://jobs\.ashbyhq\.com/([^/\s?]+)"),
        (_GREENHOUSE_SITE, "greenhouse",
         r"https?://(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io/([^/\s?]+)"),
        ("site:jobs.lever.co", "lever",
         r"https?://jobs\.lever\.co/([^/\s?]+)"),
        ("site:recruitee.com", "recruitee",
         r"https?://([\w-]+)\.recruitee\.com/"),
        ("site:jobs.smartrecruiters.com", "smartrecruiters",
         r"https?://jobs\.smartrecruiters\.com/([^/\s?]+)/"),
    ]

    dorks: list[tuple[str, str, str]] = []

    # --- Personio: German-native ATS, no location filter needed ---------------
    personio_pat = r"https?://([^.]+)\.jobs\.personio\.(?:de|com)"
    for role in roles_core + roles_extra:
        dorks.append((
            f"site:jobs.personio.de {role} {neg}",
            "personio", personio_pat,
        ))

    # --- International ATS: need explicit DE/CH/remote location ---------------
    for site, ats, pat in intl_targets:
        for role in roles_core:
            dorks.append((f"{site} {role} {de_ch} {neg}", ats, pat))
        # Broader EU sweep for LLM/AI roles with pure remote
        dorks.append((f"{site} \"AI Engineer\" {eu_broad} {neg}", ats, pat))

    return dorks


async def serper_find_jobs(
    api_key: str,
    max_queries: int | None = None,
) -> list:
    """
    Run targeted job-search dorks via Serper, extract company tokens,
    fetch full jobs from those companies immediately, and return Job objects
    ready for the normal filter + score pipeline.

    Companies already in companies.py are skipped here (the main pipeline
    fetches them); only newly discovered companies are fetched.

    Returns a list of fetchers.Job instances.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from fetchers import fetch_company, Job  # noqa: F401

    dorks = _build_job_search_dorks()
    if max_queries and max_queries > 0:
        dorks = dorks[:max_queries]

    existing = load_existing_tokens()
    found: dict[tuple[str, str], dict] = {}

    log.info("serper_find_jobs: running %d targeted dorks…", len(dorks))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(2)

        async def one(query: str, ats: str, pattern: str):
            async with sem:
                await asyncio.sleep(0.5)
                results = await serper_search(client, query, api_key)
                for r in results:
                    url = r.get("link", "")
                    token = extract_token_from_url(url, pattern)
                    if not token:
                        continue
                    key = (ats, token)
                    if key in existing or key in found:
                        continue
                    name = extract_company_name_from_result(r, token)
                    found[key] = {"ats": ats, "token": token, "name": name}
                    log.debug(
                        "serper_find_jobs: new [%s] %s (%s)", ats, token, name
                    )

        await asyncio.gather(*[one(q, a, p) for q, a, p in dorks])

    new_companies = list(found.values())
    log.info(
        "serper_find_jobs: %d dorks → %d new company tokens (not in companies.py)",
        len(dorks), len(new_companies),
    )

    if not new_companies:
        return []

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
    headers = {"User-Agent": "job-scout/0.1 (+personal-use)"}
    jobs: list = []

    async with httpx.AsyncClient(
        limits=limits, headers=headers, follow_redirects=True
    ) as client:
        sem = asyncio.Semaphore(6)

        async def fetch_one(c: dict) -> list:
            async with sem:
                return await fetch_company(
                    client, c["ats"], c["token"], c.get("name")
                )

        results = await asyncio.gather(
            *[fetch_one(c) for c in new_companies],
            return_exceptions=True,
        )

    for c, result in zip(new_companies, results):
        if isinstance(result, Exception):
            log.warning(
                "serper_find_jobs: fetch error %s/%s: %s",
                c["ats"], c["token"], result,
            )
        elif result:
            jobs.extend(result)
            log.debug(
                "serper_find_jobs: %s/%s → %d raw jobs",
                c["ats"], c["token"], len(result),
            )

    log.info(
        "serper_find_jobs: %d new companies → %d raw jobs fetched",
        len(new_companies), len(jobs),
    )
    return jobs


# ---------------------------------------------------------------------------
# Dorks-only mode — pure Google-Dorks based discovery
#
# Independent of companies.py. We search Google for individual job postings,
# group results by (ats, token), fetch each board exactly once, and return
# ONLY the jobs whose URL/ID we actually saw in the search results.
#
# Why this is more reliable than the companies.py path:
# - Google indexes any public ATS posting, so we are not blocked by missing
#   tokens in companies.py.
# - We fetch the board only when Google already confirmed a relevant job.
# - One run typically burns ~35 Serper credits and returns 50-150 jobs
#   (vs. fetching the full companies.py which yields thousands of raw jobs).
# ---------------------------------------------------------------------------

# (ats_name, regex with group(1) = token, optional group(2) = job_id_or_slug)
_JOB_URL_PATTERNS: list[tuple[str, str]] = [
    ("personio",
     r"https?://([^.]+)\.jobs\.personio\.(?:de|com)(?:/job/(\d+))?"),
    ("greenhouse",
     r"https?://(?:job-boards|boards)(?:\.eu)?\.greenhouse\.io/"
     r"([^/?#\s]+)/jobs/(\d+)"),
    ("greenhouse",
     r"https?://careers\.datadoghq\.com/detail/(\d+)"),
    ("lever",
     r"https?://jobs\.lever\.co/([^/?#\s]+)/([^/?#\s]+)"),
    ("ashby",
     r"https?://jobs\.ashbyhq\.com/([^/?#\s]+)(?:/([^/?#\s]+))?"),
    ("recruitee",
     r"https?://([\w-]+)\.recruitee\.com/o/([^/?#\s]+)"),
    ("smartrecruiters",
     r"https?://jobs\.smartrecruiters\.com/([^/?#\s]+)/(\d+)"),
]


def _extract_ats_token_jobid(url: str) -> tuple[str, str, str | None] | None:
    """Return (ats, token, job_id_or_slug) for a job URL, or None.

    The datadog-special pattern has token implicit (always 'datadog') and the
    captured group is the job id — handle that case explicitly.
    """
    for ats, pattern in _JOB_URL_PATTERNS:
        m = re.search(pattern, url, re.IGNORECASE)
        if not m:
            continue
        if ats == "greenhouse" and "datadoghq.com" in url.lower():
            return ats, "datadog", m.group(1)
        token = m.group(1)
        if not token or token.lower() in SKIP_TOKENS:
            continue
        job_id = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        return ats, token, job_id
    return None


def _job_matches_hit(job, hit: dict) -> bool:
    """True if a fetched Job corresponds to a Serper search-result hit."""
    job_url = (job.url or "").strip()
    hit_url = (hit.get("url") or "").strip()
    hit_id = hit.get("job_id")
    job_id = str(job.raw_id) if job.raw_id else ""

    if hit_id:
        if job_id and str(hit_id) == job_id:
            return True
        if str(hit_id) in job_url:
            return True
    if hit_url:
        if job_url == hit_url:
            return True
        # Trailing slashes, query params, language suffixes all vary slightly
        # between the URL Google indexes and the URL the ATS API returns.
        # A prefix match on the path is the most forgiving compromise that
        # still avoids false positives.
        if hit_url.split("?")[0].rstrip("/") in job_url:
            return True
        if job_url and job_url.split("?")[0].rstrip("/") in hit_url:
            return True
    return False


async def dorks_find_jobs(
    api_key: str,
    max_queries: int | None = None,
) -> list:
    """
    Pure Google-Dorks-based job discovery.

    Returns a list of fetchers.Job ready to flow into the normal
    filter / scorer / DB pipeline. Does NOT consult companies.py.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from fetchers import fetch_company

    dorks = _build_job_search_dorks()
    if max_queries and max_queries > 0:
        dorks = dorks[:max_queries]

    # (ats, token) -> list of hit dicts {url, job_id, title, snippet}
    hits: dict[tuple[str, str], list[dict]] = {}

    log.info("dorks_find_jobs: running %d Google Dorks via Serper", len(dorks))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(2)

        async def one_dork(query: str):
            async with sem:
                await asyncio.sleep(0.5)
                results = await serper_search(client, query, api_key)
                for r in results:
                    url = r.get("link", "")
                    if not url:
                        continue
                    extracted = _extract_ats_token_jobid(url)
                    if not extracted:
                        continue
                    ats, token, job_id = extracted
                    key = (ats, token)
                    hits.setdefault(key, []).append({
                        "url": url,
                        "job_id": job_id,
                        "title": r.get("title", "").strip(),
                        "snippet": r.get("snippet", "").strip(),
                    })

        await asyncio.gather(*[one_dork(q) for q, _, _ in dorks])

    total_hits = sum(len(v) for v in hits.values())
    log.info(
        "dorks_find_jobs: %d Treffer in %d (ats, token) Gruppen",
        total_hits, len(hits),
    )

    if not hits:
        return []

    limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
    headers = {"User-Agent": "job-scout/0.1 (+personal-use)"}
    matched_jobs: list = []

    async with httpx.AsyncClient(
        limits=limits, headers=headers, follow_redirects=True,
    ) as client:
        sem = asyncio.Semaphore(6)

        async def fetch_and_match(
            ats: str, token: str, hit_list: list[dict],
        ) -> list:
            async with sem:
                board = await fetch_company(client, ats, token, None)
            kept = [
                j for j in board
                if any(_job_matches_hit(j, h) for h in hit_list)
            ]
            log.debug(
                "dorks: %s/%s board=%d matched=%d hits=%d",
                ats, token, len(board), len(kept), len(hit_list),
            )
            return kept

        results = await asyncio.gather(
            *[
                fetch_and_match(ats, tok, hl)
                for (ats, tok), hl in hits.items()
            ],
            return_exceptions=True,
        )

    for ((ats, tok), _hl), result in zip(hits.items(), results):
        if isinstance(result, Exception):
            log.warning("dorks fetch error for %s/%s: %s", ats, tok, result)
            continue
        matched_jobs.extend(result)

    log.info(
        "dorks_find_jobs: %d Gruppen → %d gematchte Jobs",
        len(hits), len(matched_jobs),
    )
    return matched_jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main_async(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.print_query:
        print(google_query_template(
            location=args.location or "München OR Munich",
            require_remote=not args.no_remote,
        ))
        return

    api_key = args.api_key or os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        print(
            "\n✗ Serper API key fehlt.\n"
            "  Kostenlos holen:  https://serper.dev  (2.500 Suchen/Monat gratis)\n"
            "  Dann in .env:     SERPER_API_KEY=dein-key\n"
            "  Oder direkt:      --api-key dein-key\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.live:
        mode = "full" if args.full else "focused"
        print(f"Live-Discovery: mode={mode}, alle 4 ATS via per-city Dorks...\n")
        candidates = await discover_live(
            api_key,
            save_to_companies=args.update,
            mode=mode,
            max_queries=args.max_queries,
        )
        print(f"\nGefunden: {len(candidates)} neue Firmen "
              f"({'companies.py aktualisiert' if args.update else 'dry-run'})\n")
        for c in sorted(candidates, key=lambda x: (x["ats"], x["name"]))[:40]:
            print(f"  [{c['ats']:12s}] {c['token']:30s} {c['name'][:40]}")
        if len(candidates) > 40:
            print(f"  ... +{len(candidates) - 40} weitere")
        return

    print(f"Starte {len(DORKS)} klassische Google Dorks via Serper...\n")
    candidates = await run_dorks(api_key)
    print(f"\nGefunden: {len(candidates)} neue Kandidaten (noch nicht in companies.py)\n")

    if not candidates:
        print("Keine neuen Firmen entdeckt. companies.py ist aktuell.")
        return

    if args.dry_run:
        print("DRY RUN — keine Änderungen:\n")
        for c in sorted(candidates, key=lambda x: (x["ats"], x["name"])):
            print(f"  [{c['ats']:12s}] {c['token']:30s} {c['name']}")
        return

    print("Verifiziere Tokens via ATS-APIs...\n")
    verified, unverified = await verify_candidates(candidates)

    print(f"{'ATS':<12} {'TOKEN':<30} {'JOBS':>5}  NAME")
    print("-" * 65)
    for e in sorted(verified, key=lambda x: x["name"].lower()):
        print(f"{e['ats']:<12} {e['token']:<30} {e['job_count']:>5}  {e['name']}")
    if unverified:
        print(f"\n  Nicht verifiziert (URL gesehen, Token aber kein API-Hit): {len(unverified)}")
        for e in unverified:
            print(f"    [{e['ats']}] {e['token']}  ({e['_source_url'][:70]})")
    print("-" * 65)
    print(f"Verifiziert: {len(verified)}  |  Nicht verifiziert: {len(unverified)}\n")

    if not verified:
        print("Nichts zu aktualisieren.")
        return

    if args.update:
        added = append_to_companies(verified)
        print(f"✓ {added} Einträge zu src/companies.py hinzugefügt.")
    else:
        print("Füge --update hinzu um companies.py automatisch zu aktualisieren.")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Entdecke neue Firmen via Google Dorks + Serper API",
        epilog=(
            "Beispiele:\n"
            "  python serper_discover.py                  # klassische Dorks\n"
            "  python serper_discover.py --live --update  # focused per-city, alle 4 ATS\n"
            "  python serper_discover.py --live --full    # comprehensive (teuer!)\n"
            "  python serper_discover.py --print-query --location 'Stuttgart'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--api-key", default=None,
                   help="Serper API Key (oder SERPER_API_KEY in .env)")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur anzeigen was gefunden würde, nichts schreiben")
    p.add_argument("--update", action="store_true",
                   help="Verifizierte Firmen direkt in companies.py schreiben")
    p.add_argument("--live", action="store_true",
                   help="Verwende den per-city Live-Discovery-Modus statt der "
                        "klassischen Dorks (alle 4 ATS, DE + CH).")
    p.add_argument("--full", action="store_true",
                   help="Mit --live: comprehensive mode (alle Städte/Rollen, "
                        "verbrennt ~2000+ Serper-Aufrufe).")
    p.add_argument("--max-queries", type=int, default=None,
                   help="Hard cap auf Anzahl Serper-Aufrufe (Cost-Control).")
    p.add_argument("--print-query", action="store_true",
                   help="Druckt eine fertige Google-Query zum Copy-Paste statt "
                        "Serper aufzurufen.")
    p.add_argument("--location", default=None,
                   help="Mit --print-query: Standort (z.B. 'Stuttgart' oder "
                        "'München OR Munich').")
    p.add_argument("--no-remote", action="store_true",
                   help="Mit --print-query: Remote-Filter weglassen.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()