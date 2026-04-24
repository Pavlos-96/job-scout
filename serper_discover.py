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
import json
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
            json={"q": query, "num": num, "gl": "de", "hl": "de"},
            timeout=15.0,
        )
        r.raise_for_status()
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
    snippet = result.get("snippet", "")

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
        sem = asyncio.Semaphore(3)  # Serper rate limit: be gentle

        async def one_dork(query: str, ats: str, pattern: str):
            async with sem:
                await asyncio.sleep(0.3)  # small delay between requests
                results = await serper_search(client, query, api_key, num=15)
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
    from discover import discover_company

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
# CLI
# ---------------------------------------------------------------------------

async def main_async(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

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

    print(f"Starte {len(DORKS)} Google Dorks via Serper...\n")
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
        print("\nSnippet zum manuellen Einfügen in companies.py:\n")
        for e in sorted(verified, key=lambda x: (x["ats"], x["name"].lower())):
            print(
                f'    {{"ats": "{e["ats"]}", "token": "{e["token"]}", '
                f'"name": "{e["name"]}", "tier": "scaleup", '
                f'"verified": True, "notes": "{e["notes"]}"}},\n'
            )


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Entdecke neue Firmen via Google Dorks + Serper API"
    )
    p.add_argument("--api-key", default=None,
                   help="Serper API Key (oder SERPER_API_KEY in .env)")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur anzeigen was gefunden würde, nichts schreiben")
    p.add_argument("--update", action="store_true",
                   help="Verifizierte Firmen direkt in companies.py schreiben")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()