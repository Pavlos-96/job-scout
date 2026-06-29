#!/usr/bin/env python3
"""
discover.py — ATS-Erkennung für neue Firmen.

Nimmt eine Liste von Firmennamen, findet ihren ATS-Token und fügt neue Einträge
direkt in src/companies.py ein (sofern noch nicht vorhanden).

Usage:
    python discover.py "Celonis" "Aleph Alpha" "DeepL"
    python discover.py companies_to_discover.txt
    python discover.py "Celonis" --dry-run
    python discover.py "Celonis" --debug
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

import httpx

log = logging.getLogger("discover")

ROOT = Path(__file__).parent
COMPANIES_FILE = ROOT / "src" / "companies.py"

# Marker line in companies.py where new entries are inserted
INSERT_MARKER = "    # <<< new entries are appended above this line >>>"


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\b(gmbh|ag|se|inc|ltd|llc|kg|ohg|corp|sarl|sas|bv|nv|ab)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s-]", "", s)
    return s.strip()


def _token_variants(name: str) -> list[str]:
    """Generate candidate ATS board tokens from a human-readable company name."""
    base = _slugify(name)
    seen: dict[str, None] = {}

    def add(s: str):
        s = s.strip("-_ ").lower()
        if s and s not in seen:
            seen[s] = None

    no_space = re.sub(r"[\s_]+", "", base)
    add(no_space)

    hyphen = re.sub(r"[\s_]+", "-", base)
    add(hyphen)

    under = re.sub(r"[\s_]+", "_", base)
    add(under)

    add(name.lower().strip())

    words = base.split()
    if words:
        add(words[0])
        if len(words) > 1:
            add("".join(words[:2]))
            add("-".join(words[:2]))

    umlaut_map = str.maketrans({
        "ä": "ae", "ö": "oe", "ü": "ue",
        "Ä": "ae", "Ö": "oe", "Ü": "ue", "ß": "ss",
    })
    no_umlaut = base.translate(umlaut_map)
    if no_umlaut != base:
        add(re.sub(r"[\s_]+", "", no_umlaut))
        add(re.sub(r"[\s_]+", "-", no_umlaut))

    cleaned = re.sub(r"[°.,!?]", "", base)
    cleaned = re.sub(r"[\s_]+", "", cleaned)
    if cleaned != no_space:
        add(cleaned)

    return list(seen.keys())


# ---------------------------------------------------------------------------
# ATS probers
# ---------------------------------------------------------------------------

def _log_probe_miss(ats: str, token: str, status: int | None, reason: str = ""):
    if status == 404:
        log.debug("miss [%s] %s -> 404", ats, token)
    elif status is not None:
        log.debug("miss [%s] %s -> HTTP %s %s", ats, token, status, reason)
    else:
        log.debug("miss [%s] %s -> %s", ats, token, reason or "no response")


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 15.0,
    follow_redirects: bool = True,
    max_retries: int = 3,
) -> httpx.Response | None:
    """GET with simple exponential backoff on 429 responses."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            r = await client.get(url, timeout=timeout, follow_redirects=follow_redirects)
            if r.status_code == 429:
                if attempt < max_retries:
                    log.debug("429 rate-limit on %s — retrying in %.1fs", url, delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                log.warning("429 rate-limit on %s — giving up after %d retries", url, max_retries)
                return None
            return r
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            log.debug("request failed for %s: %s", url, e)
            return None
    return None


async def _probe_greenhouse(client: httpx.AsyncClient, token: str) -> int | None:
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    r = await _get_with_retry(client, url)
    if r is None:
        return None
    if r.status_code == 200:
        return len(r.json().get("jobs", []))
    _log_probe_miss("greenhouse", token, r.status_code)
    return None


async def _probe_lever(client: httpx.AsyncClient, token: str) -> int | None:
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    r = await _get_with_retry(client, url)
    if r is None:
        return None
    if r.status_code == 200:
        data = r.json()
        return len(data) if isinstance(data, list) else None
    _log_probe_miss("lever", token, r.status_code)
    return None


async def _probe_personio(client: httpx.AsyncClient, token: str) -> int | None:
    import xml.etree.ElementTree as ET
    for tld in ("de", "com"):
        url = f"https://{token}.jobs.personio.{tld}/xml?language=en"
        try:
            # Never follow redirects: a 301 means the subdomain doesn't exist and
            # Personio redirects to personio.de — following it in bulk causes 429 storms.
            r = await client.get(url, timeout=15.0, follow_redirects=False)
            if r.status_code == 200 and r.content:
                try:
                    root = ET.fromstring(r.content)
                    return len(root.findall(".//position"))
                except ET.ParseError:
                    return 0
            if r.status_code in (301, 302):
                log.debug("personio: %s redirect for %s (token not found on .%s)", r.status_code, token, tld)
                continue
            if tld == "de":
                _log_probe_miss("personio", token, r.status_code)
        except Exception as e:
            if tld == "de":
                _log_probe_miss("personio", token, None, str(e))
            continue
    return None


async def _probe_ashby(client: httpx.AsyncClient, token: str) -> int | None:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=false"
    r = await _get_with_retry(client, url)
    if r is None:
        return None
    if r.status_code == 200:
        return len(r.json().get("jobs", []))
    _log_probe_miss("ashby", token, r.status_code)
    return None


async def _probe_recruitee(client: httpx.AsyncClient, token: str) -> int | None:
    url = f"https://{token}.recruitee.com/api/offers/"
    r = await _get_with_retry(client, url)
    if r is None:
        return None
    if r.status_code == 200:
        try:
            return len(r.json().get("offers", []))
        except (ValueError, AttributeError):
            return None
    _log_probe_miss("recruitee", token, r.status_code)
    return None


async def _probe_smartrecruiters(client: httpx.AsyncClient, token: str) -> int | None:
    url = (f"https://api.smartrecruiters.com/v1/companies/"
           f"{token}/postings?limit=1")
    r = await _get_with_retry(client, url)
    if r is None:
        return None
    if r.status_code == 200:
        try:
            return r.json().get("totalFound", 0)
        except (ValueError, AttributeError):
            return None
    _log_probe_miss("smartrecruiters", token, r.status_code)
    return None


PROBERS: dict = {
    "greenhouse": _probe_greenhouse,
    "lever": _probe_lever,
    "personio": _probe_personio,
    "ashby": _probe_ashby,
    "recruitee": _probe_recruitee,
    "smartrecruiters": _probe_smartrecruiters,
}


# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------

async def discover_company(
    client: httpx.AsyncClient,
    name: str,
    sem: asyncio.Semaphore,
    debug: bool = False,
) -> list[dict]:
    """Try all ATS × token combinations. Returns found entries (usually 0–1)."""
    variants = _token_variants(name)
    if debug:
        log.debug("[%s] token variants: %s", name, variants)

    found: list[dict] = []

    async def try_one(ats: str, token: str):
        async with sem:
            n = await PROBERS[ats](client, token)
            if n is not None:
                if debug:
                    log.debug("[%s] HIT  %s / %s  (%d jobs)", name, ats, token, n)
                found.append({
                    "ats": ats, "token": token, "name": name,
                    "job_count": n, "tier": "scaleup", "verified": True,
                    "notes": f"auto-discovered, {n} open jobs",
                })

    await asyncio.gather(*[try_one(ats, token) for ats in PROBERS for token in variants])

    # Keep one entry per ATS (the token with most jobs)
    best: dict[str, dict] = {}
    for entry in found:
        key = entry["ats"]
        if key not in best or entry["job_count"] > best[key]["job_count"]:
            best[key] = entry
    return list(best.values())


async def discover_all(
    names: list[str],
    concurrency: int = 12,
    debug: bool = False,
) -> tuple[list[dict], list[str]]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
    headers = {"User-Agent": "job-scout-discover/0.1 (+personal-use)"}

    all_found: list[dict] = []
    not_found: list[str] = []

    async with httpx.AsyncClient(limits=limits, headers=headers,
                                 follow_redirects=True) as client:
        results = await asyncio.gather(
            *[discover_company(client, n, sem, debug) for n in names],
            return_exceptions=True,
        )

    for name, result in zip(names, results):
        if isinstance(result, Exception):
            log.warning("Error discovering %s: %s", name, result)
            not_found.append(name)
        elif result:
            all_found.extend(result)
        else:
            not_found.append(name)

    return all_found, not_found


# ---------------------------------------------------------------------------
# companies.py helpers
# ---------------------------------------------------------------------------

def _parse_companies_file() -> list[dict]:
    """Parse all company dicts from companies.py via ast.literal_eval."""
    if not COMPANIES_FILE.exists():
        return []
    import ast
    entries: list[dict] = []
    for line in COMPANIES_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().rstrip(",")
        if stripped.startswith('{"ats":') or stripped.startswith("{'ats':"):
            try:
                entries.append(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                pass
    return entries


def load_existing_keys() -> set[tuple[str, str]]:
    """Return set of (ats, token.lower()) already in companies.py."""
    return {
        (e.get("ats", ""), e.get("token", "").lower())
        for e in _parse_companies_file()
    }


def load_existing_names() -> set[str]:
    """
    Return normalised set of company names already in companies.py.

    Normalisation: lowercase, strip, collapse spaces — good enough for
    "Ippen Digital" == "ippen digital" == "IPPEN DIGITAL".
    Also includes the slugified token so "ippen-digital" matches too.
    """
    names: set[str] = set()
    for e in _parse_companies_file():
        raw_name = e.get("name", "")
        if raw_name:
            names.add(raw_name.lower().strip())
        token = e.get("token", "")
        if token:
            names.add(token.lower())
            names.add(token.lower().replace("-", " "))
    return names


def append_to_companies(entries: list[dict]) -> int:
    """
    Insert new entries into src/companies.py just before INSERT_MARKER.
    Returns number of entries actually written (skips already-present keys).
    """
    if not entries:
        return 0

    src = COMPANIES_FILE.read_text(encoding="utf-8")
    existing = load_existing_keys()

    new_lines: list[str] = []
    for e in sorted(entries, key=lambda x: (x["ats"], x.get("name", "").lower())):
        if (e["ats"], e["token"].lower()) in existing:
            continue
        notes = e.get("notes", "auto-discovered").replace('"', "'")
        new_lines.append(
            f'    {{"ats": "{e["ats"]}", "token": "{e["token"]}", '
            f'"name": "{e["name"]}", "tier": "{e.get("tier", "scaleup")}", '
            f'"verified": True, "notes": "{notes}"}},\n'
        )

    if not new_lines:
        return 0

    idx = src.find(INSERT_MARKER)
    if idx == -1:
        log.error(
            "Insertion marker not found in %s — please add this line:\n  %s",
            COMPANIES_FILE, INSERT_MARKER,
        )
        return 0

    block = "".join(new_lines)
    updated = src[:idx] + block + src[idx:]
    COMPANIES_FILE.write_text(updated, encoding="utf-8")
    return len(new_lines)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_table(found: list[dict], not_found: list[str]) -> str:
    lines = [f"\n{'RESULT':<10} {'ATS':<12} {'TOKEN':<30} {'JOBS':>5}  NAME"]
    lines.append("-" * 72)
    for e in sorted(found, key=lambda x: x.get("name", "").lower()):
        lines.append(
            f"{'FOUND':<10} {e['ats']:<12} {e['token']:<30} "
            f"{e.get('job_count', 0):>5}  {e.get('name', '')}"
        )
    for name in sorted(not_found):
        lines.append(f"{'NOT FOUND':<10} {'—':<12} {'—':<30} {'':>5}  {name}")
    lines.append("-" * 72)
    lines.append(f"Found: {len(found)}  |  Not found: {len(not_found)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_names(args) -> list[str]:
    names: list[str] = []

    if args.input:
        p = Path(args.input)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line)
        else:
            names.append(args.input)  # treated as a single company name

    for a in args.names or []:
        a = a.strip()
        if a:
            names.append(a)

    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)
    return unique


async def main_async(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    names = load_names(args)
    if not names:
        print("Keine Firmennamen angegeben. Übergib Namen als Argumente oder Textdatei.",
              file=sys.stderr)
        sys.exit(1)

    # Skip companies already in companies.py — match by stored name or token
    existing_names = load_existing_names()

    def _is_known(name: str) -> bool:
        n = name.lower().strip()
        return (
            n in existing_names
            or _slugify(name) in existing_names
            or _slugify(name).replace("-", " ") in existing_names
        )

    unknown = [n for n in names if not _is_known(n)]
    already_known = [n for n in names if _is_known(n)]

    if already_known:
        print(f"Bereits bekannt (übersprungen): {', '.join(already_known)}")

    if not unknown:
        print("Alle Firmen sind bereits in companies.py. Nichts zu tun.")
        return

    print(f"\nSuche ATS für {len(unknown)} Firma(s)...\n")
    found, not_found = await discover_all(unknown, concurrency=args.concurrency,
                                          debug=args.debug)

    print(render_table(found, not_found))

    if not found:
        return

    if args.dry_run:
        print("\n(Dry run — nichts geschrieben. Ohne --dry-run werden neue Einträge")
        print(f" direkt in {COMPANIES_FILE.relative_to(ROOT)} eingefügt.)")
        return

    added = append_to_companies(found)
    if added:
        print(f"\n✓ {added} neue Einträge in {COMPANIES_FILE.relative_to(ROOT)} eingefügt.")
    else:
        print("\nKeine neuen Einträge (alle bereits vorhanden).")


def main():
    p = argparse.ArgumentParser(
        description="Neue Firmen in ATS entdecken und direkt in companies.py eintragen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="Textdatei (eine Firma pro Zeile) oder einzelner Firmenname")
    p.add_argument("names", nargs="*",
                   help="Weitere Firmennamen als Argumente")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur anzeigen was gefunden würde, nichts in companies.py schreiben")
    p.add_argument("--debug", action="store_true",
                   help="Alle Token-Varianten anzeigen die probiert werden")
    p.add_argument("--concurrency", type=int, default=6,
                   help="Parallele HTTP-Probes (Standard: 6)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
