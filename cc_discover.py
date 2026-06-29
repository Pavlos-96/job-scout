"""
cc_discover.py - Tenant discovery via Common Crawl CDX index.

Crawls the Common Crawl CDX API for each ATS provider's tenant URL pattern,
extracts the company tokens, deduplicates them, and writes tenants.json.

This is the replacement for the (non-existent) "ATS master sitemaps".
Common Crawl is a free, open web index that has indexed millions of these
URLs over many years. We use it as our source of truth for "which company
slugs exist on which ATS".

Usage:
    python cc_discover.py                  # latest crawl, all providers
    python cc_discover.py --crawls 3       # last 3 monthly crawls (merged)
    python cc_discover.py --only greenhouse,personio
    python cc_discover.py --out tenants.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("cc_discover")

CDX_COLLINFO = "https://index.commoncrawl.org/collinfo.json"

# Per-ATS extraction rules.
#   url_pattern: CDX wildcard query
#   token_re:    regex on found URLs, group(1) = tenant token
#   bad_chars:   tokens containing any of these chars are dropped
@dataclass(frozen=True)
class Provider:
    ats: str
    url_pattern: str
    token_re: re.Pattern[str]
    min_token_len: int = 2
    max_token_len: int = 64
    # Path-based providers (jobs.lever.co/<tenant>) extract tokens from URL
    # paths and the token must NOT contain dots (those are files like
    # `robots.txt` or `sitemap.xml`). Subdomain-based providers
    # (<tenant>.recruitee.com) already disallow dots by virtue of the
    # subdomain grammar, but we set the flag explicitly for clarity.
    path_based: bool = True


PROVIDERS: dict[str, list[Provider]] = {
    "greenhouse": [
        Provider(
            "greenhouse",
            "boards.greenhouse.io/*",
            re.compile(r"^https?://boards\.greenhouse\.io/([^/?#]+)", re.I),
        ),
        Provider(
            "greenhouse",
            "job-boards.greenhouse.io/*",
            re.compile(r"^https?://job-boards\.greenhouse\.io/([^/?#]+)", re.I),
        ),
    ],
    "lever": [
        Provider(
            "lever",
            "jobs.lever.co/*",
            re.compile(r"^https?://jobs\.lever\.co/([^/?#]+)", re.I),
        ),
    ],
    "ashby": [
        Provider(
            "ashby",
            "jobs.ashbyhq.com/*",
            re.compile(r"^https?://jobs\.ashbyhq\.com/([^/?#]+)", re.I),
        ),
    ],
    "personio": [
        Provider(
            "personio",
            "*.jobs.personio.de/*",
            re.compile(r"^https?://([^.]+)\.jobs\.personio\.de", re.I),
            path_based=False,
        ),
        Provider(
            "personio",
            "*.jobs.personio.com/*",
            re.compile(r"^https?://([^.]+)\.jobs\.personio\.com", re.I),
            path_based=False,
        ),
    ],
    "recruitee": [
        Provider(
            "recruitee",
            "*.recruitee.com/*",
            re.compile(r"^https?://([^.]+)\.recruitee\.com", re.I),
            path_based=False,
        ),
    ],
    "smartrecruiters": [
        Provider(
            "smartrecruiters",
            "jobs.smartrecruiters.com/*",
            # First path segment is the company slug. Subsequent segments
            # are job IDs (e.g. /Adidas/744000088889249-custodian-...).
            re.compile(
                r"^https?://jobs\.smartrecruiters\.com/([^/?#]+)",
                re.I,
            ),
        ),
    ],
}

# Reserved or non-tenant path segments that show up but aren't companies.
# These leak in from things like /robots.txt, /assets/, /support, etc.
RESERVED_TOKENS: dict[str, set[str]] = {
    "ashby": {"login", "signup", "auth", "assets", "static", "embed",
              "robots", "favicon", "sitemap", "manifest"},
    "lever": {"hire", "lever-api", "assets", "static", "robots",
              "favicon", "sitemap"},
    "greenhouse": {"embed", "assets", "static", "robots", "favicon",
                   "sitemap", "departments", "offices"},
    "smartrecruiters": {"assets", "static", "api", "robots", "favicon",
                        "sitemap", "external-referrals", "oneclick-ui",
                        "ui", "auth", "login", "signup"},
    "recruitee": {"www", "api", "assets", "support", "help", "robots",
                  "favicon", "sitemap"},
    "personio": {"www", "api", "support", "help", "robots", "favicon"},
}


# Common Crawl is paginated per index. We pull every page sequentially.
# CDX is flaky under load: 504s and incomplete chunked reads happen.
def fetch_cdx_page(
    client: httpx.Client,
    index_id: str,
    url_pattern: str,
    page: int,
    max_retries: int = 3,
) -> list[str]:
    """Fetch one page of CDX results as raw URLs, retrying transient errors."""
    api = f"https://index.commoncrawl.org/{index_id}-index"
    params = {
        "url": url_pattern,
        "output": "json",
        "fl": "url",
        "page": str(page),
    }
    backoff = 2.0
    for attempt in range(max_retries + 1):
        urls: list[str] = []
        try:
            with client.stream("GET", api, params=params, timeout=180.0) as r:
                if r.status_code == 404:
                    return []
                if r.status_code in (502, 503, 504, 429):
                    raise httpx.HTTPStatusError(
                        f"transient {r.status_code}", request=r.request, response=r,
                    )
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    u = rec.get("url")
                    if u:
                        urls.append(u)
            return urls
        except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
            if attempt < max_retries:
                log.info("  retry %d/%d after %.1fs (%s)",
                         attempt + 1, max_retries, backoff,
                         type(e).__name__)
                time.sleep(backoff)
                backoff *= 2
                continue
            log.warning("cdx fetch gave up: %s %s page=%d: %s",
                        index_id, url_pattern, page, e)
            return urls
    return []


def num_pages(client: httpx.Client, index_id: str, url_pattern: str) -> int:
    """Ask CDX how many pages of results this query has."""
    api = f"https://index.commoncrawl.org/{index_id}-index"
    params = {"url": url_pattern, "output": "json", "showNumPages": "true"}
    try:
        r = client.get(api, params=params, timeout=30.0)
        r.raise_for_status()
        return int(r.json().get("pages", 0))
    except (httpx.HTTPError, ValueError) as e:
        log.warning("page count failed for %s %s: %s",
                    index_id, url_pattern, e)
        return 0


def list_recent_crawls(client: httpx.Client, n: int) -> list[str]:
    """Return the N most recent Common Crawl monthly crawl IDs."""
    r = client.get(CDX_COLLINFO, timeout=30.0)
    r.raise_for_status()
    return [d["id"] for d in r.json()[:n]]


def extract_tokens(
    urls: list[str],
    provider: Provider,
    reserved: set[str],
) -> dict[str, int]:
    """URLs -> {token: hit_count}, filtered."""
    counts: dict[str, int] = defaultdict(int)
    # Path-based tenants never contain dots (those are filenames like
    # robots.txt). Subdomain-based ones still allow internal dots in theory
    # but never in practice — subdomain labels can't have dots.
    if provider.path_based:
        slug_re = re.compile(r"[a-z0-9][a-z0-9_-]*")
    else:
        slug_re = re.compile(r"[a-z0-9][a-z0-9-]*")
    for u in urls:
        m = provider.token_re.match(u)
        if not m:
            continue
        tok = m.group(1).strip().lower()
        if not tok or tok in reserved:
            continue
        if len(tok) < provider.min_token_len or len(tok) > provider.max_token_len:
            continue
        if not slug_re.fullmatch(tok):
            continue
        counts[tok] += 1
    return counts


def discover_all(
    only: list[str] | None,
    crawls: int,
    out_path: Path,
) -> dict[str, dict[str, int]]:
    """Return {ats: {token: hit_count}} aggregated across selected crawls.

    Writes the output file after each provider completes so partial progress
    survives crashes / interrupts.
    """
    headers = {"User-Agent": "job-scout-cc/0.1 (+personal-use)"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        index_ids = list_recent_crawls(client, crawls)
        log.info("Using crawls: %s", ", ".join(index_ids))

        # ats -> token -> count
        result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for ats, providers in PROVIDERS.items():
            if only and ats not in only:
                continue
            reserved = RESERVED_TOKENS.get(ats, set())
            for prov in providers:
                for idx in index_ids:
                    pages = num_pages(client, idx, prov.url_pattern)
                    log.info("[%s] %s @ %s: %d pages",
                             ats, prov.url_pattern, idx, pages)
                    for p in range(pages):
                        urls = fetch_cdx_page(client, idx, prov.url_pattern, p)
                        toks = extract_tokens(urls, prov, reserved)
                        for t, c in toks.items():
                            result[ats][t] += c
                        log.info("  page %d/%d: %d URLs -> %d new toks "
                                 "(running total %d)",
                                 p + 1, pages, len(urls), len(toks),
                                 len(result[ats]))
                        time.sleep(0.2)
            # Snapshot after each provider completes.
            write_tenants({k: dict(v) for k, v in result.items()}, out_path)
            log.info("snapshot written after %s (%d tenants so far)",
                     ats, len(result.get(ats, {})))
        return {k: dict(v) for k, v in result.items()}


def write_tenants(data: dict[str, dict[str, int]], path: Path) -> None:
    """
    Output format:
        {
          "generated_at": "2026-05-22T...",
          "tenants": {
            "greenhouse": [{"token": "10xgenomics", "hits": 42}, ...],
            "lever":      [...],
            ...
          },
          "totals": {"greenhouse": 1234, ...}
        }
    """
    from datetime import datetime, timezone

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totals": {ats: len(toks) for ats, toks in data.items()},
        "tenants": {
            ats: sorted(
                ({"token": t, "hits": c} for t, c in toks.items()),
                key=lambda x: (-x["hits"], x["token"]),
            )
            for ats, toks in data.items()
        },
    }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crawls", type=int, default=1,
                   help="Number of most recent monthly crawls to merge")
    p.add_argument("--only", default="",
                   help="Comma-separated subset, e.g. 'greenhouse,personio'")
    p.add_argument("--out", default="tenants.json")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    if only:
        unknown = set(only) - set(PROVIDERS)
        if unknown:
            raise SystemExit(f"Unknown ATS in --only: {unknown}. "
                             f"Available: {list(PROVIDERS)}")

    out_path = Path(args.out)
    data = discover_all(only, args.crawls, out_path)
    write_tenants(data, out_path)

    print()
    print(f"Wrote {out_path}")
    for ats, toks in sorted(data.items()):
        sample = sorted(toks.items(), key=lambda x: -x[1])[:5]
        print(f"  {ats:18s} {len(toks):5d} tenants   "
              f"top: {', '.join(f'{t}({n})' for t, n in sample)}")


if __name__ == "__main__":
    main()
