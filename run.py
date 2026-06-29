#!/usr/bin/env python3
"""
Job Scout CLI.

Usage:
    python run.py                           # run with heuristic filter only
    python run.py --score                   # + LLM scoring (needs OPENAI_API_KEY)
    python run.py --score --api-key sk-...  # explicit key instead of env var
    python run.py --min-salary 85000
    python run.py --senior-only
    python run.py --munich-only
    python run.py --test-tokens
    python run.py --companies path.json

Output:
    reports/report_YYYYMMDD_HHMM.md
    reports/report_YYYYMMDD_HHMM.json
    reports/latest.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env file if present (no extra dependencies needed)
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from src.fetchers import fetch_all
from src.filters import FilterConfig, filter_jobs
from src.report import write_reports
from src.companies import COMPANIES


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


async def probe_tokens(companies: list[dict]) -> None:
    import httpx
    from src.fetchers import fetch_company
    print(f"Probing {len(companies)} companies...\n")
    ok = empty = 0
    limits = httpx.Limits(max_keepalive_connections=16, max_connections=32)
    async with httpx.AsyncClient(limits=limits, headers={"User-Agent": "job-scout/0.1"},
                                 follow_redirects=True) as client:
        sem = asyncio.Semaphore(8)
        async def one(c):
            async with sem:
                return c, await fetch_company(client, c["ats"], c["token"], c["name"])
        results = await asyncio.gather(*[one(c) for c in companies])

    print(f"{'ATS':<12}{'TOKEN':<28}{'NAME':<28}{'JOBS':>5}  STATUS")
    print("-" * 88)
    for c, jobs in results:
        if jobs:
            ok += 1
            print(f"{c['ats']:<12}{c['token']:<28}{c['name'][:27]:<28}{len(jobs):>5}  OK")
        else:
            empty += 1
            print(f"{c['ats']:<12}{c['token']:<28}{c['name'][:27]:<28}{'0':>5}  EMPTY/404")
    print("-" * 88)
    print(f"Total: {ok} with jobs, {empty} empty/404.")


async def main_async(args):
    setup_logging(args.verbose)
    log = logging.getLogger("run")

    if args.companies:
        data = json.loads(Path(args.companies).read_text())
        companies = data if isinstance(data, list) else data.get("companies", [])
    else:
        companies = list(COMPANIES)

    if args.test_tokens:
        await probe_tokens(companies)
        return

    serper_key = os.environ.get("SERPER_API_KEY", "").strip()

    # --- Dorks-only mode: skip companies.py entirely ------------------------
    # Pure Google-Dorks discovery — search Google, follow each hit back to its
    # ATS, fetch exactly the matching jobs. Independent of any company list.
    if args.dorks_only:
        if not serper_key:
            print(
                "\n✗ --dorks-only braucht SERPER_API_KEY in .env\n"
                "  Kostenlos: https://serper.dev (2.500 Suchen/Monat gratis)\n",
                file=sys.stderr,
            )
            sys.exit(1)
        log.info("Dorks-only Modus: companies.py wird ignoriert.")
        from serper_discover import dorks_find_jobs
        jobs = await dorks_find_jobs(serper_key, max_queries=args.max_queries)
        log.info("Dorks-only: %d Jobs via Google → ATS gefunden", len(jobs))
    else:
        # --- Serper job search: targeted dorks → full jobs for NEW companies -
        extra_jobs: list = []
        if args.serper_jobs and serper_key:
            log.info(
                "Serper-Suche: starte ~35 gezielte Job-Dorks auf 6 ATS..."
            )
            from serper_discover import serper_find_jobs
            extra_jobs = await serper_find_jobs(serper_key)
            log.info(
                "Serper-Suche: %d zusätzliche Jobs aus neu entdeckten Firmen",
                len(extra_jobs),
            )
        elif args.serper_jobs and not serper_key:
            log.warning(
                "--serper-jobs: SERPER_API_KEY nicht gesetzt, übersprungen"
            )

        # --- Live discovery (manual/scheduled — adds to companies.py) ------
        if args.live_discover and serper_key:
            mode = "full" if args.live_discover_full else "focused"
            log.info("Live-Discover: starte per-city Dorks (mode=%s)...", mode)
            from serper_discover import discover_live
            new_companies = await discover_live(
                serper_key,
                save_to_companies=args.save_discovered,
                mode=mode,
            )
            if new_companies:
                existing_keys = {(c["ats"], c["token"]) for c in companies}
                added = [
                    c for c in new_companies
                    if (c["ats"], c["token"]) not in existing_keys
                ]
                companies.extend(added)
                log.info(
                    "Live-Discover: %d neue Firmen zur Fetch-Liste hinzugefügt",
                    len(added),
                )
        elif args.live_discover and not serper_key:
            log.warning(
                "--live-discover: SERPER_API_KEY nicht gesetzt, übersprungen"
            )

        log.info("Fetching jobs from %d companies...", len(companies))
        jobs = await fetch_all(companies, concurrency=args.concurrency)
        jobs.extend(extra_jobs)
        log.info(
            "Raw jobs fetched: %d (incl. %d Serper-Suche)",
            len(jobs), len(extra_jobs),
        )

    cfg = FilterConfig(
        require_senior_title=args.senior_only,
        require_germany_or_remote=not args.worldwide,
        prefer_munich=True,
        min_salary=args.min_salary,
        exclude_junior=True,
        require_posted_in_current_year=not args.no_post_year_filter,
        exclude_unknown_post_date=args.exclude_undated_jobs,
    )
    matches = filter_jobs(jobs, cfg)
    if args.munich_only:
        matches = [m for m in matches if m.location_flags.get("has_munich")]
    log.info("Matches after filtering: %d", len(matches))

    out_dir = Path(args.out) if args.out else ROOT / "reports"
    scored_pairs = None

    if args.score:
        # Resolve API key
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print(
                "\n✗ --score requires an OpenAI API key.\n"
                "  Set it via:  export OPENAI_API_KEY='sk-...'\n"
                "  Or pass:     --api-key sk-...\n"
                "  Get one at:  https://platform.openai.com/api-keys\n",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            import openai  # noqa: F401
        except ImportError:
            print("\n✗ openai package missing. Run: pip install openai\n", file=sys.stderr)
            sys.exit(1)

        from src.scorer import score_all, DEFAULT_MODEL
        from src.cache import ScoreCache

        cache = None if args.no_cache else ScoreCache(ROOT / "cache" / "scores.json")
        if cache:
            log.info("Score cache: %s", ROOT / "cache" / "scores.json")

        model = args.model or DEFAULT_MODEL
        log.info("Scoring %d matches with LLM (%s)...", len(matches), model)
        scored_pairs = await score_all(
            matches,
            api_key=api_key,
            model=model,
            concurrency=args.score_concurrency,
            cache=cache,
        )
        apply_n = sum(1 for _, s in scored_pairs if s and s.recommendation == "apply")
        maybe_n = sum(1 for _, s in scored_pairs if s and s.recommendation == "maybe")
        log.info("Scoring done: %d apply  %d maybe", apply_n, maybe_n)
        if cache:
            log.info("Score cache stats: %s", cache.stats)

    md_path, json_path = write_reports(matches, out_dir, scored_pairs)
    print(f"\n✓ Wrote {md_path}")
    print(f"✓ Wrote {json_path}")
    print(f"✓ Matches: {len(matches)} / {len(jobs)} raw jobs")

    from src.db import init_db, upsert_jobs
    await init_db()
    pairs_for_db = scored_pairs if scored_pairs else [(m, None) for m in matches]
    new_in_db = await upsert_jobs(pairs_for_db)
    log.info("DB: %d neue Stellen gespeichert", new_in_db)
    print(f"✓ DB: {new_in_db} neue Stellen gespeichert")

    if scored_pairs:
        apply_n = sum(1 for _, s in scored_pairs if s and s.recommendation == "apply")
        maybe_n = sum(1 for _, s in scored_pairs if s and s.recommendation == "maybe")
        print(f"✓ LLM scores: {apply_n} apply  {maybe_n} maybe\n")
    else:
        print()


def main():
    p = argparse.ArgumentParser(description="Job Scout — AI Engineer roles in DE/EU")
    p.add_argument("--min-salary", type=int, default=85000)
    p.add_argument("--senior-only", action="store_true")
    p.add_argument("--munich-only", action="store_true")
    p.add_argument("--worldwide", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--companies", type=str, default=None)
    p.add_argument("--test-tokens", action="store_true")
    p.add_argument("--out", type=str, default=None)
    # LLM scoring
    p.add_argument("--score", action="store_true",
                   help="Score matches with LLM (requires OpenAI API key)")
    p.add_argument("--api-key", type=str, default=None,
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--model", type=str, default=None,
                   help="OpenAI model to use (default: src.scorer.DEFAULT_MODEL)")
    p.add_argument("--score-concurrency", type=int, default=5,
                   help="Parallel LLM calls (default: 5)")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip score cache, always re-score with LLM")
    p.add_argument("--no-post-year-filter", action="store_true",
                   help="Keep listings whose posted year is not the current year")
    p.add_argument("--exclude-undated-jobs", action="store_true",
                   help="Drop jobs when the ATS provides no posted/created date")
    # Serper job search (primary Serper path — ~35 credits per run)
    p.add_argument("--serper-jobs", action="store_true",
                   help="Run ~35 targeted dorks via Serper, fetch full jobs from "
                        "newly found companies (requires SERPER_API_KEY in .env)")
    # Dorks-only mode: skip companies.py entirely, search Google for individual
    # job postings and fetch only those exact matches.
    p.add_argument("--dorks-only", action="store_true",
                   help="Pure Google Dorks: ignore companies.py, find jobs "
                        "directly via Serper and fetch only the matching "
                        "postings from each ATS (requires SERPER_API_KEY)")
    p.add_argument("--max-queries", type=int, default=None,
                   help="With --dorks-only: hard cap on Serper queries "
                        "(default: ~35 from _build_job_search_dorks)")
    # Live discovery (periodic manual task — adds to companies.py)
    p.add_argument("--live-discover", action="store_true",
                   help="Run per-city × role Serper dorks to expand companies.py "
                        "(use manually / weekly, not on every run)")
    p.add_argument("--live-discover-full", action="store_true",
                   help="With --live-discover: use comprehensive mode "
                        "(~2600 Serper queries, slow + expensive)")
    p.add_argument("--save-discovered", action="store_true",
                   help="Save newly discovered companies to src/companies.py "
                        "(use together with --live-discover)")
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
