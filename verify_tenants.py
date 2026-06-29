"""
Quick smoke test: sample N tokens per ATS from tenants.json, hit the real
API for each, count how many return live job data. This proves the CC-derived
tokens are real tenants, not noise.
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from discover import (  # noqa: E402
    _probe_greenhouse, _probe_ashby, _probe_personio,
    _probe_recruitee, _probe_smartrecruiters,
)
import httpx

PROBERS = {
    "greenhouse": _probe_greenhouse,
    "ashby": _probe_ashby,
    "personio": _probe_personio,
    "recruitee": _probe_recruitee,
    "smartrecruiters": _probe_smartrecruiters,
}


async def verify_ats(client: httpx.AsyncClient, ats: str,
                     tokens: list[str], sample: int) -> tuple[int, int, list[tuple[str, int]]]:
    """Returns (hits, total_sampled, top_hits)."""
    if not tokens:
        return 0, 0, []
    sampled = random.sample(tokens, min(sample, len(tokens)))
    sem = asyncio.Semaphore(8)
    results: list[tuple[str, int | None]] = []

    async def one(tok: str):
        async with sem:
            n = await PROBERS[ats](client, tok)
            results.append((tok, n))

    await asyncio.gather(*[one(t) for t in sampled])
    # Require at least one job posting -- a 200 with totalFound=0 means the
    # API accepted the name but no real company sits behind it.
    hits = [(t, n) for t, n in results if n is not None and n > 0]
    return len(hits), len(sampled), sorted(hits, key=lambda x: -x[1])[:5]


async def main():
    data = json.loads(Path("tenants.json").read_text())
    print(f"Verifying tenants.json from {data['generated_at']}")
    print()
    sample = 30  # per ATS

    async with httpx.AsyncClient(
        headers={"User-Agent": "job-scout-verify/0.1"},
        follow_redirects=True,
    ) as client:
        for ats, probe in PROBERS.items():
            toks = [t["token"] for t in data["tenants"].get(ats, [])]
            if not toks:
                print(f"{ats:18s} no tokens")
                continue
            t0 = time.time()
            hits, total, top = await verify_ats(client, ats, toks, sample)
            dt = time.time() - t0
            pct = 100 * hits / total if total else 0
            samples_str = ", ".join(f"{t}({n})" for t, n in top)
            print(f"{ats:18s} {hits}/{total} valid  ({pct:.0f}%)  "
                  f"[{dt:.1f}s]  e.g. {samples_str}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
