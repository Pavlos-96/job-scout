# Job Scout

**Automated job discovery and AI-powered relevance scoring for AI/ML Engineering roles in Germany and the EU.**

Fetches job postings directly from company ATS APIs (Greenhouse, Lever, Personio, Ashby) — bypassing LinkedIn — and scores each one with GPT-4o-mini against a detailed candidate profile. Matched roles appear in a local web UI where you can review scores, hide irrelevant postings, track applications, and generate tailored cover letters with Claude.

---

## Features

- **Multi-ATS fetching** — Parallel HTTP requests to 270+ companies across Greenhouse, Lever, Personio, and Ashby (US + EU endpoints)
- **LLM scoring** — GPT-4o-mini rates each role 1–10 with an `apply / maybe / skip` recommendation based on your profile and strict exclusion rules (no lead roles, no pure infra, no non-technical)
- **Smart filtering** — title regex, seniority detection, EU location classification, 365-day freshness window, US work-authorization detection
- **Web UI** — FastAPI + Jinja2 + HTMX, no JavaScript framework; shows scores, salaries, dates; hide jobs, mark applied
- **Cover letter generation** — Claude 3.5 Sonnet drafts a cover letter matching the job description and your writing style
- **Company discovery** — Serper-powered Google Dork queries find new companies and their ATS tokens automatically

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12, fully async |
| HTTP | `httpx` (async, connection pooling) |
| Web framework | FastAPI + Jinja2 + HTMX |
| Database | SQLite via `aiosqlite` |
| LLM scoring | OpenAI `gpt-4o-mini` |
| Cover letters | Anthropic `claude-3-5-sonnet` |
| Company discovery | Serper API (Google Dorks) |

---

## How It Works

```
                    ┌─────────────────────────────────────┐
                    │         270+ Companies               │
                    │  (src/companies.py)                  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │    Async ATS Fetchers                │
                    │  Greenhouse · Lever · Personio       │
                    │  Ashby  (US + EU endpoints)          │
                    └──────────────┬──────────────────────┘
                                   │  ~17,000 raw jobs
                    ┌──────────────▼──────────────────────┐
                    │    Deterministic Pre-Filter          │
                    │  title regex · location · seniority  │
                    │  age (365d) · US-auth detection      │
                    └──────────────┬──────────────────────┘
                                   │  ~50–80 candidates
                    ┌──────────────▼──────────────────────┐
                    │    GPT-4o-mini Scorer                │
                    │  score 1–10 + apply/maybe/skip       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         SQLite DB                    │
                    │  jobs · scores · cover letters       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Web UI (localhost:8000)        │
                    │  review · hide · apply · cover letter│
                    └─────────────────────────────────────┘
```

---

## Setup

**1. Clone and create a virtual environment:**

```bash
git clone https://github.com/YOUR_USERNAME/job-scout.git
cd job-scout
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure API keys:**

```bash
cp .env.example .env
# Edit .env and add your keys
```

Required keys in `.env`:

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...   # optional, only needed for cover letters
SERPER_API_KEY=...              # optional, only needed for company discovery
```

---

## Usage

### Web UI (recommended)

```bash
python web.py
# Opens http://localhost:8000
```

From the UI you can trigger a full pipeline run (fetch → filter → score), browse results sorted by score, hide irrelevant jobs, mark applications, and generate cover letters.

### CLI

```bash
# Full run: fetch all companies, filter, score, save to DB
python run.py

# Skip the LLM scoring step (faster, just filtering)
python run.py --no-score

# Disable the 365-day age filter (show all dates)
python run.py --no-post-year-filter

# Also include jobs with no posted date
python run.py --exclude-undated-jobs
```

### Company Discovery

Finds new companies on Greenhouse/Lever/Personio/Ashby using Google Dorks:

```bash
python serper_discover.py
```

Results are printed as `companies.py`-compatible entries you can paste directly into `src/companies.py`.

---

## Project Structure

```
job-scout/
├── run.py                  # CLI entry point
├── web.py                  # FastAPI web application
├── write_cover_letter.py   # Claude-based cover letter generator
├── serper_discover.py      # Company discovery via Serper + Google Dorks
├── requirements.txt
├── .env.example
├── src/
│   ├── companies.py        # 270+ companies with ATS type and token
│   ├── fetchers.py         # ATS adapters (Greenhouse US+EU, Lever, Personio, Ashby)
│   ├── filters.py          # Deterministic pre-filter (title, location, age)
│   ├── scorer.py           # GPT-4o-mini scoring with structured output
│   ├── db.py               # SQLite schema, CRUD, migrations
│   ├── cache.py            # HTTP response cache
│   └── report.py           # Markdown/JSON report generation
├── templates/
│   ├── jobs.html           # Main job list view (HTMX)
│   ├── job_detail.html     # Single job detail
│   └── ...
└── static/
    └── style.css
```

---

## Design Decisions

**Direct ATS API access instead of LinkedIn** — Most job boards aggregate the same postings. Going directly to the source via public ATS APIs means you see jobs before aggregators index them, and with far less competition from other applicants.

**Deterministic pre-filter before LLM** — With 17,000+ raw jobs per run, calling GPT on everything would be slow and expensive. A regex + location filter narrows it down to ~50–80 candidates first; the LLM only sees those.

**Rolling 365-day freshness window** — Calendar-year filtering would drop valid late-2025 postings that are only months old. A rolling window keeps anything recent while removing truly stale listings.

**HTMX over a SPA framework** — The UI is server-rendered Jinja2 with HTMX for inline updates (hide job, mark applied, load cover letter). No build step, no bundler, no JS framework.

---

## Customizing for Your Profile

Edit the `CANDIDATE_PROFILE` constant in `src/scorer.py` to match your background. The scoring prompt instructs the model to rate fit against that profile with explicit hard exclusion rules (management-first, pure infra, non-technical).

To add companies, append to `src/companies.py`:

```python
{"ats": "greenhouse", "token": "acme-corp", "name": "Acme Corp",
 "tier": "scaleup", "verified": False, "notes": "Berlin; AI platform team"},
```

The `token` is the identifier from the company's ATS URL:

| ATS | URL pattern | Token |
|---|---|---|
| Greenhouse | `job-boards.greenhouse.io/TOKEN` | `TOKEN` |
| Lever | `jobs.lever.co/TOKEN/...` | `TOKEN` |
| Personio | `TOKEN.jobs.personio.de` | `TOKEN` |
| Ashby | `jobs.ashbyhq.com/TOKEN` | `TOKEN` |

---

## Limitations

- Only works with companies that use one of the four supported ATS platforms. Workday, SAP SuccessFactors, Taleo, and in-house systems are not covered.
- LLM scoring is English/German only and tuned for AI/ML engineering roles in Germany and the EU. Adapt the prompt in `src/scorer.py` for other markets or roles.
- Serper API has a request quota; discovery runs should be used sparingly.
