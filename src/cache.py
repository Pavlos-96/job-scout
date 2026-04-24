"""
JSON-based score cache.

Keyed by job URL (stable per posting). Entries expire after TTL days.
File: cache/scores.json (auto-created, safe to delete to reset).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TTL_DAYS = 30


class ScoreCache:
    """
    Transparent LLM score cache backed by a single JSON file.

    Usage:
        cache = ScoreCache(Path("cache/scores.json"))
        hit = cache.get(job_url)   # None on miss / expiry
        cache.set(job_url, scored_job.as_dict())
        cache.save()               # flush to disk once after batch
    """

    def __init__(self, path: Path, ttl_days: int = DEFAULT_TTL_DAYS):
        self.path = path
        self._ttl = timedelta(days=ttl_days)
        self._data: dict[str, dict] = self._load()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------

    def get(self, url: str) -> dict | None:
        """Return cached score dict, or None on miss / expiry."""
        if not url:
            return None
        entry = self._data.get(url)
        if entry is None:
            self._misses += 1
            return None
        cached_at = entry.get("_cached_at")
        if cached_at:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)
                if age > self._ttl:
                    self._misses += 1
                    return None
            except ValueError:
                pass
        self._hits += 1
        return entry

    def set(self, url: str, score: dict):
        """Store score dict under url. Call save() when done."""
        if not url:
            return
        self._data[url] = {
            **score,
            "_cached_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self):
        """Flush all entries to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.debug("score cache saved: %d entries → %s", len(self._data), self.path)

    def evict_expired(self):
        """Remove stale entries. Optional maintenance — not required for correctness."""
        cutoff = datetime.now(timezone.utc) - self._ttl
        before = len(self._data)
        self._data = {
            url: entry for url, entry in self._data.items()
            if self._is_fresh(entry, cutoff)
        }
        removed = before - len(self._data)
        if removed:
            log.info("score cache: evicted %d expired entries", removed)

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        return (
            f"{self._hits}/{total} cache hits  "
            f"({len(self._data)} total entries in {self.path})"
        )

    # ------------------------------------------------------------------

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("score cache load failed (%s): %s — starting fresh", self.path, e)
            return {}

    @staticmethod
    def _is_fresh(entry: dict, cutoff: datetime) -> bool:
        cached_at = entry.get("_cached_at")
        if not cached_at:
            return True
        try:
            return datetime.fromisoformat(cached_at) >= cutoff
        except ValueError:
            return True
