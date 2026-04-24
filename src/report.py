"""
Report generator: renders matched jobs as a Markdown report + JSON dump.
Supports optional LLM scores (ScoredJob) alongside MatchedJob entries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from filters import MatchedJob, rank_score


def rank_with_score(pair: tuple) -> tuple:
    m, s = pair
    llm_score = s.score if s else 0
    return (llm_score, rank_score(m))


RECOMMENDATION_EMOJI = {"apply": "🟢", "maybe": "🟡", "skip": "🔴"}


def _render_entry(m: MatchedJob, s=None) -> list[str]:
    j = m.job
    out: list[str] = []

    if s:
        emoji = RECOMMENDATION_EMOJI.get(s.recommendation, "⚪")
        out.append(f"### {emoji} [{s.score}/10] {j.title} — {j.company_display}")
    else:
        out.append(f"### {j.title} — {j.company_display}")

    meta_bits = []
    if j.location:    meta_bits.append(f"📍 {j.location}")
    if j.workplace_type: meta_bits.append(f"🏢 {j.workplace_type}")
    if j.commitment:  meta_bits.append(f"⏱ {j.commitment}")
    if j.posted_at:   meta_bits.append(f"🗓 {j.posted_at[:10]}")
    if meta_bits:
        out.append(" · ".join(meta_bits))

    if s:
        out.append(f"_{s.summary}_")
        facts = [
            f"**Erfahrung gefordert:** {s.years_required}",
            f"**Python:** {'✓' if s.python_required else '✗'}",
        ]
        sal_icons = {"passt": "✓", "zu niedrig": "✗", "unklar": "?", "kein Gehalt angegeben": "—"}
        facts.append(f"**Gehalt:** {s.salary_assessment} {sal_icons.get(s.salary_assessment, '?')}")
        out.append("  ".join(facts))
        if s.strengths:
            out.append("**Stärken:** " + " · ".join(s.strengths))
        if s.concerns:
            out.append("**Bedenken:** " + " · ".join(s.concerns))
        if s.python_signal and s.python_signal != "nicht erwähnt":
            out.append(f"_Python-Signal: {s.python_signal[:120]}_")
    else:
        if m.reasons:
            out.append(f"**Match reasons:** {', '.join(m.reasons)}")

    if m.salary_signals:
        parts = []
        for sig in m.salary_signals:
            lo = sig[0]; hi = sig[1] if len(sig) > 1 else None
            parts.append(f"{lo:,}–{hi:,}" if hi else f"{lo:,}+")
        icon = "✓" if m.salary_ok is True else ("✗" if m.salary_ok is False else "")
        out.append(f"**💰 Gehaltsangabe:** {'; '.join(parts)} EUR {icon}".strip())

    out.append(f"**Apply:** {j.url}")
    if not s:
        excerpt = j.description_text[:280].rstrip()
        if excerpt:
            out.append(f"> {excerpt}…")
    out.append(f"_Quelle: {j.source}_")
    out.append("")
    return out


def render_markdown(matches: list[MatchedJob], scored_pairs: list[tuple] | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = ["# Job Scout Report", f"Generated: {now}"]

    if scored_pairs is not None:
        ranked = sorted(scored_pairs, key=rank_with_score, reverse=True)
        apply_c = sum(1 for _, s in ranked if s and s.recommendation == "apply")
        maybe_c = sum(1 for _, s in ranked if s and s.recommendation == "maybe")
        out.append(f"Total: {len(ranked)}  |  🟢 Apply: {apply_c}  |  🟡 Maybe: {maybe_c}")
        out.append("")

        for label, recs in [("🟢 Bewerben", ["apply"]), ("🟡 Vielleicht", ["maybe"]), ("🔴 Skip", ["skip", None])]:
            bucket = [(m, s) for m, s in ranked
                      if (s.recommendation if s else None) in recs]
            if not bucket:
                continue
            out.append(f"## {label} ({len(bucket)})")
            out.append("")
            for m, s in bucket:
                out.extend(_render_entry(m, s))
    else:
        ranked = sorted(matches, key=rank_score, reverse=True)
        out.append(f"Total matches: {len(ranked)}")
        out.append("")
        if not ranked:
            out.append("_No matches found._")
            return "\n".join(out)

        munich, remote_de, remote_emea, other_de, other = [], [], [], [], []
        for m in ranked:
            f = m.location_flags
            if f.get("has_munich"):             munich.append(m)
            elif f.get("is_remote") and f.get("in_germany"): remote_de.append(m)
            elif f.get("in_germany"):           other_de.append(m)
            elif f.get("is_remote"):            remote_emea.append(m)
            else:                               other.append(m)

        for title, bucket in [
            ("Munich / München", munich),
            ("Remote in Germany", remote_de),
            ("Germany (onsite / hybrid)", other_de),
            ("Remote EMEA / Europe", remote_emea),
            ("Other", other),
        ]:
            if not bucket:
                continue
            out.append(f"## {title} ({len(bucket)})")
            out.append("")
            for m in bucket:
                out.extend(_render_entry(m, None))

    return "\n".join(out)


def render_json(matches: list[MatchedJob], scored_pairs: list[tuple] | None = None) -> str:
    if scored_pairs is not None:
        ranked = sorted(scored_pairs, key=rank_with_score, reverse=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(ranked),
            "matches": [{**m.as_dict(), "llm_score": s.as_dict() if s else None} for m, s in ranked],
        }
    else:
        ranked = sorted(matches, key=rank_score, reverse=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(ranked),
            "matches": [m.as_dict() for m in ranked],
        }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def write_reports(matches: list[MatchedJob], out_dir: Path,
                  scored_pairs: list[tuple] | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    md_path  = out_dir / f"report_{ts}.md"
    json_path = out_dir / f"report_{ts}.json"
    md = render_markdown(matches, scored_pairs)
    js = render_json(matches, scored_pairs)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(js, encoding="utf-8")
    (out_dir / "latest.md").write_text(md, encoding="utf-8")
    return md_path, json_path
