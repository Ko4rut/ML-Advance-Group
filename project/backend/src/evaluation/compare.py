"""Side-by-side comparison of two tournament runs (e.g. ``default`` vs ``equal_time``).

    python -m src tournament-compare RUN_A RUN_B

Reads only ``manifest.json`` and ``games.jsonl`` of each run (like ``tournament-report``)
and recomputes Elo/ranks with the same :class:`~src.evaluation.metrics.MetricsCollector`.
Elo is anchored at ``random`` = 0 inside each run, so it is comparable between runs as a
*relative* strength within that run's field of opponents; ranks and rank changes are the
more robust reading when the two runs have different agents or budgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import report
from .metrics import MetricsCollector, fmt_num


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Standings of one run plus the facts needed to label it."""
    run_dir = Path(run_dir)
    manifest = report.load_manifest(run_dir)
    records, _notes = report.load_games(run_dir)
    config = manifest.get("config", {}) or {}
    collector = MetricsCollector(
        agents=list((manifest.get("agents") or {}).keys()),
        master_seed=int(config.get("master_seed", 0)),
        openings=list(config.get("openings", [])),
        anchor="random",
    )
    collector.add_matches(records)
    summary = collector.summarize()
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "budget_mode": report.budget_mode_of(manifest),
        "time_per_move": manifest.get("time_per_move"),
        "completed": len(records),
        "total": manifest.get("total_games", len(records)),
        "config_name": manifest.get("name"),
        "standings": {row["agent"]: row for row in summary["standings"]},
        "order": list(summary["agents"]),
    }


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """Per-agent Elo and rank in both runs, and their differences (B minus A)."""
    a, b = summarize_run(run_a), summarize_run(run_b)
    names = list(a["order"]) + [name for name in b["order"] if name not in a["standings"]]
    rows = []
    for name in names:
        row_a, row_b = a["standings"].get(name), b["standings"].get(name)
        elo_a = row_a["elo"] if row_a else None
        elo_b = row_b["elo"] if row_b else None
        rank_a = row_a["rank"] if row_a else None
        rank_b = row_b["rank"] if row_b else None
        rows.append({
            "agent": name,
            "elo_a": elo_a, "elo_a_low": row_a["elo_low"] if row_a else None,
            "elo_a_high": row_a["elo_high"] if row_a else None, "rank_a": rank_a,
            "elo_b": elo_b, "elo_b_low": row_b["elo_low"] if row_b else None,
            "elo_b_high": row_b["elo_high"] if row_b else None, "rank_b": rank_b,
            "delta_elo": None if elo_a is None or elo_b is None else elo_b - elo_a,
            "delta_rank": None if rank_a is None or rank_b is None else rank_b - rank_a,
        })
    rows.sort(key=lambda r: (r["rank_a"] is None, r["rank_a"] or 0, r["agent"]))
    return {"a": a, "b": b, "rows": rows}


def _elo(value: float | None, low: float | None, high: float | None) -> str:
    if value is None:
        return "–"
    return f"{fmt_num(value, 0)} [{fmt_num(low, 0)}; {fmt_num(high, 0)}]"


def _signed(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "–"
    text = fmt_num(abs(value), digits)
    if float(text) == 0:
        return text
    return ("+" if value > 0 else "-") + text


def _label(run: dict[str, Any]) -> str:
    mode = run["budget_mode"]
    if mode == "equal_time":
        mode += f" (T={run['time_per_move']:g}s)"
    return f"{run['run_id']}  [{mode}]  {run['completed']}/{run['total']} games"


def format_comparison(comparison: dict[str, Any]) -> str:
    a, b = comparison["a"], comparison["b"]
    lines = [f"A: {_label(a)}", f"B: {_label(b)}", ""]
    headers = ["agent", "rank A", "Elo A [95% CI]", "rank B", "Elo B [95% CI]", "d rank", "d Elo (B-A)"]
    table = [headers]
    for r in comparison["rows"]:
        table.append([
            r["agent"],
            "–" if r["rank_a"] is None else str(r["rank_a"]), _elo(r["elo_a"], r["elo_a_low"], r["elo_a_high"]),
            "–" if r["rank_b"] is None else str(r["rank_b"]), _elo(r["elo_b"], r["elo_b_low"], r["elo_b_high"]),
            _signed(r["delta_rank"]), _signed(r["delta_elo"]),
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    for index, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                               for i, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    lines.append("")
    lines.append("Rank = standing by score (T counted as half a point). A negative d rank means the "
                 "agent moved up in B. Elo is anchored at random = 0 in each run, so compare it as relative "
                 "strength within the field; overlapping 95% intervals are not evidence of a change.")
    if a["completed"] < a["total"] or b["completed"] < b["total"]:
        lines.append("WARNING: at least one run is incomplete; its numbers are provisional.")
    return "\n".join(lines)
