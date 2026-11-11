"""Statistics for round-robin tournaments, computed from ``games.jsonl`` records.

Everything here is pure Python (stdlib only, no numpy/scipy) and deterministic:
the same records always give byte-identical CSV output.

Conventions
-----------
* Per game, from one agent's point of view: ``W`` win, ``D`` draw, ``L`` loss,
  ``T`` truncated (hit ``max_plies``, *not* a draw) and ``E`` error.
* Win = 1 point, draw = 0.5, loss = 0.
* ``T`` is reported two ways: ``T_half`` (worth 0.5 point, the "main" figure)
  and ``excl_T`` (truncated games removed from the computation).
* ``E`` games are removed from every score computation and listed separately.
* Wilson intervals and Elo are computed on the valid (non-error) games.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BOOTSTRAP_SAMPLES = 1000
TRUNCATED_VALUE = 0.5  # score of a truncated game in the "T_half" convention
SKEW_THRESHOLD = 0.70  # an opening is "skewed" if one color wins >= 70% of valid games
FINISHED_CODES = ("W", "D", "L")

STANDINGS_COLUMNS = (
    "agent", "games", "W", "D", "L", "T", "E", "points", "score_pct", "score_pct_excl_T",
    "rank", "rank_excl_T", "elo", "elo_low", "elo_high",
)
PAIR_COLUMNS = (
    "A", "B", "games", "W", "D", "L", "T", "E", "score", "score_pct", "score_pct_excl_T",
    "wilson_low", "wilson_high", "A_as_red_score", "A_as_black_score", "avg_plies",
    "distinct_games", "distinct_ratio", "conclusion",
)
COLOR_COLUMNS = (
    "agent", "red_games", "red_points", "red_score_pct",
    "black_games", "black_points", "black_score_pct", "red_minus_black_pct",
)
COST_COLUMNS = (
    "agent", "moves", "think_mean_s", "think_median_s", "think_p95_s", "think_total_s",
    "nodes_mean", "simulations_mean", "fallback_rate",
)
OPENING_COLUMNS = (
    "opening_id", "family", "name", "games", "red_win_pct", "draw_pct", "black_win_pct",
    "truncated_pct", "error_pct", "skew",
)
ERROR_COLUMNS = (
    "game_index", "game_key", "red_agent", "black_agent", "opening_id", "ply", "actor",
    "error_type", "message",
)

# Digits for float columns (anything not listed uses DEFAULT_DIGITS).
DEFAULT_DIGITS = 4
COLUMN_DIGITS = {
    "points": 1, "score_pct": 2, "score_pct_excl_T": 2, "elo": 1, "elo_low": 1, "elo_high": 1,
    "score": 1, "wilson_low": 2, "wilson_high": 2, "A_as_red_score": 1, "A_as_black_score": 1,
    "avg_plies": 1, "distinct_ratio": 4, "red_points": 1, "black_points": 1,
    "red_score_pct": 2, "black_score_pct": 2, "red_minus_black_pct": 2,
    "think_mean_s": 4, "think_median_s": 4, "think_p95_s": 4, "think_total_s": 2,
    "nodes_mean": 1, "simulations_mean": 1, "fallback_rate": 4,
    "red_win_pct": 2, "draw_pct": 2, "black_win_pct": 2, "truncated_pct": 2, "error_pct": 2,
}


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def fmt_num(value: Any, digits: int) -> str:
    """Fixed-format number; ``None`` -> empty; never ``-0.00``."""
    if value is None:
        return ""
    text = f"{float(value):.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (fractions in [0, 1]).

    ``successes`` may be fractional (a draw counts 0.5), which makes this an
    approximation. ``n == 0`` returns the uninformative interval (0, 1).
    """
    if n <= 0:
        return 0.0, 1.0
    successes = min(max(float(successes), 0.0), float(n))
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile (0..100) with linear interpolation between ranks."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def classify_result(record: Mapping[str, Any], agent: str) -> str | None:
    """W/D/L/T/E for ``agent`` in this game, or ``None`` if it did not play."""
    if agent == record["red_agent"]:
        side = "red"
    elif agent == record["black_agent"]:
        side = "black"
    else:
        return None
    outcome = record["outcome"]
    status = outcome["status"]
    if status == "error":
        return "E"
    if status == "truncated":
        return "T"
    winner = outcome["winner"]
    if winner is None:
        return "D"
    return "W" if winner == side else "L"


def game_score(code: str, truncated_value: float | None = TRUNCATED_VALUE) -> float | None:
    """Points for a result code; ``None`` means "leave this game out"."""
    if code == "W":
        return 1.0
    if code == "D":
        return 0.5
    if code == "L":
        return 0.0
    if code == "T":
        return truncated_value
    return None  # E


def rank_values(values: Mapping[str, float | None], order: Sequence[str]) -> dict[str, int]:
    """Competition ranking (1,2,2,4) by descending value; ``None`` sorts last."""
    def key(name: str) -> tuple[int, float, int]:
        value = values[name]
        return (1, 0.0, order.index(name)) if value is None else (0, -value, order.index(name))

    ranks: dict[str, int] = {}
    previous: float | None = None
    previous_rank = 0
    for position, name in enumerate(sorted(order, key=key), start=1):
        value = values[name]
        if value is not None and previous is not None and abs(value - previous) < 1e-9:
            ranks[name] = previous_rank
        else:
            ranks[name] = position
            previous_rank = position
        previous = value
    return ranks


# --------------------------------------------------------------------------
# Bradley-Terry / Elo
# --------------------------------------------------------------------------


def bradley_terry(
    points: Sequence[Sequence[float]],
    games: Sequence[Sequence[float]],
    *,
    start: Sequence[float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> list[float]:
    """Bradley-Terry strengths by the MM algorithm (Hunter 2004).

    ``points[i][j]`` is what ``i`` scored against ``j`` (draw = 0.5) and
    ``games[i][j]`` the number of games between them (symmetric). Strengths are
    normalised to geometric mean 1 after each iteration. Every agent must have
    positive points and positive games overall (the caller adds a prior).
    """
    n = len(points)
    strengths = [1.0] * n if start is None else [float(value) for value in start]
    totals = [math.fsum(points[i][j] for j in range(n) if j != i) for i in range(n)]
    for _ in range(max_iter):
        updated = []
        for i in range(n):
            denominator = math.fsum(
                games[i][j] / (strengths[i] + strengths[j]) for j in range(n) if j != i and games[i][j] > 0
            )
            updated.append(totals[i] / denominator)
        log_mean = math.fsum(math.log(value) for value in updated) / n
        scale = math.exp(log_mean)
        updated = [value / scale for value in updated]
        change = max(abs(math.log(new / old)) for new, old in zip(updated, strengths))
        strengths = updated
        if change < tol:
            break
    return strengths


def strengths_to_elo(strengths: Sequence[float], anchor: int) -> list[float]:
    """``Elo = 400 * log10(strength)`` relative to the anchor (anchor = 0)."""
    base = math.log10(strengths[anchor])
    return [400.0 * (math.log10(value) - base) for value in strengths]


def _build_matrices(
    n_agents: int,
    pair_scores: Mapping[tuple[int, int], Sequence[float]],
    prior: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Points/games matrices with ``prior`` virtual draw(s) for EVERY pair."""
    points = [[0.0] * n_agents for _ in range(n_agents)]
    games = [[0.0] * n_agents for _ in range(n_agents)]
    for i, j in itertools.combinations(range(n_agents), 2):
        scores = pair_scores.get((i, j), ())
        total = math.fsum(scores)
        count = len(scores)
        points[i][j] = total + 0.5 * prior
        points[j][i] = (count - total) + 0.5 * prior
        games[i][j] = games[j][i] = count + prior
    return points, games


def elo_ratings(
    agents: Sequence[str],
    pair_scores: Mapping[tuple[str, str], Sequence[float]],
    *,
    anchor: str = "random",
    prior: float = 1.0,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Elo (Bradley-Terry MLE with a light prior) and bootstrap 95% intervals.

    ``pair_scores[(A, B)]`` lists A's score (1/0.5/0) in each valid game of the
    pair. The prior adds ``prior`` virtual draw(s) to every pair, even pairs
    without games, so ratings stay finite when someone wins every game. The
    bootstrap resamples games *within each pair* with replacement and uses
    ``random.Random(seed)``, so it is reproducible.
    """
    names = list(agents)
    n = len(names)
    if n == 0:
        return {}
    index = {name: i for i, name in enumerate(names)}
    anchor_index = index.get(anchor, 0)
    indexed = {
        (index[a], index[b]): list(scores) for (a, b), scores in pair_scores.items()
        if a in index and b in index and index[a] < index[b]
    }
    for (a, b), scores in pair_scores.items():  # tolerate pairs given as (later, earlier)
        if a in index and b in index and index[a] > index[b]:
            indexed[(index[b], index[a])] = [1.0 - value for value in scores]

    points, games = _build_matrices(n, indexed, prior)
    base = bradley_terry(points, games)
    base_elo = strengths_to_elo(base, anchor_index)

    rng = random.Random(seed)
    replicates: list[list[float]] = [[] for _ in range(n)]
    pair_order = sorted(indexed)
    for _ in range(bootstrap_samples if n > 1 else 0):
        resampled: dict[tuple[int, int], list[float]] = {}
        for pair in pair_order:
            scores = indexed[pair]
            resampled[pair] = rng.choices(scores, k=len(scores)) if scores else []
        sample_points, sample_games = _build_matrices(n, resampled, prior)
        elo = strengths_to_elo(
            bradley_terry(sample_points, sample_games, start=base, tol=1e-8), anchor_index)
        for i in range(n):
            replicates[i].append(elo[i])

    result: dict[str, dict[str, float]] = {}
    for name in names:
        i = index[name]
        if replicates[i]:
            low, high = percentile(replicates[i], 2.5), percentile(replicates[i], 97.5)
        else:
            low = high = base_elo[i]
        result[name] = {"elo": base_elo[i], "elo_low": low, "elo_high": high}
    return result


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------


class MetricsCollector:
    """Aggregates ``games.jsonl`` records into the tables of the report.

    ``add_match`` takes one record, ``summarize`` returns all tables as lists
    of dicts (raw numbers) and ``export`` writes them as CSV files.
    """

    def __init__(
        self,
        agents: Sequence[str] | None = None,
        *,
        master_seed: int = 0,
        bootstrap_samples: int = BOOTSTRAP_SAMPLES,
        anchor: str = "random",
        openings: Sequence[str] | None = None,
    ) -> None:
        self._agents_hint = list(agents) if agents else []
        self.master_seed = master_seed
        self.bootstrap_samples = bootstrap_samples
        self.anchor = anchor
        self._openings_hint = list(openings) if openings else []
        self._records: dict[int, Mapping[str, Any]] = {}
        self._summary: dict[str, Any] | None = None

    def add_match(self, result: Mapping[str, Any]) -> None:
        index = result["game_index"]
        if index in self._records:
            raise ValueError(f"duplicate game_index {index}")
        self._records[index] = result
        self._summary = None

    def add_matches(self, results: Iterable[Mapping[str, Any]]) -> None:
        for result in results:
            self.add_match(result)

    def _ordered(self) -> list[Mapping[str, Any]]:
        return [self._records[index] for index in sorted(self._records)]

    def agent_names(self) -> list[str]:
        names = list(self._agents_hint)
        for record in self._ordered():
            for name in record["pair"]:
                if name not in names:
                    names.append(name)
        return names

    # -- summary -----------------------------------------------------------

    def summarize(self) -> dict[str, Any]:
        if self._summary is None:
            self._summary = self._compute()
        return self._summary

    def _compute(self) -> dict[str, Any]:
        records = self._ordered()
        agents = self.agent_names()
        pairs = list(itertools.combinations(agents, 2))

        tallies = {name: dict.fromkeys("WDLTE", 0) for name in agents}
        for record in records:
            for name in record["pair"]:
                tallies[name][classify_result(record, name)] += 1

        # Per-pair scores of A's valid games (T counts 0.5), in game_index order.
        pair_games: dict[tuple[str, str], list[tuple[Mapping[str, Any], str]]] = {p: [] for p in pairs}
        for record in records:
            key = tuple(record["pair"])
            if key in pair_games:
                pair_games[key].append((record, classify_result(record, key[0])))

        # standings (rank needs everything; Elo is added below)
        standings = []
        for name in agents:
            c = tallies[name]
            valid_t = c["W"] + c["D"] + c["L"] + c["T"]
            valid_x = c["W"] + c["D"] + c["L"]
            points = c["W"] + 0.5 * c["D"] + TRUNCATED_VALUE * c["T"]
            points_x = c["W"] + 0.5 * c["D"]
            standings.append({
                "agent": name, "games": valid_t + c["E"],
                "W": c["W"], "D": c["D"], "L": c["L"], "T": c["T"], "E": c["E"],
                "points": points,
                "score_pct": 100.0 * points / valid_t if valid_t else None,
                "score_pct_excl_T": 100.0 * points_x / valid_x if valid_x else None,
            })
        ranks = rank_values({r["agent"]: r["score_pct"] for r in standings}, agents)
        ranks_x = rank_values({r["agent"]: r["score_pct_excl_T"] for r in standings}, agents)
        for row in standings:
            row["rank"] = ranks[row["agent"]]
            row["rank_excl_T"] = ranks_x[row["agent"]]

        # pair table, crosstable
        pair_rows = []
        pair_score_lists: dict[tuple[str, str], list[float]] = {}
        cross: dict[tuple[str, str], float | None] = {}
        for a, b in pairs:
            games = pair_games[(a, b)]
            counts = dict.fromkeys("WDLTE", 0)
            scores: list[float] = []
            red_points = black_points = 0.0
            red_games = black_games = 0
            plies: list[int] = []
            fingerprints = set()
            for record, code in games:
                counts[code] += 1
                score = game_score(code)
                if score is None:
                    continue
                scores.append(score)
                plies.append(record["plies"])
                fingerprints.add(record["fingerprint"])
                if record["a_color"] == "red":
                    red_points += score
                    red_games += 1
                else:
                    black_points += score
                    black_games += 1
            pair_score_lists[(a, b)] = scores
            valid = len(scores)
            points = math.fsum(scores)
            points_x = counts["W"] + 0.5 * counts["D"]
            valid_x = counts["W"] + counts["D"] + counts["L"]
            low, high = wilson_interval(points, valid)
            if valid == 0:
                conclusion = "inconclusive"
            elif low > 0.5:
                conclusion = "A_stronger"
            elif high < 0.5:
                conclusion = "B_stronger"
            else:
                conclusion = "inconclusive"
            pct = 100.0 * points / valid if valid else None
            cross[(a, b)] = pct
            cross[(b, a)] = 100.0 - pct if pct is not None else None
            pair_rows.append({
                "A": a, "B": b, "games": len(games),
                "W": counts["W"], "D": counts["D"], "L": counts["L"], "T": counts["T"], "E": counts["E"],
                "score": points, "score_pct": pct,
                "score_pct_excl_T": 100.0 * points_x / valid_x if valid_x else None,
                "wilson_low": 100.0 * low if valid else None,
                "wilson_high": 100.0 * high if valid else None,
                "A_as_red_score": red_points if red_games else None,
                "A_as_black_score": black_points if black_games else None,
                "avg_plies": mean(plies),
                "distinct_games": len(fingerprints),
                "distinct_ratio": len(fingerprints) / valid if valid else None,
                "conclusion": conclusion,
                "_valid": valid, "_red_games": red_games, "_black_games": black_games,
            })
        crosstable = [
            {"agent": a, **{b: (cross.get((a, b)) if a != b else None) for b in agents}}
            for a in agents
        ]

        # Elo
        elo = elo_ratings(
            agents, pair_score_lists, anchor=self.anchor,
            bootstrap_samples=self.bootstrap_samples, seed=self.master_seed,
        )
        for row in standings:
            row.update(elo.get(row["agent"], {"elo": None, "elo_low": None, "elo_high": None}))

        return {
            "agents": agents,
            "anchor": self.anchor if self.anchor in agents else (agents[0] if agents else None),
            "bootstrap_samples": self.bootstrap_samples,
            "standings": standings,
            "pairs": pair_rows,
            "crosstable": crosstable,
            "color": self._color_rows(records, agents),
            "cost": self._cost_rows(records, agents),
            "openings": self._opening_rows(records),
            "errors": self._error_rows(records),
            "totals": self._totals(records),
            "rank_differs": [r["agent"] for r in standings if r["rank"] != r["rank_excl_T"]],
        }

    @staticmethod
    def _totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        totals = {"games": len(records), "finished": 0, "draws": 0, "truncated": 0, "error": 0}
        for record in records:
            outcome = record["outcome"]
            if outcome["status"] == "finished":
                totals["finished"] += 1
                totals["draws"] += outcome["winner"] is None
            elif outcome["status"] in totals:
                totals[outcome["status"]] += 1
        return totals

    @staticmethod
    def _color_rows(records, agents) -> list[dict[str, Any]]:
        acc = {name: {"red": [0, 0.0], "black": [0, 0.0]} for name in agents}
        total = {"red": [0, 0.0], "black": [0, 0.0]}
        for record in records:
            for side in ("red", "black"):
                name = record[f"{side}_agent"]
                score = game_score(classify_result(record, name))
                if score is None:
                    continue
                acc[name][side][0] += 1
                acc[name][side][1] += score
                total[side][0] += 1
                total[side][1] += score  # score of the side's seat, i.e. first-move advantage for red

        def row(label: str, data) -> dict[str, Any]:
            red_games, red_points = data["red"]
            black_games, black_points = data["black"]
            red_pct = 100.0 * red_points / red_games if red_games else None
            black_pct = 100.0 * black_points / black_games if black_games else None
            return {
                "agent": label, "red_games": red_games, "red_points": red_points,
                "red_score_pct": red_pct, "black_games": black_games, "black_points": black_points,
                "black_score_pct": black_pct,
                "red_minus_black_pct": red_pct - black_pct if None not in (red_pct, black_pct) else None,
            }

        rows = [row(name, acc[name]) for name in agents]
        rows.append(row("TOTAL", total))
        return rows

    @staticmethod
    def _cost_rows(records, agents) -> list[dict[str, Any]]:
        rows = []
        for name in agents:
            thinks: list[float] = []
            nodes: list[float] = []
            sims: list[float] = []
            supported: list[bool] = []
            for record in records:
                for move in record["moves"]:
                    if move["actor"] != name:  # skips "opening" plies and the opponent
                        continue
                    thinks.append(move["think_seconds"])
                    nodes.append(move.get("nodes", 0))
                    sims.append(move.get("simulations", 0))
                    if "supported" in move:
                        supported.append(bool(move["supported"]))
            rows.append({
                "agent": name, "moves": len(thinks),
                "think_mean_s": mean(thinks),
                "think_median_s": percentile(thinks, 50) if thinks else None,
                "think_p95_s": percentile(thinks, 95) if thinks else None,
                "think_total_s": math.fsum(thinks) if thinks else None,
                "nodes_mean": mean(nodes) if any(nodes) else None,
                "simulations_mean": mean(sims) if any(sims) else None,
                "fallback_rate": 1.0 - sum(supported) / len(supported) if supported else None,
            })
        return rows

    def _opening_rows(self, records) -> list[dict[str, Any]]:
        order = list(self._openings_hint)
        info: dict[str, Mapping[str, Any]] = {}
        stats: dict[str, dict[str, int]] = {}
        for record in records:
            opening = record["opening"]
            oid = opening["id"]
            if oid not in info:
                info[oid] = opening
                stats[oid] = {"games": 0, "red": 0, "draw": 0, "black": 0, "truncated": 0, "error": 0}
                if oid not in order:
                    order.append(oid)
            bucket = stats[oid]
            bucket["games"] += 1
            outcome = record["outcome"]
            if outcome["status"] == "finished":
                bucket["draw" if outcome["winner"] is None else outcome["winner"]] += 1
            else:
                bucket[outcome["status"]] += 1
        rows = []
        for oid in order:
            if oid not in stats:
                continue
            b = stats[oid]
            n = b["games"]
            valid = n - b["error"]
            skew = ""
            if valid and b["red"] / valid >= SKEW_THRESHOLD:
                skew = "red"
            elif valid and b["black"] / valid >= SKEW_THRESHOLD:
                skew = "black"
            rows.append({
                "opening_id": oid, "family": info[oid]["family"], "name": info[oid]["name"],
                "games": n,
                "red_win_pct": 100.0 * b["red"] / n, "draw_pct": 100.0 * b["draw"] / n,
                "black_win_pct": 100.0 * b["black"] / n, "truncated_pct": 100.0 * b["truncated"] / n,
                "error_pct": 100.0 * b["error"] / n, "skew": skew,
            })
        return rows

    @staticmethod
    def _error_rows(records) -> list[dict[str, Any]]:
        rows = []
        for record in records:
            if record["outcome"]["status"] != "error":
                continue
            error = record.get("error") or {}
            message = " ".join(str(error.get("message", "")).split())
            rows.append({
                "game_index": record["game_index"], "game_key": record["game_key"],
                "red_agent": record["red_agent"], "black_agent": record["black_agent"],
                "opening_id": record["opening"]["id"], "ply": error.get("ply"),
                "actor": error.get("actor"), "error_type": error.get("type"), "message": message[:300],
            })
        return rows

    # -- export ------------------------------------------------------------

    def export(self, path: str | Path) -> list[Path]:
        """Write the CSV tables into directory ``path``; returns the files written."""
        summary = self.summarize()
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        agents = summary["agents"]
        tables = {
            "standings.csv": (STANDINGS_COLUMNS, summary["standings"]),
            "pair_results.csv": (PAIR_COLUMNS, summary["pairs"]),
            "crosstable.csv": (("agent", *agents), summary["crosstable"]),
            "color_breakdown.csv": (COLOR_COLUMNS, summary["color"]),
            "cost.csv": (COST_COLUMNS, summary["cost"]),
            "openings.csv": (OPENING_COLUMNS, summary["openings"]),
        }
        if summary["errors"]:
            tables["errors.csv"] = (ERROR_COLUMNS, summary["errors"])
        else:
            (directory / "errors.csv").unlink(missing_ok=True)  # never leave a stale file
        written = []
        for filename, (columns, rows) in tables.items():
            target = directory / filename
            write_csv(target, columns, rows, crosstable=filename == "crosstable.csv")
            written.append(target)
        return written


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]], *, crosstable: bool = False
) -> None:
    """Write ``rows`` with fixed number formats and ``\\n`` line endings (byte-stable)."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            cells = []
            for column in columns:
                value = row.get(column)
                if value is None:
                    cells.append("")
                elif isinstance(value, bool):
                    cells.append(str(value))
                elif isinstance(value, int):
                    cells.append(str(value))
                elif isinstance(value, float):
                    digits = 2 if crosstable else COLUMN_DIGITS.get(column, DEFAULT_DIGITS)
                    cells.append(fmt_num(value, digits))
                else:
                    cells.append(str(value))
            writer.writerow(cells)
