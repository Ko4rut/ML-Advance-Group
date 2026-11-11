"""Pure statistics: Wilson, percentile, classification, Bradley-Terry Elo, collector."""

import math
import random
import tempfile
import unittest
from pathlib import Path

from src.evaluation import metrics
from src.evaluation.metrics import (
    MetricsCollector, classify_result, elo_ratings, fmt_num, game_score, percentile, rank_values,
    wilson_interval,
)


def record(index, a, b, a_color, status="finished", winner=None, *, fingerprint=None, plies=30,
           moves=(), opening="o1", family="fam", error=None):
    """Minimal games.jsonl-like record. ``winner`` is "A", "B" or None."""
    red, black = (a, b) if a_color == "red" else (b, a)
    win_side = None
    if winner == "A":
        win_side = a_color
    elif winner == "B":
        win_side = "black" if a_color == "red" else "red"
    rec = {
        "game_index": index, "game_key": f"{a}|{b}|{opening}|{a_color}", "pair": [a, b],
        "red_agent": red, "black_agent": black, "a_color": a_color,
        "opening": {"id": opening, "family": family, "name": opening},
        "outcome": {"status": status, "winner": win_side, "reason": "x"},
        "plies": plies, "fingerprint": fingerprint or f"fp{index}", "moves": list(moves),
    }
    if error:
        rec["error"] = error
    return rec


class WilsonTests(unittest.TestCase):
    def test_hand_computed_value(self):
        low, high = wilson_interval(25, 50)
        self.assertAlmostEqual(low, 0.366, places=3)
        self.assertAlmostEqual(high, 0.634, places=3)

    def test_symmetry_and_bounds(self):
        low, high = wilson_interval(10, 40)
        low2, high2 = wilson_interval(30, 40)
        self.assertAlmostEqual(low, 1 - high2)
        self.assertAlmostEqual(high, 1 - low2)
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_fractional_successes_allowed(self):
        low, high = wilson_interval(12.5, 25)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


class PercentileTests(unittest.TestCase):
    def test_linear_interpolation(self):
        data = [1, 2, 3, 4]
        self.assertEqual(percentile(data, 0), 1)
        self.assertEqual(percentile(data, 100), 4)
        self.assertAlmostEqual(percentile(data, 50), 2.5)
        self.assertAlmostEqual(percentile(data, 25), 1.75)
        self.assertAlmostEqual(percentile([5, 1, 3], 50), 3)  # unsorted input

    def test_single_value_and_errors(self):
        self.assertEqual(percentile([7], 95), 7)
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1], 101)


class ClassificationTests(unittest.TestCase):
    def test_codes_from_each_agent_point_of_view(self):
        win = record(0, "a", "b", "red", winner="A")
        self.assertEqual(classify_result(win, "a"), "W")
        self.assertEqual(classify_result(win, "b"), "L")
        self.assertIsNone(classify_result(win, "c"))
        black_win = record(1, "a", "b", "black", winner="A")
        self.assertEqual(classify_result(black_win, "a"), "W")
        self.assertEqual(classify_result(record(2, "a", "b", "red"), "a"), "D")
        self.assertEqual(classify_result(record(3, "a", "b", "red", "truncated"), "b"), "T")
        self.assertEqual(classify_result(record(4, "a", "b", "red", "error"), "a"), "E")

    def test_scores(self):
        self.assertEqual([game_score(c) for c in "WDLT"], [1.0, 0.5, 0.0, 0.5])
        self.assertIsNone(game_score("T", None))
        self.assertIsNone(game_score("E"))

    def test_fmt_num_never_negative_zero(self):
        self.assertEqual(fmt_num(-0.0001, 2), "0.00")
        self.assertEqual(fmt_num(None, 2), "")
        self.assertEqual(fmt_num(1.005, 1), "1.0")

    def test_rank_values_ties_share_rank(self):
        order = ["a", "b", "c", "d"]
        ranks = rank_values({"a": 50.0, "b": 70.0, "c": 50.0, "d": None}, order)
        self.assertEqual(ranks, {"b": 1, "a": 2, "c": 2, "d": 4})


def toy_pair_scores():
    """strong beats mid beats weak, with margins."""
    def scores(wins, draws, losses):
        return [1.0] * wins + [0.5] * draws + [0.0] * losses
    return {
        ("weak", "mid"): scores(5, 5, 40),
        ("weak", "strong"): scores(1, 4, 45),
        ("mid", "strong"): scores(10, 10, 30),
    }


class EloTests(unittest.TestCase):
    AGENTS = ["weak", "mid", "strong"]

    def test_mle_recovers_known_ordering(self):
        elo = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="weak", bootstrap_samples=0)
        self.assertEqual(elo["weak"]["elo"], 0.0)
        self.assertGreater(elo["mid"]["elo"], elo["weak"]["elo"])
        self.assertGreater(elo["strong"]["elo"], elo["mid"]["elo"])

    def test_equal_agents_get_equal_elo(self):
        pair_scores = {
            ("a", "b"): [1.0, 0.0] * 10,                      # a and b split
            ("a", "c"): [1.0] * 15 + [0.0] * 5,
            ("b", "c"): [1.0] * 15 + [0.0] * 5,
        }
        elo = elo_ratings(["a", "b", "c"], pair_scores, anchor="c", bootstrap_samples=0)
        self.assertAlmostEqual(elo["a"]["elo"], elo["b"]["elo"], places=6)
        self.assertGreater(elo["a"]["elo"], 0)

    def test_two_agents_match_closed_form_with_prior(self):
        # 30 wins of 40 plus 1 virtual draw: p = (30.5)/(41); Elo = 400*log10(p/(1-p)).
        scores = {("a", "b"): [1.0] * 30 + [0.0] * 10}
        elo = elo_ratings(["a", "b"], scores, anchor="b", bootstrap_samples=0)
        p = 30.5 / 41
        self.assertAlmostEqual(elo["a"]["elo"], 400 * math.log10(p / (1 - p)), places=4)

    def test_total_domination_stays_finite_thanks_to_prior(self):
        scores = {("a", "random"): [1.0] * 50}
        elo = elo_ratings(["a", "random"], scores, bootstrap_samples=0)
        self.assertTrue(math.isfinite(elo["a"]["elo"]))
        self.assertEqual(elo["random"]["elo"], 0.0)
        self.assertGreater(elo["a"]["elo"], 300)

    def test_pair_without_games_still_gets_prior(self):
        scores = {("a", "b"): [1.0] * 10, ("b", "c"): [1.0] * 10}  # a-c never played
        elo = elo_ratings(["a", "b", "c"], scores, anchor="c", bootstrap_samples=0)
        self.assertTrue(all(math.isfinite(v["elo"]) for v in elo.values()))
        self.assertGreater(elo["a"]["elo"], elo["b"]["elo"])
        self.assertGreater(elo["b"]["elo"], elo["c"]["elo"])

    def test_anchor_falls_back_to_first_agent(self):
        elo = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="random", bootstrap_samples=0)
        self.assertEqual(elo["weak"]["elo"], 0.0)

    def test_bootstrap_is_reproducible_and_seed_dependent(self):
        first = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="weak", bootstrap_samples=60, seed=7)
        second = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="weak", bootstrap_samples=60, seed=7)
        other = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="weak", bootstrap_samples=60, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first["strong"]["elo_low"], other["strong"]["elo_low"])

    def test_confidence_interval_brackets_estimate(self):
        elo = elo_ratings(self.AGENTS, toy_pair_scores(), anchor="weak", bootstrap_samples=200, seed=1)
        for name in self.AGENTS:
            row = elo[name]
            self.assertLessEqual(row["elo_low"], row["elo"] + 1e-6)
            self.assertGreaterEqual(row["elo_high"], row["elo"] - 1e-6)
        self.assertEqual((elo["weak"]["elo_low"], elo["weak"]["elo_high"]), (0.0, 0.0))
        self.assertGreater(elo["strong"]["elo_high"], elo["strong"]["elo_low"])


class CollectorTests(unittest.TestCase):
    def build(self):
        collector = MetricsCollector(agents=["random", "rules"], master_seed=5, bootstrap_samples=30)
        moves_a = [
            {"ply": 1, "actor": "opening", "think_seconds": 0.0, "nodes": 0, "simulations": 0},
            {"ply": 2, "actor": "rules", "think_seconds": 0.2, "nodes": 10, "simulations": 0, "supported": True},
            {"ply": 3, "actor": "random", "think_seconds": 0.1, "nodes": 0, "simulations": 0},
            {"ply": 4, "actor": "rules", "think_seconds": 0.4, "nodes": 30, "simulations": 0, "supported": False},
        ]
        collector.add_match(record(0, "random", "rules", "red", winner="B", moves=moves_a, opening="o1"))
        collector.add_match(record(1, "random", "rules", "black", winner="B", opening="o1"))
        collector.add_match(record(2, "random", "rules", "red", "truncated", opening="o2", plies=300))
        collector.add_match(record(3, "random", "rules", "black", winner=None, opening="o2"))
        collector.add_match(record(4, "random", "rules", "red", "error", opening="o3",
                                   error={"type": "ValueError", "message": "boom\nline2", "ply": 12, "actor": "rules"}))
        return collector

    def test_tallies_exclude_errors_from_scores(self):
        summary = self.build().summarize()
        rows = {r["agent"]: r for r in summary["standings"]}
        rules = rows["rules"]
        self.assertEqual((rules["W"], rules["D"], rules["L"], rules["T"], rules["E"]), (2, 1, 0, 1, 1))
        self.assertEqual(rules["games"], 5)
        self.assertAlmostEqual(rules["points"], 2 + 0.5 + 0.5)
        self.assertAlmostEqual(rules["score_pct"], 100 * 3 / 4)
        self.assertAlmostEqual(rules["score_pct_excl_T"], 100 * 2.5 / 3)
        self.assertEqual(rows["rules"]["rank"], 1)
        self.assertEqual(rows["random"]["rank"], 2)

    def test_pair_row_and_conclusion(self):
        pair = self.build().summarize()["pairs"][0]
        self.assertEqual((pair["A"], pair["B"], pair["games"], pair["E"]), ("random", "rules", 5, 1))
        self.assertAlmostEqual(pair["score"], 1.0)  # draw 0.5 + truncated 0.5
        self.assertEqual(pair["_valid"], 4)
        self.assertEqual(pair["conclusion"], "inconclusive")
        self.assertEqual(pair["distinct_games"], 4)  # error game excluded
        self.assertAlmostEqual(pair["distinct_ratio"], 1.0)

    def test_clear_winner_conclusion_and_distinct_ratio(self):
        collector = MetricsCollector(agents=["a", "b"], bootstrap_samples=0)
        for i in range(40):
            collector.add_match(record(i, "a", "b", "red" if i % 2 == 0 else "black",
                                       winner="A", fingerprint="same" if i < 4 else None))
        pair = collector.summarize()["pairs"][0]
        self.assertEqual(pair["conclusion"], "A_stronger")
        self.assertEqual(pair["distinct_games"], 37)
        self.assertLess(pair["distinct_ratio"], 1)
        flipped = MetricsCollector(agents=["a", "b"], bootstrap_samples=0)
        for i in range(40):
            flipped.add_match(record(i, "a", "b", "red", winner="B"))
        self.assertEqual(flipped.summarize()["pairs"][0]["conclusion"], "B_stronger")

    def test_cost_counts_only_agent_moves(self):
        cost = {r["agent"]: r for r in self.build().summarize()["cost"]}
        self.assertEqual(cost["rules"]["moves"], 2)
        self.assertAlmostEqual(cost["rules"]["think_mean_s"], 0.3)
        self.assertAlmostEqual(cost["rules"]["think_median_s"], 0.3)
        self.assertAlmostEqual(cost["rules"]["think_p95_s"], 0.2 + 0.2 * 0.95)
        self.assertAlmostEqual(cost["rules"]["nodes_mean"], 20)
        self.assertAlmostEqual(cost["rules"]["fallback_rate"], 0.5)
        self.assertEqual(cost["random"]["moves"], 1)
        self.assertIsNone(cost["random"]["fallback_rate"])
        self.assertIsNone(cost["random"]["nodes_mean"])

    def test_colors_openings_errors(self):
        summary = self.build().summarize()
        color = {r["agent"]: r for r in summary["color"]}
        self.assertEqual(color["TOTAL"]["red_games"], 4)
        self.assertEqual(summary["totals"], {"games": 5, "finished": 3, "draws": 1, "truncated": 1, "error": 1})
        openings = {r["opening_id"]: r for r in summary["openings"]}
        self.assertEqual(list(openings), ["o1", "o2", "o3"])
        self.assertAlmostEqual(openings["o1"]["black_win_pct"] + openings["o1"]["red_win_pct"], 100)
        self.assertEqual(openings["o3"]["error_pct"], 100.0)
        errors = summary["errors"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["message"], "boom line2")

    def test_skewed_opening_flagged_at_70_percent(self):
        collector = MetricsCollector(agents=["a", "b"], bootstrap_samples=0)
        # red wins 7 of 10 valid games
        for i in range(10):
            collector.add_match(record(i, "a", "b", "red", winner="A" if i < 7 else "B"))
        self.assertEqual(collector.summarize()["openings"][0]["skew"], "red")

    def test_rank_differs_when_truncation_convention_matters(self):
        collector = MetricsCollector(agents=["x", "y", "z"], bootstrap_samples=0)
        i = 0
        # x: wins 1 of 1 non-truncated, then many truncated; y: wins 3 of 4
        for status, winner, a, b in [("finished", "A", "x", "z"), ("truncated", None, "x", "y"),
                                     ("truncated", None, "x", "y"), ("truncated", None, "x", "y"),
                                     ("finished", "A", "y", "z"), ("finished", "A", "y", "z"),
                                     ("finished", "A", "y", "z"), ("finished", "B", "y", "z")]:
            collector.add_match(record(i, a, b, "red", status, winner))
            i += 1
        summary = collector.summarize()
        self.assertTrue(summary["rank_differs"])

    def test_duplicate_game_index_rejected(self):
        collector = MetricsCollector()
        collector.add_match(record(0, "a", "b", "red"))
        with self.assertRaises(ValueError):
            collector.add_match(record(0, "a", "b", "red"))

    def test_export_is_byte_stable_and_independent_of_insertion_order(self):
        records = [record(i, "a", "b", "red" if i % 2 == 0 else "black",
                          winner=random.Random(i).choice(["A", "B", None])) for i in range(12)]

        def export(items):
            collector = MetricsCollector(agents=["a", "b"], master_seed=3, bootstrap_samples=40)
            collector.add_matches(items)
            with tempfile.TemporaryDirectory() as tmp:
                collector.export(tmp)
                return {p.name: p.read_bytes() for p in Path(tmp).iterdir()}

        first = export(records)
        self.assertEqual(first, export(records))
        self.assertEqual(first, export(list(reversed(records))))
        self.assertNotIn("errors.csv", first)
        self.assertEqual(set(first), {"standings.csv", "pair_results.csv", "crosstable.csv",
                                      "color_breakdown.csv", "cost.csv", "openings.csv"})
        for payload in first.values():
            self.assertNotIn(b"\r", payload)
        self.assertTrue(first["standings.csv"].startswith(b"agent,games,W,D,L,T,E,points,score_pct,"))

    def test_crosstable_is_complementary_with_empty_diagonal(self):
        summary = self.build().summarize()
        table = {r["agent"]: r for r in summary["crosstable"]}
        self.assertIsNone(table["random"]["random"])
        self.assertAlmostEqual(table["random"]["rules"] + table["rules"]["random"], 100.0)


if __name__ == "__main__":
    unittest.main()
