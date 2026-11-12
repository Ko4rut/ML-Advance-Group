"""Tournament agents must be configured exactly like the agents the web API serves.

Only the RNG seeds may differ (the API leaves them unset / fixed; the tournament
derives one per game so results are reproducible and independent).
"""

import os
import unittest
from unittest import mock

try:
    from src.api.models.match import ParticipantModel
    from src.api.services.matches import MatchService
    from src.api.services.mcts import MCTSService
except ImportError:  # API extras (fastapi, PyYAML, dotenv) not installed
    MatchService = None

from src.environment import XiangqiEnv
from src.evaluation import agent_registry as registry
from src.evaluation import tournament as t

# tournament agent name -> id used by the API catalogue
API_IDS = {
    "random": "random", "rules": "rules", "alphabeta": "alphabeta",
    "heuristic": "astar", "mcts": "mcts", "tabular_q": "tabular_q",
}
SEED_KEYS = {"seed", "_rng", "rng", "last_action_supported"}
SKIP_KEYS = {"engine", "last_root", "last_nodes", "last_duration_seconds", "last_cost",
             "last_score", "table", "last_simulations", "last_elapsed", "eval_fn"}


def settings(obj):
    """Public scalar settings of an agent/search component, minus RNG state."""
    return {
        key: (value if isinstance(value, (int, float, str, bool, type(None))) else type(value).__name__)
        for key, value in vars(obj).items()
        if not key.startswith("_") and key not in SEED_KEYS | SKIP_KEYS
    }


@unittest.skipIf(MatchService is None, "API dependencies not installed")
class ApiParityTests(unittest.TestCase):
    def setUp(self):
        checkpoint = t.BACKEND_ROOT / "models" / "tabular_q" / "q_table.json.gz"
        if not checkpoint.is_file():
            self.skipTest("Tabular Q checkpoint not present")
        env_patch = mock.patch.dict(os.environ, {"XIANGQI_TABULAR_Q_CHECKPOINT": str(checkpoint)})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.service = MatchService()
        self.config = t.resolve_config(t.load_raw_config(t.BACKEND_ROOT / "configs" / "tournament.yaml"))

    def test_every_agent_is_built_like_the_api_builds_it(self):
        for name, api_id in API_IDS.items():
            with self.subTest(agent=name):
                api_agent = self.service._make_agent(
                    ParticipantModel(type="agent", agent_id=api_id), XiangqiEnv())
                mine = registry.make_agent(
                    name, self.config.data["agents"][name], XiangqiEnv().engine, 12345)
                self.assertIs(type(mine), type(api_agent))
                self.assertEqual(settings(mine), settings(api_agent))
                if name == "mcts":
                    for part in ("selection", "evaluator"):
                        self.assertEqual(settings(getattr(mine.search, part)),
                                         settings(getattr(api_agent.search, part)))
                if name == "tabular_q":
                    self.assertIs(type(mine.fallback_agent), type(api_agent.fallback_agent))
                    self.assertEqual(len(mine.table.values), len(api_agent.table.values))

    def test_budgets_match_what_the_api_passes(self):
        """API: select_action(state) with no budget, except MCTS (time_limit from its yaml)."""
        for name in ("random", "rules", "alphabeta", "heuristic", "tabular_q"):
            self.assertIsNone(registry.budget_for(name, self.config.data["agents"][name]), name)
        api = MCTSService()._settings()
        mcts = registry.budget_for("mcts", self.config.data["agents"]["mcts"])
        self.assertEqual(mcts.time_limit, api.get("time_limit", 1.0))
        self.assertEqual(mcts.max_simulations, api.get("max_simulations"))


if __name__ == "__main__":
    unittest.main()
