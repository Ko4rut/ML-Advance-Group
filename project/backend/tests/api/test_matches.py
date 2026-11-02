"""Run: python -m unittest discover -s tests -v (from project/backend)."""

import unittest

from fastapi.testclient import TestClient

from project.backend.src.api.main import create_app

class MatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def create_match(self, red="human", black="human"):
        def participant(agent_id):
            return {"type": "human"} if agent_id == "human" else {"type": "agent", "agent_id": agent_id}
        result = self.client.post("/api/v1/matches", json={"red": participant(red), "black": participant(black)})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def test_initial_snapshot_contains_all_legal_moves(self):
        match = self.create_match()
        self.assertEqual(len(match["board"]), 90)
        self.assertEqual(match["board"][0], -5)
        self.assertEqual(match["board"][81], 5)
        self.assertEqual(match["side_to_move"], "red")
        self.assertEqual(match["ply"], 0)
        self.assertEqual(match["status"], "playing")
        self.assertEqual(len(match["legal_moves"]), 44)
        self.assertIn({"source": 54, "target": 45}, match["legal_moves"])
        mid = match["match_id"]
        filtered = self.client.get(f"/api/v1/matches/{mid}/legal-moves", params={"source": 54})
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["legal_moves"], [{"source": 54, "target": 45}])
        self.assertEqual(self.client.get(f"/api/v1/matches/{mid}").json()["board"], match["board"])

    def test_human_move_and_stale_ply_rejected(self):
        match = self.create_match()
        mid = match["match_id"]
        endpoint = f"/api/v1/matches/{mid}/moves"
        moved = self.client.post(endpoint, json={"source": 54, "target": 45, "expected_ply": 0})
        self.assertEqual(moved.status_code, 200, moved.text)
        state = moved.json()
        self.assertEqual(state["ply"], 1)
        self.assertEqual(state["side_to_move"], "black")
        self.assertEqual(state["board"][54], 0)
        self.assertEqual(state["board"][45], 7)
        self.assertEqual(state["last_move"], {"source": 54, "target": 45})
        stale = self.client.post(endpoint, json={"source": 54, "target": 45, "expected_ply": 0})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.client.get(f"/api/v1/matches/{mid}").json()["ply"], 1)

    def test_illegal_move_does_not_mutate_and_invalid_shape_rejected(self):
        match = self.create_match()
        mid = match["match_id"]
        endpoint = f"/api/v1/matches/{mid}/moves"
        self.assertEqual(self.client.post(endpoint, json={"source": 54, "target": 55, "expected_ply": 0}).status_code, 422)
        self.assertEqual(self.client.post(endpoint, json={"source": 90, "target": 0, "expected_ply": 0}).status_code, 422)
        self.assertEqual(self.client.post(endpoint, json={"source": "54", "target": 45, "expected_ply": 0}).status_code, 422)
        self.assertEqual(self.client.get(f"/api/v1/matches/{mid}").json()["board"], match["board"])

    def test_agent_catalog_and_agent_step(self):
        agents = {a["id"]: a for a in self.client.get("/api/v1/agents").json()}
        self.assertTrue(agents["random"]["available"])
        self.assertTrue(agents["rules"]["available"])
        self.assertTrue(agents["astar"]["available"])
        self.assertTrue(agents["minimax"]["available"])
        self.assertTrue(agents["alphabeta"]["available"])
        self.assertFalse(agents["dqn"]["available"])
        match = self.create_match(red="random")
        mid = match["match_id"]
        self.assertEqual(self.client.post(f"/api/v1/matches/{mid}/moves", json={"source": 54, "target": 45, "expected_ply": 0}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 1}).status_code, 409)
        result = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 0})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], match["legal_moves"])
        self.assertEqual(result.json()["ply"], 1)
        self.assertEqual(self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 1}).status_code, 409)

    def test_minimax_agent_can_play_a_move(self):
        match = self.create_match(red="minimax")
        mid = match["match_id"]

        result = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 0})

        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], match["legal_moves"])
        self.assertEqual(result.json()["ply"], 1)

    def test_alpha_beta_agent_can_play_a_move(self):
        match = self.create_match(red="alphabeta")
        mid = match["match_id"]

        result = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 0})

        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn(result.json()["last_move"], match["legal_moves"])
        self.assertEqual(result.json()["ply"], 1)

    def test_rule_agent_reset_missing_match_and_cors(self):
        match = self.create_match(red="rules", black="random")
        mid = match["match_id"]
        response = self.client.post(f"/api/v1/matches/{mid}/step", json={"expected_ply": 0})
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["ply"], 1)
        reset = self.client.post(f"/api/v1/matches/{mid}/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["ply"], 0)
        self.assertEqual(reset.json()["last_move"], None)
        self.assertEqual(reset.json()["board"], match["board"])
        self.assertEqual(self.client.get("/api/v1/matches/not-found").status_code, 404)
        cors = self.client.options("/api/v1/matches", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        self.assertEqual(cors.status_code, 200)
        self.assertEqual(cors.headers.get("access-control-allow-origin"), "http://localhost:3000")

if __name__ == "__main__":
    unittest.main()