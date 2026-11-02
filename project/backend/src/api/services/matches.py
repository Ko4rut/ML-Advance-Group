"""Own one XiangqiEnv per match; never duplicate chess rules in HTTP handlers."""

from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from ...agents import (
    Agent,
    AlphaBetaAgent,
    HeuristicAgent,
    MinimaxAgent,
    RandomAgent,
    RuleBasedAgent,
)
from ...environment import Move, Side, XiangqiEnv
from ..models.match import (
    AgentInfo, CreateMatchRequest, LegalMovesResponse, MatchSnapshot,
    MoveModel, MoveRequest, ParticipantModel, StepRequest,
)

class MatchError(Exception):
    """A domain error that the HTTP boundary can map to an appropriate status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail

AGENT_CATALOG = (
    AgentInfo(id="random", name="Random", available=True),
    AgentInfo(id="rules", name="Rule-Based", available=True),
    AgentInfo(id="astar", name="A*", available=True),
    AgentInfo(id="minimax", name="Minimax", available=True),
    AgentInfo(id="alphabeta", name="Alpha-Beta", available=True),
    AgentInfo(id="mcts", name="MCTS", available=False),
    AgentInfo(id="neural", name="Neural MCTS", available=False),
    AgentInfo(id="dqn", name="DQN", available=False),
)

def side_name(side: Side) -> str:
    return "red" if side is Side.RED else "black"

@dataclass(slots=True)
class Match:
    env: XiangqiEnv
    red: ParticipantModel
    black: ParticipantModel
    red_agent: Agent | None
    black_agent: Agent | None
    last_move: Move | None = None

    def participant(self, side: Side) -> ParticipantModel:
        return self.red if side is Side.RED else self.black

    def agent(self, side: Side) -> Agent | None:
        return self.red_agent if side is Side.RED else self.black_agent

class MatchService:
    """In-memory MVP store. A process restart deletes matches; use one worker."""

    def __init__(self) -> None:
        self._matches: dict[str, Match] = {}
        self._lock = RLock()

    def agents(self) -> list[AgentInfo]:
        return list(AGENT_CATALOG)

    @staticmethod
    def _make_agent(participant: ParticipantModel, env: XiangqiEnv) -> Agent | None:
        if participant.type == "human":
            return None
        agent_id = participant.agent_id
        catalog = {agent.id: agent for agent in AGENT_CATALOG}
        if agent_id not in catalog:
            raise MatchError(422, f"unknown agent: {agent_id}")
        if not catalog[agent_id].available:
            raise MatchError(503, f"agent not implemented yet: {agent_id}")
        if agent_id == "random":
            return RandomAgent()
        if agent_id == "rules":
            return RuleBasedAgent(env.engine)
        if agent_id == "astar":
            return HeuristicAgent(env.engine)
        if agent_id == "minimax":
            return MinimaxAgent(env.engine)
        if agent_id == "alphabeta":
            return AlphaBetaAgent(env.engine)
        raise MatchError(503, f"agent not implemented yet: {agent_id}")

    def create(self, request: CreateMatchRequest) -> MatchSnapshot:
        env = XiangqiEnv()
        red_agent = self._make_agent(request.red, env)
        black_agent = self._make_agent(request.black, env)
        match = Match(env, request.red, request.black, red_agent, black_agent)
        with self._lock:
            match_id = uuid4().hex
            self._matches[match_id] = match
            return self._snapshot(match_id, match)

    def get(self, match_id: str) -> MatchSnapshot:
        with self._lock:
            match = self._require(match_id)
            return self._snapshot(match_id, match)

    def legal_moves(self, match_id: str, source: int | None = None) -> LegalMovesResponse:
        with self._lock:
            match = self._require(match_id)
            moves = match.env.get_legal_actions()
            if source is not None:
                moves = [move for move in moves if move.source == source]
            return LegalMovesResponse(
                match_id=match_id,
                ply=match.env.state.ply_count,
                legal_moves=[MoveModel(source=m.source, target=m.target) for m in moves],
            )

    def move(self, match_id: str, request: MoveRequest) -> MatchSnapshot:
        with self._lock:
            match = self._require(match_id)
            self._check_ply(match, request.expected_ply)
            side = match.env.state.side_to_move
            if match.participant(side).type != "human":
                raise MatchError(409, "current turn belongs to an AI agent")
            if match.env.is_terminal():
                raise MatchError(409, "match is already finished")
            action = Move(request.source, request.target)
            if action not in match.env.get_legal_actions():
                raise MatchError(422, "illegal move")
            match.env.step(action)
            match.last_move = action
            return self._snapshot(match_id, match)

    def step(self, match_id: str, request: StepRequest) -> MatchSnapshot:
        with self._lock:
            match = self._require(match_id)
            self._check_ply(match, request.expected_ply)
            if match.env.is_terminal():
                raise MatchError(409, "match is already finished")
            agent = match.agent(match.env.state.side_to_move)
            if agent is None:
                raise MatchError(409, "current turn belongs to a human")
            decision = agent.select_action(match.env.clone_state())
            action = decision.move
            if action not in match.env.get_legal_actions():
                raise MatchError(422, "agent produced an illegal move")
            match.env.step(action)
            match.last_move = action
            return self._snapshot(match_id, match)

    def reset(self, match_id: str) -> MatchSnapshot:
        with self._lock:
            match = self._require(match_id)
            match.env.reset()
            match.last_move = None
            for agent in (match.red_agent, match.black_agent):
                if agent is not None:
                    agent.reset()
            return self._snapshot(match_id, match)

    def _require(self, match_id: str) -> Match:
        match = self._matches.get(match_id)
        if match is None:
            raise MatchError(404, "match not found")
        return match

    @staticmethod
    def _check_ply(match: Match, expected_ply: int) -> None:
        if match.env.state.ply_count != expected_ply:
            raise MatchError(409, "stale match state; fetch the latest snapshot")

    @staticmethod
    def _snapshot(match_id: str, match: Match) -> MatchSnapshot:
        state = match.env.state
        outcome = match.env.engine.outcome(state)
        return MatchSnapshot(
            match_id=match_id,
            red=match.red,
            black=match.black,
            board=list(state.board),
            side_to_move=side_name(state.side_to_move),
            ply=state.ply_count,
            status="finished" if outcome else "playing",
            winner=side_name(outcome.winner) if outcome and outcome.winner is not None else None,
            reason=outcome.reason if outcome else None,
            last_move=MoveModel(source=match.last_move.source, target=match.last_move.target) if match.last_move else None,
            legal_moves=[MoveModel(source=m.source, target=m.target) for m in match.env.get_legal_actions()],
        )