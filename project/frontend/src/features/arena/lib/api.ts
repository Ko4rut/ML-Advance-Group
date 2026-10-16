/**
 * @deprecated Import from api/agents, api/matches, api/moves, or api/types.
 * Kept temporarily for consumers outside the Arena feature.
 */
export { ApiError } from "../api/client";
export { getAgents } from "../api/agents";
export { createMatch, getMatch, resetMatch } from "../api/matches";
export { makeMove, stepAgent } from "../api/moves";
export type {
  AgentAvailability,
  LegalMove,
  MatchSnapshot,
  Participant,
} from "../api/types";

