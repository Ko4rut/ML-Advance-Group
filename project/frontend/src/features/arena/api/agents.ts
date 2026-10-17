import { requestJson } from "./client";
import type { AgentAvailability } from "./types";

/** GET /agents - which agents are implemented on the backend? */
export function getAgents(): Promise<AgentAvailability[]> {
  return requestJson<AgentAvailability[]>("/agents");
}

