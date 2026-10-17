import { postJson, requestJson } from "./client";
import type { MatchSnapshot, Participant } from "./types";

/** POST /matches - create a new server-authoritative match. */
export function createMatch(red: Participant, black: Participant): Promise<MatchSnapshot> {
  return postJson<MatchSnapshot>("/matches", { red, black });
}

/** GET /matches/{id} - refresh after conflicts or reconnects. */
export function getMatch(matchId: string): Promise<MatchSnapshot> {
  return requestJson<MatchSnapshot>(`/matches/${encodeURIComponent(matchId)}`);
}

/** POST /matches/{id}/reset - reset an existing match. */
export function resetMatch(matchId: string): Promise<MatchSnapshot> {
  return postJson<MatchSnapshot>(`/matches/${encodeURIComponent(matchId)}/reset`, {});
}

