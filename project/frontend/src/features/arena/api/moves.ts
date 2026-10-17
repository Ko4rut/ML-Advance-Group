import { postJson } from "./client";
import type { LegalMove, MatchSnapshot } from "./types";

/** POST /matches/{id}/moves - human move with optimistic ply protection. */
export function makeMove(
  matchId: string,
  move: LegalMove,
  expectedPly: number,
): Promise<MatchSnapshot> {
  return postJson<MatchSnapshot>(`/matches/${encodeURIComponent(matchId)}/moves`, {
    ...move,
    expected_ply: expectedPly,
  });
}

/** POST /matches/{id}/step - take exactly one AI turn. */
export function stepAgent(matchId: string, expectedPly: number): Promise<MatchSnapshot> {
  return postJson<MatchSnapshot>(`/matches/${encodeURIComponent(matchId)}/step`, {
    expected_ply: expectedPly,
  });
}

