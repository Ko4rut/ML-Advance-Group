export type Participant =
  | { type: "human"; agent_id?: null }
  | { type: "agent"; agent_id: string };

export type LegalMove = {
  source: number;
  target: number;
};

export type AgentAvailability = {
  id: string;
  name: string;
  available: boolean;
  unavailable_reason?: string | null;
};

export type MatchSnapshot = {
  match_id: string;
  red: Participant;
  black: Participant;
  board: number[];
  side_to_move: "red" | "black";
  ply: number;
  status: "playing" | "finished";
  winner: "red" | "black" | null;
  reason: string | null;
  last_move: LegalMove | null;
  legal_moves: LegalMove[];
};

