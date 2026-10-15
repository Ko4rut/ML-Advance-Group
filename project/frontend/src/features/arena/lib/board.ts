import { AGENTS } from "../constants";
import type { AgentId, Board, Side } from "../types";

export function initialBoard(): Board {
  const board: Board = Array.from({ length: 90 }, () => null);
  const backRank = [5, 4, 3, 2, 1, 2, 3, 4, 5];
  backRank.forEach((kind, col) => {
    board[col] = { side: "top", kind };
    board[9 * 9 + col] = { side: "bottom", kind };
  });
  [1, 7].forEach((col) => {
    board[2 * 9 + col] = { side: "top", kind: 6 };
    board[7 * 9 + col] = { side: "bottom", kind: 6 };
  });
  [0, 2, 4, 6, 8].forEach((col) => {
    board[3 * 9 + col] = { side: "top", kind: 7 };
    board[6 * 9 + col] = { side: "bottom", kind: 7 };
  });
  return board;
}


export function getAgentName(id: AgentId, side: Side): string {
  return id === "human" ? (side === "top" ? "Player 1" : "Player 2") : AGENTS.find((agent) => agent.id === id)!.label;
}
