import { AGENTS } from "../constants";
import type { AgentId, Board, Side } from "../types";

export function initialBoard(): Board {
  const board: Board = Array.from({ length: 90 }, function emptySquare() {
    return null;
  });
  const backRank = [5, 4, 3, 2, 1, 2, 3, 4, 5];

  for (let col = 0; col < backRank.length; col += 1) {
    const kind = backRank[col];
    board[col] = { side: "top", kind };
    board[9 * 9 + col] = { side: "bottom", kind };
  }

  for (const col of [1, 7]) {
    board[2 * 9 + col] = { side: "top", kind: 6 };
    board[7 * 9 + col] = { side: "bottom", kind: 6 };
  }

  for (const col of [0, 2, 4, 6, 8]) {
    board[3 * 9 + col] = { side: "top", kind: 7 };
    board[6 * 9 + col] = { side: "bottom", kind: 7 };
  }

  return board;
}

/** The UI puts red on top; the Python engine puts red on the bottom. Flip rows only. */
export function toDisplaySquare(engineIndex: number): number {
  return (9 - Math.floor(engineIndex / 9)) * 9 + (engineIndex % 9);
}

export function toEngineSquare(displayIndex: number): number {
  return toDisplaySquare(displayIndex);
}

export function boardFromEngine(squares: number[]): Board {
  if (squares.length !== 90) {
    throw new Error("API trả về bàn cờ không đủ 90 ô.");
  }

  return squares.map(function convertSquare(_code, displayIndex) {
    const code = squares[toEngineSquare(displayIndex)];
    if (code === 0) {
      return null;
    }

    let side: Side = "bottom";
    if (code > 0) {
      side = "top";
    }
    return { side, kind: Math.abs(code) };
  });
}

export function getAgentName(id: AgentId, side: Side): string {
  if (id === "human") {
    if (side === "top") {
      return "Player 1";
    }
    return "Player 2";
  }

  const agent = AGENTS.find(function matchesAgent(entry) {
    return entry.id === id;
  });
  if (!agent) {
    return id;
  }
  return agent.label;
}

