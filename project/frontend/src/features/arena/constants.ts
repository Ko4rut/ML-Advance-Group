import type { Agent } from "./types";

export const AGENTS: Agent[] = [
  { id: "human", label: "Player", symbol: "user", hint: "Human player" },
  { id: "random", label: "Random", symbol: "✳", hint: "Random policy" },
  { id: "rules", label: "Rule-Based", symbol: "≡", hint: "Handcrafted rules" },
  { id: "astar", label: "A*", symbol: "A*", hint: "Shortest predicted win" },
  { id: "minimax", label: "Minimax", symbol: "M", hint: "Game tree search" },
  { id: "alphabeta", label: "Alpha-Beta", symbol: "αβ", hint: "Pruned search" },
  { id: "mcts", label: "MCTS", symbol: "⌘", hint: "Monte Carlo search" },
  { id: "neural", label: "Neural MCTS", symbol: "N", hint: "Neural-guided search" },
  { id: "dqn", label: "DQN", symbol: "D", hint: "Deep Q-network" },
];

export const PIECE_NAMES: Record<number, string> = {
  1: "Tướng", 2: "Sĩ", 3: "Tượng", 4: "Mã", 5: "Xe", 6: "Pháo", 7: "Tốt",
};
export const PIECE_FILES: Record<number, string> = {
  1: "tuong", 2: "si", 3: "tuongj", 4: "ma", 5: "xe", 6: "phao", 7: "tot",
};
