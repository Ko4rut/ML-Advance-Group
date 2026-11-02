export type Side = "top" | "bottom";
export type AgentId = "human" | "random" | "rules" | "astar" | "minimax" | "alphabeta" | "mcts" | "neural" | "dqn";
export type Agent = { id: AgentId; label: string; symbol: string; hint: string };

export type Piece = { side: Side; kind: number };
export type Board = Array<Piece | null>;
