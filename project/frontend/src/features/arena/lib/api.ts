export type Participant = { type: "human"; agent_id?: null } | { type: "agent"; agent_id: string };
export type LegalMove = { source: number; target: number };
export type AgentAvailability = { id: string; name: string; available: boolean };
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

const BASE = (process.env.NEXT_PUBLIC_XIANGQI_API_URL || "http://localhost:8000/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, "Không kết nối được FastAPI. Hãy chạy backend tại localhost:8000 và kiểm tra CORS.");
  }
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => null);
    const detail = data && typeof data === "object" && "detail" in data ? (data as { detail: unknown }).detail : null;
    throw new ApiError(response.status, typeof detail === "string" ? detail : `API lỗi HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

const json = (body: object): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
export const getAgents = () => request<AgentAvailability[]>("/agents");
export const createMatch = (red: Participant, black: Participant) =>
  request<MatchSnapshot>("/matches", json({ red, black }));
export const getMatch = (id: string) => request<MatchSnapshot>(`/matches/${encodeURIComponent(id)}`);
export const makeMove = (id: string, move: LegalMove, expected_ply: number) =>
  request<MatchSnapshot>(`/matches/${encodeURIComponent(id)}/moves`, json({ ...move, expected_ply }));
export const stepAgent = (id: string, expected_ply: number) =>
  request<MatchSnapshot>(`/matches/${encodeURIComponent(id)}/step`, json({ expected_ply }));
export const resetMatch = (id: string) =>
  request<MatchSnapshot>(`/matches/${encodeURIComponent(id)}/reset`, json({}));
