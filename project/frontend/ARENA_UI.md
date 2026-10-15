# Xiangqi Arena frontend (UI-only MVP)

The arena UI is organized by feature under `src/features/arena`:

- `types.ts`: shared types for players, sides, and board state.
- `constants.ts`: the seven AI agent choices and piece asset definitions.
- `lib/board.ts`: deterministic opening board preview and agent display names.
- `hooks/useArenaPreview.ts`: local preview interactions and UI state (replace with API-backed state later).
- `components/`: focused presentational components for the board, agent selection, sidebar, header, matchup, match controls, statistics, and modal.
- `src/app/page.tsx`: small composition layer for the arena screen.

## Features
- Dark/light responsive minimalist dashboard.
- Choose any of seven AI agents or a human for each side.
- Xiangqi 9×10 board from `public/assets/pieces`, now larger on desktop with larger pieces.
- Click pieces for selection only; no simulated rule enforcement.
- Local preview create/reset and explicit empty-state statistics.

## Important
No backend/API calls, legal move validation, AI turns, real win rates, or game persistence have been implemented. The Start Battle button reports the integration requirement rather than simulating a result.

## API follow-up
1. Add `GET /agents` and `POST /matches`.
2. Load authoritative board state and legal moves from `XiangqiEnv`.
3. Add `POST /matches/{id}/moves` for human actions and `POST /matches/{id}/step` (or a WebSocket) for AI turns.
4. Feed backend match statistics into the overview and statistics dialog.

From `project/frontend`, run `npm ci`, then `npm run dev`.
