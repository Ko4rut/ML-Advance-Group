# Xiangqi Arena frontend (UI-only MVP)

This screen is a frontend-only prototype for `project/frontend`.

## Included
- Minimal dark/light responsive arena layout.
- Select any of seven agent types or a human on either side.
- Render a 9×10 Xiangqi board using the existing `/public/assets/pieces` files.
- Select a board piece for inspection (no move execution without backend).
- Local "Generate match" / "Reset" preview controls; statistics dialog with explicit empty state.

## Not yet connected
- No backend calls, move validation, AI turns, live statistics, persistence, or real game execution.
- The Start Battle action explains which APIs are required; it does not fake a game.

## Next steps
1. Connect `GET /agents` and a `POST /matches` creation endpoint.
2. Fetch authoritative board state + legal moves from `XiangqiEnv`, rather than maintaining a local board.
3. Add `POST /matches/:id/moves` and `POST /matches/:id/step` or WebSocket for AI-vs-AI autoplay.
4. Wire match statistics from real backend results.

Run from `project/frontend` with `npm install` and `npm run dev`.
