# Arena UX: move indicators and pre-match state

The match-integration UI uses an explicit board marker layer inside
`ChessBoard.tsx`. It is separate from the clickable 90 square buttons:
each legal destination has an absolutely positioned marker at the same
coordinate as its board square, and `pointer-events: none` lets the click
reach the underlying square.

- **Empty destination:** a dark teal dot with a light outline.
- **Capturable destination:** a restrained teal ring around the enemy piece.
- **Current selection:** thin teal outline (separate from legal markers).
- **No match yet:** board previews the start position under a visibly dimmed
  overlay saying "Chưa tạo trận đấu", with a working create-match button.
- **After creation:** overlay disappears and legal moves appear on selecting
  a piece of the side to move.
- The existing `width` and desktop/mobile board dimensions are unchanged.

To verify:
1. Start FastAPI, start Next.js, open Arena. The overlay and create button
   must be visible even before creating any match.
2. Create Human vs Human. Select the red pawn on the top row of pawns;
   its forward destination should have a visible dot.
3. Play a legal red move, then select a black cannon on the bottom half.
   The status line reports the number of legal moves and all empty legal
   destinations show dots.
4. In a position with a legal capture, the target enemy piece gets a thin ring.
5. Check mobile sizing and verify indicators do not intercept clicks.

If status shows a positive legal move count but no dots, inspect the DOM for
`.board-legal-moves .board-move-marker` and the matching element positions.
The marker DOM is generated directly from the authoritative snapshot's
`legal_moves`, not computed by CSS or by duplicating backend chess rules.
