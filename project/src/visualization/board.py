"""Text rendering for debugging Xiangqi positions and replays."""

from collections.abc import Iterable

from ..environment.state import GameState


PIECE_SYMBOLS = {
    1: "K", 2: "A", 3: "E", 4: "H", 5: "R", 6: "C", 7: "P",
    -1: "k", -2: "a", -3: "e", -4: "h", -5: "r", -6: "c", -7: "p",
}


class BoardRenderer:
    """Render a state as a coordinate-labelled Unicode board."""

    def render(self, state: GameState) -> str:
        rows = ["    " + " ".join(str(col) for col in range(9))]
        for row in range(10):
            pieces = state.board[row * 9:(row + 1) * 9]
            rows.append(f"{row:>2}  " + " ".join(PIECE_SYMBOLS.get(piece, ".") for piece in pieces))
            if row == 4:
                rows.append("    -------- RIVER --------")
        rows.append(
            f"Turn: {state.side_to_move.name} | "
            f"Ply: {state.ply_count} | "
            f"No capture: {state.no_capture_count}"
        )
        return "\n".join(rows)

    def render_replay(self, states: Iterable[GameState]) -> str:
        return "\n\n".join(
            f"Move {index}\n{self.render(state)}"
            for index, state in enumerate(states)
        )
