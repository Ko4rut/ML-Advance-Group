"""Interactive Streamlit board for manually exercising the environment."""

import sys
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# Streamlit executes this file as a script, so ``project`` is not always on
# sys.path. Add it explicitly before importing the local ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.backend.src.environment import Move, Side, XiangqiEnv, is_in_check


ASSET_DIR = PROJECT_ROOT / "assets" / "pieces"
PIECE_ASSETS = {
    1: "red-tuong.png", 2: "red-si.png", 3: "red-tuongj.png",
    4: "red-ma.png", 5: "red-xe.png", 6: "red-phao.png", 7: "red-tot.png",
    -1: "black-tuong.png", -2: "black-si.png", -3: "black-tuongj.png",
    -4: "black-ma.png", -5: "black-xe.png", -6: "black-phao.png", -7: "black-tot.png",
}

BOARD_WIDTH = 720
BOARD_HEIGHT = 796
MARGIN = 56
SPACING = 76
PIECE_SIZE = 68


def piece_side(piece: int) -> Side | None:
    if piece > 0:
        return Side.RED
    if piece < 0:
        return Side.BLACK
    return None


def reset_game() -> None:
    st.session_state.env = XiangqiEnv()
    st.session_state.selected = None
    st.session_state.last_move = None
    st.session_state.notice = "Đỏ đi trước. Chọn một quân để xem các nước hợp lệ."


def select_square(index: int) -> None:
    env = st.session_state.env
    state = env.state
    selected = st.session_state.selected
    legal_moves = env.get_legal_actions()

    if selected is not None:
        move = Move(selected, index)
        if move in legal_moves:
            result = env.step(move)
            st.session_state.last_move = move
            st.session_state.selected = None
            if result.terminated:
                winner = "Hòa" if result.result.winner is None else result.result.winner.name
                st.session_state.notice = f"Kết thúc: {result.result.reason}. Kết quả: {winner}."
            else:
                st.session_state.notice = f"Đã đi {selected} → {index}."
            return

    piece = state.board[index]
    if piece_side(piece) is state.side_to_move:
        st.session_state.selected = index
        count = sum(move.source == index for move in legal_moves)
        st.session_state.notice = f"Đã chọn ô {index}; có {count} nước hợp lệ."
    elif selected is not None:
        st.session_state.notice = "Ô đích này không hợp lệ."
    else:
        st.session_state.notice = "Hãy chọn quân của phe đang đến lượt."


@lru_cache(maxsize=None)
def load_piece(filename: str) -> Image.Image:
    return Image.open(ASSET_DIR / filename).convert("RGBA").resize(
        (PIECE_SIZE, PIECE_SIZE), Image.Resampling.LANCZOS
    )


def center(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * SPACING, MARGIN + row * SPACING


def draw_marker(draw: ImageDraw.ImageDraw, index: int, color: str, width: int = 5) -> None:
    row, col = divmod(index, 9)
    x, y = center(row, col)
    radius = PIECE_SIZE // 2 + 3
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def render_board_image() -> Image.Image:
    state = st.session_state.env.state
    selected = st.session_state.selected
    last_move = st.session_state.last_move
    legal_targets = {
        move.target
        for move in st.session_state.env.get_legal_actions()
        if move.source == selected
    }

    board = Image.new("RGB", (BOARD_WIDTH, BOARD_HEIGHT), "#e3b96d")
    draw = ImageDraw.Draw(board)

    # Subtle wood grain.
    for x in range(0, BOARD_WIDTH, 16):
        draw.line((x, 0, x + 40, BOARD_HEIGHT), fill="#ddaF63", width=2)

    line_color = "#603918"
    for row in range(10):
        x1, y = center(row, 0)
        x2, _ = center(row, 8)
        draw.line((x1, y, x2, y), fill=line_color, width=4)
    for col in range(9):
        x, y1 = center(0, col)
        _, y4 = center(4, col)
        _, y5 = center(5, col)
        _, y9 = center(9, col)
        if col in (0, 8):
            draw.line((x, y1, x, y9), fill=line_color, width=4)
        else:
            draw.line((x, y1, x, y4), fill=line_color, width=4)
            draw.line((x, y5, x, y9), fill=line_color, width=4)

    # Palace diagonals.
    draw.line((*center(0, 3), *center(2, 5)), fill=line_color, width=4)
    draw.line((*center(0, 5), *center(2, 3)), fill=line_color, width=4)
    draw.line((*center(7, 3), *center(9, 5)), fill=line_color, width=4)
    draw.line((*center(7, 5), *center(9, 3)), fill=line_color, width=4)

    river_y = (center(4, 0)[1] + center(5, 0)[1]) // 2
    draw.text((200, river_y - 10), "CHU HA", fill=line_color, anchor="mm", stroke_width=1)
    draw.text((520, river_y - 10), "HAN GIOI", fill=line_color, anchor="mm", stroke_width=1)

    if last_move is not None:
        draw_marker(draw, last_move.source, "#1870b8", 4)
        draw_marker(draw, last_move.target, "#1870b8", 4)
    if selected is not None:
        draw_marker(draw, selected, "#f4c400", 7)
    for index in legal_targets:
        row, col = divmod(index, 9)
        x, y = center(row, col)
        radius = 10 if state.board[index] == 0 else PIECE_SIZE // 2 + 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#20a447" if state.board[index] == 0 else None,
            outline="#20a447",
            width=6,
        )

    for index, piece in enumerate(state.board):
        if piece == 0:
            continue
        row, col = divmod(index, 9)
        x, y = center(row, col)
        image = load_piece(PIECE_ASSETS[piece])
        board.paste(image, (x - PIECE_SIZE // 2, y - PIECE_SIZE // 2), image)

    return board


def clicked_square(point: dict[str, int] | None) -> int | None:
    if not point:
        return None
    col = round((point["x"] - MARGIN) / SPACING)
    row = round((point["y"] - MARGIN) / SPACING)
    if 0 <= row < 10 and 0 <= col < 9:
        return row * 9 + col
    return None


st.set_page_config(page_title="Xiangqi Environment", page_icon="♟️", layout="centered")
if "env" not in st.session_state:
    reset_game()

st.title("Cờ Tướng – Environment Viewer")


@st.fragment
def game_view() -> None:
    """Rerun only the interactive game area when the board is clicked."""
    env = st.session_state.env
    state = env.state
    legal_moves = env.get_legal_actions()

    left, middle, right = st.columns([2, 2, 1])
    left.metric("Lượt", "ĐỎ" if state.side_to_move is Side.RED else "ĐEN")
    middle.metric("Số nước", state.ply_count)
    if right.button("Ván mới", use_container_width=True):
        reset_game()
        st.rerun(scope="fragment")

    if is_in_check(state, state.side_to_move):
        st.warning(f"{state.side_to_move.name} đang bị chiếu!")
    st.info(st.session_state.notice)
    st.caption("Đen ở phía trên · Đỏ ở phía dưới")

    board_key = f"board-{state.ply_count}-{st.session_state.selected}"
    point = streamlit_image_coordinates(render_board_image(), width=BOARD_WIDTH, key=board_key)
    square = clicked_square(point)
    if square is not None:
        select_square(square)
        st.rerun(scope="fragment")

    with st.expander("Thông tin trạng thái"):
        st.write({
            "side_to_move": state.side_to_move.name,
            "ply_count": state.ply_count,
            "no_capture_count": state.no_capture_count,
            "legal_moves": len(legal_moves),
        })


game_view()
