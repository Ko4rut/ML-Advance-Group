"use client";

import { useEffect, useState } from "react";
import { PIECE_NAMES } from "../constants";
import { initialBoard } from "../lib/board";
import type { AgentId, Board, Side } from "../types";

export function useArenaPreview() {
  const [top, setTop] = useState<AgentId>("alphabeta");
  const [bottom, setBottom] = useState<AgentId>("mcts");
  const [board, setBoard] = useState<Board>(initialBoard);
  const [selection, setSelection] = useState<number | null>(null);
  const [matchNumber, setMatchNumber] = useState<number>(0);
  const [notice, setNotice] = useState("Chọn hai đối thủ và tạo một trận đấu mới.");
  const [modal, setModal] = useState(false);
  const [lightMode, setLightMode] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setModal(false); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const resetBoard = () => { setBoard(initialBoard()); setSelection(null); setNotice("Bàn cờ đã trở về vị trí ban đầu."); };
  const chooseAgent = (side: Side, id: AgentId) => {
    if (side === "top") setTop(id); else setBottom(id);
    setNotice("Đã cập nhật lựa chọn. Nhấn Tạo trận để xác nhận cặp đấu.");
  };
  const createMatch = () => {
    setBoard(initialBoard()); setSelection(null); setMatchNumber((n) => n + 1);
    setNotice("Đã tạo bản xem trước trận đấu. Cần kết nối API để bắt đầu chơi.");
  };
  const selectSquare = (index: number) => {
    if (!board[index]) { setSelection(null); setNotice("Chưa kết nối game engine. Bàn cờ hiện chỉ dùng để xem và chọn quân."); return; }
    setSelection((current) => current === index ? null : index);
    setNotice(`${PIECE_NAMES[board[index]!.kind]} ${board[index]!.side === "top" ? "đỏ" : "đen"} · Cần API game engine để thực hiện nước đi hợp lệ.`);
  };

  const startBattle = () => setNotice("Chưa thể bắt đầu: cần endpoint tạo trận và endpoint yêu cầu nước đi từ agent.");

  return { top, bottom, board, selection, matchNumber, notice, modal, lightMode, setModal, setLightMode, resetBoard, chooseAgent, createMatch, selectSquare, startBattle };

}
