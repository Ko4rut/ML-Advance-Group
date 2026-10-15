import Image from "next/image";
import { PIECE_FILES, PIECE_NAMES } from "../constants";
import type { Board } from "../types";
import type { LegalMove } from "../lib/api";

function squarePosition(index: number) {
  const col = index % 9;
  const row = Math.floor(index / 9);
  return {
    left: `${((40 + col * 65) / 600) * 100}%`,
    top: `${((40 + row * 64) / 656) * 100}%`,
  };
}

function BoardGrid() {
  const horizontal = Array.from({ length: 10 }, (_, n) => n);
  const vertical = Array.from({ length: 9 }, (_, n) => n);
  return <svg className="board-lines" viewBox="0 0 600 656" role="img" aria-label="Lưới bàn cờ tướng 9 cột, 10 hàng, có sông ở giữa">
    <rect x="18" y="18" width="564" height="620" rx="5" fill="none" stroke="currentColor" strokeWidth="2" opacity=".22"/>
    {horizontal.map((n) => <line key={`row-${n}`} x1="40" x2="560" y1={40 + n * 64} y2={40 + n * 64} stroke="currentColor" strokeWidth="1.3"/>)}
    {vertical.map((n) => <g key={`col-${n}`}><line x1={40 + n * 65} x2={40 + n * 65} y1="40" y2="296" stroke="currentColor" strokeWidth="1.3"/><line x1={40 + n * 65} x2={40 + n * 65} y1="360" y2="616" stroke="currentColor" strokeWidth="1.3"/></g>)}
    <line x1="40" y1="296" x2="40" y2="360" stroke="currentColor" strokeWidth="1.3"/><line x1="560" y1="296" x2="560" y2="360" stroke="currentColor" strokeWidth="1.3"/>
    <path d="M235 40 365 168M365 40 235 168M235 488 365 616M365 488 235 616" stroke="currentColor" strokeWidth="1.3" fill="none"/>
    <text x="169" y="333" textAnchor="middle" dominantBaseline="middle" className="river-label">楚 河</text>
    <text x="427" y="333" textAnchor="middle" dominantBaseline="middle" className="river-label">漢 界</text>
  </svg>;
}

type Props = {
  board: Board;
  selection: number | null;
  legalTargets: ReadonlySet<number>;
  lastMove: LegalMove | null;
  disabled?: boolean;
  preview?: boolean;
  onCreateMatch?: () => void;
  onSquare: (index: number) => void;
};

export function ChessBoard({ board, selection, legalTargets, lastMove, disabled, preview = false, onCreateMatch, onSquare }: Props) {
  return <div className="board-frame">
    <div className={`board-canvas ${preview ? "board-canvas--preview" : ""}`}>
      <BoardGrid />
      {board.map((piece, index) => {
        const col = index % 9;
        const row = Math.floor(index / 9);
        const legal = legalTargets.has(index);
        const last = lastMove?.source === index || lastMove?.target === index;
        return <button key={index} type="button"
          disabled={disabled}
          className={`board-intersection ${selection === index ? "board-intersection--selected" : ""} ${piece ? "has-piece" : ""} ${legal ? (piece ? "board-intersection--capture" : "board-intersection--legal") : ""} ${last ? "board-intersection--last" : ""}`}
          style={squarePosition(index)}
          onClick={() => onSquare(index)}
          aria-label={piece ? `${PIECE_NAMES[piece.kind]} ${piece.side === "top" ? "đỏ" : "đen"}, hàng ${row + 1} cột ${col + 1}${legal ? ", có thể ăn" : ""}` : `Ô trống hàng ${row + 1} cột ${col + 1}${legal ? ", có thể đi" : ""}`}
          aria-pressed={selection === index}>
          {piece && <Image unoptimized src={`/assets/pieces/${piece.side === "top" ? "red" : "black"}-${PIECE_FILES[piece.kind]}.png`} alt="" width={72} height={72} draggable={false}/>}
        </button>;
      })}

      {/* A dedicated topmost, non-interactive layer prevents grid-button pseudo-elements
          from hiding behind neighboring buttons or images. */}
      {!preview && legalTargets.size > 0 && (
        <div className="board-legal-moves" aria-hidden="true">
          {Array.from(legalTargets).map((index) => (
            <span
              key={index}
              className={`board-move-marker ${board[index] ? "board-move-marker--capture" : "board-move-marker--empty"}`}
              style={squarePosition(index)}
            />
          ))}
        </div>
      )}

      {preview && (
        <div className="board-start-overlay">
          <div className="board-start-card" role="status">
            <span className="board-start-kicker">XIANGQI ARENA</span>
            <strong>Chưa tạo trận đấu</strong>
            <p>Chọn hai đối thủ ở trên và dưới, sau đó tạo trận để bắt đầu chơi.</p>
            <button type="button" onClick={onCreateMatch} disabled={disabled || !onCreateMatch}>
              {disabled ? "Đang tạo trận..." : "Tạo trận đấu"}
              <span aria-hidden="true">→</span>
            </button>
          </div>
        </div>
      )}
    </div>
  </div>;
}
