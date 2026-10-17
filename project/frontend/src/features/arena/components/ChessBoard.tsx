
import Image from "next/image";
import type { LegalMove } from "../api/types";
import { PIECE_FILES, PIECE_NAMES } from "../constants";
import type { Board, Piece } from "../types";

function squarePosition(index: number) {
  const col = index % 9;
  const row = Math.floor(index / 9);
  return {
    left: `${((40 + col * 65) / 600) * 100}%`,
    top: `${((40 + row * 64) / 656) * 100}%`,
  };
}

function BoardGrid() {
  const horizontal = Array.from({ length: 10 }, function rowNumber(_value, index) {
    return index;
  });
  const vertical = Array.from({ length: 9 }, function columnNumber(_value, index) {
    return index;
  });

  return (
    <svg
      className="board-lines"
      viewBox="0 0 600 656"
      role="img"
      aria-label="Lưới bàn cờ tướng 9 cột, 10 hàng, có sông ở giữa"
    >
      <rect x="18" y="18" width="564" height="620" rx="5" fill="none" stroke="currentColor" strokeWidth="2" opacity=".22" />
      {horizontal.map(function renderRow(index) {
        return <line key={`row-${index}`} x1="40" x2="560" y1={40 + index * 64} y2={40 + index * 64} stroke="currentColor" strokeWidth="1.3" />;
      })}
      {vertical.map(function renderColumn(index) {
        return (
          <g key={`col-${index}`}>
            <line x1={40 + index * 65} x2={40 + index * 65} y1="40" y2="296" stroke="currentColor" strokeWidth="1.3" />
            <line x1={40 + index * 65} x2={40 + index * 65} y1="360" y2="616" stroke="currentColor" strokeWidth="1.3" />
          </g>
        );
      })}
      <line x1="40" y1="296" x2="40" y2="360" stroke="currentColor" strokeWidth="1.3" />
      <line x1="560" y1="296" x2="560" y2="360" stroke="currentColor" strokeWidth="1.3" />
      <path d="M235 40 365 168M365 40 235 168M235 488 365 616M365 488 235 616" stroke="currentColor" strokeWidth="1.3" fill="none" />
      <text x="169" y="333" textAnchor="middle" dominantBaseline="middle" className="river-label">楚 河</text>
      <text x="427" y="333" textAnchor="middle" dominantBaseline="middle" className="river-label">漢 界</text>
    </svg>
  );
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

type SquareProps = {
  index: number;
  piece: Piece | null;
  selection: number | null;
  legal: boolean;
  last: boolean;
  disabled: boolean;
  onSquare: (index: number) => void;
};

function BoardSquare({ index, piece, selection, legal, last, disabled, onSquare }: SquareProps) {
  const col = index % 9;
  const row = Math.floor(index / 9);
  let classes = "board-intersection";
  let accessibleName = `Ô trống hàng ${row + 1} cột ${col + 1}`;
  let pieceImage: React.ReactNode = null;

  if (selection === index) {
    classes += " board-intersection--selected";
  }
  if (piece) {
    classes += " has-piece";
    let color = "black";
    let sideLabel = "đen";
    if (piece.side === "top") {
      color = "red";
      sideLabel = "đỏ";
    }
    accessibleName = `${PIECE_NAMES[piece.kind]} ${sideLabel}, hàng ${row + 1} cột ${col + 1}`;
    pieceImage = (
      <Image
        unoptimized
        src={`/assets/pieces/${color}-${PIECE_FILES[piece.kind]}.png`}
        alt=""
        width={72}
        height={72}
        draggable={false}
      />
    );
  }
  if (legal) {
    if (piece) {
      classes += " board-intersection--capture";
      accessibleName += ", có thể ăn";
    } else {
      classes += " board-intersection--legal";
      accessibleName += ", có thể đi";
    }
  }
  if (last) {
    classes += " board-intersection--last";
  }

  function handleSquareClick(): void {
    onSquare(index);
  }

  return (
    <button
      type="button"
      disabled={disabled}
      className={classes}
      style={squarePosition(index)}
      onClick={handleSquareClick}
      aria-label={accessibleName}
      aria-pressed={selection === index}
    >
      {pieceImage}
    </button>
  );
}

export function ChessBoard({
  board,
  selection,
  legalTargets,
  lastMove,
  disabled = false,
  preview = false,
  onCreateMatch,
  onSquare,
}: Props) {
  let boardCanvasClass = "board-canvas";
  if (preview) {
    boardCanvasClass += " board-canvas--preview";
  }
  let createLabel = "Tạo trận đấu";
  if (disabled) {
    createLabel = "Đang tạo trận...";
  }

  return (
    <div className="board-frame">
      <div className={boardCanvasClass}>
        <BoardGrid />
        {board.map(function renderSquare(piece, index) {
          const last = lastMove?.source === index || lastMove?.target === index;
          return (
            <BoardSquare
              key={index}
              index={index}
              piece={piece}
              selection={selection}
              legal={legalTargets.has(index)}
              last={last}
              disabled={disabled}
              onSquare={onSquare}
            />
          );
        })}
        {!preview && legalTargets.size > 0 && (
          <div className="board-legal-moves" aria-hidden="true">
            {Array.from(legalTargets).map(function renderMarker(index) {
              let markerClass = "board-move-marker board-move-marker--empty";
              if (board[index]) {
                markerClass = "board-move-marker board-move-marker--capture";
              }
              return <span key={index} className={markerClass} style={squarePosition(index)} />;
            })}
          </div>
        )}
        {preview && (
          <div className="board-start-overlay">
            <div className="board-start-card" role="status">
              <span className="board-start-kicker">XIANGQI ARENA</span>
              <strong>Chưa tạo trận đấu</strong>
              <p>Chọn hai đối thủ ở trên và dưới, sau đó tạo trận để bắt đầu chơi.</p>
              <button type="button" onClick={onCreateMatch} disabled={disabled || !onCreateMatch}>
                {createLabel}
                <span aria-hidden="true">→</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
