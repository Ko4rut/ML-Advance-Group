"use client";

import { useState } from "react";
import type { ChangeEvent } from "react";
import type { LegalMove } from "../../arena/api/types";
import { ArenaSidebar } from "../../arena/components/ArenaSidebar";
import { ChessBoard } from "../../arena/components/ChessBoard";
import { Icon } from "../../arena/components/Icon";
import { boardFromEngine, initialBoard, toDisplaySquare } from "../../arena/lib/board";
import type { Board } from "../../arena/types";
import { useGameReplay } from "../hooks/useGameReplay";
import type { ReplayMove, ReplaySide } from "../lib/replay";

const EMPTY_LEGAL_TARGETS = new Set<number>();

function ignoreBoardInteraction(_index: number): void {
  // Historical board states are read-only.
}

function nextSide(side: ReplaySide): ReplaySide {
  if (side === "red") {
    return "black";
  }
  return "red";
}

function sideLabel(side: ReplaySide): string {
  if (side === "red") {
    return "Đỏ";
  }
  return "Đen";
}

export function ReplayViewer() {
  const replay = useGameReplay();
  const [lightMode, setLightMode] = useState(false);

  let siteClass = "site";
  let board: Board = initialBoard();
  let lastMove: LegalMove | null = null;
  let activeSide: ReplaySide = "red";
  let currentMove: ReplayMove | null = null;
  let playbackLabel = "Phát";
  let playbackIcon = "play";
  let fileLabel = "Chưa chọn tệp";

  if (lightMode) {
    siteClass += " site--light";
  }
  if (replay.playing) {
    playbackLabel = "Tạm dừng";
    playbackIcon = "pause";
  }
  if (replay.fileName) {
    fileLabel = replay.fileName;
  }
  if (replay.game) {
    const frame = replay.game.frames[replay.currentPly];
    board = boardFromEngine(frame.board);
    activeSide = replay.game.initialSide;
    if (frame.lastMove) {
      currentMove = frame.lastMove;
      lastMove = {
        source: toDisplaySquare(frame.lastMove.source),
        target: toDisplaySquare(frame.lastMove.target),
      };
    }
    if (replay.currentPly % 2 === 1) {
      activeSide = nextSide(activeSide);
    }
  }

  function toggleTheme(): void {
    setLightMode(function invert(previous) {
      return !previous;
    });
  }

  function handleUpload(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0];
    void replay.loadFile(file);
    event.target.value = "";
  }

  function handleGameSelect(event: ChangeEvent<HTMLSelectElement>): void {
    replay.chooseGame(Number(event.target.value));
  }

  function handleSeek(event: ChangeEvent<HTMLInputElement>): void {
    replay.seek(Number(event.target.value));
  }

  function handleSpeed(event: ChangeEvent<HTMLSelectElement>): void {
    replay.setSpeed(Number(event.target.value));
  }

  let themeLabel = "Bật giao diện sáng";
  let themeIcon = "sun";
  if (lightMode) {
    themeLabel = "Bật giao diện tối";
    themeIcon = "moon";
  }

  return (
    <div className={siteClass}>
      <ArenaSidebar
        activePage="replay"
        onToggleTheme={toggleTheme}
      />
      <main className="workspace">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace <span>/</span> Replay</span>
            <h1>Game replay</h1>
            <p>Tải lịch sử ván cờ và mô phỏng lại từng nước ngay trên trình duyệt.</p>
          </div>
          <div className="topbar-actions">
            <span className="preview-pill"><span /> OFFLINE REPLAY</span>
            <button
              type="button"
              className="icon-button"
              aria-label={themeLabel}
              onClick={toggleTheme}
            >
              <Icon name={themeIcon} size={19} />
            </button>
          </div>
        </header>

        <div className="replay-layout">
          <div className="replay-main">
            <section className="board-section" aria-label="Bàn cờ replay">
              <div className="board-title">
                <span><span className="board-title-icon">✦</span> XIANGQI BOARD</span>
                <span className="board-mode">9 × 10 · Nước {replay.currentPly} / {replay.totalPlies}</span>
              </div>
              <div className="replay-board">
                <ChessBoard
                  board={board}
                  selection={null}
                  legalTargets={EMPTY_LEGAL_TARGETS}
                  lastMove={lastMove}
                  disabled
                  onSquare={ignoreBoardInteraction}
                />
              </div>
              <div className="board-caption" role="status">
                <span className="board-caption-dot" />
                {replay.game
                  ? "Đang xem " + replay.game.label + " · Lượt tiếp theo: " + sideLabel(activeSide)
                  : "Tải một ván cờ để bắt đầu mô phỏng. Bàn cờ hiện ở vị trí khai cuộc."}
              </div>
            </section>

            <section className="panel replay-transport" aria-label="Điều khiển replay">
              <div className="replay-transport-heading">
                <div>
                  <span className="panel-kicker">TIMELINE</span>
                  <h2>Điều khiển phát lại</h2>
                </div>
                <strong className="replay-ply-counter" aria-live="polite">
                  {replay.currentPly} <span>/ {replay.totalPlies}</span>
                </strong>
              </div>
              <input
                className="replay-range"
                type="range"
                min={0}
                max={Math.max(1, replay.totalPlies)}
                step={1}
                value={replay.currentPly}
                aria-label="Tua đến nước đi"
                disabled={!replay.game || replay.totalPlies === 0}
                onChange={handleSeek}
              />
              <div className="replay-transport-bottom">
                <div className="replay-transport-buttons">
                  <button type="button" aria-label="Về đầu" title="Về đầu" disabled={!replay.game || replay.currentPly === 0} onClick={replay.goToStart}>|◀</button>
                  <button type="button" aria-label="Lùi một nước" title="Lùi một nước" disabled={!replay.game || replay.currentPly === 0} onClick={replay.previous}>◀</button>
                  <button
                    type="button"
                    className="replay-play-button"
                    aria-label={playbackLabel}
                    disabled={!replay.game || replay.totalPlies === 0}
                    onClick={replay.togglePlayback}
                  >
                    <Icon name={playbackIcon} size={16} /> {playbackLabel}
                  </button>
                  <button type="button" aria-label="Tới một nước" title="Tới một nước" disabled={!replay.game || replay.currentPly >= replay.totalPlies} onClick={replay.next}>▶</button>
                  <button type="button" aria-label="Đến cuối" title="Đến cuối" disabled={!replay.game || replay.currentPly >= replay.totalPlies} onClick={replay.goToEnd}>▶|</button>
                </div>
                <label className="replay-speed">
                  Tốc độ
                  <select aria-label="Tốc độ phát" value={replay.speed} onChange={handleSpeed}>
                    <option value={0.5}>0.5×</option>
                    <option value={1}>1×</option>
                    <option value={1.5}>1.5×</option>
                    <option value={2}>2×</option>
                  </select>
                </label>
              </div>
              <p className="replay-current-move">
                {currentMove
                  ? "Nước " + currentMove.ply + " · " + sideLabel(currentMove.side) + " · " + currentMove.notation
                  : "Vị trí bắt đầu ván cờ"}
              </p>
            </section>
          </div>

          <aside className="replay-aside" aria-label="Dữ liệu và lịch sử ván cờ">
            <section className="panel replay-upload-panel">
              <div className="panel-heading">
                <h2>Tải ván cờ</h2>
                <span className="panel-kicker">JSON / JSONL</span>
              </div>
              <label className="replay-upload-box">
                <span className="replay-upload-icon"><Icon name="upload" size={23} /></span>
                <strong>{replay.loading ? "Đang đọc dữ liệu..." : "Chọn file từ thiết bị"}</strong>
                <span>Alpha-Beta, Silver hoặc nhật ký Neural MCTS</span>
                <input
                  type="file"
                  accept=".json,.jsonl,application/json,application/x-ndjson"
                  aria-label="Tải file ván cờ"
                  disabled={replay.loading}
                  onChange={handleUpload}
                />
              </label>
              <p className="replay-file-name" title={fileLabel}>{fileLabel}</p>
              {replay.error && <p className="replay-error" role="alert">{replay.error}</p>}
              <p className="replay-helper">
                File được đọc cục bộ; không gửi lên backend. Giới hạn 30 MB.
              </p>
              {replay.games.length > 0 && (
                <div className="replay-game-selection">
                  <label htmlFor="replay-game-select">Chọn ván ({replay.games.length})</label>
                  <select
                    id="replay-game-select"
                    value={replay.selectedIndex}
                    onChange={handleGameSelect}
                  >
                    {replay.games.map(function renderGameOption(game, index) {
                      return <option key={game.id} value={index}>{index + 1}. {game.label}</option>;
                    })}
                  </select>
                </div>
              )}
            </section>

            <section className="panel replay-history-panel">
              <div className="panel-heading">
                <h2>Lịch sử nước đi</h2>
                <span className="small-id">{replay.totalPlies} PLY</span>
              </div>
              {replay.game?.outcome && <p className="replay-outcome">{replay.game.outcome}</p>}
              {!replay.game && (
                <div className="replay-empty-history">
                  <Icon name="clock" size={24} />
                  <p>Chưa có lịch sử. Sau khi tải file, bạn có thể nhấn từng nước để tua bàn cờ.</p>
                </div>
              )}
              {replay.game && replay.game.moves.length === 0 && (
                <p className="replay-empty-history">Ván này chưa có nước đi nào.</p>
              )}
              {replay.game && replay.game.moves.length > 0 && (
                <ol className="replay-move-list" aria-label="Danh sách nước đi">
                  {replay.game.moves.map(function renderMove(move) {
                    let className = "replay-move";
                    let current: "step" | undefined;
                    if (move.ply === replay.currentPly) {
                      className += " replay-move--current";
                      current = "step";
                    }
                    if (move.ply < replay.currentPly) {
                      className += " replay-move--visited";
                    }

                    function jumpToMove(): void {
                      replay.seek(move.ply);
                    }

                    return (
                      <li key={move.ply}>
                        <button
                          type="button"
                          className={className}
                          onClick={jumpToMove}
                          aria-current={current}
                        >
                          <span className="replay-move-number">{String(move.ply).padStart(3, "0")}</span>
                          <span className={"replay-move-side replay-move-side--" + move.side}>{sideLabel(move.side)}</span>
                          <strong>{move.notation}</strong>
                          {move.actor && <span className="replay-move-actor">{move.actor}</span>}
                        </button>
                      </li>
                    );
                  })}
                </ol>
              )}
              <p className="replay-helper">
                Replay tái tạo vị trí quân cờ; việc kiểm tra đầy đủ luật chơi thuộc pipeline backend.
              </p>
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}
