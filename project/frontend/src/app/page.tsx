"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

type Side = "top" | "bottom";
type AgentId = "human" | "random" | "rules" | "minimax" | "alphabeta" | "mcts" | "neural" | "dqn";
type Agent = { id: AgentId; label: string; symbol: string; hint: string };

type Piece = { side: Side; kind: number };
type Board = Array<Piece | null>;

const AGENTS: Agent[] = [
  { id: "human", label: "Player", symbol: "user", hint: "Human player" },
  { id: "random", label: "Random", symbol: "✳", hint: "Random policy" },
  { id: "rules", label: "Rule-Based", symbol: "≡", hint: "Handcrafted rules" },
  { id: "minimax", label: "Minimax", symbol: "M", hint: "Game tree search" },
  { id: "alphabeta", label: "Alpha-Beta", symbol: "αβ", hint: "Pruned search" },
  { id: "mcts", label: "MCTS", symbol: "⌘", hint: "Monte Carlo search" },
  { id: "neural", label: "Neural MCTS", symbol: "N", hint: "Neural-guided search" },
  { id: "dqn", label: "DQN", symbol: "D", hint: "Deep Q-network" },
];

const PIECE_NAMES: Record<number, string> = {
  1: "Tướng", 2: "Sĩ", 3: "Tượng", 4: "Mã", 5: "Xe", 6: "Pháo", 7: "Tốt",
};
const PIECE_FILES: Record<number, string> = {
  1: "tuong", 2: "si", 3: "tuongj", 4: "ma", 5: "xe", 6: "phao", 7: "tot",
};

function initialBoard(): Board {
  const board: Board = Array.from({ length: 90 }, () => null);
  const backRank = [5, 4, 3, 2, 1, 2, 3, 4, 5];
  backRank.forEach((kind, col) => {
    board[col] = { side: "top", kind };
    board[9 * 9 + col] = { side: "bottom", kind };
  });
  [1, 7].forEach((col) => {
    board[2 * 9 + col] = { side: "top", kind: 6 };
    board[7 * 9 + col] = { side: "bottom", kind: 6 };
  });
  [0, 2, 4, 6, 8].forEach((col) => {
    board[3 * 9 + col] = { side: "top", kind: 7 };
    board[6 * 9 + col] = { side: "bottom", kind: 7 };
  });
  return board;
}

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  switch (name) {
    case "user": return <svg {...common}><circle cx="12" cy="8" r="3.5"/><path d="M5.5 20c.2-4 2.4-6 6.5-6s6.3 2 6.5 6"/></svg>;
    case "grid": return <svg {...common}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>;
    case "chart": return <svg {...common}><path d="M4 20V11m6 9V4m5 16v-8m5 8V7"/></svg>;
    case "layers": return <svg {...common}><path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/></svg>;
    case "settings": return <svg {...common}><circle cx="12" cy="12" r="3.5"/><path d="M12 2.5v2m0 15v2M2.5 12h2m15 0h2M5.3 5.3l1.5 1.5m10.4 10.4 1.5 1.5m0-13.4-1.5 1.5M6.8 17.2l-1.5 1.5"/></svg>;
    case "play": return <svg {...common}><path fill="currentColor" stroke="none" d="m8 5 11 7-11 7V5Z"/></svg>;
    case "refresh": return <svg {...common}><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.8 9a7 7 0 0 1 12.5-2L20 12M4 12l1.7 5a7 7 0 0 0 12.5-2"/></svg>;
    case "dice": return <svg {...common}><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="8.5" cy="8.5" r="1" fill="currentColor" stroke="none"/><circle cx="15.5" cy="8.5" r="1" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/><circle cx="8.5" cy="15.5" r="1" fill="currentColor" stroke="none"/><circle cx="15.5" cy="15.5" r="1" fill="currentColor" stroke="none"/></svg>;
    case "clock": return <svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>;
    case "moon": return <svg {...common}><path d="M20 15.8A8.5 8.5 0 0 1 8.2 4 8.5 8.5 0 1 0 20 15.8Z"/></svg>;
    case "sun": return <svg {...common}><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4m0-14.2-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>;
    case "close": return <svg {...common}><path d="m5 5 14 14M19 5 5 19"/></svg>;
    case "arrow": return <svg {...common}><path d="M5 12h14m-6-6 6 6-6 6"/></svg>;
    default: return <svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 10v6m0-9v.5"/></svg>;
  }
}

function Selector({ side, value, onChange }: { side: Side; value: AgentId; onChange: (id: AgentId) => void }) {
  return <section className={`selector selector--${side}`} aria-label={side === "top" ? "Chọn đối thủ phía trên" : "Chọn đối thủ phía dưới"}>
    <div className="selector-heading">
      <div className="section-eyebrow"><span className="side-dot"/> {side === "top" ? "ĐỐI THỦ PHÍA TRÊN" : "ĐỐI THỦ PHÍA DƯỚI"}</div>
      <span className="side-caption">{side === "top" ? "Quân đỏ" : "Quân đen"} · Chọn một đối thủ</span>
    </div>
    <div className="agent-grid">
      {AGENTS.map((agent) => <button key={agent.id} type="button" title={agent.hint} aria-pressed={value === agent.id} onClick={() => onChange(agent.id)} className={`agent-option ${value === agent.id ? "is-active" : ""}`}>
        <span className={`agent-glyph ${agent.id === "human" ? "agent-glyph--human" : ""}`}>{agent.id === "human" ? <Icon name="user" size={23}/> : agent.symbol}</span>
        <span className="agent-name">{agent.id === "human" ? (side === "top" ? "Player 1" : "Player 2") : agent.label}</span>
      </button>)}
    </div>
  </section>;
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

function ChessBoard({ board, selection, onSquare }: { board: Board; selection: number | null; onSquare: (index: number) => void }) {
  return <div className="board-frame">
    <div className="board-canvas">
      <BoardGrid/>
      {board.map((piece, index) => {
        const col = index % 9;
        const row = Math.floor(index / 9);
        return <button
          key={index}
          type="button"
          className={`board-intersection ${selection === index ? "board-intersection--selected" : ""} ${piece ? "has-piece" : ""}`}
          style={{ left: `${((40 + col * 65) / 600) * 100}%`, top: `${((40 + row * 64) / 656) * 100}%` }}
          onClick={() => onSquare(index)}
          aria-label={piece ? `${PIECE_NAMES[piece.kind]} ${piece.side === "top" ? "đỏ" : "đen"}, hàng ${row + 1} cột ${col + 1}` : `Ô trống hàng ${row + 1} cột ${col + 1}`}
          aria-pressed={selection === index}
        >{piece && <Image unoptimized src={`/assets/pieces/${piece.side === "top" ? "red" : "black"}-${PIECE_FILES[piece.kind]}.png`} alt="" width={72} height={72} draggable={false}/>}</button>;
      })}
    </div>
  </div>;
}

function LabelValue({ label, value, icon }: { label: string; value: string; icon: string }) {
  return <div className="metric"><span className="metric-caption"><Icon name={icon} size={15}/>{label}</span><strong>{value}</strong></div>;
}

export default function Home() {
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

  const agentName = (id: AgentId, side: Side) => id === "human" ? side === "top" ? "Player 1" : "Player 2" : AGENTS.find((a) => a.id === id)!.label;
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

  return <div className={`site ${lightMode ? "site--light" : ""}`}>
    <aside className="sidebar" aria-label="Thanh điều hướng">
      <div className="brand"><span className="brand-mark">棋</span><span className="brand-copy"><strong>XIANGQI</strong><small>AI ARENA</small></span></div>
      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="navigation">
        <button className="nav-item nav-item--active" type="button" aria-current="page"><Icon name="grid"/> Arena</button>
        <button className="nav-item" type="button" onClick={() => document.getElementById("agents")?.scrollIntoView({ behavior: "smooth" })}><Icon name="layers"/> Agents</button>
        <button className="nav-item" type="button" onClick={() => setModal(true)}><Icon name="chart"/> Statistics</button>
        <button className="nav-item" type="button" onClick={() => setLightMode((v) => !v)}><Icon name="settings"/> Theme</button>
      </nav>
      <div className="sidebar-footer"><span className="live-indicator"/> FRONTEND PREVIEW <small>Backend not connected</small></div>
    </aside>

    <main className="workspace">
      <header className="topbar"><div><span className="breadcrumb">Workspace <span>/</span> Arena</span><h1>Battle arena</h1><p>Thử nghiệm, so sánh và theo dõi các Xiangqi AI agent.</p></div><div className="topbar-actions"><span className="preview-pill"><span/> UI PREVIEW</span><button type="button" className="icon-button" aria-label={lightMode ? "Bật giao diện tối" : "Bật giao diện sáng"} onClick={() => setLightMode((v) => !v)}><Icon name={lightMode ? "moon" : "sun"} size={19}/></button></div></header>

      <div className="content-layout">
        <div className="arena-main" id="agents">
          <Selector side="top" value={top} onChange={(id) => chooseAgent("top", id)}/>
          <div className="board-section">
            <div className="board-title"><span><span className="board-title-icon">✦</span> XIANGQI BOARD</span><span className="board-mode">9 × 10 · Trạng thái ban đầu</span></div>
            <ChessBoard board={board} selection={selection} onSquare={selectSquare}/>
            <div className="board-caption"><span className="board-caption-dot"/> {notice}</div>
          </div>
          <Selector side="bottom" value={bottom} onChange={(id) => chooseAgent("bottom", id)}/>
        </div>

        <aside className="control-column" aria-label="Thông tin trận đấu">
          <section className="panel versus-panel"><div className="panel-header"><span className="panel-kicker">MATCH SETUP</span><span className="outline-tag">LOCAL PREVIEW</span></div><h2>Cặp đấu đã chọn</h2><div className="versus-players"><div className="versus-player"><span className="versus-label">RED · TOP</span><strong>{agentName(top, "top")}</strong></div><span className="versus-badge">VS</span><div className="versus-player versus-player--bottom"><span className="versus-label">BLACK · BOTTOM</span><strong>{agentName(bottom, "bottom")}</strong></div></div><p className="panel-helper">Chọn một đối thủ ở mỗi hàng để thiết lập trận đấu.</p></section>
          <section className="panel controls-panel"><div className="panel-heading"><h2>Điều khiển trận đấu</h2><span className="small-id">{matchNumber ? `#${String(matchNumber).padStart(3, "0")}` : "NEW"}</span></div><button type="button" className="action-button action-button--primary" onClick={createMatch}><Icon name="dice" size={19}/> Tạo trận đấu <Icon name="arrow" size={17}/></button><button type="button" className="action-button action-button--secondary" onClick={() => setNotice("Chưa thể bắt đầu: cần endpoint tạo trận và endpoint yêu cầu nước đi từ agent.")}><Icon name="play" size={17}/> Bắt đầu trận <span className="action-suffix">API required</span></button><div className="button-pair"><button type="button" className="quiet-button" onClick={resetBoard}><Icon name="refresh" size={17}/> Đặt lại</button><button type="button" className="quiet-button" onClick={() => setModal(true)}><Icon name="chart" size={17}/> Thống kê</button></div><div className="api-note"><span className="api-note-icon">i</span><span>Bản giao diện thử nghiệm. Chưa có endpoint điều khiển agent.</span></div></section>
          <section className="panel stats-panel"><div className="panel-heading"><h2>Tổng quan trận đấu</h2><span className="stats-live">Chưa bắt đầu</span></div><div className="stats-grid"><LabelValue label="Số nước đi" value="0" icon="layers"/><LabelValue label="Thời gian" value="00:00" icon="clock"/><LabelValue label="Lượt hiện tại" value="Đỏ" icon="refresh"/><LabelValue label="Trạng thái" value="Sẵn sàng" icon="grid"/></div><div className="panel-divider"/><div className="score-row"><div><span className="score-dot score-dot--red"/> Đỏ thắng<strong>0</strong></div><div><span className="score-dot score-dot--gray"/> Hòa<strong>0</strong></div><div><span className="score-dot score-dot--blue"/> Đen thắng<strong>0</strong></div></div></section>
          <div className="roadmap-note"><span className="roadmap-mark">↗</span><div><strong>Ready for integration</strong><p>Kết nối XiangqiEnv và agent qua FastAPI ở bước tiếp theo.</p></div></div>
        </aside>
      </div>
    </main>

    {modal && <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) setModal(false); }}><section role="dialog" aria-modal="true" aria-labelledby="stats-modal-title" className="modal"><div className="modal-head"><div><span className="panel-kicker">MATCH STATISTICS</span><h2 id="stats-modal-title">Thống kê trận đấu</h2></div><button type="button" className="icon-button" aria-label="Đóng thống kê" onClick={() => setModal(false)}><Icon name="close"/></button></div><div className="modal-matchup">{agentName(top, "top")} <span>vs</span> {agentName(bottom, "bottom")}</div><div className="stats-grid"><LabelValue label="Ván đã chơi" value="0" icon="grid"/><LabelValue label="Số nước đi" value="0" icon="layers"/><LabelValue label="Đỏ thắng" value="0" icon="chart"/><LabelValue label="Đen thắng" value="0" icon="chart"/></div><p className="modal-footnote">Chưa có dữ liệu thực tế. Thống kê sẽ được tải từ backend khi kết nối API trận đấu.</p></section></div>}
  </div>;
}
