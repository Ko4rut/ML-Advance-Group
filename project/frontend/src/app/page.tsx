"use client";

import { AgentSelector } from "@/features/arena/components/AgentSelector";
import { ArenaHeader } from "@/features/arena/components/ArenaHeader";
import { ArenaSidebar } from "@/features/arena/components/ArenaSidebar";
import { ChessBoard } from "@/features/arena/components/ChessBoard";
import { IntegrationNote } from "@/features/arena/components/IntegrationNote";
import { MatchControls } from "@/features/arena/components/MatchControls";
import { MatchOverview } from "@/features/arena/components/MatchOverview";
import { MatchupCard } from "@/features/arena/components/MatchupCard";
import { StatisticsDialog } from "@/features/arena/components/StatisticsDialog";
import { useArenaMatch } from "@/features/arena/hooks/useArenaMatch";

export default function Home() {
  const arena = useArenaMatch();
  const toggleTheme = () => arena.setLightMode((previous) => !previous);
  const openStatistics = () => arena.setModal(true);
  const actor = arena.snapshot?.side_to_move === "red" ? arena.snapshot?.red : arena.snapshot?.black;
  const canRun = Boolean(arena.snapshot && arena.snapshot.status === "playing" && actor?.type === "agent" && !arena.pending);

  return <div className={`site ${arena.lightMode ? "site--light" : ""}`}>
    <ArenaSidebar onOpenStatistics={openStatistics} onToggleTheme={toggleTheme}/>
    <main className="workspace">
      <ArenaHeader lightMode={arena.lightMode} onToggleTheme={toggleTheme}/>
      <div className="content-layout">
        <div className="arena-main" id="agents">
          <AgentSelector side="top" value={arena.top} available={arena.available} disabled={arena.pending} onChange={(id) => arena.chooseAgent("top", id)}/>
          <div className="board-section">
            <div className="board-title">
              <span><span className="board-title-icon">✦</span> XIANGQI BOARD</span>
              <span className="board-mode">9 × 10 · {arena.snapshot ? `Nước ${arena.snapshot.ply}` : "Chưa tạo trận"}</span>
            </div>
            <ChessBoard
              board={arena.board}
              selection={arena.selection}
              legalTargets={arena.legalTargets}
              lastMove={arena.lastMove}
              disabled={arena.pending}
              preview={!arena.snapshot}
              onCreateMatch={arena.createMatch}
              onSquare={arena.selectSquare}
            />
            <div className="board-caption" role="status"><span className="board-caption-dot"/> {arena.notice}</div>
          </div>
          <AgentSelector side="bottom" value={arena.bottom} available={arena.available} disabled={arena.pending} onChange={(id) => arena.chooseAgent("bottom", id)}/>
        </div>
        <aside className="control-column" aria-label="Thông tin trận đấu">
          <MatchupCard top={arena.top} bottom={arena.bottom}/>
          <MatchControls matchNumber={arena.snapshot ? 1 : 0} hasMatch={Boolean(arena.snapshot)} pending={arena.pending}
            running={arena.running} canRun={canRun} onCreateMatch={arena.createMatch} onStartBattle={arena.startBattle}
            onResetBoard={arena.resetBoard} onOpenStatistics={openStatistics}/>
          <MatchOverview snapshot={arena.snapshot}/>
          <IntegrationNote connected={arena.connected}/>
        </aside>
      </div>
    </main>
    {arena.modal && <StatisticsDialog top={arena.top} bottom={arena.bottom} snapshot={arena.snapshot} onClose={() => arena.setModal(false)}/>}
  </div>;
}
