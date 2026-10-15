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
import { useArenaPreview } from "@/features/arena/hooks/useArenaPreview";

export default function Home() {
  const arena = useArenaPreview();
  const toggleTheme = () => arena.setLightMode((previous) => !previous);
  const openStatistics = () => arena.setModal(true);

  return (
    <div className={`site ${arena.lightMode ? "site--light" : ""}`}>
      <ArenaSidebar onOpenStatistics={openStatistics} onToggleTheme={toggleTheme} />
      <main className="workspace">
        <ArenaHeader lightMode={arena.lightMode} onToggleTheme={toggleTheme} />
        <div className="content-layout">
          <div className="arena-main" id="agents">
            <AgentSelector side="top" value={arena.top} onChange={(id) => arena.chooseAgent("top", id)} />
            <div className="board-section">
              <div className="board-title">
                <span><span className="board-title-icon">✦</span> XIANGQI BOARD</span>
                <span className="board-mode">9 × 10 · Trạng thái ban đầu</span>
              </div>
              <ChessBoard board={arena.board} selection={arena.selection} onSquare={arena.selectSquare} />
              <div className="board-caption"><span className="board-caption-dot" /> {arena.notice}</div>
            </div>
            <AgentSelector side="bottom" value={arena.bottom} onChange={(id) => arena.chooseAgent("bottom", id)} />
          </div>
          <aside className="control-column" aria-label="Thông tin trận đấu">
            <MatchupCard top={arena.top} bottom={arena.bottom} />
            <MatchControls
              matchNumber={arena.matchNumber}
              onCreateMatch={arena.createMatch}
              onStartBattle={arena.startBattle}
              onResetBoard={arena.resetBoard}
              onOpenStatistics={openStatistics}
            />
            <MatchOverview />
            <IntegrationNote />
          </aside>
        </div>
      </main>
      {arena.modal && <StatisticsDialog top={arena.top} bottom={arena.bottom} onClose={() => arena.setModal(false)} />}
    </div>
  );
}
