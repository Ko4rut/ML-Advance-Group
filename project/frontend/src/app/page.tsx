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

  let siteClass = "site";
  let moveLabel = "Chưa tạo trận";
  if (arena.lightMode) {
    siteClass += " site--light";
  }
  if (arena.snapshot) {
    moveLabel = `Nước ${arena.snapshot.ply}`;
  }

  return (
    <div className={siteClass}>
      <ArenaSidebar
        onOpenStatistics={arena.openStatistics}
        onToggleTheme={arena.toggleTheme}
      />
      <main className="workspace">
        <ArenaHeader
          lightMode={arena.lightMode}
          onToggleTheme={arena.toggleTheme}
        />
        <div className="content-layout">
          <div className="arena-main" id="agents">
            <AgentSelector
              side="top"
              value={arena.top}
              available={arena.available}
              unavailableReasons={arena.unavailableReasons}
              disabled={arena.pending}
              onChange={arena.chooseTopAgent}
            />
            <div className="board-section">
              <div className="board-title">
                <span>
                  <span className="board-title-icon">✦</span> XIANGQI BOARD
                </span>
                <span className="board-mode">9 × 10 · {moveLabel}</span>
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
              <div className="board-caption" role="status">
                <span className="board-caption-dot" /> {arena.notice}
              </div>
            </div>
            <AgentSelector
              side="bottom"
              value={arena.bottom}
              available={arena.available}
              unavailableReasons={arena.unavailableReasons}
              disabled={arena.pending}
              onChange={arena.chooseBottomAgent}
            />
          </div>
          <aside className="control-column" aria-label="Thông tin trận đấu">
            <MatchupCard top={arena.top} bottom={arena.bottom} />
            <MatchControls
              matchNumber={1}
              hasMatch={Boolean(arena.snapshot)}
              pending={arena.pending}
              running={arena.running}
              canRun={arena.canRun}
              onCreateMatch={arena.createMatch}
              onStartBattle={arena.startBattle}
              onResetBoard={arena.resetBoard}
              onOpenStatistics={arena.openStatistics}
            />
            <MatchOverview snapshot={arena.snapshot} />
            <IntegrationNote connected={arena.connected} />
          </aside>
        </div>
      </main>
      {arena.modal && (
        <StatisticsDialog
          top={arena.top}
          bottom={arena.bottom}
          snapshot={arena.snapshot}
          onClose={arena.closeStatistics}
        />
      )}
    </div>
  );
}

