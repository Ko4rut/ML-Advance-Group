import Link from "next/link";
import { Icon } from "./Icon";

type Props = {
  activePage?: "arena" | "replay";
  onOpenStatistics?: () => void;
  onToggleTheme: () => void;
};

export function ArenaSidebar({
  activePage = "arena",
  onOpenStatistics,
  onToggleTheme,
}: Props) {
  let arenaClass = "nav-item";
  let replayClass = "nav-item";
  let arenaCurrent: "page" | undefined;
  let replayCurrent: "page" | undefined;

  if (activePage === "arena") {
    arenaClass += " nav-item--active";
    arenaCurrent = "page";
  } else {
    replayClass += " nav-item--active";
    replayCurrent = "page";
  }

  return (
    <aside className="sidebar" aria-label="Thanh điều hướng">
      <div className="brand">
        <span className="brand-mark">棋</span>
        <span className="brand-copy">
          <strong>XIANGQI</strong>
          <small>AI ARENA</small>
        </span>
      </div>
      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="navigation">
        <Link className={arenaClass} aria-current={arenaCurrent} href="/">
          <Icon name="grid" /> Arena
        </Link>
        <Link className={replayClass} aria-current={replayCurrent} href="/replay">
          <Icon name="replay" /> Replay
        </Link>
        <Link className="nav-item" href="/#agents">
          <Icon name="layers" /> Agents
        </Link>
        {onOpenStatistics && (
          <button className="nav-item" type="button" onClick={onOpenStatistics}>
            <Icon name="chart" /> Statistics
          </button>
        )}
        <button className="nav-item" type="button" onClick={onToggleTheme}>
          <Icon name="settings" /> Theme
        </button>
      </nav>
      <div className="sidebar-footer">
        <span className="live-indicator" /> XIANGQI API
        <small>Game state from FastAPI</small>
      </div>
    </aside>
  );
}
