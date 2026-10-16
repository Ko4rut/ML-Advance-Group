import { Icon } from "./Icon";

export function ArenaSidebar({ onOpenStatistics, onToggleTheme }: { onOpenStatistics: () => void; onToggleTheme: () => void }) {
  return <aside className="sidebar" aria-label="Thanh điều hướng">
    <div className="brand"><span className="brand-mark">棋</span><span className="brand-copy"><strong>XIANGQI</strong><small>AI ARENA</small></span></div>
    <div className="sidebar-caption">WORKSPACE</div>
    <nav className="navigation">
      <button className="nav-item nav-item--active" type="button" aria-current="page"><Icon name="grid"/> Arena</button>
      <button className="nav-item" type="button" onClick={() => document.getElementById("agents")?.scrollIntoView({ behavior: "smooth" })}><Icon name="layers"/> Agents</button>
      <button className="nav-item" type="button" onClick={onOpenStatistics}><Icon name="chart"/> Statistics</button>
      <button className="nav-item" type="button" onClick={onToggleTheme}><Icon name="settings"/> Theme</button>
    </nav>
    <div className="sidebar-footer"><span className="live-indicator"/> XIANGQI API <small>Game state from FastAPI</small></div>
  </aside>;
}
