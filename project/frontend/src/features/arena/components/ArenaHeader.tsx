import { Icon } from "./Icon";

export function ArenaHeader({ lightMode, onToggleTheme }: { lightMode: boolean; onToggleTheme: () => void }) {
  return (
    <header className="topbar"><div><span className="breadcrumb">Workspace <span>/</span> Arena</span><h1>Battle arena</h1><p>Thử nghiệm, so sánh và theo dõi các Xiangqi AI agent.</p></div><div className="topbar-actions"><span className="preview-pill"><span/> UI PREVIEW</span><button type="button" className="icon-button" aria-label={lightMode ? "Bật giao diện tối" : "Bật giao diện sáng"} onClick={() => onToggleTheme()}><Icon name={lightMode ? "moon" : "sun"} size={19}/></button></div></header>
  );
}
