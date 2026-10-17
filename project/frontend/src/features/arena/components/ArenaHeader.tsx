import { Icon } from "./Icon";

type Props = {
  lightMode: boolean;
  onToggleTheme: () => void;
};

export function ArenaHeader({ lightMode, onToggleTheme }: Props) {
  let themeLabel = "Bật giao diện sáng";
  let themeIcon = "sun";

  if (lightMode) {
    themeLabel = "Bật giao diện tối";
    themeIcon = "moon";
  }

  return (
    <header className="topbar">
      <div>
        <span className="breadcrumb">
          Workspace <span>/</span> Arena
        </span>
        <h1>Battle arena</h1>
        <p>Thử nghiệm, so sánh và theo dõi các Xiangqi AI agent.</p>
      </div>
      <div className="topbar-actions">
        <span className="preview-pill">
          <span /> UI PREVIEW
        </span>
        <button
          type="button"
          className="icon-button"
          aria-label={themeLabel}
          onClick={onToggleTheme}
        >
          <Icon name={themeIcon} size={19} />
        </button>
      </div>
    </header>
  );
}

