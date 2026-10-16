import { Icon } from "./Icon";

type Props = {
  matchNumber: number;
  hasMatch: boolean;
  pending: boolean;
  running: boolean;
  canRun: boolean;
  onCreateMatch: () => void;
  onStartBattle: () => void;
  onResetBoard: () => void;
  onOpenStatistics: () => void;
};

export function MatchControls({ matchNumber, hasMatch, pending, running, canRun, onCreateMatch, onStartBattle, onResetBoard, onOpenStatistics }: Props) {
  return <section className="panel controls-panel">
    <div className="panel-heading"><h2>Điều khiển trận đấu</h2><span className="small-id">{hasMatch ? `#${String(matchNumber).padStart(3, "0")}` : "NEW"}</span></div>
    <button type="button" className="action-button action-button--primary" disabled={pending} onClick={onCreateMatch}>
      <Icon name="dice" size={19}/> Tạo trận đấu <Icon name="arrow" size={17}/>
    </button>
    <button type="button" className="action-button action-button--secondary" disabled={!canRun && !running} onClick={onStartBattle}>
      <Icon name="play" size={17}/> {running ? "Tạm dừng AI" : "Chạy AI"} <span className="action-suffix">{running ? "Đang chạy" : "Theo lượt"}</span>
    </button>
    <div className="button-pair">
      <button type="button" className="quiet-button" disabled={!hasMatch || pending} onClick={onResetBoard}><Icon name="refresh" size={17}/> Đặt lại</button>
      <button type="button" className="quiet-button" onClick={onOpenStatistics}><Icon name="chart" size={17}/> Thống kê</button>
    </div>
    <div className="api-note"><span className="api-note-icon">i</span><span>Trận đấu được xử lý bởi FastAPI / XiangqiEnv. AI hỗ trợ Random và Rule-Based.</span></div>
  </section>;
}
