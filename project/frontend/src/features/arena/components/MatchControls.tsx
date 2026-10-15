import { Icon } from "./Icon";

type MatchControlsProps = { matchNumber: number; onCreateMatch: () => void; onStartBattle: () => void; onResetBoard: () => void; onOpenStatistics: () => void };

export function MatchControls({ matchNumber, onCreateMatch, onStartBattle, onResetBoard, onOpenStatistics }: MatchControlsProps) {
  return (
    <section className="panel controls-panel"><div className="panel-heading"><h2>Điều khiển trận đấu</h2><span className="small-id">{matchNumber ? `#${String(matchNumber).padStart(3, "0")}` : "NEW"}</span></div><button type="button" className="action-button action-button--primary" onClick={onCreateMatch}><Icon name="dice" size={19}/> Tạo trận đấu <Icon name="arrow" size={17}/></button><button type="button" className="action-button action-button--secondary" onClick={onStartBattle}><Icon name="play" size={17}/> Bắt đầu trận <span className="action-suffix">API required</span></button><div className="button-pair"><button type="button" className="quiet-button" onClick={onResetBoard}><Icon name="refresh" size={17}/> Đặt lại</button><button type="button" className="quiet-button" onClick={onOpenStatistics}><Icon name="chart" size={17}/> Thống kê</button></div><div className="api-note"><span className="api-note-icon">i</span><span>Bản giao diện thử nghiệm. Chưa có endpoint điều khiển agent.</span></div></section>
  );
}
