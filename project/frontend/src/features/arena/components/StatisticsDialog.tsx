import type { AgentId } from "../types";
import { getAgentName } from "../lib/board";
import { Icon } from "./Icon";
import { Metric } from "./Metric";

export function StatisticsDialog({ top, bottom, onClose }: { top: AgentId; bottom: AgentId; onClose: () => void }) {
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}><section role="dialog" aria-modal="true" aria-labelledby="stats-modal-title" className="modal"><div className="modal-head"><div><span className="panel-kicker">MATCH STATISTICS</span><h2 id="stats-modal-title">Thống kê trận đấu</h2></div><button type="button" className="icon-button" aria-label="Đóng thống kê" onClick={() => onClose()}><Icon name="close"/></button></div><div className="modal-matchup">{getAgentName(top, "top")} <span>vs</span> {getAgentName(bottom, "bottom")}</div><div className="stats-grid"><Metric label="Ván đã chơi" value="0" icon="grid"/><Metric label="Số nước đi" value="0" icon="layers"/><Metric label="Đỏ thắng" value="0" icon="chart"/><Metric label="Đen thắng" value="0" icon="chart"/></div><p className="modal-footnote">Chưa có dữ liệu thực tế. Thống kê sẽ được tải từ backend khi kết nối API trận đấu.</p></section></div>
  );
}
