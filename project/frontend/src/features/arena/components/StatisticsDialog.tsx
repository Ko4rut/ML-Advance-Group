import type { AgentId } from "../types";
import type { MatchSnapshot } from "../lib/api";
import { getAgentName } from "../lib/board";
import { Icon } from "./Icon";
import { Metric } from "./Metric";

type Props = { top: AgentId; bottom: AgentId; snapshot: MatchSnapshot | null; onClose: () => void };

export function StatisticsDialog({ top, bottom, snapshot, onClose }: Props) {
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section role="dialog" aria-modal="true" aria-labelledby="stats-modal-title" className="modal">
      <div className="modal-head"><div><span className="panel-kicker">MATCH STATISTICS</span><h2 id="stats-modal-title">Thống kê trận đấu</h2></div>
        <button type="button" className="icon-button" aria-label="Đóng thống kê" onClick={onClose}><Icon name="close"/></button>
      </div>
      <div className="modal-matchup">{getAgentName(top, "top")} <span>vs</span> {getAgentName(bottom, "bottom")}</div>
      <div className="stats-grid">
        <Metric label="Ván hiện tại" value={snapshot ? "1" : "0"} icon="grid"/>
        <Metric label="Số nước đi" value={String(snapshot?.ply ?? 0)} icon="layers"/>
        <Metric label="Đỏ thắng" value={snapshot?.status === "finished" && snapshot.winner === "red" ? "1" : "0"} icon="chart"/>
        <Metric label="Đen thắng" value={snapshot?.status === "finished" && snapshot.winner === "black" ? "1" : "0"} icon="chart"/>
      </div>
      <p className="modal-footnote">{snapshot?.status === "finished" ? `Kết quả: ${snapshot.winner ? (snapshot.winner === "red" ? "Đỏ thắng" : "Đen thắng") : "Hòa"} (${snapshot.reason ?? "unknown"}).` : "Thống kê chỉ áp dụng cho ván hiện tại. Backend chưa lưu tổng lịch sử hay thời gian thi đấu."}</p>
    </section>
  </div>;
}
