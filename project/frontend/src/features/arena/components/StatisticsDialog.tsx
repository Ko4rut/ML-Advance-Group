import type { MouseEvent } from "react";
import type { MatchSnapshot } from "../api/types";
import { getAgentName } from "../lib/board";
import type { AgentId } from "../types";
import { Icon } from "./Icon";
import { Metric } from "./Metric";

type Props = {
  top: AgentId;
  bottom: AgentId;
  snapshot: MatchSnapshot | null;
  onClose: () => void;
};

export function StatisticsDialog({ top, bottom, snapshot, onClose }: Props) {
  let games = "0";
  let moves = "0";
  let redWins = "0";
  let blackWins = "0";
  let footnote = "Thống kê chỉ áp dụng cho ván hiện tại. Backend chưa lưu tổng lịch sử hay thời gian thi đấu.";

  if (snapshot) {
    games = "1";
    moves = String(snapshot.ply);

    if (snapshot.status === "finished") {
      let result = "Hòa";
      if (snapshot.winner === "red") {
        result = "Đỏ thắng";
        redWins = "1";
      } else if (snapshot.winner === "black") {
        result = "Đen thắng";
        blackWins = "1";
      }
      footnote = `Kết quả: ${result} (${snapshot.reason ?? "unknown"}).`;
    }
  }

  function handleBackdropMouseDown(event: MouseEvent<HTMLDivElement>): void {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={handleBackdropMouseDown}>
      <section role="dialog" aria-modal="true" aria-labelledby="stats-modal-title" className="modal">
        <div className="modal-head">
          <div>
            <span className="panel-kicker">MATCH STATISTICS</span>
            <h2 id="stats-modal-title">Thống kê trận đấu</h2>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Đóng thống kê"
            onClick={onClose}
          >
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-matchup">
          {getAgentName(top, "top")} <span>vs</span> {getAgentName(bottom, "bottom")}
        </div>
        <div className="stats-grid">
          <Metric label="Ván hiện tại" value={games} icon="grid" />
          <Metric label="Số nước đi" value={moves} icon="layers" />
          <Metric label="Đỏ thắng" value={redWins} icon="chart" />
          <Metric label="Đen thắng" value={blackWins} icon="chart" />
        </div>
        <p className="modal-footnote">{footnote}</p>
      </section>
    </div>
  );
}

