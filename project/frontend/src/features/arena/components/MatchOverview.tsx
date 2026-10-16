import type { MatchSnapshot } from "../api/types";
import { Metric } from "./Metric";

export function MatchOverview({ snapshot }: { snapshot: MatchSnapshot | null }) {
  let status = "Chưa tạo trận";
  let turn = "—";
  let moves = "0";
  let redWins = 0;
  let blackWins = 0;
  let draws = 0;

  if (snapshot) {
    moves = String(snapshot.ply);
    status = "Đang chơi";
    turn = "Đen";
    if (snapshot.side_to_move === "red") {
      turn = "Đỏ";
    }
    if (snapshot.status === "finished") {
      status = "Đã kết thúc";
      turn = "Kết thúc";
      if (snapshot.winner === "red") {
        redWins = 1;
      } else if (snapshot.winner === "black") {
        blackWins = 1;
      } else {
        draws = 1;
      }
    }
  }

  return (
    <section className="panel stats-panel">
      <div className="panel-heading">
        <h2>Tổng quan trận đấu</h2>
        <span className="stats-live">{status}</span>
      </div>
      <div className="stats-grid">
        <Metric label="Số nước đi" value={moves} icon="layers" />
        <Metric label="Thời gian" value="—" icon="clock" />
        <Metric label="Lượt hiện tại" value={turn} icon="refresh" />
        <Metric label="Trạng thái" value={status} icon="grid" />
      </div>
      <div className="panel-divider" />
      <div className="score-row">
        <div><span className="score-dot score-dot--red" /> Đỏ thắng<strong>{redWins}</strong></div>
        <div><span className="score-dot score-dot--gray" /> Hòa<strong>{draws}</strong></div>
        <div><span className="score-dot score-dot--blue" /> Đen thắng<strong>{blackWins}</strong></div>
      </div>
    </section>
  );
}

