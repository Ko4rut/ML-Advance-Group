import { Metric } from "./Metric";
import type { MatchSnapshot } from "../lib/api";

export function MatchOverview({ snapshot }: { snapshot: MatchSnapshot | null }) {
  const result = snapshot?.status === "finished" ? snapshot.winner : null;
  const draws = snapshot?.status === "finished" && !result ? 1 : 0;
  return <section className="panel stats-panel">
    <div className="panel-heading"><h2>Tổng quan trận đấu</h2><span className="stats-live">{!snapshot ? "Chưa tạo trận" : snapshot.status === "finished" ? "Đã kết thúc" : "Đang chơi"}</span></div>
    <div className="stats-grid">
      <Metric label="Số nước đi" value={String(snapshot?.ply ?? 0)} icon="layers"/>
      <Metric label="Thời gian" value="—" icon="clock"/>
      <Metric label="Lượt hiện tại" value={snapshot ? (snapshot.status === "finished" ? "Kết thúc" : snapshot.side_to_move === "red" ? "Đỏ" : "Đen") : "—"} icon="refresh"/>
      <Metric label="Trạng thái" value={!snapshot ? "Chưa tạo" : snapshot.status === "finished" ? "Kết thúc" : "Đang chơi"} icon="grid"/>
    </div>
    <div className="panel-divider"/>
    <div className="score-row">
      <div><span className="score-dot score-dot--red"/> Đỏ thắng<strong>{result === "red" ? 1 : 0}</strong></div>
      <div><span className="score-dot score-dot--gray"/> Hòa<strong>{draws}</strong></div>
      <div><span className="score-dot score-dot--blue"/> Đen thắng<strong>{result === "black" ? 1 : 0}</strong></div>
    </div>
  </section>;
}
