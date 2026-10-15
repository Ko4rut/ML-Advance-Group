import { Metric } from "./Metric";

export function MatchOverview() {
  return (
    <section className="panel stats-panel"><div className="panel-heading"><h2>Tổng quan trận đấu</h2><span className="stats-live">Chưa bắt đầu</span></div><div className="stats-grid"><Metric label="Số nước đi" value="0" icon="layers"/><Metric label="Thời gian" value="00:00" icon="clock"/><Metric label="Lượt hiện tại" value="Đỏ" icon="refresh"/><Metric label="Trạng thái" value="Sẵn sàng" icon="grid"/></div><div className="panel-divider"/><div className="score-row"><div><span className="score-dot score-dot--red"/> Đỏ thắng<strong>0</strong></div><div><span className="score-dot score-dot--gray"/> Hòa<strong>0</strong></div><div><span className="score-dot score-dot--blue"/> Đen thắng<strong>0</strong></div></div></section>
  );
}
