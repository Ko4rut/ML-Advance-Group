import type { AgentId } from "../types";
import { getAgentName } from "../lib/board";

export function MatchupCard({ top, bottom }: { top: AgentId; bottom: AgentId }) {
  return (
    <section className="panel versus-panel"><div className="panel-header"><span className="panel-kicker">MATCH SETUP</span><span className="outline-tag">LOCAL PREVIEW</span></div><h2>Cặp đấu đã chọn</h2><div className="versus-players"><div className="versus-player"><span className="versus-label">RED · TOP</span><strong>{getAgentName(top, "top")}</strong></div><span className="versus-badge">VS</span><div className="versus-player versus-player--bottom"><span className="versus-label">BLACK · BOTTOM</span><strong>{getAgentName(bottom, "bottom")}</strong></div></div><p className="panel-helper">Chọn một đối thủ ở mỗi hàng để thiết lập trận đấu.</p></section>
  );
}
