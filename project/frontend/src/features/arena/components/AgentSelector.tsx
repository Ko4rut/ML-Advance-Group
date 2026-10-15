import { AGENTS } from "../constants";
import type { AgentId, Side } from "../types";
import { Icon } from "./Icon";

export function AgentSelector({ side, value, onChange }: { side: Side; value: AgentId; onChange: (id: AgentId) => void }) {
  return <section className={`selector selector--${side}`} aria-label={side === "top" ? "Chọn đối thủ phía trên" : "Chọn đối thủ phía dưới"}>
    <div className="selector-heading">
      <div className="section-eyebrow"><span className="side-dot"/> {side === "top" ? "ĐỐI THỦ PHÍA TRÊN" : "ĐỐI THỦ PHÍA DƯỚI"}</div>
      <span className="side-caption">{side === "top" ? "Quân đỏ" : "Quân đen"} · Chọn một đối thủ</span>
    </div>
    <div className="agent-grid">
      {AGENTS.map((agent) => <button key={agent.id} type="button" title={agent.hint} aria-pressed={value === agent.id} onClick={() => onChange(agent.id)} className={`agent-option ${value === agent.id ? "is-active" : ""}`}>
        <span className={`agent-glyph ${agent.id === "human" ? "agent-glyph--human" : ""}`}>{agent.id === "human" ? <Icon name="user" size={23}/> : agent.symbol}</span>
        <span className="agent-name">{agent.id === "human" ? (side === "top" ? "Player 1" : "Player 2") : agent.label}</span>
      </button>)}
    </div>
  </section>;
}
