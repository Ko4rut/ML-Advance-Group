import { AGENTS } from "../constants";
import { getAgentName } from "../lib/board";
import type { Agent, AgentId, Side } from "../types";
import { Icon } from "./Icon";

type Props = {
  side: Side;
  value: AgentId;
  available: ReadonlySet<AgentId>;
  unavailableReasons?: Partial<Record<AgentId, string>>;
  disabled?: boolean;
  onChange: (id: AgentId) => void;
};

type OptionProps = {
  agent: Agent;
  side: Side;
  selected: boolean;
  enabled: boolean;
  disabled: boolean;
  reason?: string;
  onChange: (id: AgentId) => void;
};

function AgentOption({ agent, side, selected, enabled, disabled, reason, onChange }: OptionProps) {
  let optionClass = "agent-option";
  let glyphClass = "agent-glyph";
  let glyph: React.ReactNode = agent.symbol;
  let hint = agent.hint;

  if (selected) {
    optionClass += " is-active";
  }
  if (agent.id === "human") {
    glyphClass += " agent-glyph--human";
    glyph = <Icon name="user" size={23} />;
  }
  if (!enabled) {
    hint = reason || `${agent.label}: chưa khả dụng trên backend`;
  }

  function handleSelect(): void {
    onChange(agent.id);
  }

  return (
    <button
      type="button"
      className={optionClass}
      disabled={disabled || !enabled}
      title={hint}
      aria-pressed={selected}
      onClick={handleSelect}
    >
      <span className={glyphClass}>{glyph}</span>
      <span className="agent-name">{getAgentName(agent.id, side)}</span>
      {!enabled && <span className="agent-unavailable">Chưa khả dụng</span>}
    </button>
  );
}

export function AgentSelector({
  side,
  value,
  available,
  unavailableReasons = {},
  disabled = false,
  onChange,
}: Props) {
  let heading = "ĐỐI THỦ PHÍA DƯỚI";
  let caption = "Quân đen";
  let ariaLabel = "Chọn đối thủ phía dưới";

  if (side === "top") {
    heading = "ĐỐI THỦ PHÍA TRÊN";
    caption = "Quân đỏ";
    ariaLabel = "Chọn đối thủ phía trên";
  }

  return (
    <section className={`selector selector--${side}`} aria-label={ariaLabel}>
      <div className="selector-heading">
        <div className="section-eyebrow">
          <span className="side-dot" /> {heading}
        </div>
        <span className="side-caption">{caption} · Chọn một đối thủ</span>
      </div>
      <div className="agent-grid">
        {AGENTS.map(function renderAgent(agent) {
          return (
            <AgentOption
              key={agent.id}
              agent={agent}
              side={side}
              selected={value === agent.id}
              enabled={available.has(agent.id)}
              reason={unavailableReasons[agent.id]}
              disabled={disabled}
              onChange={onChange}
            />
          );
        })}
      </div>
      {!available.has("neural") && unavailableReasons.neural && (
        <p className="neural-status" role="note">{unavailableReasons.neural}</p>
      )}
    </section>
  );
}

