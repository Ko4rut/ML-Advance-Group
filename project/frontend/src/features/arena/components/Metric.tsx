import { Icon } from "./Icon";

type Props = { label: string; value: string; icon: string };

export function Metric({ label, value, icon }: Props) {
  return (
    <div className="metric">
      <span className="metric-caption">
        <Icon name={icon} size={15} />
        {label}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

