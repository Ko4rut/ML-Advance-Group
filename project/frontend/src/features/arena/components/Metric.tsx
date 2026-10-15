import { Icon } from "./Icon";

export function Metric({ label, value, icon }: { label: string; value: string; icon: string }) {
  return <div className="metric"><span className="metric-caption"><Icon name={icon} size={15}/>{label}</span><strong>{value}</strong></div>;
}
