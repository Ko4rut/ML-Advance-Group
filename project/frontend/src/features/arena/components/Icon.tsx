export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  switch (name) {
    case "user": return <svg {...common}><circle cx="12" cy="8" r="3.5"/><path d="M5.5 20c.2-4 2.4-6 6.5-6s6.3 2 6.5 6"/></svg>;
    case "grid": return <svg {...common}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>;
    case "chart": return <svg {...common}><path d="M4 20V11m6 9V4m5 16v-8m5 8V7"/></svg>;
    case "layers": return <svg {...common}><path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/></svg>;
    case "settings": return <svg {...common}><circle cx="12" cy="12" r="3.5"/><path d="M12 2.5v2m0 15v2M2.5 12h2m15 0h2M5.3 5.3l1.5 1.5m10.4 10.4 1.5 1.5m0-13.4-1.5 1.5M6.8 17.2l-1.5 1.5"/></svg>;
    case "play": return <svg {...common}><path fill="currentColor" stroke="none" d="m8 5 11 7-11 7V5Z"/></svg>;
    case "refresh": return <svg {...common}><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.8 9a7 7 0 0 1 12.5-2L20 12M4 12l1.7 5a7 7 0 0 0 12.5-2"/></svg>;
    case "dice": return <svg {...common}><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="8.5" cy="8.5" r="1" fill="currentColor" stroke="none"/><circle cx="15.5" cy="8.5" r="1" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/><circle cx="8.5" cy="15.5" r="1" fill="currentColor" stroke="none"/><circle cx="15.5" cy="15.5" r="1" fill="currentColor" stroke="none"/></svg>;
    case "clock": return <svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>;
    case "moon": return <svg {...common}><path d="M20 15.8A8.5 8.5 0 0 1 8.2 4 8.5 8.5 0 1 0 20 15.8Z"/></svg>;
    case "sun": return <svg {...common}><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4m0-14.2-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>;
    case "close": return <svg {...common}><path d="m5 5 14 14M19 5 5 19"/></svg>;
    case "arrow": return <svg {...common}><path d="M5 12h14m-6-6 6 6-6 6"/></svg>;
    default: return <svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 10v6m0-9v.5"/></svg>;
  }
}
