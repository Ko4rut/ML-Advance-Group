import Link from "next/link";
import { ThemeToggle } from "@/features/theme/components/ThemeToggle";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[color-mix(in_srgb,var(--background)_82%,transparent)] backdrop-blur-xl">
      <div className="site-width flex h-16 items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-3">
          <span className="grid size-9 place-items-center rounded-xl bg-[var(--accent)] text-sm font-bold text-white">帥</span>
          <span className="font-pixel text-[10px] sm:text-xs">XIANGQI AI</span>
        </Link>
        <nav className="hidden gap-6 text-sm text-[var(--muted)] md:flex">
          <Link href="/home">Home</Link>
          <Link href="/home#agents">Agents</Link>
          <Link href="/home#modes">Modes</Link>
          <Link href="/home#arena">Arena</Link>
        </nav>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Link href="/home" className="hidden rounded-xl bg-[var(--foreground)] px-4 py-2 text-sm font-semibold text-[var(--background)] sm:block">
            Enter Arena
          </Link>
        </div>
      </div>
    </header>
  );
}
