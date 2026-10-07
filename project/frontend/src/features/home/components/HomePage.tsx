import Image from "next/image";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { MiniBoard } from "@/components/shared/MiniBoard";
import { agents } from "@/features/agents/data";

export function HomePage() {
  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main className="site-width space-y-12 py-8 md:py-10">
        <section className="grid gap-5 lg:grid-cols-[1.15fr_.85fr]">
          <div className="rounded-[28px] border border-[var(--border)] bg-[var(--surface)] p-7 shadow-[var(--shadow)]">
            <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">ARENA CONTROL</p>
            <h1 className="mt-4 text-4xl font-black tracking-[-.05em] md:text-5xl">Choose a side.<span className="block text-[var(--muted)]">Choose an intelligence.</span></h1>
            <p className="mt-5 max-w-xl leading-7 text-[var(--muted)]">Trang chính tập trung vào ba workflow: chơi với AI, xem AI đấu và phân tích agent.</p>
            <div className="mt-8 grid grid-cols-3 gap-3">
              {[["2","Ready"],["5","Planned"],["90","Squares"]].map(([v,l])=><div key={l} className="rounded-2xl bg-[var(--surface-2)] p-4"><strong className="font-pixel text-sm text-[var(--accent)]">{v}</strong><span className="mt-2 block text-xs text-[var(--muted)]">{l}</span></div>)}
            </div>
          </div>
          <div className="rounded-[28px] border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow)]"><MiniBoard/></div>
        </section>

        <section id="agents">
          <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">ROSTER</p>
          <h2 className="mt-3 text-3xl font-bold tracking-[-.04em]">AI Agents</h2>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {agents.map(([name,label,piece,status])=>(
              <article key={name} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
                <div className="flex items-start justify-between"><Image src={`/assets/pieces/${piece}`} alt="" width={54} height={54}/><span className={`rounded-full bg-[var(--surface-2)] px-2.5 py-1 text-[9px] uppercase ${status==="ready"?"text-[var(--positive)]":"text-[var(--muted)]"}`}>{status}</span></div>
                <h3 className="mt-5 text-lg font-bold">{name}</h3><p className="mt-1 text-xs text-[var(--muted)]">{label}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="modes">
          <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">PLAY</p>
          <h2 className="mt-3 text-3xl font-bold tracking-[-.04em]">Choose a mode</h2>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {[["PVE","Human vs AI"],["AVA","AI vs AI"],["LAB","Analysis"]].map(([code,title],i)=>(
              <article key={code} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6">
                <span className="font-pixel text-[9px] text-[var(--accent)]">{code}</span><span className="float-right text-sm text-[var(--muted)]">0{i+1}</span>
                <h3 className="mt-10 text-xl font-bold">{title}</h3><p className="mt-3 text-sm leading-6 text-[var(--muted)]">Workflow được chuẩn bị để nối vào Python API ở bước tiếp theo.</p>
              </article>
            ))}
          </div>
        </section>

        <section id="arena">
          <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">LIVE</p>
          <h2 className="mt-3 text-3xl font-bold tracking-[-.04em]">Arena preview</h2>
          <div className="mt-6 grid gap-5 rounded-[28px] border border-[var(--border)] bg-[var(--surface)] p-5 lg:grid-cols-[330px_1fr]">
            <MiniBoard/>
            <div className="min-w-0 rounded-2xl bg-[var(--surface-2)] p-5">
              <div className="flex items-center justify-between gap-4"><h3 className="text-xl font-bold">Rule-Based <span className="text-[var(--muted)]">vs</span> Random</h3><span className="rounded-full bg-[color-mix(in_srgb,var(--positive)_15%,transparent)] px-3 py-1 text-xs font-semibold text-[var(--positive)]">● LIVE</span></div>
              <div className="mt-6 grid gap-3 sm:grid-cols-3">{[["Move","12 / 200"],["Thinking","3.8 ms"],["Legal moves","31"]].map(([l,v])=><div key={l} className="rounded-xl bg-[var(--surface)] p-4"><span className="text-[11px] text-[var(--muted)]">{l}</span><strong className="mt-2 block text-sm">{v}</strong></div>)}</div>
              <div className="mt-6 flex h-2 overflow-hidden rounded-full bg-[var(--border)]"><span className="w-[48%] bg-[var(--accent)]"/><span className="flex-1 bg-[var(--accent-2)]"/></div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
