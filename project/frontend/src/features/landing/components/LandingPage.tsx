import Image from "next/image";
import Link from "next/link";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { agents } from "@/features/agents/data";

const floating = [
  ["/assets/pieces/red-xe.png","left-[6%] top-[22%]"],
  ["/assets/pieces/black-ma.png","right-[7%] top-[16%]"],
  ["/assets/pieces/red-phao.png","right-[17%] bottom-[10%]"],
  ["/assets/pieces/black-tuong.png","left-[18%] bottom-[8%]"],
];

export function LandingPage() {
  return (
    <div className="min-h-screen">
      <SiteHeader />
      <main>
        <section className="relative isolate overflow-hidden border-b border-[var(--border)]">
          <div className="ambient-grid"/>
          <div className="glow right-[-100px] top-[8%] bg-[var(--accent)]"/>
          <div className="glow bottom-[6%] left-[-120px] bg-[var(--accent-2)]"/>
          <div className="pointer-events-none absolute inset-0">
            {floating.map(([src,pos])=><Image key={src} src={src} alt="" width={80} height={80} className={`float-piece ${pos}`}/>)}
          </div>
          <div className="site-width relative z-10 grid min-h-[calc(100vh-4rem)] items-center gap-12 py-20 lg:grid-cols-[1.08fr_.92fr]">
            <div>
              <span className="inline-flex rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--muted)]">Traditional game × modern AI</span>
              <h1 className="mt-6 text-5xl font-black leading-[.95] tracking-[-.055em] sm:text-6xl lg:text-7xl">
                Xiangqi <span className="block text-[var(--accent)]">AI Arena</span>
              </h1>
              <p className="font-pixel mt-6 text-[9px] leading-6 tracking-[.08em]">PLAY · WATCH · TRAIN · COMPARE</p>
              <p className="mt-6 max-w-2xl text-base leading-7 text-[var(--muted)] sm:text-lg">
                Một nền tảng trực quan để chơi với AI, xem agent đối đầu và khám phá cách các chiến lược Cờ Tướng đưa ra quyết định.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link href="/home" className="rounded-xl bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-white shadow-lg">Enter Arena →</Link>
                <a href="#agents" className="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-5 py-3 text-sm font-semibold">Explore agents</a>
              </div>
            </div>
            <div className="relative mx-auto hidden h-[470px] w-full max-w-[480px] lg:block">
              <div className="absolute inset-10 rounded-[36px] border border-[var(--border)] bg-[color-mix(in_srgb,var(--surface)_82%,transparent)] shadow-[var(--shadow)] backdrop-blur-xl"/>
              <Image src="/assets/pieces/red-tuong.png" alt="Red king" width={150} height={150} className="absolute left-[20%] top-[16%] w-32 -rotate-6 drop-shadow-2xl"/>
              <Image src="/assets/pieces/black-tuong.png" alt="Black king" width={150} height={150} className="absolute bottom-[16%] right-[20%] w-32 rotate-6 drop-shadow-2xl"/>
              <div className="font-pixel absolute bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap text-[8px] tracking-[.18em] text-[var(--muted)]">THINK · MOVE · EVOLVE</div>
            </div>
          </div>
        </section>

        <section id="agents" className="site-width py-20">
          <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">AGENT ROSTER</p>
          <h2 className="mt-3 text-3xl font-bold tracking-[-.04em]">Seven ways to think.</h2>
          <p className="mt-3 max-w-2xl text-[var(--muted)]">Hai baseline đã sẵn sàng; năm agent còn lại tạo nên roadmap từ search cổ điển đến deep reinforcement learning.</p>
          <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
            {agents.map(([name,label,piece,status])=>(
              <article key={name} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-[var(--shadow)]">
                <Image src={`/assets/pieces/${piece}`} alt="" width={50} height={50} className="size-12 object-contain"/>
                <h3 className="mt-4 text-sm font-bold">{name}</h3>
                <p className="mt-1 text-[11px] text-[var(--muted)]">{label}</p>
                <span className={`mt-3 inline-block text-[9px] uppercase ${status==="ready"?"text-[var(--positive)]":"text-[var(--muted)]"}`}>{status}</span>
              </article>
            ))}
          </div>
        </section>

        <section className="border-y border-[var(--border)] bg-[var(--surface-2)] py-20">
          <div className="site-width">
            <p className="font-pixel text-[9px] tracking-[.16em] text-[var(--accent)]">EXPERIENCE</p>
            <h2 className="mt-3 text-3xl font-bold tracking-[-.04em]">More than a board viewer.</h2>
            <div className="mt-8 grid gap-4 md:grid-cols-3">
              {[
                ["01","Human vs AI","Chọn agent, chọn phe và trực tiếp thử sức."],
                ["02","AI vs AI","Theo dõi hai agent thi đấu theo thời gian thực."],
                ["03","Analysis","So sánh move, timing và metric giữa các agent."]
              ].map(([n,t,d])=>(
                <article key={n} className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6">
                  <span className="font-pixel text-[9px] text-[var(--accent)]">{n}</span>
                  <h3 className="mt-12 text-xl font-bold">{t}</h3>
                  <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{d}</p>
                </article>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
