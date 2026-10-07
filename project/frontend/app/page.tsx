export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center gap-6 px-6 py-16">
      <p className="text-sm font-medium uppercase tracking-[0.2em] text-neutral-500">
        ML Advance Group
      </p>
      <h1 className="max-w-3xl text-5xl font-semibold tracking-tight">
        Xiangqi AI Arena
      </h1>
      <p className="max-w-2xl text-lg leading-8 text-neutral-600 dark:text-neutral-300">
        Next.js frontend scaffold for visualizing Human vs AI and AI vs AI
        matches. The Python agents and Xiangqi environment live in the backend.
      </p>
      <div className="flex flex-wrap gap-3 text-sm">
        <span className="rounded-full border border-neutral-300 px-4 py-2 dark:border-neutral-700">
          Next.js App Router
        </span>
        <span className="rounded-full border border-neutral-300 px-4 py-2 dark:border-neutral-700">
          TypeScript
        </span>
        <span className="rounded-full border border-neutral-300 px-4 py-2 dark:border-neutral-700">
          Tailwind CSS
        </span>
      </div>
    </main>
  );
}
