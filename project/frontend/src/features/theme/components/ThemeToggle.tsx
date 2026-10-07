"use client";
import { useContext } from "react";
import { ThemeContext } from "./ThemeProvider";

export function ThemeToggle() {
  const {theme,toggle} = useContext(ThemeContext);
  return (
    <button onClick={toggle} aria-label="Toggle theme" className="grid size-10 place-items-center rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      {theme==="dark" ? "☀" : "◐"}
    </button>
  );
}
