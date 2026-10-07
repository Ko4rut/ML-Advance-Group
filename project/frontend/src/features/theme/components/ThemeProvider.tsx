"use client";
import { createContext, useEffect, useState } from "react";
export const ThemeContext = createContext({ theme:"dark", toggle:()=>{} });

export function ThemeProvider({children}:{children:React.ReactNode}) {
  const [theme,setTheme] = useState("dark");
  useEffect(()=>{
    const saved = localStorage.getItem("xiangqi-theme");
    const resolved = saved === "light" || saved === "dark"
      ? saved
      : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    setTheme(resolved);
  },[]);
  useEffect(()=>{
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("xiangqi-theme", theme);
  },[theme]);
  return <ThemeContext.Provider value={{theme,toggle:()=>setTheme(v=>v==="dark"?"light":"dark")}}>{children}</ThemeContext.Provider>;
}
