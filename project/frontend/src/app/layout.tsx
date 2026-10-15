import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Xiangqi AI Arena | Battle Arena",
  description: "Giao diện thi đấu và so sánh AI Agent Cờ Tướng",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
