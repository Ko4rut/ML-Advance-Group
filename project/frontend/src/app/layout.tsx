import type { Metadata } from "next";
import { Geist, Press_Start_2P } from "next/font/google";
import { ThemeProvider } from "@/features/theme/components/ThemeProvider";
import "@/styles/tokens.css";
import "./globals.css";

const bodyFont = Geist({ variable: "--font-body", subsets: ["latin"] });
const pixelFont = Press_Start_2P({
  variable: "--font-pixel",
  subsets: ["latin"],
  weight: "400",
});

export const metadata: Metadata = {
  title: "Xiangqi AI Arena",
  description: "Play, watch and compare Xiangqi AI agents.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="vi"
      data-theme="dark"
      suppressHydrationWarning
      className={`${bodyFont.variable} ${pixelFont.variable}`}
    >
      <body>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
