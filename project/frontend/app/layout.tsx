import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Xiangqi AI Arena",
  description: "Visual arena for Xiangqi AI agents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
