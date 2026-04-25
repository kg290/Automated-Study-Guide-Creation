import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Space_Grotesk } from "next/font/google";

import { SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const bodyFont = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-body",
});

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "StudyGuide Agent",
  description:
    "Generative AI agent for automated study guide creation from PDF and DOCX documents.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${bodyFont.variable} ${displayFont.variable}`}>
        <div className="background-orb background-orb-a" />
        <div className="background-orb background-orb-b" />
        <div className="background-orb background-orb-c" />
        <div className="app-shell">
          <SiteHeader />
          <main className="page-width">{children}</main>
        </div>
      </body>
    </html>
  );
}
