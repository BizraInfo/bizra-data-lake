import type { Metadata } from "next";
import {
  Geist,
  Geist_Mono,
  Playfair_Display,
  Inter,
  Amiri,
  JetBrains_Mono,
} from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { Providers } from "@/components/providers";
import { ThemeProvider } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ErrorBoundary } from "@/components/error-boundary";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// BIZRA brand fonts — loaded via next/font/google (self-hosted, no CSS
// @import timing hazard). Exposed as CSS vars; Tailwind `@theme` wires
// utilities (font-brand-serif, font-brand-arabic, etc.).
const playfair = Playfair_Display({
  variable: "--font-brand-serif",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const brandSans = Inter({
  variable: "--font-brand-sans",
  subsets: ["latin"],
  weight: ["200", "300", "400", "500", "600"],
  display: "swap",
});

const amiri = Amiri({
  variable: "--font-brand-arabic",
  subsets: ["arabic"],
  weight: ["400", "700"],
  display: "swap",
});

const brandMono = JetBrains_Mono({
  variable: "--font-brand-mono",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DEMA — Sovereign Operator Face",
  description: "The one visible face of BIZRA. Calm, sovereign, full-stack operator with constitutional trust, receipt chains, and lawful state.",
  keywords: ["DEMA", "BIZRA", "operator", "AI", "trust", "receipts", "manifest"],
  authors: [{ name: "BIZRA" }],
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%23000'/><text x='4' y='24' font-size='22' font-weight='bold' fill='%23D4A853' font-family='monospace'>D</text></svg>",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${playfair.variable} ${brandSans.variable} ${amiri.variable} ${brandMono.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <Providers>
            <TooltipProvider delayDuration={200}>
              <ErrorBoundary>
                {children}
              </ErrorBoundary>
            </TooltipProvider>
          </Providers>
        </ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
