import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { Analytics } from "@vercel/analytics/react";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://nn-visual.com"),
  title: {
    template: "%s | Neural Network Visual",
    default: "Neural Network Visual — Interactive ML Visualizations",
  },
  description:
    "Free interactive visualizations to help students understand neural networks, backpropagation, and transformer attention.",
  authors: [{ name: "Grant Wasserman", url: "https://grantwasserman.com" }],
  creator: "Grant Wasserman",
  openGraph: {
    type: "website",
    siteName: "Neural Network Visual",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <header className="w-full bg-white/90 dark:bg-[hsl(224,40%,5%)]/90 backdrop-blur-md sticky top-0 z-50 border-b border-indigo-100 dark:border-indigo-950">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5 group">
              <span className="inline-flex h-6 w-6 rounded-md bg-indigo-500 items-center justify-center shrink-0">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <circle cx="3" cy="7" r="2" fill="white" />
                  <circle cx="11" cy="4" r="1.5" fill="white" fillOpacity="0.7" />
                  <circle cx="11" cy="10" r="1.5" fill="white" fillOpacity="0.7" />
                  <line x1="5" y1="6.5" x2="9.5" y2="4.5" stroke="white" strokeWidth="1" strokeOpacity="0.6" />
                  <line x1="5" y1="7.5" x2="9.5" y2="9.5" stroke="white" strokeWidth="1" strokeOpacity="0.6" />
                </svg>
              </span>
              <span className="text-base font-semibold tracking-tight">Neural Network Visual</span>
            </Link>
            <nav className="flex items-center gap-1">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-md text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-indigo-50 dark:hover:bg-indigo-950/50 transition-colors"
              >
                Neural Network
              </Link>
              <Link
                href="/transformer"
                className="px-3 py-1.5 rounded-md text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-indigo-50 dark:hover:bg-indigo-950/50 transition-colors"
              >
                Transformers
              </Link>
            </nav>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 pt-6">
          {children}
        </main>
        <Toaster />
        <Analytics />
      </body>
    </html>
  );
}
