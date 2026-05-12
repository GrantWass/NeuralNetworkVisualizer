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
        <header className="w-full bg-white/80 dark:bg-gray-900/80 backdrop-blur sticky top-0 z-50 border-b border-gray-200 dark:border-gray-800">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <span className="text-lg font-bold">Neural Network Visual</span>
            </Link>
            <nav className="flex items-center gap-4">
              <Link href="/" className="text-sm font-medium hover:underline">Neural Network</Link>
              <Link href="/transformer" className="text-sm font-medium hover:underline">Transformers</Link>
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
