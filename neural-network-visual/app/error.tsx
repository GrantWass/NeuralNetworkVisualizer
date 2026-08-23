"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled page error:", error);
  }, [error]);

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 px-4 text-center">
      <h1 className="text-2xl font-bold tracking-tight">Something went wrong</h1>
      <p className="text-muted-foreground max-w-md leading-relaxed">
        An unexpected error occurred while rendering this page. Reloading usually fixes it.
      </p>
      <div className="flex gap-3">
        <button
          onClick={reset}
          className="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors"
        >
          Try again
        </button>
        <Link
          href="/"
          className="px-4 py-2 rounded-md border border-border hover:bg-muted text-sm font-medium transition-colors"
        >
          Back to home
        </Link>
      </div>
    </div>
  );
}
