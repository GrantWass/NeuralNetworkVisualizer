import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 px-4 text-center">
      <h1 className="text-3xl font-bold tracking-tight">Page not found</h1>
      <p className="text-muted-foreground max-w-md leading-relaxed">
        That page doesn&apos;t exist. Check out the visualizations instead:
      </p>
      <div className="flex gap-3">
        <Link
          href="/"
          className="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors"
        >
          Neural Networks
        </Link>
        <Link
          href="/attention"
          className="px-4 py-2 rounded-md border border-border hover:bg-muted text-sm font-medium transition-colors"
        >
          Attention
        </Link>
        <Link
          href="/transformers"
          className="px-4 py-2 rounded-md border border-border hover:bg-muted text-sm font-medium transition-colors"
        >
          Transformers
        </Link>
      </div>
    </div>
  );
}
