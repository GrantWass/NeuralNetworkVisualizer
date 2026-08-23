"use client";

import { useEffect, useState } from "react";
import type { TransformerExample } from "./types";

// Precomputed BERT examples are split into per-example files under
// /data/attention-examples/ and fetched on demand instead of being bundled
// into the page JS. The manifest is tiny; each example is ~100–250 KB.
export type ExampleManifestEntry = { id: string; label: string; file: string };

let manifestCache: ExampleManifestEntry[] | null = null;
let manifestPending: Promise<ExampleManifestEntry[]> | null = null;
const exampleCache = new Map<string, TransformerExample>();
const examplePending = new Map<string, Promise<TransformerExample>>();

function fetchManifest(): Promise<ExampleManifestEntry[]> {
  if (manifestCache) return Promise.resolve(manifestCache);
  if (!manifestPending) {
    manifestPending = fetch("/data/attention-examples/manifest.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        return res.json() as Promise<ExampleManifestEntry[]>;
      })
      .then((data) => {
        if (!Array.isArray(data) || data.length === 0) throw new Error("Example data is empty");
        manifestCache = data;
        return data;
      })
      .catch((e) => {
        manifestPending = null; // allow retry
        throw e;
      });
  }
  return manifestPending;
}

function fetchExample(entry: ExampleManifestEntry): Promise<TransformerExample> {
  const cached = exampleCache.get(entry.id);
  if (cached) return Promise.resolve(cached);
  let pending = examplePending.get(entry.id);
  if (!pending) {
    pending = (async () => {
      const res = await fetch(entry.file);
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const example = (await res.json()) as TransformerExample;
      exampleCache.set(entry.id, example);
      examplePending.delete(entry.id);
      return example;
    })();
    pending.catch(() => examplePending.delete(entry.id)); // allow retry
    examplePending.set(entry.id, pending);
  }
  return pending;
}

export function useAttentionManifest() {
  const [manifest, setManifest] = useState<ExampleManifestEntry[] | null>(manifestCache);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (manifestCache) return;
    let cancelled = false;
    fetchManifest()
      .then((entries) => { if (!cancelled) setManifest(entries); })
      .catch((e) => {
        console.error(e);
        if (!cancelled) setError("Couldn't load the attention examples. Please try again.");
      });
    return () => { cancelled = true; };
  }, [attempt]);

  const retry = () => {
    setError("");
    setAttempt((n) => n + 1);
  };
  return { manifest, error, retry };
}

/** Loads a single example by manifest entry, with cache + error state. */
export function useAttentionExample(entry: ExampleManifestEntry | null) {
  // The module-level exampleCache is the source of truth; bumping state just
  // triggers a re-render when an async load lands in it.
  const [, bumpVersion] = useState(0);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!entry || exampleCache.has(entry.id)) return;
    let cancelled = false;
    fetchExample(entry)
      .then(() => { if (!cancelled) bumpVersion((v) => v + 1); })
      .catch((e) => {
        console.error(e);
        if (!cancelled) setError("Couldn't load this example. Please try again.");
      });
    return () => { cancelled = true; };
  }, [entry, attempt]);

  const retry = () => setAttempt((n) => n + 1);
  const example = entry ? exampleCache.get(entry.id) ?? null : null;
  return { example, error, retry };
}

export function ExamplesError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="space-y-2 py-8 text-center">
      <p className="text-sm text-red-500">{message}</p>
      <button
        onClick={onRetry}
        className="text-xs underline underline-offset-4 hover:text-foreground"
      >
        Try again
      </button>
    </div>
  );
}

export function ExamplesLoading() {
  return (
    <div className="h-48 rounded-lg border border-border bg-muted/40 animate-pulse flex items-center justify-center">
      <p className="text-sm text-muted-foreground">Loading attention examples…</p>
    </div>
  );
}
