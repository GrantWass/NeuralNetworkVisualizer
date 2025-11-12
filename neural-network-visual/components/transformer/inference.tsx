"use client";
import React, { useMemo, useState } from "react";

// -----------------------------
// Small utility helpers
// -----------------------------
const softmax = (xs: number[]) => {
  const max = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - max));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
};

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

const scalarMultiply = (a: number[], s: number) => a.map((v) => v * s);

const add = (a: number[], b: number[]) => a.map((v, i) => v + b[i]);

const rand = (seed = Math.random()) => {
  // simple xorshift-like pseudo RNG (deterministic only if seed provided as number)
  let x = Math.floor(seed * 2147483647) || 1234567;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) % 1000000) / 1000000;
  };
};

// -----------------------------
// Fake "vocabulary" helpers
// -----------------------------
const DEFAULT_VOCAB = ["I", "love", "transformers", "visualization", "!", "<eos>"];

const tokenize = (text: string) => text.split(/\s+/).filter(Boolean);

export const InferenceSection: React.FC<{
  prompt: string;
  vocab?: string[];
  dim?: number;
  seed?: number;
}> = ({ prompt, vocab = DEFAULT_VOCAB, dim = 16, seed = 0.9 }) => {
  const tokens = tokenize(prompt);
  const rng = useMemo(() => rand(seed + 50), [seed]);
  // small random projection to produce "logits" per candidate token
  const stepLogits = (contextVectors: number[]) => {
    // produce a logit for each vocab token via a pseudo similarity
    return vocab.map((_, i) => {
      // make deterministic-ish variation per token
      const bias = (i - vocab.length / 2) * 0.1;
      const noise = (rng() - 0.5) * 0.8;
      const score = dot(contextVectors.slice(0, dim), Array.from({ length: dim }, () => 0.2)) + bias + noise;
      return score;
    });
  };

  const [generated, setGenerated] = useState<string[]>([]);
  const [history, setHistory] = useState<{ step: number; probs: number[]; pick: string }[]>([]);
  const [running, setRunning] = useState(false);

  const stepOnce = () => {
    // context vector: hash of (tokens + generated) to fixed vector
    const ctxSeed = (tokens.join(" ") + " " + generated.join(" ")).length * 13 + seed;
    const r2 = rand(ctxSeed);
    const ctx = Array.from({ length: dim }, () => r2() - 0.5);
    const logits = stepLogits(ctx);
    const probs = softmax(logits);
    // pick greedy for clarity
    const bestIdx = probs.indexOf(Math.max(...probs));
    const pick = vocab[bestIdx];
    setGenerated((g) => [...g, pick]);
    setHistory((h) => [...h, { step: h.length + 1, probs, pick }]);
    return pick;
  };

  const runN = (n: number) => {
    setRunning(true);
    for (let i = 0; i < n; i++) {
      stepOnce();
    }
    setRunning(false);
  };

  return (
    <section className="p-4 border rounded-md mt-4">
      <h2 className="text-xl font-semibold mb-2">Autoregressive Inference</h2>
      <div className="flex gap-4">
        <div className="w-1/2">
          <h3 className="font-medium">Prompt</h3>
          <div className="p-2 bg-slate-50 rounded mb-2">{prompt}</div>
          <h3 className="font-medium">Generated</h3>
          <div className="p-2 bg-white border rounded min-h-[72px]">{generated.join(" ") || <span className="text-slate-400">(empty)</span>}</div>

          <div className="mt-3 flex gap-2">
            <button className="px-3 py-1 bg-sky-500 text-white rounded" onClick={() => stepOnce()} disabled={running}>Step</button>
            <button className="px-3 py-1 bg-emerald-500 text-white rounded" onClick={() => runN(3)} disabled={running}>Run 3</button>
            <button className="px-3 py-1 bg-gray-200 rounded" onClick={() => { setGenerated([]); setHistory([]); }}>Reset</button>
          </div>
        </div>

        <div className="w-1/2">
          <h3 className="font-medium">Per-step Probabilities</h3>
          <div className="space-y-2 max-h-48 overflow-auto">
            {history.slice().reverse().map((h, idx) => (
              <div key={`hist-${idx}`} className="p-2 border rounded bg-slate-50">
                <div className="text-sm font-medium">Step {h.step}: picked "{h.pick}"</div>
                <div className="flex gap-2 mt-2 flex-wrap">
                  {vocab.map((v, i) => (
                    <div key={v} className="text-xs">
                      <div className="font-mono">{v}</div>
                      <div style={{ width: Math.max(40, h.probs[i] * 120), background: "#60a5fa", height: 8, borderRadius: 4 }} title={h.probs[i].toFixed(3)} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

    </section>
  );
};