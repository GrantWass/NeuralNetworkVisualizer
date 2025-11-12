"use client";
import React, { useMemo, useState } from "react";
import ContactInfo from "../contact";

// NEED TO ADD METADATA

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

// -----------------------------
// Component: Embeddings
// - shows tokens + 2D projection (random projection of high-dim vectors)
// - shows vector numeric view for selected token
// -----------------------------
const EmbeddingsSection: React.FC<{
  tokens: string[];
  dim: number;
  seed?: number;
}> = ({ tokens, dim, seed = 0.42 }) => {
  // deterministic embeddings per token for demo
  const rng = useMemo(() => rand(seed), [seed]);

  const embeddingMap = useMemo(() => {
    const map: Record<string, number[]> = {};
    tokens.forEach((t, i) => {
      // generate a vector of length dim
      const v = Array.from({ length: dim }, () => (rng() - 0.5) * 2);
      // normalize a bit
      const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
      map[t] = v.map((x) => x / norm);
    });
    return map;
  }, [tokens.join(" ")]);

  // Two random projection vectors to reduce dim -> 2
  const proj = useMemo(() => {
    const r = rand(seed + 1);
    const p1 = Array.from({ length: dim }, () => r() - 0.5);
    const r2 = rand(seed + 2);
    const p2 = Array.from({ length: dim }, () => r2() - 0.5);
    return [p1, p2];
  }, [dim, seed]);

  const coords = useMemo(() => {
    return tokens.map((t) => {
      const v = embeddingMap[t];
      const x = dot(v, proj[0]);
      const y = dot(v, proj[1]);
      return { token: t, x, y };
    });
  }, [tokens, embeddingMap, proj]);

  const [selected, setSelected] = useState<string | null>(tokens[0] ?? null);

  // compute bbox for svg
  const xs = coords.map((c) => c.x);
  const ys = coords.map((c) => c.y);
  const minX = Math.min(...xs, -1);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1);
  const maxY = Math.max(...ys, 1);

  return (
    <section className="p-4 border rounded-md">
      <h2 className="text-xl font-semibold mb-2">Word Embeddings</h2>
      <div className="flex gap-4">
        <svg viewBox={`0 0 400 300`} className="w-2/3 border rounded-md bg-white">
          <g transform={`translate(20,20)`}> 
            {coords.map((c, i) => {
              const cx = ((c.x - minX) / (maxX - minX || 1)) * 350;
              const cy = ((c.y - minY) / (maxY - minY || 1)) * 260;
              const isSel = selected === c.token;
              return (
                <g key={`emb-${i}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSel ? 8 : 6}
                    fill={isSel ? "#0ea5e9" : "#60a5fa"}
                    stroke="#0f172a"
                    strokeWidth={isSel ? 2 : 1}
                    onClick={() => setSelected(c.token)}
                    style={{ cursor: "pointer" }}
                  />
                  <text x={cx + 10} y={cy + 4} fontSize={12}>
                    {c.token}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        <div className="w-1/3">
          <h3 className="font-medium">Selected token</h3>
          <p className="mb-2">{selected}</p>
          <h4 className="font-medium">Vector (first 12 dims)</h4>
          <pre className="text-sm bg-slate-50 p-2 rounded h-48 overflow-auto">
            {embeddingMap[selected || tokens[0]]?.slice(0, 12).map((v, i) => `${i}: ${v.toFixed(4)}`).join("\n")}
          </pre>
          <p className="text-xs text-slate-500 mt-2">Projection: 2D random projection for visualization</p>
        </div>
      </div>
    </section>
  );
};

// -----------------------------
// Component: Attention
// - computes scaled dot-product attention between tokens using the embeddings
// - displays a heatmap and draws attention arcs
// -----------------------------
const AttentionSection: React.FC<{
  tokens: string[];
  dim: number;
  seed?: number;
  heads?: number;
}> = ({ tokens, dim, seed = 0.42, heads = 2 }) => {
  const rng = useMemo(() => rand(seed + 10), [seed]);
  // For demo create random Q/K/V per head
  const headsData = useMemo(() => {
    return Array.from({ length: heads }, (_, h) => {
      const Q = tokens.map(() => Array.from({ length: dim }, () => rng() - 0.5));
      const K = tokens.map(() => Array.from({ length: dim }, () => rng() - 0.5));
      const V = tokens.map(() => Array.from({ length: dim }, () => rng() - 0.5));
      return { Q, K, V };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens.join(" "), heads]);

  const [headIndex, setHeadIndex] = useState(0);
  const [temp, setTemp] = useState(1.0);

  const scores = useMemo(() => {
    const { Q, K } = headsData[headIndex];
    const s = Q.map((q) => K.map((k) => dot(q, k) / Math.sqrt(dim)));
    return s.map((row) => softmax(row.map((v) => v / temp)));
  }, [headIndex, temp, headsData, dim]);

  const selectedCellColor = (v: number) => {
    const shade = Math.round(255 - v * 200);
    return `rgb(${shade}, ${shade}, 255)`;
  };

  return (
    <section className="p-4 border rounded-md mt-4">
      <h2 className="text-xl font-semibold mb-2">Scaled Dot-Product Attention</h2>
      <div className="flex gap-4">
        <div className="w-1/2">
          <div className="flex items-center gap-2 mb-2">
            <label className="text-sm">Head:</label>
            <select value={headIndex} onChange={(e) => setHeadIndex(Number(e.target.value))} className="border px-2 py-1 rounded">
              {Array.from({ length: heads }).map((_, i) => (
                <option key={i} value={i}>Head {i + 1}</option>
              ))}
            </select>
            <label className="text-sm ml-4">Temperature</label>
            <input type="range" min={0.2} max={2} step={0.1} value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
            <span className="text-sm ml-2">{temp.toFixed(1)}</span>
          </div>

          <div className="grid grid-cols-1 gap-2">
            <div className="overflow-auto border rounded">
              <table className="table-auto text-xs w-full border-collapse">
                <thead>
                  <tr>
                    <th className="p-1">→ / ↓</th>
                    {tokens.map((t, i) => (
                      <th key={`h-${i}`} className="p-1 text-left">{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {scores.map((row, i) => (
                    <tr key={`r-${i}`}>
                      <td className="p-1 font-mono">{tokens[i]}</td>
                      {row.map((v, j) => (
                        <td key={`c-${i}-${j}`} className="p-1 font-mono">
                          <div style={{ width: 60, height: 20, background: selectedCellColor(v), borderRadius: 4 }} title={v.toFixed(3)}>
                            <div style={{ padding: 2, fontSize: 11 }}>{v.toFixed(2)}</div>
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="w-1/2">
          <svg viewBox={`0 0 400 120`} className="w-full border rounded bg-white">
            {tokens.map((t, i) => {
              const x = 40 + (i * 300) / Math.max(1, tokens.length - 1);
              const y = 40;
              return (
                <g key={`token-${i}`}>
                  <circle cx={x} cy={y} r={14} fill="#fde68a" stroke="#92400e" />
                  <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fontWeight={600}>{t}</text>
                </g>
              );
            })}

            {/* draw attention arcs from each query token to keys */}
            {scores.map((row, i) =>
              row.map((v, j) => {
                if (v < 0.03) return null;
                const x1 = 40 + (i * 300) / Math.max(1, tokens.length - 1);
                const x2 = 40 + (j * 300) / Math.max(1, tokens.length - 1);
                const stroke = `rgba(59,130,246,${v * 1.4})`;
                const width = 1 + v * 6;
                const path = `M ${x1} 50 Q ${(x1 + x2) / 2} ${20 - Math.abs(x2 - x1) * 0.08} ${x2} 50`;
                return <path key={`arc-${i}-${j}`} d={path} stroke={stroke} strokeWidth={width} fill="none" />;
              })
            )}
          </svg>
        </div>
      </div>
    </section>
  );
};

// -----------------------------
// Component: Inference
// - shows iterative autoregressive decoding using a small toy vocab
// - visualizes probabilities and selected token each step
// -----------------------------
const InferenceSection: React.FC<{
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

// -----------------------------
// Main Page Component
// -----------------------------
const TransformerVizPage: React.FC = () => {
  const [prompt, setPrompt] = useState("I love transformers");
  const tokens = tokenize(prompt);

  return (
    <div className="p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Transformer Visualization *IN PROGRESS*</h1>
        <p className="text-sm text-slate-500">Interactive demo: embeddings → attention → inference</p>
      </header>

      <div>
        <label className="block text-sm font-medium">Prompt</label>
        <input className="border p-2 rounded w-full mt-2" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      </div>

      {/* <EmbeddingsSection tokens={tokens} dim={32} seed={1.2} />
      <AttentionSection tokens={tokens} dim={32} heads={3} seed={2.1} />
      <InferenceSection prompt={prompt} vocab={DEFAULT_VOCAB} dim={16} seed={3.7} /> */}

      <ContactInfo />

    </div>
  );
};

export default TransformerVizPage;
