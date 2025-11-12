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

export const AttentionSection: React.FC<{
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