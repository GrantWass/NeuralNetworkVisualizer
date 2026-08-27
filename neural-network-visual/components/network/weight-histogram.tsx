"use client";

import { useMemo, useState } from "react";
import type { NetworkState } from "@/components/network/static/types";

interface WeightHistogramProps {
  network: NetworkState | null;
}

const BIN_COUNT = 20;
// Show at least [-1,1] so initialization is visible; expand if weights grow larger.
const MIN_RANGE = 1.0;

function collectWeights(network: NetworkState, layerFilter: number | null): number[] {
  const layers = network.layers;
  // layers includes last placeholder; skip if no weights
  const filtered = layerFilter !== null ? [layers[layerFilter]].filter(Boolean) : layers;
  const out: number[] = [];
  for (const layer of filtered) {
    if (!layer.weights?.length) continue;
    for (const row of layer.weights) {
      for (const w of row) out.push(w);
    }
  }
  return out;
}

export function WeightHistogram({ network }: WeightHistogramProps) {
  const [filter, setFilter] = useState<number | null>(null);
  // Exclude the last "output placeholder" style? All layers except perhaps last if empty.
  // Keep all layers that have weights.
  const availableLayers = useMemo(() => {
    if (!network) return [];
    return network.layers
      .map((l, i) => ({ i, size: l.size, weights: l.weights }))
      .filter((l) => l.weights?.length > 0);
  }, [network]);

  const { bins, range, maxCount, stats } = useMemo(() => {
    if (!network || availableLayers.length === 0) {
      return { bins: [] as number[], range: [-MIN_RANGE, MIN_RANGE] as [number, number], maxCount: 0, stats: null as null | { mean: number; std: number; count: number; min: number; max: number } };
    }
    const weights = collectWeights(network, filter);
    if (weights.length === 0) {
      return { bins: new Array(BIN_COUNT).fill(0), range: [-MIN_RANGE, MIN_RANGE] as [number, number], maxCount: 0, stats: null };
    }
    let maxAbs = 0;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    for (const w of weights) {
      const a = Math.abs(w);
      if (a > maxAbs) maxAbs = a;
      sum += w;
      if (w < min) min = w;
      if (w > max) max = w;
    }
    const half = Math.max(MIN_RANGE, maxAbs * 1.05);
    const lo = -half;
    const hi = half;
    const binsArr = new Array(BIN_COUNT).fill(0);
    const width = (hi - lo) / BIN_COUNT;
    for (const w of weights) {
      let idx = Math.floor((w - lo) / width);
      if (idx < 0) idx = 0;
      if (idx >= BIN_COUNT) idx = BIN_COUNT - 1;
      binsArr[idx]++;
    }
    const mean = sum / weights.length;
    let varSum = 0;
    for (const w of weights) varSum += (w - mean) ** 2;
    const std = Math.sqrt(varSum / weights.length);
    return {
      bins: binsArr,
      range: [lo, hi] as [number, number],
      maxCount: Math.max(...binsArr),
      stats: { mean, std, count: weights.length, min, max },
    };
  }, [network, availableLayers.length, filter]);

  if (!network || availableLayers.length === 0) {
    return (
      <div className="bg-white p-3 sm:p-4 rounded-lg border border-gray-200">
        <h3 className="text-sm font-semibold text-gray-800">Weight Distribution</h3>
        <p className="text-xs text-gray-400 mt-1">Initialize the model to see the weight histogram.</p>
        <div className="h-28 mt-3 flex items-center justify-center text-sm text-gray-400 border border-dashed border-gray-200 rounded">
          No weights yet
        </div>
      </div>
    );
  }

  const totalLayers = availableLayers.length;
  const filterLabel = filter === null ? "All layers" : `Layer ${filter + 1}`;

  return (
    <div className="bg-white p-3 sm:p-4 rounded-lg border border-gray-200">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h3 className="text-sm font-semibold text-gray-800">Weight Distribution</h3>
        <span className="text-xs font-mono text-gray-500">{stats?.count} weights · {filterLabel}</span>
      </div>
      <p className="text-xs text-gray-400 mt-0.5">
        Random initialization → trained distribution. Collapse or explosion is visible before the loss curve shows it.
      </p>

      {/* Layer filter */}
      <div className="flex flex-wrap gap-1.5 mt-3">
        <button
          onClick={() => setFilter(null)}
          className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${filter === null ? "bg-gray-900 text-white border-gray-900" : "bg-white text-gray-600 border-gray-200 hover:bg-gray-50"}`}
        >
          All layers
        </button>
        {availableLayers.map(({ i }) => (
          <button
            key={i}
            onClick={() => setFilter(i)}
            className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${filter === i ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:bg-gray-50"}`}
          >
            Layer {i + 1}
          </button>
        ))}
      </div>

      {/* Histogram bars */}
      <div className="mt-3">
        <div className="flex items-end gap-px h-28 px-1">
          {bins.map((count, idx) => {
            const h = maxCount > 0 ? (count / maxCount) * 100 : 0;
            // Center bins (near 0) in gray, tails in indigo to highlight outliers
            const center = BIN_COUNT / 2;
            const dist = Math.abs(idx - center + 0.5) / center;
            const opacity = 0.35 + dist * 0.65;
            return (
              <div
                key={idx}
                className="flex-1 rounded-t"
                style={{
                  height: `${h}%`,
                  minHeight: count > 0 ? 2 : 0,
                  backgroundColor: `rgba(99,102,241,${opacity.toFixed(2)})`,
                  borderTop: count > 0 ? "1px solid rgba(99,102,241,0.6)" : undefined,
                }}
                title={`bin ${idx + 1}: ${count} weights`}
              />
            );
          })}
        </div>
        {/* X axis labels */}
        <div className="flex justify-between text-[10px] font-mono text-gray-400 px-1 mt-1">
          <span>{range[0].toFixed(1)}</span>
          <span>0</span>
          <span>{range[1].toFixed(1)}</span>
        </div>
        <div className="text-[10px] text-gray-400 text-center mt-0.5">weight value</div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="mt-3 grid grid-cols-4 gap-2 text-center border-t border-gray-100 pt-3">
          <div>
            <p className="text-[10px] text-gray-400 uppercase tracking-wide">Mean</p>
            <p className="text-xs font-mono font-semibold text-gray-700">{stats.mean.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-[10px] text-gray-400 uppercase tracking-wide">Std</p>
            <p className="text-xs font-mono font-semibold text-gray-700">{stats.std.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-[10px] text-gray-400 uppercase tracking-wide">Min</p>
            <p className="text-xs font-mono font-semibold text-gray-700">{stats.min.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-[10px] text-gray-400 uppercase tracking-wide">Max</p>
            <p className="text-xs font-mono font-semibold text-gray-700">{stats.max.toFixed(2)}</p>
          </div>
        </div>
      )}

      {totalLayers > 1 && filter === null && stats && stats.std < 0.08 && (
        <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 mt-3">
          Weights are tightly clustered near zero — possible vanishing gradients. Try a larger learning rate or a different activation (Leaky ReLU / Tanh).
        </p>
      )}
      {stats && stats.std > 1.2 && (
        <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded px-2 py-1.5 mt-3">
          Weights have exploded (std &gt; 1.2) — loss may diverge. Try a smaller learning rate.
        </p>
      )}
    </div>
  );
}
