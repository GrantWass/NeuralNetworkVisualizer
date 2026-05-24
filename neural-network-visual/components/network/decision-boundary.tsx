"use client";
import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { forwardPassSingle } from "@/components/network/lib/store";
import type { NeuronLayer } from "@/components/network/static/types";

const XOR_COLORS = [
  [99, 102, 241],   // indigo
  [239, 68, 68],    // red
] as const;

const IRIS_COLORS = [
  [59, 130, 246],   // blue  — setosa
  [34, 197, 94],    // green — versicolor
  [249, 115, 22],   // orange — virginica
] as const;

const IRIS_CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"];
const IRIS_FEATURE_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"];

interface HoverInfo {
  cx: number; // canvas-relative x pixel
  cy: number; // canvas-relative y pixel
  probs: number[];
  featA: number;
  featB: number;
}

interface DecisionBoundaryProps {
  layers: NeuronLayer[];
  dataset: "xor" | "iris";
  originalData: number[][];
}

function computeIrisStats(originalData: number[][]) {
  const n = originalData.length;
  if (n === 0) return null;
  const means = [0, 1, 2, 3].map(
    (i) => originalData.reduce((s, r) => s + (r[i] ?? 0), 0) / n
  );
  const stds = [0, 1, 2, 3].map((i) => {
    const m = means[i];
    const variance = originalData.reduce((s, r) => s + ((r[i] ?? 0) - m) ** 2, 0) / n;
    return Math.sqrt(variance) || 1;
  });
  return { means, stds };
}

const GRID = 60;
const CANVAS = 200;

export function DecisionBoundary({ layers, dataset, originalData }: DecisionBoundaryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const gridCacheRef = useRef<{ imageData: ImageData; key: string } | null>(null);

  // Iris axis selectors — default to petal length (2) × petal width (3)
  const [axisA, setAxisA] = useState(2); // X axis feature index
  const [axisB, setAxisB] = useState(3); // Y axis feature index

  // Stable fingerprint of the inputs that determine the grid colors.
  // Sampling a few weights per layer is enough to detect any training update.
  const weightsKey = useMemo(() => {
    if (!layers.length) return "";
    const sample = layers.slice(1, 3).flatMap((l) => (l.weights?.[0] ?? []).slice(0, 8));
    return `${dataset}|${axisA}|${axisB}|${sample.map((v) => v.toFixed(4)).join(",")}`;
  }, [layers, dataset, axisA, axisB]);

  const getRange = useCallback(() => {
    if (dataset === "xor") {
      return { minA: -0.1, maxA: 1.1, minB: -0.1, maxB: 1.1 };
    }
    if (originalData.length === 0) return null;
    const valsA = originalData.map((r) => r[axisA] ?? 0);
    const valsB = originalData.map((r) => r[axisB] ?? 0);
    const pad = 0.1;
    const minA = Math.min(...valsA), maxA = Math.max(...valsA);
    const minB = Math.min(...valsB), maxB = Math.max(...valsB);
    return {
      minA: minA - (maxA - minA) * pad,
      maxA: maxA + (maxA - minA) * pad,
      minB: minB - (maxB - minB) * pad,
      maxB: maxB + (maxB - minB) * pad,
    };
  }, [dataset, originalData, axisA, axisB]);

  const predict = useCallback(
    (rawA: number, rawB: number): number[] => {
      let input: number[];
      if (dataset === "xor") {
        input = [rawA, rawB];
      } else {
        const stats = computeIrisStats(originalData);
        if (!stats) return [0, 0, 0];
        const { means, stds } = stats;
        // Build full 4-feature input; fix non-displayed features at their mean (= 0 after z-score)
        input = [0, 1, 2, 3].map((i) => {
          if (i === axisA) return (rawA - means[i]) / stds[i];
          if (i === axisB) return (rawB - means[i]) / stds[i];
          return 0; // fixed at mean
        });
      }
      const { A } = forwardPassSingle(layers, input);
      return A[A.length - 1] ?? [];
    },
    [layers, dataset, originalData, axisA, axisB]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const range = getRange();
    if (!range) return;
    const { minA, maxA, minB, maxB } = range;

    // Only recompute the expensive grid when weights/config changed
    if (!gridCacheRef.current || gridCacheRef.current.key !== weightsKey) {
      const imageData = ctx.createImageData(CANVAS, CANVAS);
      const cellSize = CANVAS / GRID;

      for (let gy = 0; gy < GRID; gy++) {
        for (let gx = 0; gx < GRID; gx++) {
          const rawA = minA + (gx / (GRID - 1)) * (maxA - minA);
          const rawB = maxB - (gy / (GRID - 1)) * (maxB - minB); // flip y
          const probs = predict(rawA, rawB);

          let r = 200, g = 200, b = 200;
          if (dataset === "xor") {
            const p = probs[0] ?? 0.5;
            const [c0, c1] = [XOR_COLORS[0], XOR_COLORS[1]];
            r = Math.round(c0[0] * (1 - p) + c1[0] * p);
            g = Math.round(c0[1] * (1 - p) + c1[1] * p);
            b = Math.round(c0[2] * (1 - p) + c1[2] * p);
          } else {
            const cls = probs.reduce((best, v, i) => (v > probs[best] ? i : best), 0);
            const conf = probs[cls] ?? 0.5;
            const alpha = 0.3 + conf * 0.6;
            const col = IRIS_COLORS[cls % IRIS_COLORS.length];
            r = Math.round(col[0] * alpha + 245 * (1 - alpha));
            g = Math.round(col[1] * alpha + 245 * (1 - alpha));
            b = Math.round(col[2] * alpha + 245 * (1 - alpha));
          }

          for (let dy = 0; dy < cellSize; dy++) {
            for (let dx = 0; dx < cellSize; dx++) {
              const px = Math.floor(gx * cellSize) + dx;
              const py = Math.floor(gy * cellSize) + dy;
              if (px >= CANVAS || py >= CANVAS) continue;
              const idx = (py * CANVAS + px) * 4;
              imageData.data[idx] = r;
              imageData.data[idx + 1] = g;
              imageData.data[idx + 2] = b;
              imageData.data[idx + 3] = 255;
            }
          }
        }
      }
      gridCacheRef.current = { imageData, key: weightsKey };
    }

    ctx.putImageData(gridCacheRef.current.imageData, 0, 0);

    // Draw training data points
    ctx.save();
    if (dataset === "xor") {
      const pts = [[0, 0], [0, 1], [1, 0], [1, 1]];
      const labels = [0, 1, 1, 0];
      for (let i = 0; i < pts.length; i++) {
        const [a, bv] = pts[i];
        const px = ((a - minA) / (maxA - minA)) * CANVAS;
        const py = (1 - (bv - minB) / (maxB - minB)) * CANVAS;
        const col = labels[i] === 0 ? XOR_COLORS[0] : XOR_COLORS[1];
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
      }
    } else {
      for (const row of originalData) {
        const rawA = row[axisA] ?? 0;
        const rawBv = row[axisB] ?? 0;
        const cls = [row[4] ?? 0, row[5] ?? 0, row[6] ?? 0].reduce(
          (best, v, i, arr) => (v > arr[best] ? i : best),
          0
        );
        const px = ((rawA - minA) / (maxA - minA)) * CANVAS;
        const py = (1 - (rawBv - minB) / (maxB - minB)) * CANVAS;
        const col = IRIS_COLORS[cls % IRIS_COLORS.length];
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
  }, [weightsKey, originalData, getRange, predict, dataset, axisA, axisB]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS / rect.width;
      const scaleY = CANVAS / rect.height;
      const px = (e.clientX - rect.left) * scaleX;
      const py = (e.clientY - rect.top) * scaleY;

      const range = getRange();
      if (!range) return;
      const { minA, maxA, minB, maxB } = range;
      const rawA = minA + (px / CANVAS) * (maxA - minA);
      const rawB = maxB - (py / CANVAS) * (maxB - minB);
      const probs = predict(rawA, rawB);
      setHover({ cx: e.clientX - rect.left, cy: e.clientY - rect.top, probs, featA: rawA, featB: rawB });
    },
    [getRange, predict]
  );

  const initialized = layers.length > 0 && layers[0].weights?.length > 0;

  const xLabel = dataset === "xor" ? "A" : IRIS_FEATURE_NAMES[axisA];
  const yLabel = dataset === "xor" ? "B" : IRIS_FEATURE_NAMES[axisB];

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Decision Boundary
      </p>

      {/* Iris axis selectors */}
      {dataset === "iris" && (
        <div className="flex gap-1.5 text-[10px] text-gray-500">
          <label className="flex items-center gap-1">
            X
            <select
              value={axisA}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v !== axisB) setAxisA(v);
              }}
              className="border border-gray-200 rounded px-1 py-0.5 text-[10px] bg-white"
            >
              {IRIS_FEATURE_NAMES.map((name, i) => (
                <option key={i} value={i} disabled={i === axisB}>{name}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1">
            Y
            <select
              value={axisB}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v !== axisA) setAxisB(v);
              }}
              className="border border-gray-200 rounded px-1 py-0.5 text-[10px] bg-white"
            >
              {IRIS_FEATURE_NAMES.map((name, i) => (
                <option key={i} value={i} disabled={i === axisA}>{name}</option>
              ))}
            </select>
          </label>
        </div>
      )}

      {!initialized ? (
        <div className="flex items-center justify-center h-[200px] text-sm text-gray-400">
          Initialize model to see boundary
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1">
          {/* Canvas with floating tooltip */}
          <div className="relative">
            {/* Y axis label */}
            <span className="absolute -left-5 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] text-gray-400 whitespace-nowrap">
              {yLabel}
            </span>
            <canvas
              ref={canvasRef}
              width={CANVAS}
              height={CANVAS}
              className="rounded border border-gray-200 cursor-crosshair block w-full h-auto"
              style={{ maxWidth: CANVAS }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHover(null)}
            />
            {/* Floating tooltip — follows cursor, no layout shift */}
            {hover && (
              <div
                className="pointer-events-none absolute z-10 rounded border border-gray-200 bg-white/95 px-2 py-1.5 text-xs shadow-md"
                style={{
                  left: hover.cx + (hover.cx > CANVAS * 0.6 ? -130 : 12),
                  top: hover.cy + (hover.cy > CANVAS * 0.7 ? -80 : 8),
                  minWidth: 110,
                }}
              >
                {dataset === "xor" ? (
                  <div className="flex flex-col gap-0.5">
                    <span className="text-gray-400">{xLabel}={hover.featA.toFixed(2)}, {yLabel}={hover.featB.toFixed(2)}</span>
                    <span className="font-mono font-semibold text-gray-800">
                      P(1)={(((hover.probs[0] ?? 0)) * 100).toFixed(1)}%
                    </span>
                  </div>
                ) : (
                  <div className="flex flex-col gap-0.5">
                    {IRIS_CLASS_NAMES.map((name, i) => (
                      <div key={name} className="flex justify-between gap-2">
                        <span className="text-gray-500">{name}</span>
                        <span className="font-mono font-semibold" style={{ color: `rgb(${IRIS_COLORS[i][0]},${IRIS_COLORS[i][1]},${IRIS_COLORS[i][2]})` }}>
                          {((hover.probs[i] ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
          {/* X axis label */}
          <span className="text-[9px] text-gray-400">{xLabel}</span>

          {/* Legend */}
          <div className="flex flex-wrap gap-2 justify-center mt-0.5">
            {dataset === "xor" ? (
              <>
                <LegendDot color="rgb(99,102,241)" label="Output 0" />
                <LegendDot color="rgb(239,68,68)" label="Output 1" />
              </>
            ) : (
              IRIS_CLASS_NAMES.map((name, i) => (
                <LegendDot
                  key={name}
                  color={`rgb(${IRIS_COLORS[i][0]},${IRIS_COLORS[i][1]},${IRIS_COLORS[i][2]})`}
                  label={name}
                />
              ))
            )}
          </div>

          {/* Iris 2D disclaimer */}
          {dataset === "iris" && (
            <p className="text-[10px] text-gray-400 text-center mt-1.5 leading-snug">
              The network uses all 4 features to classify — this slice shows only 2, so boundaries may look imperfect.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />
      <span className="text-[10px] text-gray-500">{label}</span>
    </div>
  );
}
