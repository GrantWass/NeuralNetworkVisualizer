"use client";
import { useRef, useEffect, useState, useCallback } from "react";
import { forwardPassSingle } from "@/components/network/lib/store";
import type { NeuronLayer } from "@/components/network/static/types";

// XOR: class 0 = blue, class 1 = red
// Iris: class 0 (setosa) = blue, class 1 (versicolor) = green, class 2 (virginica) = orange
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

interface HoverInfo {
  x: number;
  y: number;
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

export function DecisionBoundary({ layers, dataset, originalData }: DecisionBoundaryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const GRID = 60;
  const CANVAS = 200;

  const getRange = useCallback(() => {
    if (dataset === "xor") {
      return { minA: -0.1, maxA: 1.1, minB: -0.1, maxB: 1.1 };
    }
    // Iris: petal_length (col 2), petal_width (col 3)
    if (originalData.length === 0) return null;
    const vals2 = originalData.map((r) => r[2] ?? 0);
    const vals3 = originalData.map((r) => r[3] ?? 0);
    const pad = 0.1;
    const minA = Math.min(...vals2);
    const maxA = Math.max(...vals2);
    const minB = Math.min(...vals3);
    const maxB = Math.max(...vals3);
    const rangeA = (maxA - minA) * pad;
    const rangeB = (maxB - minB) * pad;
    return { minA: minA - rangeA, maxA: maxA + rangeA, minB: minB - rangeB, maxB: maxB + rangeB };
  }, [dataset, originalData]);

  const predict = useCallback(
    (rawA: number, rawB: number): number[] => {
      let input: number[];
      if (dataset === "xor") {
        input = [rawA, rawB];
      } else {
        const stats = computeIrisStats(originalData);
        if (!stats) return [0, 0, 0];
        const { means, stds } = stats;
        input = [
          (means[0] - means[0]) / stds[0], // sepal_length at mean → 0
          (means[1] - means[1]) / stds[1], // sepal_width at mean → 0
          (rawA - means[2]) / stds[2],
          (rawB - means[3]) / stds[3],
        ];
      }
      const { A } = forwardPassSingle(layers, input);
      return A[A.length - 1] ?? [];
    },
    [layers, dataset, originalData]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const range = getRange();
    if (!range) return;
    const { minA, maxA, minB, maxB } = range;

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

        // Fill the cell
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
    ctx.putImageData(imageData, 0, 0);

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
        const rawA = row[2] ?? 0;
        const rawBv = row[3] ?? 0;
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
  }, [layers, dataset, originalData, getRange, predict]);

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
      setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top, probs, featA: rawA, featB: rawB });
    },
    [getRange, predict]
  );

  const initialized = layers.length > 0 && layers[0].weights?.length > 0;

  const axisLabels =
    dataset === "xor"
      ? { a: "A", b: "B" }
      : { a: "Petal length", b: "Petal width" };

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Decision Boundary
      </p>
      {!initialized ? (
        <div className="flex items-center justify-center h-[200px] text-sm text-gray-400">
          Initialize model to see boundary
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1">
          <div className="relative">
            <canvas
              ref={canvasRef}
              width={CANVAS}
              height={CANVAS}
              className="rounded border border-gray-200 cursor-crosshair"
              style={{ width: CANVAS, height: CANVAS }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHover(null)}
            />
            {/* Y axis label */}
            <span className="absolute -left-5 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] text-gray-400 whitespace-nowrap">
              {axisLabels.b}
            </span>
          </div>
          {/* X axis label */}
          <span className="text-[9px] text-gray-400">{axisLabels.a}</span>

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

          {/* Hover tooltip */}
          {hover && (
            <div className="w-full rounded border border-gray-200 bg-white px-2 py-1.5 text-xs shadow-sm">
              {dataset === "xor" ? (
                <div className="flex justify-between gap-3">
                  <span className="text-gray-500">
                    A={hover.featA.toFixed(2)}, B={hover.featB.toFixed(2)}
                  </span>
                  <span className="font-mono font-semibold text-gray-800">
                    P(1)={((hover.probs[0] ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
              ) : (
                <div className="flex flex-col gap-0.5">
                  <span className="text-gray-400 text-[10px]">
                    Petal {hover.featA.toFixed(2)} × {hover.featB.toFixed(2)} cm
                  </span>
                  {IRIS_CLASS_NAMES.map((name, i) => (
                    <div key={name} className="flex justify-between">
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
