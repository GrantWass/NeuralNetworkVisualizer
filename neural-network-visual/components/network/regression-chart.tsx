"use client";
import { useRef, useEffect } from "react";
import type { NetworkState } from "@/components/network/static/types";

const CANVAS = 200;

interface RegressionChartProps {
  network: NetworkState;
  originalData: number[][];
  yMean: number | null;
  yStd: number | null;
}

type Point = { actual: number; pred: number };

function buildPoints(network: NetworkState, originalData: number[][], yMean: number | null, yStd: number | null): Point[] {
  const outputLayerIdx = network.layers.length - 2;
  const A = network.layers[outputLayerIdx]?.A;
  if (!A?.length) return [];
  const points: Point[] = [];
  for (let i = 0; i < originalData.length; i++) {
    const actualRaw = originalData[i]?.[4];
    const predNorm = A[i]?.[0];
    if (actualRaw === undefined || predNorm === undefined) continue;
    const pred = yMean !== null && yStd !== null ? predNorm * yStd + yMean : predNorm;
    points.push({ actual: actualRaw, pred });
  }
  return points;
}

export function RegressionChart({ network, originalData, yMean, yStd }: RegressionChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const points = buildPoints(network, originalData, yMean, yStd);
    if (points.length === 0) return;

    const actuals = points.map(p => p.actual);
    const preds = points.map(p => p.pred);
    const minV = Math.min(...actuals, ...preds);
    const maxV = Math.max(...actuals, ...preds);
    const pad = (maxV - minV) * 0.08;
    const lo = minV - pad, hi = maxV + pad;

    const toX = (v: number) => ((v - lo) / (hi - lo)) * CANVAS;
    const toY = (v: number) => CANVAS - ((v - lo) / (hi - lo)) * CANVAS;

    ctx.clearRect(0, 0, CANVAS, CANVAS);
    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(0, 0, CANVAS, CANVAS);

    // Perfect prediction line (y = x)
    ctx.beginPath();
    ctx.moveTo(toX(lo), toY(lo));
    ctx.lineTo(toX(hi), toY(hi));
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data points, colored by absolute error
    const MAX_ERR = 10;
    for (const { actual, pred } of points) {
      const t = Math.min(1, Math.abs(pred - actual) / MAX_ERR);
      const r = Math.round(22 + t * (239 - 22));
      const g = Math.round(163 + t * (68 - 163));
      const b = Math.round(74 + t * (68 - 74));
      ctx.beginPath();
      ctx.arc(toX(actual), toY(pred), 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
      ctx.fill();
    }

    // Axis tick labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(minV.toFixed(0), 2, CANVAS - 3);
    ctx.textAlign = "right";
    ctx.fillText(maxV.toFixed(0), CANVAS - 2, CANVAS - 3);
    ctx.textAlign = "right";
    ctx.fillText(maxV.toFixed(0), CANVAS - 2, 10);
  }, [network.layers, originalData, yMean, yStd]);

  const points = buildPoints(network, originalData, yMean, yStd);
  const initialized = points.length > 0;

  let r2: number | null = null;
  if (initialized) {
    const meanActual = points.reduce((s, p) => s + p.actual, 0) / points.length;
    const ssTot = points.reduce((s, p) => s + (p.actual - meanActual) ** 2, 0);
    const ssRes = points.reduce((s, p) => s + (p.actual - p.pred) ** 2, 0);
    r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Prediction vs Actual
      </p>

      {!initialized ? (
        <div className="flex items-center justify-center h-[200px] text-sm text-gray-400">
          Train model to see predictions
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1">
          <div className="relative">
            <span className="absolute -left-5 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] text-gray-400 whitespace-nowrap">
              Predicted MPG
            </span>

            {r2 !== null && (
              <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded px-2 py-1 border border-gray-200 shadow-sm">
                <p className="text-[8px] text-gray-400 uppercase tracking-wide leading-none">
                  R²
                </p>
                <p
                  className={`font-mono text-xs font-semibold leading-none mt-0.5 ${
                    r2 > 0.8
                      ? "text-green-600"
                      : r2 > 0.5
                        ? "text-yellow-600"
                        : "text-red-500"
                  }`}
                >
                  {r2.toFixed(3)}
                </p>
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={CANVAS}
              height={CANVAS}
              className="rounded border border-gray-200 block w-full h-auto cursor-crosshair"
              style={{ maxWidth: CANVAS }}
            />
          </div>
          <span className="text-[9px] text-gray-400">Actual MPG</span>

          <div className="flex gap-3 text-[9px] text-gray-500 mt-0.5">
            <span className="flex items-center gap-1">
              <span className="inline-block w-5 border-t border-dashed border-gray-400" />
              Perfect
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-green-500" />
              Close
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-400" />
              Off
            </span>
          </div>

          <p className="text-[10px] text-gray-400 text-center mt-1 leading-snug">
            Points on the diagonal = perfect predictions. Scatter = error.
          </p>
        </div>
      )}
    </div>
  );
}
