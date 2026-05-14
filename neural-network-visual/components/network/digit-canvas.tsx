"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import useStore from "@/components/network/lib/store";

const CANVAS_SIZE = 280; // display size in px
const GRID = 28;
const CELL = CANVAS_SIZE / GRID; // 10px per cell

function downsample(ctx: CanvasRenderingContext2D): number[] {
  const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const pixels: number[] = new Array(GRID * GRID).fill(0);

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      let sum = 0;
      let count = 0;
      const startY = row * CELL;
      const startX = col * CELL;
      for (let dy = 0; dy < CELL; dy++) {
        for (let dx = 0; dx < CELL; dx++) {
          const px = Math.floor(startX + dx);
          const py = Math.floor(startY + dy);
          const idx = (py * CANVAS_SIZE + px) * 4;
          // canvas is white bg, black strokes — invert so digit=bright like MNIST
          const alpha = imageData.data[idx + 3];
          const r = imageData.data[idx];
          // white background (r=255,a=255) → 0; black stroke (r=0,a=255) → 1
          sum += alpha > 0 ? (255 - r) / 255 : 0;
          count++;
        }
      }
      pixels[row * GRID + col] = sum / count;
    }
  }
  return pixels;
}

export default function DigitCanvas({ hidePrediction = false }: { hidePrediction?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);
  const { predictDigit, clearDigitPrediction, drawnDigitPrediction, sessionId } = useStore();
  const [hasStrokes, setHasStrokes] = useState(false);

  // White background on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    if ("touches" in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const stampBrush = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
    const r = 18;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0,   "rgba(0,0,0,1)");
    g.addColorStop(0.35,"rgba(0,0,0,0.85)");
    g.addColorStop(0.7, "rgba(0,0,0,0.35)");
    g.addColorStop(1,   "rgba(0,0,0,0)");
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
  };

  const draw = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    if (lastPos.current) {
      const dx = x - lastPos.current.x;
      const dy = y - lastPos.current.y;
      const steps = Math.max(1, Math.ceil(Math.sqrt(dx * dx + dy * dy) / 3));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        stampBrush(ctx, lastPos.current.x + dx * t, lastPos.current.y + dy * t);
      }
    } else {
      stampBrush(ctx, x, y);
    }
    lastPos.current = { x, y };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDrawing.current = true;
    lastPos.current = null;
    const pos = getPos(e);
    draw(pos.x, pos.y);
    setHasStrokes(true);
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing.current) return;
    draw(getPos(e).x, getPos(e).y);
  };
  const handleMouseUp = () => { isDrawing.current = false; lastPos.current = null; };

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    isDrawing.current = true;
    lastPos.current = null;
    const pos = getPos(e);
    draw(pos.x, pos.y);
    setHasStrokes(true);
  };
  const handleTouchMove = (e: React.TouchEvent) => {
    e.preventDefault();
    if (!isDrawing.current) return;
    draw(getPos(e).x, getPos(e).y);
  };
  const handleTouchEnd = () => { isDrawing.current = false; lastPos.current = null; };

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasStrokes(false);
    clearDigitPrediction();
  };

  const handlePredict = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const pixels = downsample(ctx);
    predictDigit(pixels);
  };

  return (
    <div className="flex flex-col items-center gap-3">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="border-2 border-gray-300 rounded-lg cursor-crosshair touch-none"
        style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      />
      <div className="flex gap-2 w-full max-w-[280px]">
        <Button variant="outline" onClick={handleClear} className="flex-1 text-sm">
          Clear
        </Button>
        <Button
          onClick={handlePredict}
          disabled={!sessionId || !hasStrokes}
          className="flex-1 text-sm"
        >
          Predict
        </Button>
      </div>

      {/* Prediction result */}
      {!hidePrediction && drawnDigitPrediction && (
        <div className="w-full max-w-[340px] space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-700">Prediction</p>
            <span className="text-2xl font-bold text-indigo-600">
              {drawnDigitPrediction.predictedClass}
            </span>
          </div>
          <div className="space-y-1">
            {drawnDigitPrediction.confidences.map((conf, digit) => {
              const isPredicted = digit === drawnDigitPrediction.predictedClass;
              const pct = (conf * 100).toFixed(1);
              return (
                <div key={digit} className="flex items-center gap-2">
                  <span className={`w-4 text-xs font-mono text-right flex-shrink-0 ${isPredicted ? "font-bold text-indigo-700" : "text-gray-500"}`}>
                    {digit}
                  </span>
                  <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${isPredicted ? "bg-indigo-500" : "bg-gray-300"}`}
                      style={{ width: `${(conf * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className={`w-10 text-xs text-right flex-shrink-0 ${isPredicted ? "font-bold text-indigo-700" : "text-gray-400"}`}>
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!sessionId && (
        <p className="text-xs text-gray-400 text-center">Initialize the model first to enable prediction.</p>
      )}
    </div>
  );
}
