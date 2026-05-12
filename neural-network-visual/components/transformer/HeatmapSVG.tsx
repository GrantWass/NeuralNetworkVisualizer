"use client";
import React, { useState } from "react";

type HeatmapSVGProps = {
  tokens: string[];
  matrix: number[][];
  cellSize?: number;
};

// Interpolate white → indigo-500 (#6366f1) based on value in [0,1]
function cellColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  const r = Math.round(255 - v * (255 - 99));
  const g = Math.round(255 - v * (255 - 102));
  const b = Math.round(255 - v * (255 - 241));
  return `rgb(${r},${g},${b})`;
}

// Filter [CLS], [SEP], and other BERT special tokens from the display
function stripSpecial(tokens: string[], matrix: number[][]): { tokens: string[]; matrix: number[][] } {
  const keep = tokens.map((t) => !/^\[.*\]$/.test(t));
  const indices = keep.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
  return {
    tokens: indices.map((i) => tokens[i]),
    matrix: indices.map((i) => indices.map((j) => matrix[i][j])),
  };
}

export const HeatmapSVG: React.FC<HeatmapSVGProps> = ({
  tokens: rawTokens,
  matrix: rawMatrix,
  cellSize = 30,
}) => {
  const { tokens, matrix: rawFiltered } = stripSpecial(rawTokens, rawMatrix);

  // Renormalize each row so weights sum to 1 after special tokens are stripped.
  // [CLS]/[SEP] absorbed part of the original softmax probability mass, so the
  // remaining values would otherwise sum to less than 1.
  const normalized = rawFiltered.map((row) => {
    const rowSum = row.reduce((a, b) => a + b, 0) || 1;
    return row.map((v) => v / rowSum);
  });

  // Rescale to local max so the full color range is used
  const localMax = Math.max(...normalized.flat(), 1e-9);
  const matrix = normalized.map((row) => row.map((v) => v / localMax));
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);

  const n = tokens.length;
  const labelPad = 70; // left labels
  const topPad = 70;   // top labels
  const width = labelPad + n * cellSize;
  const height = topPad + n * cellSize;

  // Truncate long tokens for display
  const display = (t: string) => (t.length > 8 ? t.slice(0, 7) + "…" : t);

  return (
    <div className="relative overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        style={{ fontFamily: "var(--font-geist-mono, monospace)" }}
      >
        {/* Top (column) labels — rotated 45° */}
        {tokens.map((t, j) => (
          <text
            key={`col-${j}`}
            x={labelPad + j * cellSize + cellSize / 2}
            y={topPad - 6}
            fontSize={cellSize < 24 ? 9 : 11}
            textAnchor="start"
            fill="currentColor"
            className="text-foreground"
            transform={`rotate(-45, ${labelPad + j * cellSize + cellSize / 2}, ${topPad - 6})`}
          >
            {display(t)}
          </text>
        ))}

        {/* Left (row) labels */}
        {tokens.map((t, i) => (
          <text
            key={`row-${i}`}
            x={labelPad - 6}
            y={topPad + i * cellSize + cellSize / 2 + 4}
            fontSize={cellSize < 24 ? 9 : 11}
            textAnchor="end"
            fill="currentColor"
            className="text-foreground"
          >
            {display(t)}
          </text>
        ))}

        {/* Heatmap cells — color uses normalized value, tooltip shows raw value */}
        {matrix.map((row, i) =>
          row.map((normVal, j) => {
            const rawVal = normalized[i][j];
            const x = labelPad + j * cellSize;
            const y = topPad + i * cellSize;
            const label = `${tokens[i]} attends to ${tokens[j]} with weight ${rawVal.toFixed(3)}`;
            return (
              <rect
                key={`cell-${i}-${j}`}
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                fill={cellColor(normVal)}
                stroke="rgba(0,0,0,0.06)"
                strokeWidth={0.5}
                role="img"
                aria-label={label}
                onMouseEnter={() => {
                  setTooltip({
                    x: x + cellSize / 2,
                    y: y,
                    text: `${tokens[i]} → ${tokens[j]}: ${rawVal.toFixed(3)}`,
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
                style={{ cursor: "default" }}
              />
            );
          })
        )}

        {/* Tooltip rendered in SVG */}
        {tooltip && (
          <g>
            <rect
              x={Math.min(tooltip.x - 60, width - 130)}
              y={Math.max(tooltip.y - 30, 2)}
              width={125}
              height={22}
              rx={4}
              fill="hsl(var(--popover))"
              stroke="hsl(var(--border))"
              strokeWidth={1}
            />
            <text
              x={Math.min(tooltip.x - 60, width - 130) + 6}
              y={Math.max(tooltip.y - 30, 2) + 15}
              fontSize={10}
              fill="hsl(var(--popover-foreground))"
            >
              {tooltip.text}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
};
