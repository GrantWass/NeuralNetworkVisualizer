"use client";
import React, { useState } from "react";

type QKVBreakdownProps = {
  tokens: string[];            // special tokens already stripped
  attentionMatrix: number[][];  // stripped matrix, rows/cols match tokens
  rawScoresMatrix?: number[][]; // pre-softmax scores (optional; falls back to log reconstruction)
  defaultSelected?: string;
};

export const QKVBreakdown: React.FC<QKVBreakdownProps> = ({
  tokens,
  attentionMatrix,
  rawScoresMatrix,
  defaultSelected = "it",
}) => {
  const defaultIdx = Math.max(
    0,
    tokens.findIndex((t) => t.toLowerCase() === defaultSelected.toLowerCase())
  );
  const [selectedIdx, setSelectedIdx] = useState(defaultIdx);

  const row = attentionMatrix[selectedIdx];
  const maxWeight = Math.max(...row, 1e-9);

  const label = (t: string) => (t.length > 10 ? t.slice(0, 9) + "…" : t);

  return (
    <div className="space-y-6">
      {/* Token selector */}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Select a token">
        {tokens.map((t, i) => (
          <button
            key={i}
            onClick={() => setSelectedIdx(i)}
            aria-label={`Select token ${t}`}
            aria-pressed={i === selectedIdx}
            className={[
              "px-3 py-1 rounded-md text-sm font-mono border transition-colors",
              i === selectedIdx
                ? "bg-indigo-500 text-white border-indigo-500"
                : "bg-background border-border hover:border-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-950",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="space-y-2 rounded-lg border border-border p-4 bg-card">
        {/* Step 2: softmax weights */}
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Attention weights
            <span className="ml-2 font-normal normal-case">softmax(scores)</span>
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            Softmax turns similarity scores into a probability distribution that
            sums to 1. Higher-scoring tokens capture more weight, which is why a
            few bars dominate while the rest stay small.
          </p>
          <div className="space-y-1 mt-2">
            {tokens.map((t, i) => (
              <div key={i} className="flex items-center gap-3">
                <span
                  className={[
                    "text-xs font-mono w-20 shrink-0 text-right",
                    i === selectedIdx
                      ? "text-indigo-500 font-semibold"
                      : "text-muted-foreground",
                  ].join(" ")}
                >
                  {label(t)}
                </span>
                <div className="h-3 bg-indigo-100 dark:bg-indigo-950 rounded overflow-hidden w-40">
                  <div
                    className="h-full bg-indigo-500 rounded"
                    style={{ width: `${((row[i] / maxWeight) * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs font-mono w-12 text-right tabular-nums text-muted-foreground">
                  {(row[i] * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
