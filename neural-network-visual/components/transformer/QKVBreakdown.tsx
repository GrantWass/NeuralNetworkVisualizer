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

  // Use real pre-softmax scores when available; otherwise reconstruct from
  // log(softmax_weight) which preserves relative ordering.
  const rawRow = rawScoresMatrix
    ? rawScoresMatrix[selectedIdx]
    : row.map((w) => Math.log(Math.max(w, 1e-9)));

  const scoreMax = Math.max(...rawRow);
  const scoreMin = Math.min(...rawRow);
  const scores = rawRow.map((s) => s - scoreMax); // shift so max = 0

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

      <div className="space-y-6 rounded-lg border border-border p-4 bg-card">
        {/* Step 1: pre-softmax scores */}
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Step 1 — Similarity scores
            <span className="ml-2 font-normal normal-case">
              Q<sub>{tokens[selectedIdx]}</sub> · K<sub>i</sub> / √d
            </span>
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            Each token projects itself into a <strong>Key</strong> vector.{" "}
            <em>{tokens[selectedIdx]}</em>'s <strong>Query</strong> is compared
            against each Key via a dot product. Higher scores mean stronger
            alignment — the softmax output below is derived directly from these.
          </p>
          <div className="space-y-1 mt-2">
            {tokens.map((t, i) => {
              const norm = (scores[i] - scoreMin) / (-scoreMin || 1);
              return (
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
                      className="h-full bg-indigo-400 rounded"
                      style={{ width: `${(norm * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono w-12 text-right tabular-nums text-muted-foreground">
                    {rawRow[i].toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Step 2: softmax weights */}
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Step 2 — Attention weights
            <span className="ml-2 font-normal normal-case">softmax(scores)</span>
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            Softmax converts the raw scores into a probability distribution
            summing to 1. Exponential amplification means the highest-scoring
            token captures a disproportionate share of the weight.
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

        {/* Step 3: output */}
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Step 3 — Output
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            Each token also projects itself into a <strong>Value</strong> vector.
            The output for <em>{tokens[selectedIdx]}</em> is a weighted sum of
            all Value vectors using the weights above — tokens with higher
            attention weight contribute more of their meaning to the result. This
            is how context from across the sentence is folded into a single
            representation.
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            These are real attention weights from BERT layer 6, averaged across
            all 12 heads.
          </p>
        </div>
      </div>
    </div>
  );
};
