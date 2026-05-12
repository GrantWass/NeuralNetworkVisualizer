"use client";
import React, { useState } from "react";
import { Gloss } from "./Gloss";

// ─── Color helpers ────────────────────────────────────────────────────────────

// Diverging scale: negative → slate-400, zero → white, positive → indigo-500
function vecColor(v: number, maxAbs: number): string {
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0;
  if (t >= 0) {
    const r = Math.round(255 - t * (255 - 99));
    const g = Math.round(255 - t * (255 - 102));
    const b = Math.round(255 - t * (255 - 241));
    return `rgb(${r},${g},${b})`;
  }
  const s = -t;
  return `rgb(${Math.round(255 - s * (255 - 148))},${Math.round(255 - s * (255 - 163))},${Math.round(255 - s * (255 - 184))})`;
}

// ─── Vector strip ─────────────────────────────────────────────────────────────

function VectorStrip({
  vec,
  maxAbs,
  cellW = 8,
  cellH = 16,
}: {
  vec: number[];
  maxAbs: number;
  cellW?: number;
  cellH?: number;
}) {
  const w = vec.length * cellW;
  return (
    <svg width={w} height={cellH} viewBox={`0 0 ${w} ${cellH}`} style={{ display: "block", flexShrink: 0 }}>
      {vec.map((v, i) => (
        <rect key={i} x={i * cellW} y={0} width={cellW - 0.5} height={cellH} fill={vecColor(v, maxAbs)} />
      ))}
    </svg>
  );
}

// ─── Props ────────────────────────────────────────────────────────────────────

type QKVBreakdownProps = {
  tokens: string[];
  attentionMatrix: number[][];
  rawScoresMatrix?: number[][];
  queryVectors?: number[][][];      // 4 heads × seq_len × head_dim
  keyVectors?: number[][][];        // 4 heads × seq_len × head_dim
  multiHeadAttention?: number[][][]; // 4 heads × seq_len × seq_len
  multiHeadRawScores?: number[][][]; // 4 heads × seq_len × seq_len
  headIndices?: number[];
  defaultSelected?: string;
};

export const QKVBreakdown: React.FC<QKVBreakdownProps> = ({
  tokens,
  attentionMatrix,
  rawScoresMatrix,
  queryVectors,
  keyVectors,
  multiHeadAttention,
  multiHeadRawScores,
  headIndices,
  defaultSelected = "it",
}) => {
  const defaultIdx = Math.max(
    0,
    tokens.findIndex((t) => t.toLowerCase() === defaultSelected.toLowerCase())
  );
  const [selectedTokenIdx, setSelectedTokenIdx] = useState(defaultIdx);
  const [selectedHeadSlot, setSelectedHeadSlot] = useState(0);

  const hasVectors = !!queryVectors && !!keyVectors;
  const numHeads = queryVectors?.length ?? 0;

  // Per-head data for the selected head slot
  const qVecs = hasVectors ? queryVectors![selectedHeadSlot] : null; // seq_len × head_dim
  const kVecs = hasVectors ? keyVectors![selectedHeadSlot] : null;
  const headScoreRow = multiHeadRawScores?.[selectedHeadSlot]?.[selectedTokenIdx]
    ?? rawScoresMatrix?.[selectedTokenIdx];

  // Renormalize weights after [CLS]/[SEP] have been stripped — the remaining
  // values no longer sum to 1 because the special tokens absorbed part of the
  // probability mass in the original softmax output.
  const rawHeadWeightRow = multiHeadAttention?.[selectedHeadSlot]?.[selectedTokenIdx]
    ?? attentionMatrix[selectedTokenIdx];
  const weightRowSum = rawHeadWeightRow.reduce((a, b) => a + b, 0) || 1;
  const headWeightRow = rawHeadWeightRow.map((w) => w / weightRowSum);

  // Global max absolute value across all Q and K vectors for this head (consistent color scale)
  const maxAbsVec = hasVectors
    ? Math.max(...qVecs!.flat().map(Math.abs), ...kVecs!.flat().map(Math.abs), 1e-9)
    : 1;

  // Score bar normalization
  const scoreMax = headScoreRow ? Math.max(...headScoreRow) : 1;
  const scoreMin = headScoreRow ? Math.min(...headScoreRow) : 0;
  const scorePct = (s: number) =>
    headScoreRow ? ((s - scoreMin) / (scoreMax - scoreMin || 1)) * 100 : 0;

  const maxWeight = Math.max(...headWeightRow, 1e-9);

  const truncate = (t: string) => (t.length > 10 ? t.slice(0, 9) + "…" : t);
  const headLabel = (slot: number) =>
    headIndices ? `Head ${headIndices[slot] + 1}` : `Head ${slot + 1}`;

  return (
    <div className="space-y-6">
      {/* Token selector */}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Select a token">
        {tokens.map((t, i) => (
          <button
            key={i}
            onClick={() => setSelectedTokenIdx(i)}
            aria-pressed={i === selectedTokenIdx}
            className={[
              "px-3 py-1 rounded-md text-sm font-mono border transition-colors",
              i === selectedTokenIdx
                ? "bg-indigo-500 text-white border-indigo-500"
                : "bg-background border-border hover:border-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-950",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Head selector — only shown when per-head data is available */}
      {hasVectors && numHeads > 1 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-muted-foreground">Head:</span>
          {Array.from({ length: numHeads }, (_, slot) => (
            <button
              key={slot}
              onClick={() => setSelectedHeadSlot(slot)}
              aria-pressed={slot === selectedHeadSlot}
              className={[
                "px-2.5 py-0.5 rounded text-xs font-mono border transition-colors",
                slot === selectedHeadSlot
                  ? "bg-indigo-500 text-white border-indigo-500"
                  : "bg-background border-border hover:border-indigo-400",
              ].join(" ")}
            >
              {headLabel(slot)}
            </button>
          ))}
        </div>
      )}

      <div className="space-y-6 rounded-lg border border-border p-4 bg-card">
        {/* Q vector */}
        {hasVectors && qVecs && (
          <div className="space-y-2">
            <p className="text-xs  text-muted-foreground">
              <Gloss term="Query">
                A learned linear projection of the token into a 64-d vector encoding what
                this token is searching for. At training time the weights are optimized so
                that tokens needing related information produce aligned Query and Key vectors.
              </Gloss>
              {" "}—{" "}
              <span className="font-mono font-normal normal-case">{tokens[selectedTokenIdx]}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Each cell is one dimension of the 64-d Query vector. Indigo = positive, slate = negative.
              Dimensions where this Query and a token's Key share the same sign contribute
              positively to the similarity score.
            </p>
            <div className="overflow-x-auto">
              <div className="flex items-center gap-3 min-w-0">
                <span className="text-xs font-mono text-indigo-500 w-20 shrink-0 text-right font-semibold">
                  {truncate(tokens[selectedTokenIdx])}
                </span>
                <VectorStrip vec={qVecs[selectedTokenIdx]} maxAbs={maxAbsVec} />
              </div>
            </div>
          </div>
        )}

        {/* K vectors */}
        {hasVectors && kVecs && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              <Gloss term="Keys">
                Each token projects itself into a 64-d Key vector encoding what it contains
                and is willing to share. Matching Q against K is how the model decides who
                attends to whom — it's entirely learned from data, not hand-coded rules.
              </Gloss>
              {" "}— all tokens
            </p>
            <p className="text-xs text-muted-foreground">
              Scan vertically: where a Key column looks similar to the Query above (same
              color pattern), the dot product will be high and that token will attract
              more attention weight.
            </p>
            <div className="overflow-x-auto">
              <div className="space-y-1 min-w-0">
                {tokens.map((t, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span
                      className={[
                        "text-xs font-mono w-20 shrink-0 text-right",
                        i === selectedTokenIdx ? "text-indigo-500 font-semibold" : "text-muted-foreground",
                      ].join(" ")}
                    >
                      {truncate(t)}
                    </span>
                    <VectorStrip vec={kVecs[i]} maxAbs={maxAbsVec} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Scores → Weights side by side */}
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Pre-softmax scores */}
          {headScoreRow && (
            <div className="space-y-2 flex-1">
              <p className="text-xs text-muted-foreground">
                <Gloss term="Similarity scores">
                  The dot product Q·K sums the element-wise products of the two vectors. Dimensions
                  where both are positive or both are negative add to the score; dimensions where
                  they disagree subtract. Dividing by √64 = 8 prevents scores from growing so
                  large that softmax saturates and gradients vanish during training.
                </Gloss>
                <span className="ml-2 font-normal text-muted-foreground/70">Q · K / √d</span>
              </p>
              <p className="text-xs text-muted-foreground">
                Raw values before softmax. Can be positive or negative.
              </p>
              <div className="space-y-1 mt-1">
                {tokens.map((t, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span className={["text-xs font-mono w-20 shrink-0 text-right", i === selectedTokenIdx ? "text-indigo-500 font-semibold" : "text-muted-foreground"].join(" ")}>
                      {truncate(t)}
                    </span>
                    <div className="h-3 bg-indigo-100 dark:bg-indigo-950 rounded overflow-hidden w-32">
                      <div className="h-full bg-indigo-400 rounded" style={{ width: `${scorePct(headScoreRow[i]).toFixed(1)}%` }} />
                    </div>
                    <span className="text-xs font-mono w-10 text-right tabular-nums text-muted-foreground">
                      {headScoreRow[i].toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Softmax weights */}
          <div className="space-y-2 flex-1">
            <p className="text-xs text-muted-foreground">
              <Gloss term="Attention weights">
                Softmax applies eˣⁱ / Σeˣ to the scores. The exponential amplifies differences
                non-linearly: a score advantage of 2.0 gives ~7× more weight, not 2×. This means
                attention concentrates — the highest-scoring token captures a disproportionately
                large fraction of the budget while low-scoring tokens receive almost nothing.
              </Gloss>
              <span className="ml-2 font-normal text-muted-foreground/70">softmax(scores)</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Probabilities summing to 100%. Same ordering as scores.
            </p>
            <div className="space-y-1 mt-1">
              {tokens.map((t, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className={["text-xs font-mono w-20 shrink-0 text-right", i === selectedTokenIdx ? "text-indigo-500 font-semibold" : "text-muted-foreground"].join(" ")}>
                    {truncate(t)}
                  </span>
                  <div className="h-3 bg-indigo-100 dark:bg-indigo-950 rounded overflow-hidden w-32">
                    <div className="h-full bg-indigo-500 rounded" style={{ width: `${((headWeightRow[i] / maxWeight) * 100).toFixed(1)}%` }} />
                  </div>
                  <span className="text-xs font-mono w-10 text-right tabular-nums text-muted-foreground">
                    {(headWeightRow[i] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Output */}
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">
            <Gloss term="Output">
              output = Σᵢ wᵢ · Vᵢ. Before this step, every token has a fixed context-free
              embedding — the same vector regardless of sentence. After it, the representation
              is a weighted blend of all Value vectors. The same word produces a different
              output in every sentence because the weights change with context. This is how
              "it" comes to encode that it refers to an animal rather than a street.
            </Gloss>
          </p>
          <p className="text-xs text-muted-foreground max-w-prose">
            Each token projects into a{" "}
            <Gloss term="Value">
              A third learned projection, independent of Q and K. Q and K decide who attends
              to whom; V decides what gets transferred when they do. Think of K as the reason
              a token gets selected, and V as the actual message it sends. The same token can
              have a very different Key pattern and Value pattern — matching and contributing
              are separate learned behaviors.
            </Gloss>{" "}
            vector. The output for <em>{tokens[selectedTokenIdx]}</em> is the weighted sum of
            all Value vectors using the weights above — the token's new representation is no
            longer just itself, it's a mixture of the whole sentence weighted by relevance.
          </p>

          {/* Equation */}
          <div className="rounded-md border border-border bg-muted/40 px-4 py-3 space-y-2 overflow-x-auto">
            {/* General form */}
            <p className="text-xs font-mono text-foreground whitespace-nowrap">
              output({tokens[selectedTokenIdx]}) = Σᵢ wᵢ · V(tokenᵢ)
            </p>
            {/* Expanded with real weights */}
            <p className="text-xs font-mono text-muted-foreground whitespace-nowrap">
              {"= "}
              {tokens.map((t, i) => {
                const pct = (headWeightRow[i] * 100).toFixed(1);
                return (
                  <span key={i}>
                    <span className={headWeightRow[i] === Math.max(...headWeightRow) ? "text-indigo-500 font-semibold" : ""}>
                      {pct}%·V({t})
                    </span>
                    {i < tokens.length - 1 && <span className="text-muted-foreground/50"> + </span>}
                  </span>
                );
              })}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
