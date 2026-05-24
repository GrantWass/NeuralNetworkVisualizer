"use client";
import React, { useState } from "react";
import type { VizExample } from "./TransformerArchitectureViz";

// ─── Config ──────────────────────────────────────────────────────────────────

const STRIP_SHOW    = 48;  // dims rendered in each vector strip
const DEFAULT_TOP_K = 10;  // initial top-k shown

// ─── Small inline SVG strip ──────────────────────────────────────────────────

function VecStrip({
  vec,
  rgb,
  w = 192,
  h = 20,
}: {
  vec: number[];
  rgb: [number, number, number];
  w?: number;
  h?: number;
}) {
  const dims = Math.min(STRIP_SHOW, vec.length);
  const cellW = w / dims;
  const maxAbs = Math.max(...vec.slice(0, dims).map(Math.abs), 0.1);
  const [r, g, b] = rgb;
  return (
    <svg width={w} height={h} style={{ display: "block", flexShrink: 0 }}>
      {Array.from({ length: dims }, (_, i) => {
        const v = vec[i] ?? 0;
        const t = Math.min(Math.abs(v) / maxAbs, 1);
        const a = 0.1 + t * 0.85;
        const fill =
          v >= 0
            ? `rgba(${r},${g},${b},${a.toFixed(2)})`
            : `rgba(100,116,139,${(a * 0.55).toFixed(2)})`;
        return (
          <rect key={i} x={i * cellW} y={0} width={Math.max(cellW - 0.5, 0.5)} height={h} fill={fill} />
        );
      })}
    </svg>
  );
}

// ─── Arrow SVG ───────────────────────────────────────────────────────────────

function Arrow({ label, sublabel }: { label?: string; sublabel?: string }) {
  return (
    <div className="flex flex-col items-center justify-center shrink-0 px-1 gap-1" style={{ minWidth: 64 }}>
      {label && (
        <span className="text-[9px] font-semibold text-muted-foreground text-center leading-tight whitespace-nowrap">
          {label}
        </span>
      )}
      {sublabel && (
        <span className="text-[8px] text-muted-foreground text-center leading-tight whitespace-nowrap">
          {sublabel}
        </span>
      )}
      <svg width={48} height={16} style={{ display: "block" }}>
        <defs>
          <marker id="lmh-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="hsl(var(--border))" />
          </marker>
        </defs>
        <line x1={2} y1={8} x2={42} y2={8} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#lmh-arrow)" />
      </svg>
    </div>
  );
}

// ─── Step card wrapper ────────────────────────────────────────────────────────

function StepCard({
  step,
  label,
  accent,
  children,
}: {
  step: number;
  label: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2 shrink-0">
      <div className="flex items-center gap-1.5">
        <span
          className="inline-flex h-4 w-4 rounded-full items-center justify-center text-[9px] font-bold shrink-0"
          style={{ background: accent, color: "white" }}
        >
          {step}
        </span>
        <span className="text-[10px] font-semibold tracking-wide text-muted-foreground uppercase whitespace-nowrap">
          {label}
        </span>
      </div>
      {children}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function LMHeadViz({ example }: { example: VizExample }) {
  const { layers, nextWordProbs, tokens } = example;
  const [showMath, setShowMath] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(Math.min(DEFAULT_TOP_K, nextWordProbs.length));

  const MAX_K = nextWordProbs.length;

  const lastLayer  = layers[layers.length - 1];
  const lastTokIdx = tokens.length - 1;
  const lastToken  = tokens[lastTokIdx];
  const hidden     = lastLayer.ffnOut[lastTokIdx];

  // LayerNorm approximation over the visible dims
  const visHidden = hidden.slice(0, STRIP_SHOW);
  const mean = visHidden.reduce((a, b) => a + b, 0) / visHidden.length;
  const variance =
    visHidden.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) /
    visHidden.length;
  const std = Math.sqrt(variance + 1e-5);
  const normalized = hidden.map((v) => (v - mean) / std);

  // Raw logits from model output (temperature-independent, top-20 available)
  const allRawLogits = nextWordProbs.map((p) => Math.log(Math.max(p.prob, 1e-9)));

  // Top-k slice of raw logits for the logit step display
  const kTokens    = nextWordProbs.slice(0, topK);
  const kLogits    = allRawLogits.slice(0, topK);
  const maxRawL    = kLogits[0] ?? 0;
  const minRawL    = kLogits[kLogits.length - 1] ?? 0;
  const rawSpan    = Math.max(maxRawL - minRawL, 0.01);

  // Apply temperature + top-k → recomputed probabilities
  const scaledLogits = kLogits.map((l) => l / temperature);
  const maxScaled    = Math.max(...scaledLogits);
  const exps         = scaledLogits.map((l) => Math.exp(l - maxScaled));
  const sumExps      = exps.reduce((a, b) => a + b, 0);
  const displayProbs = kTokens.map((p, i) => ({ token: p.token, prob: exps[i] / sumExps }));
  const maxProb      = displayProbs[0]?.prob ?? 1;

  // Fabricate per-token embedding strips (deterministic, no real embeddings in data)
  function fakeEmb(rank: number): number[] {
    return Array.from({ length: STRIP_SHOW }, (_, d) => {
      const signal = normalized[d] ?? 0;
      const corr   = Math.max(0, 1 - rank * 0.12);
      const noise  = Math.sin(rank * 6.73 + d * 4.19) * (1 - corr * 0.6);
      return signal * corr + noise * 0.55;
    });
  }

  const tempLabel =
    temperature < 0.7 ? "sharp / confident" :
    temperature > 1.4 ? "flat / random" :
    temperature === 1.0 ? "default" : "slightly varied";

  return (
    <div
      className="border-t border-border bg-gradient-to-b from-card to-card/60 px-5 py-4 space-y-4"
      aria-label="Language model head visualization"
    >
      {/* ── Header ── */}
      <div className="flex items-center gap-3 flex-wrap">
        <h3 className="text-xs font-bold tracking-widest text-muted-foreground uppercase">
          LM Head — How the Next Token is Predicted
        </h3>
        <div className="flex-1 h-px bg-border min-w-4" />
        <button
          onClick={() => setShowMath((v) => !v)}
          className="text-[10px] font-medium text-indigo-500 hover:text-indigo-400 transition-colors"
        >
          {showMath ? "hide math" : "show math"}
        </button>
      </div>

      {showMath && (
        <div className="rounded-lg border border-border bg-secondary/40 px-4 py-3 text-xs text-muted-foreground leading-relaxed max-w-2xl space-y-1.5">
          <p>
            <code className="font-mono text-yellow-400">x_norm = LayerNorm(x_last)</code>
            {" "}— stabilize the final hidden state
          </p>
          <p>
            <code className="font-mono text-indigo-400">logits = W_unembed · x_norm</code>
            {" "}— one dot product per vocabulary word (50,257 total)
          </p>
          <p>
            <code className="font-mono text-orange-400">scaled = logits / temperature</code>
            {" "}— divide by T before softmax; T{"<"}1 sharpens, T{">"}1 flattens
          </p>
          <p>
            <code className="font-mono text-green-400">probs = softmax(scaled[top-k]) = eˣ / Σeˣ</code>
            {" "}— renormalize over the top-k candidates only
          </p>
          <p>The word with the highest probability is sampled as the next token.</p>
        </div>
      )}

      {/* ── Sampling controls ── */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-border bg-secondary/30 px-4 py-2.5">
        {/* Temperature */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide shrink-0">
            Temperature
          </span>
          <input
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-28 h-1 accent-orange-400 cursor-pointer"
          />
          <span className="font-mono text-[11px] text-orange-400 w-8 shrink-0">
            {temperature.toFixed(2)}
          </span>
          <span className="text-[9px] text-muted-foreground shrink-0">{tempLabel}</span>
          <button
            onClick={() => setTemperature(1.0)}
            className="text-[9px] text-muted-foreground hover:text-foreground transition-colors shrink-0 underline underline-offset-2"
          >
            reset
          </button>
        </div>

        <div className="h-4 w-px bg-border shrink-0 hidden sm:block" />

        {/* Top-k */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide shrink-0">
            Top-k
          </span>
          <input
            type="range"
            min={1}
            max={MAX_K}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-28 h-1 accent-green-400 cursor-pointer"
          />
          <span className="font-mono text-[11px] text-green-400 w-5 shrink-0">{topK}</span>
          <span className="text-[9px] text-muted-foreground shrink-0">
            of 50,257 tokens
          </span>
          <button
            onClick={() => setTopK(DEFAULT_TOP_K)}
            className="text-[9px] text-muted-foreground hover:text-foreground transition-colors shrink-0 underline underline-offset-2"
          >
            reset
          </button>
        </div>
      </div>

      {/* ── Pipeline row ── */}
      <div className="flex items-start overflow-x-auto pb-1 gap-0">

        {/* ── Step 1: Last token hidden state ── */}
        <StepCard step={1} label="Hidden State" accent="rgb(139,92,246)">
          <div className="rounded-lg border border-violet-500/30 bg-violet-500/5 p-2 space-y-1">
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-[10px] text-violet-400 font-bold shrink-0">
                &ldquo;{lastToken}&rdquo;
              </span>
              <span className="text-[9px] text-muted-foreground">last token output</span>
            </div>
            <VecStrip vec={hidden} rgb={[139, 92, 246]} w={192} h={20} />
            <p className="text-[8px] text-muted-foreground">768-dim · from last FFN block</p>
          </div>
        </StepCard>

        <Arrow label="LayerNorm" sublabel="normalize" />

        {/* ── Step 2: Normalized vector ── */}
        <StepCard step={2} label="Normalized" accent="rgb(251,191,36)">
          <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/5 p-2 space-y-1">
            <p className="text-[9px] text-muted-foreground">zero mean · unit variance</p>
            <VecStrip vec={normalized} rgb={[251, 191, 36]} w={192} h={20} />
            <p className="text-[8px] text-muted-foreground">768-dim · ready for projection</p>
          </div>
        </StepCard>

        <Arrow label="W_unembed" sublabel="768 → 50,257" />

        {/* ── Step 3: Logit scores (raw, temperature-independent) ── */}
        <StepCard step={3} label="Logit Scores" accent="rgb(99,102,241)">
          <div className="rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-2 space-y-1">
            <p className="text-[9px] text-muted-foreground mb-1.5">
              dot product: hidden · word_embedding
            </p>
            <div className="space-y-[3px]">
              {kTokens.map(({ token }, rank) => {
                const logit  = kLogits[rank];
                const barPct = ((logit - minRawL) / rawSpan) * 100;
                const isTop  = rank === 0;
                return (
                  <div key={rank} className="flex items-center gap-1.5">
                    <span
                      className="font-mono text-right shrink-0"
                      style={{
                        width: 32,
                        fontSize: 9,
                        color: isTop ? "rgb(139,92,246)" : "hsl(var(--muted-foreground))",
                        fontWeight: isTop ? 700 : 400,
                      }}
                    >
                      {token || '""'}
                    </span>
                    <div
                      className="rounded overflow-hidden shrink-0"
                      style={{
                        outline: isTop ? "1px solid rgba(139,92,246,0.5)" : "1px solid transparent",
                      }}
                    >
                      <VecStrip
                        vec={fakeEmb(rank)}
                        rgb={isTop ? [139, 92, 246] : [99, 102, 241]}
                        w={72}
                        h={10}
                      />
                    </div>
                    <div className="w-20 h-2.5 rounded-sm overflow-hidden bg-muted/30 shrink-0">
                      <div
                        className="h-full rounded-sm"
                        style={{
                          width: `${barPct}%`,
                          background: isTop
                            ? "rgb(139,92,246)"
                            : `rgba(99,102,241,${0.25 + (barPct / 100) * 0.55})`,
                        }}
                      />
                    </div>
                    <span
                      className="font-mono shrink-0"
                      style={{
                        fontSize: 8,
                        width: 30,
                        color: isTop ? "rgb(139,92,246)" : "hsl(var(--muted-foreground))",
                      }}
                    >
                      {logit.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="text-[8px] text-muted-foreground pt-0.5">
              top {topK} of 50,257 · log-scale proxy
            </p>
          </div>
        </StepCard>

        <Arrow
          label="Softmax"
          sublabel={temperature !== 1.0 ? `T=${temperature.toFixed(2)}` : "eˣ / Σeˣ"}
        />

        {/* ── Step 4: Probability distribution (temperature + top-k applied) ── */}
        <StepCard step={4} label="Probabilities" accent="rgb(34,197,94)">
          <div className="rounded-lg border border-green-500/20 bg-green-500/5 p-2 space-y-1">
            <p className="text-[9px] text-muted-foreground mb-1.5">
              top-{topK} · T={temperature.toFixed(2)} · renormalized
            </p>
            <div className="space-y-[3px]">
              {displayProbs.map(({ token, prob }, rank) => {
                const barPct = (prob / maxProb) * 100;
                const isTop  = rank === 0;
                return (
                  <div key={rank} className="flex items-center gap-1.5" style={{ height: 14 }}>
                    <span
                      className="font-mono text-right shrink-0"
                      style={{
                        width: 32,
                        fontSize: 9,
                        color: isTop ? "rgb(139,92,246)" : "hsl(var(--muted-foreground))",
                        fontWeight: isTop ? 700 : 400,
                      }}
                    >
                      {token || '""'}
                    </span>
                    <div className="h-2.5 rounded-sm overflow-hidden bg-muted/30 shrink-0" style={{ width: 100 }}>
                      <div
                        className="h-full rounded-sm transition-all duration-150"
                        style={{
                          width: `${barPct}%`,
                          background: isTop
                            ? "rgb(139,92,246)"
                            : `rgba(99,102,241,${0.25 + (barPct / 100) * 0.55})`,
                        }}
                      />
                    </div>
                    <span
                      className="font-mono shrink-0"
                      style={{
                        fontSize: 8,
                        width: 38,
                        color: isTop ? "rgb(139,92,246)" : "hsl(var(--muted-foreground))",
                        fontWeight: isTop ? 600 : 400,
                      }}
                    >
                      {(prob * 100).toFixed(prob >= 0.01 ? 1 : 2)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </StepCard>

        {/* ── Final prediction callout ── */}
        <div className="flex flex-col items-center justify-center shrink-0 ml-4 gap-2">
          <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
            Predicted
          </div>
          <div
            className="rounded-xl px-5 py-3 text-center"
            style={{
              border: "2px solid rgb(139,92,246)",
              background: "rgba(139,92,246,0.10)",
              boxShadow: "0 0 18px rgba(139,92,246,0.18)",
            }}
          >
            <p className="font-mono text-2xl font-bold" style={{ color: "rgb(167,139,250)" }}>
              &ldquo;{displayProbs[0]?.token}&rdquo;
            </p>
            <p className="text-[9px] mt-1" style={{ color: "rgba(167,139,250,0.65)" }}>
              {((displayProbs[0]?.prob ?? 0) * 100).toFixed(1)}% probability
            </p>
          </div>
          {displayProbs[1] && (
            <div className="text-center">
              <p className="text-[8px] text-muted-foreground">runner-up</p>
              <p
                className="font-mono text-xs"
                style={{ color: "rgba(139,92,246,0.55)" }}
              >
                &ldquo;{displayProbs[1].token}&rdquo; {(displayProbs[1].prob * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Explanatory footnote ── */}
      <p className="text-[11px] text-muted-foreground leading-relaxed max-w-3xl border-t border-border pt-3">
        After the last transformer block, a{" "}
        <strong className="text-foreground">final layer normalization</strong> stabilizes the 768-dim hidden state.
        The <strong className="text-foreground">unembedding matrix</strong> (W_unembed, 50,257 × 768) scores each
        vocabulary word via dot product.{" "}
        <strong className="text-foreground">Temperature</strong> divides each logit before softmax — lower values
        concentrate probability on the top token, higher values spread it across more options.{" "}
        <strong className="text-foreground">Top-k</strong> restricts sampling to the k highest-scoring tokens,
        renormalizing their probabilities to sum to 1.
        Logit bars use log-probability as a proxy; word embedding strips are approximate.
      </p>
    </div>
  );
}
