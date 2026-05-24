"use client";
import React, { useState, useRef } from "react";

// ─── Types ──────────────────────────────────────────────────────────────────

export type LayerData = {
  layerIdx: number;
  multiHeadAttention: number[][][]; // 12 × n × n
  attentionMatrix: number[][];      // n × n averaged
  queryVectors: number[][][];       // 12 × n × 64
  keyVectors: number[][][];         // 12 × n × 64
  valueVectors: number[][][];       // 12 × n × 64
  ffnIn:     number[][];            // n × 768 — LayerNorm'd residual entering FFN
  ffnHidden: number[][];            // n × 3072 — post-GELU hidden activations
  ffnOut:    number[][];            // n × 768 — FFN output before residual add
};

export type VizExample = {
  id: string;
  label: string;
  tokens: string[];
  layers: LayerData[];
  nextWordProbs: Array<{ token: string; prob: number }>;
};

// ─── Layout constants ────────────────────────────────────────────────────────

const C = {
  padX: 0,
  padY: 64,
  tokenW: 30,
  embedW: 36,
  flowW: 64,
  cardPadX: 12,
  cardPadY: 10,
  stripW: 26,
  attnGap: 14,
  attnCell: 32,
  outGap: 12,
  outW: 56,
  mlpGap: 100,
  mlpW: 310,
  mlpGapR: 80,
  probLabelW: 54,
  probBarW: 92,
  probRowH: 30,
  rowH: 20,
  rowGap: 13,
  secGap: 30,
  secLabelH: 26,
  dimVis: 14,
};

const FFN_DIM_VIS      = 14;   // dims shown in input/output strips
const FFN_GELU_ROWS    = 64;   // cells sampled from 3072 for the GELU strip
const FFN_GELU_STRIP_W = 20;   // px width of the GELU heatmap column
const FFN_ARROW_N      = 10;   // arrow lines representing W1 / W2
const COLLAPSED_W = 8;
const ATTN_GAMMA  = 0.30;



function scaleAttn(w: number): number {
  return Math.pow(Math.max(w, 0), ATTN_GAMMA);
}

// ─── Geometry ────────────────────────────────────────────────────────────────

function computeLayout(n: number) {
  const secH = C.secLabelH + n * (C.rowH + C.rowGap) - C.rowGap;

  const kTop = C.padY;
  const qTop = kTop + secH + C.secGap;
  const vTop = qTop + secH + C.secGap;
  const svgH = vTop + secH + C.padY + 4;

  const rowY = (secTop: number, i: number) =>
    secTop + C.secLabelH + i * (C.rowH + C.rowGap) + C.rowH / 2;

  const totalInnerH = vTop + secH - kTop;
  const tokenY = (i: number) => kTop + totalInnerH * (2 * i + 1) / (2 * n);

  const tokenRight = C.padX + C.tokenW;
  const embedLeft  = tokenRight + 3;
  const embedRight = embedLeft + C.embedW;
  const flowLeft   = embedRight;
  const flowRight  = flowLeft + C.flowW;

  const cardLeft   = flowRight + C.cardPadX;
  const stripLeft  = cardLeft + C.cardPadX;
  const stripRight = stripLeft + C.stripW;
  const attnLeft      = stripRight + C.attnGap;
  const ATTN_FIXED_W  = C.attnCell * 6;          // 192px — bounding box never changes
  const attnCell      = ATTN_FIXED_W / n;         // cell shrinks as n grows, expands as n shrinks
  const attnRight     = attnLeft + ATTN_FIXED_W;  // always 256px
  const outLeft    = attnRight + C.outGap;
  const outRight   = outLeft + C.outW;
  const cardRight  = outRight + C.cardPadX;
  const cardTop    = kTop - C.cardPadY;
  const cardBottom = vTop + secH + C.cardPadY;

  const mlpLeft    = cardRight + C.mlpGap;
  const mlpRight   = mlpLeft + C.mlpW;
  const probLeft   = mlpRight + C.mlpGapR;
  const svgW       = probLeft + 340;

  return {
    svgH, svgW, n, secH,
    kTop, qTop, vTop,
    cardTop, cardBottom, cardLeft, cardRight,
    kRowY: (i: number) => rowY(kTop, i),
    qRowY: (i: number) => rowY(qTop, i),
    vRowY: (i: number) => rowY(vTop, i),
    tokenY,
    tokenRight, embedLeft, embedRight, flowLeft, flowRight,
    stripLeft, stripRight,
    attnLeft, attnRight, attnCell,
    attnColX: (j: number) => attnLeft + j * attnCell + attnCell / 2,
    outLeft, outRight,
    mlpLeft, mlpRight, probLeft,
    attnTopY: () => rowY(qTop, 0) - C.rowH / 2 - 2,
  };
}

function bezierH(x1: number, y1: number, x2: number, y2: number) {
  const cx = (x1 + x2) / 2;
  return `M ${x1.toFixed(1)},${y1.toFixed(1)} C ${cx.toFixed(1)},${y1.toFixed(1)} ${cx.toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
}

function renderStrip(
  vec: number[], x: number, y: number, w: number, h: number,
  rgb: [number, number, number], dims = C.dimVis
): React.ReactNode {
  const cellW  = w / dims;
  const maxAbs = Math.max(...vec.slice(0, dims).map(Math.abs), 0.1);
  const [r, g, b] = rgb;
  return vec.slice(0, dims).map((v, i) => {
    const t    = Math.min(Math.abs(v) / maxAbs, 1);
    const a    = 0.08 + t * 0.82;
    const fill = v >= 0
      ? `rgba(${r},${g},${b},${a.toFixed(2)})`
      : `rgba(100,116,139,${(a * 0.6).toFixed(2)})`;
    return <rect key={i} x={x + i * cellW} y={y} width={cellW - 0.3} height={h} fill={fill} />;
  });
}

// ─── Mini FFN diagram (for explainer) ────────────────────────────────────────

function MiniFFNDiagram() {
  const inYs  = [24, 46, 68, 90];
  const hidYs = [16, 32, 48, 64, 80, 96, 112];
  const outYs = [24, 46, 68, 90];
  const inX = 22, hidX = 90, outX = 158;

  return (
    <svg width={180} height={120} viewBox="0 0 180 120" className="shrink-0">
      {inYs.map((iy, i) => hidYs.map((hy, j) => (
        <line key={`ih-${i}-${j}`}
          x1={inX + 7} y1={iy} x2={hidX - 7} y2={hy}
          stroke="rgb(99,102,241)" strokeWidth={0.5} strokeOpacity={0.28} />
      )))}
      {hidYs.map((hy, j) => outYs.map((oy, i) => (
        <line key={`ho-${j}-${i}`}
          x1={hidX + 7} y1={hy} x2={outX - 7} y2={oy}
          stroke="rgb(139,92,246)" strokeWidth={0.5} strokeOpacity={0.28} />
      )))}
      {inYs.map((iy, i)  => <circle key={`in-${i}`}  cx={inX}  cy={iy} r={6.5} fill="rgba(139,92,246,0.75)" />)}
      {hidYs.map((hy, j) => <circle key={`h-${j}`}   cx={hidX} cy={hy} r={5.5} fill="rgba(34,197,94,0.75)"  />)}
      {outYs.map((oy, i) => <circle key={`out-${i}`} cx={outX} cy={oy} r={6.5} fill="rgba(99,102,241,0.75)" />)}
      <text x={inX}  y={113} textAnchor="middle" fontSize={8} fill="rgb(139,92,246)">in</text>
      <text x={hidX} y={113} textAnchor="middle" fontSize={8} fill="rgb(34,197,94)">GELU</text>
      <text x={outX} y={113} textAnchor="middle" fontSize={8} fill="rgb(99,102,241)">out</text>
    </svg>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────

export function TransformerArchitectureViz({ example }: { example: VizExample }) {
  const { tokens, layers, nextWordProbs } = example;
  const n         = tokens.length;
  const numLayers = layers.length;
  const numHeads  = layers[0]?.multiHeadAttention.length ?? 12;

  const [activeBlock,      setActiveBlock]      = useState(0);
  const [activeHead,       setActiveHead]        = useState(0);
  const [focusedFFNToken,  setFocusedFFNToken]   = useState<number | null>(null);
  const [showFFNExplainer,      setShowFFNExplainer]      = useState(false);
  const [showAttnMathExplainer, setShowAttnMathExplainer] = useState(false);
  const [hoveredBlock,          setHoveredBlock]           = useState<number | null>(null);
  const [hoverPos,         setHoverPos]          = useState<{ x: number; y: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverY,           setHoverY]            = useState(0);

  const layer = layers[activeBlock];
  const attn  = layer.multiHeadAttention[activeHead];
  const qVecs = layer.queryVectors[activeHead];
  const kVecs = layer.keyVectors[activeHead];
  const vVecs = layer.valueVectors[activeHead];

  const activeFocusToken = Math.min(focusedFFNToken ?? (n - 1), n - 1);
  const L = computeLayout(n);

  const K_RGB: [number, number, number] = [239, 68,  68];
  const Q_RGB: [number, number, number] = [59,  130, 246];
  const V_RGB: [number, number, number] = [34,  197, 94];

  const valueDim    = vVecs[0]?.length ?? 0;
  const attnOutVecs = tokens.map((_, i) => {
    const out = Array.from({ length: valueDim }, () => 0);
    for (let j = 0; j < n; j++) {
      const w = attn[i][j];
      vVecs[j].forEach((v, d) => { out[d] += w * v; });
    }
    return out;
  });



  const MAX_K = nextWordProbs.length;
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(Math.min(12, MAX_K));

  const lastIdx = n - 1;

  // ── LM Head data (always uses the final layer, regardless of activeBlock) ──
  const lastLayerData      = layers[numLayers - 1];
  const lastTokenHiddenVec = lastLayerData.ffnOut[lastIdx];

  // Raw pseudo-logits (temperature-independent, used for logit bars)
  const kTokens    = nextWordProbs.slice(0, topK);
  const rawLogits  = kTokens.map(p => Math.log(Math.max(p.prob, 1e-9)));
  const maxLogit   = rawLogits[0];
  const minLogit   = rawLogits[rawLogits.length - 1];
  const logitSpan  = Math.max(maxLogit - minLogit, 0.01);

  // Temperature-scaled probabilities (what the prob bars and callout show)
  const scaledLogits = rawLogits.map(l => l / temperature);
  const maxScaled    = Math.max(...scaledLogits);
  const scaledExps   = scaledLogits.map(l => Math.exp(l - maxScaled));
  const scaledSum    = scaledExps.reduce((a, b) => a + b, 0);
  const topProbs     = kTokens.map((p, i) => ({ token: p.token, prob: scaledExps[i] / scaledSum }));
  const maxProb      = topProbs[0]?.prob ?? 1;

  // Average attention weight per value position j (column mean of attn matrix)
  const avgWeights = Array.from({ length: n }, (_, j) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += attn[i][j];
    return sum / n;
  });
  const avgWeightsScaled = avgWeights.map(scaleAttn);
  const avgMax = Math.max(...avgWeightsScaled, 0.001);

  // ── FFN layout ────────────────────────────────────────────────────────────
  const ffnPad          = 10, ffnGap = 7;
  const ffnArrowRegionW = 80;
  const ffnInW          = FFN_DIM_VIS * 3;
  const ffnOutW         = FFN_DIM_VIS * 3;

  const ffnInX       = L.mlpLeft + ffnPad;
  const ffnArrowInX  = ffnInX + ffnInW + ffnGap;
  const ffnGeluX     = ffnArrowInX + ffnArrowRegionW + ffnGap;
  const ffnArrowOutX = ffnGeluX + FFN_GELU_STRIP_W + ffnGap;
  const ffnOutX      = ffnArrowOutX + ffnArrowRegionW + ffnGap;

  const ffnBoxTop   = L.cardTop + 2;
  const ffnBoxBot   = L.cardBottom - 2;
  const ffnBoxH     = ffnBoxBot - ffnBoxTop;
  const ffnHeaderH  = 26;
  const ffnContentH = ffnBoxH - ffnHeaderH - 4;
  const ffnContentY = ffnBoxTop + ffnHeaderH + 2;
  const ffnGeluCellH = ffnContentH / FFN_GELU_ROWS;

  // GELU activation data for the focused token (3072-dim, sampled evenly)
  const geluVec     = layer.ffnHidden?.[activeFocusToken] ?? [];
  const geluStride  = Math.max(1, Math.floor(geluVec.length / FFN_GELU_ROWS));
  const geluSampled = Array.from({ length: FFN_GELU_ROWS }, (_, r) => geluVec[r * geluStride] ?? 0);
  // Use 85th percentile of abs values so active neurons are visible (most are near-zero)
  const geluAbsSorted = [...geluSampled].map(Math.abs).sort((a, b) => a - b);
  const geluScale = Math.max(geluAbsSorted[Math.floor(geluAbsSorted.length * 0.85)] ?? 0, 0.01);

  // ── Head output → FFN routing geometry ────────────────────────────────────
  // Σ box — per-head weighted sum, sits close to card right edge
  const boxCy        = (L.cardTop + L.cardBottom) / 2;
  const sigmaBoxW    = 28;
  const sigmaBoxH    = 44;
  const sigmaX       = L.attnRight + Math.round((L.cardRight - L.attnRight) / 2);
  const sigmaBoxLeft  = sigmaX - sigmaBoxW / 2;
  const sigmaBoxRight = sigmaX + sigmaBoxW / 2;
  const sigmaBoxTop   = boxCy - sigmaBoxH / 2;
  const sigmaBoxBot   = boxCy + sigmaBoxH / 2;

  // Center of the active head's slice in the FFN input vector
  const headSliceX = (h: number) =>
    ffnInX + (ffnInW * (h + 0.5)) / Math.max(numHeads, 1);

  // Stack offset per ghost-card layer
  const stackDx = 5;
  const stackDy = 1;

  // ── SVG split dimensions ──────────────────────────────────────────────────
  const tokColW     = L.embedRight + 22;          // +20 for fan-start strip
  const expLeft     = L.flowLeft + 20;             // block viewport starts 20px in
  const expW        = L.mlpRight + Math.floor(C.mlpGapR * 0.6) - L.flowLeft - 20;
  const probSVGLeft = L.probLeft - 10;
  const probSVGW    = L.svgW - probSVGLeft;

  // Total min-width of the flex row (so the outer div knows how wide to scroll)
  const totalW = tokColW + numLayers * COLLAPSED_W + probSVGW;
  const sepInset = 28;
  const leftHidden = activeBlock;
  const rightHidden = numLayers - activeBlock - 1;
  const leftRegionW = leftHidden * COLLAPSED_W;
  const rightRegionW = rightHidden * COLLAPSED_W;
  const leftLabelX = tokColW + leftRegionW / 2;
  const rightLabelX = tokColW + leftRegionW + expW + rightRegionW / 2;
  const labelY = 8;
  const showHoverLabel = hoveredBlock !== null && hoveredBlock !== activeBlock && hoverPos !== null;
  const hoverLabelY = hoverPos
    ? Math.min(Math.max(hoverPos.y - 14, sepInset + 16), L.svgH - sepInset - 16)
    : 0;

  // ── Inline SVG content for the expanded block ─────────────────────────────
  const expandedBlockSVG = (
    <svg
      key={`exp-${activeBlock}-${activeHead}`}
      viewBox={`${expLeft} 0 ${expW} ${L.svgH - 38}`}
      width={expW}
      height={L.svgH - 38}
      style={{ display: "block" }}
    >
      <defs>
        <style>{`
          @keyframes headCardIn {
            from { opacity: 0.35; transform: translateX(8px); }
            to   { opacity: 1;    transform: translateX(0);   }
          }
          .head-card-animated { animation: headCardIn 0.18s ease-out forwards; }
        `}</style>
        <marker id="arrowQ" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="rgb(59,130,246)" fillOpacity="0.65" />
        </marker>
        <marker id="arrowMlp" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="rgb(139,92,246)" fillOpacity="0.8" />
        </marker>
      </defs>

      {/* Dashed outline around the entire expanded block */}
      <rect
        x={expLeft + 1} y={1} width={expW - 2} height={L.svgH - 2}
        rx={leftHidden > 0 && rightHidden > 0 ? 0 : leftHidden === 0 && rightHidden === 0 ? 6 : leftHidden === 0 ? 6 : 6}
        fill="none"
        stroke="hsl(var(--foreground))" strokeOpacity={0.1}
        strokeWidth={2} strokeDasharray="5 3"
      />

      {/* Token → K/Q/V bezier flows */}
      {tokens.map((_, i) => (
        <path key={`kf-${i}`}
          d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.kRowY(i))}
          fill="none" stroke="rgb(239,68,68)" strokeWidth={1.5} strokeOpacity={0.28} />
      ))}
      {tokens.map((_, i) => (
        <path key={`qf-${i}`}
          d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.qRowY(i))}
          fill="none" stroke="rgb(59,130,246)" strokeWidth={1.5} strokeOpacity={0.28} />
      ))}
      {tokens.map((_, i) => (
        <path key={`vf-${i}`}
          d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.vRowY(i))}
          fill="none" stroke="rgb(34,197,94)" strokeWidth={1.5} strokeOpacity={0.28} />
      ))}

      {/* 12 stacked head cards — back-to-front, each with fan lines to token columns */}
      {Array.from({ length: numHeads - 1 }, (_, pi) => {
        // pi=0 → furthest back (p = numHeads-1); pi = numHeads-2 → just behind active (p=1)
        const p  = numHeads - 1 - pi;
        const dx = p * stackDx;
        const dy = p * stackDy;
        return (
          <g key={`ghost-${p}`}>
            {/* Card rect — covers the part of lines from further-back cards */}
            <rect
              x={L.cardLeft + dx} y={L.cardTop + dy}
              width={L.cardRight - L.cardLeft} height={L.cardBottom - L.cardTop}
              rx={8} fill="hsl(var(--card))" fillOpacity={0.94}
              stroke="hsl(var(--border))" strokeWidth={1}
              strokeOpacity={0.28 + 0.42 * (1 - p / (numHeads - 1))}
            />
          </g>
        );
      })}

      {/* Main animated card */}
      <g key={`card-${activeBlock}-${activeHead}`} className="head-card-animated">
        <rect
          x={L.cardLeft} y={L.cardTop}
          width={L.cardRight - L.cardLeft} height={L.cardBottom - L.cardTop}
          rx={8} fill="hsl(var(--card))"
          stroke="hsl(var(--border))" strokeWidth={1.5}
        />

        {/* K strips */}
        {tokens.map((tok, i) => (
          <g key={`ks-${i}`}>
            <rect x={L.stripLeft} y={L.kRowY(i) - C.rowH / 2}
              width={C.stripW} height={C.rowH} rx={2}
              fill="rgba(239,68,68,0.07)" stroke="rgba(239,68,68,0.2)" strokeWidth={0.5} />
            {renderStrip(kVecs[i], L.stripLeft, L.kRowY(i) - C.rowH / 2, C.stripW, C.rowH, K_RGB)}
            <text x={L.stripLeft + C.stripW / 2} y={L.kRowY(i) - C.rowH / 2 - 4}
              textAnchor="middle" fontSize={9} fill="rgba(239,68,68,0.7)"
              fontFamily="monospace">{tok}</text>
          </g>
        ))}

        {/* Q strips */}
        {tokens.map((tok, i) => (
          <g key={`qs-${i}`}>
            <rect x={L.stripLeft} y={L.qRowY(i) - C.rowH / 2}
              width={C.stripW} height={C.rowH} rx={2}
              fill="rgba(59,130,246,0.07)" stroke="rgba(59,130,246,0.2)" strokeWidth={0.5} />
            {renderStrip(qVecs[i], L.stripLeft, L.qRowY(i) - C.rowH / 2, C.stripW, C.rowH, Q_RGB)}
            <text x={L.stripLeft + C.stripW / 2} y={L.qRowY(i) - C.rowH / 2 - 4}
              textAnchor="middle" fontSize={9} fill="rgba(59,130,246,0.7)"
              fontFamily="monospace">{tok}</text>
          </g>
        ))}

        {/* V strips */}
        {tokens.map((tok, j) => (
          <g key={`vs-${j}`}>
            <rect x={L.stripLeft} y={L.vRowY(j) - C.rowH / 2}
              width={C.stripW} height={C.rowH} rx={2}
              fill="rgba(34,197,94,0.07)" stroke="rgba(34,197,94,0.2)" strokeWidth={0.5} />
            {renderStrip(vVecs[j], L.stripLeft, L.vRowY(j) - C.rowH / 2, C.stripW, C.rowH, V_RGB)}
            <text x={L.stripLeft + C.stripW / 2} y={L.vRowY(j) - C.rowH / 2 - 4}
              textAnchor="middle" fontSize={9} fill="rgba(34,197,94,0.7)"
              fontFamily="monospace">{tok}</text>
          </g>
        ))}

        {/* K → attention columns */}
        {tokens.map((_, j) => {
          const fromX = L.stripRight, fromY = L.kRowY(j);
          const toX   = L.attnColX(j), toY  = L.attnTopY() - 1;
          const cx1   = fromX + (toX - fromX) * 0.55;
          const cx2   = toX;
          const cy2   = fromY + (toY - fromY) * 0.35;
          return (
            <path key={`kc-${j}`}
              d={`M${fromX.toFixed(1)},${fromY.toFixed(1)} C${cx1.toFixed(1)},${fromY.toFixed(1)} ${cx2.toFixed(1)},${cy2.toFixed(1)} ${toX.toFixed(1)},${toY.toFixed(1)}`}
              fill="none" stroke="rgb(239,68,68)" strokeWidth={1.4} strokeOpacity={0.55} />
          );
        })}

        {/* Q → attention rows */}
        {tokens.map((_, i) => (
          <line key={`qr-${i}`}
            x1={L.stripRight} y1={L.qRowY(i)} x2={L.attnLeft - 1} y2={L.qRowY(i)}
            stroke="rgb(59,130,246)" strokeWidth={1.8} strokeOpacity={0.65}
            markerEnd="url(#arrowQ)"
          />
        ))}

        {/* Attention dot matrix */}
        {tokens.map((_, row) =>
          tokens.map((_, col) => {
            const w = scaleAttn(attn[row][col]);
            return (
              <circle key={`dot-${row}-${col}`}
                cx={L.attnColX(col)} cy={L.qRowY(row)}
                r={Math.min(L.attnCell * 0.38, (C.rowH + C.rowGap) / 2 - 2)}
                fill="rgb(67,56,202)" fillOpacity={0.04 + w * 0.94}
              />
            );
          })
        )}


      </g>

      {/* Section labels (rendered on top of card) */}
      <text x={L.stripLeft + C.stripW / 2} y={L.kTop + 14}
        textAnchor="middle" fontSize={12} fontWeight={600} fill="rgb(239,68,68)">Key</text>
      <text x={L.stripLeft + C.stripW / 2} y={L.qTop + 14}
        textAnchor="middle" fontSize={12} fontWeight={600} fill="rgb(59,130,246)">Query</text>
      <text x={L.stripLeft + C.stripW / 2} y={L.vTop + 14}
        textAnchor="middle" fontSize={12} fontWeight={600} fill="rgb(34,197,94)">Value</text>
      <text x={(L.attnLeft + L.attnRight) / 2} y={L.qTop + 14}
        textAnchor="middle" fontSize={11} fontWeight={600} fill="hsl(var(--muted-foreground))">Attention</text>

      {/* ── V → Σ → FFN routing paths (concat + W_O implied) ── */}

      {/* Attention rows: curve from each qRowY into the left side of the Σ box */}
      {tokens.map((_, i) => {
        const sx  = L.attnRight + 1, sy = L.qRowY(i);
        const cx1 = sx + (sigmaBoxLeft - sx) * 0.5;
        const cx2 = sigmaBoxLeft - (sigmaBoxLeft - sx) * 0.25;
        return (
          <path key={`ar-${i}`}
            d={`M ${sx},${sy} C ${cx1},${sy} ${cx2},${boxCy} ${sigmaBoxLeft},${boxCy}`}
            fill="none" stroke="rgb(99,102,241)" strokeWidth={1.0} strokeOpacity={0.35}
          />
        );
      })}

      {/* V paths: curve down into the bottom of the Σ box */}
      {tokens.map((_, j) => {
        const w    = avgWeightsScaled[j];
        const t    = w / avgMax;
        const op   = 0.08 + t * 0.68;
        const sw   = 0.6 + t * 2.4;
        const sy   = L.vRowY(j);
        const cpx1 = L.stripRight + (sigmaX - L.stripRight) * 1.20;
        return (
          <path key={`vpath-${j}`}
            d={`M ${L.stripRight},${sy} C ${cpx1},${sy} ${sigmaX},${sigmaBoxBot - 50} ${sigmaX},${sigmaBoxBot}`}
            fill="none" stroke="rgb(34,197,94)" strokeWidth={sw} strokeOpacity={op}
          />
        );
      })}

      {/* Σ box */}
      <rect x={sigmaBoxLeft} y={sigmaBoxTop} width={sigmaBoxW} height={sigmaBoxH}
        rx={5} fill="hsl(var(--card))" stroke="rgb(139,92,246)" strokeWidth={1.5} />
      <text x={sigmaX} y={boxCy + 6}
        textAnchor="middle" fontSize={17} fontWeight={700} fill="rgb(139,92,246)">Σ</text>

      {/* Σ tooltip badge */}
      <g onClick={() => setShowAttnMathExplainer(v => !v)} style={{ cursor: "pointer" }}>
        <circle cx={sigmaX} cy={sigmaBoxTop - 12} r={7}
          fill={showAttnMathExplainer ? "rgb(99,102,241)" : "hsl(var(--secondary))"}
          stroke="rgb(99,102,241)" strokeWidth={1} />
        <text x={sigmaX} y={sigmaBoxTop - 8}
          textAnchor="middle" fontSize={10} fontWeight={700}
          fill={showAttnMathExplainer ? "white" : "rgb(99,102,241)"}>?</text>
      </g>

      {/* ════ FFN heatmap ════ */}
      {/* Box */}
      <rect x={L.mlpLeft} y={ffnBoxTop} width={C.mlpW} height={ffnBoxH}
        rx={7} fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={1.2} />

      {/* Head output → FFN input slice — rendered after FFN box so arrows appear on top */}
      {tokens.map((_, ti) => {
        const tx      = headSliceX(activeHead);
        const rowCy   = L.tokenY(ti);
        const isAbove = rowCy < boxCy;
        const ty      = isAbove ? rowCy + C.rowH / 2 : rowCy - C.rowH / 2;
        const cp1x    = sigmaBoxRight + Math.abs(tx - sigmaBoxRight) * 0.45;
        const cp2y    = isAbove ? ty + 24 : ty - 24;
        return (
          <path key={`ac-${ti}`}
            d={`M ${sigmaBoxRight},${boxCy} C ${cp1x},${boxCy} ${tx},${cp2y} ${tx},${ty}`}
            fill="none" stroke="rgb(139,92,246)" strokeWidth={1.4} strokeOpacity={0.65}
          />
        );
      })}

      {/* Label + ⓘ toggle */}
      <text x={(L.mlpLeft + L.mlpRight) / 2 - 6} y={L.cardTop - 32}
        textAnchor="middle" fontSize={9} fontWeight={700} letterSpacing={0.8}
        fill="hsl(var(--muted-foreground))" opacity={0.6}>FEED-FORWARD NETWORK</text>

            {/* FFN token stepper — below FFN box, centered on GELU column */}
      {(() => {
        const geluCx  = ffnGeluX + FFN_GELU_STRIP_W / 2;
        const navY    = ffnBoxBot + 26;
        const labelY  = L.cardTop - 18;
        const ry      = navY - 11;
        const rh      = 14;
        const tokLabel = tokens[activeFocusToken] ?? `#${activeFocusToken}`;
        return (
          <>
            <text x={(L.mlpLeft + L.mlpRight) / 2 - 6} y={labelY}
              textAnchor="middle" fontSize={9} fontWeight={700} letterSpacing={0.8} opacity={0.6}>
              Runs One Token at a Time
            </text>
            <text x={(L.mlpLeft + L.mlpRight) / 2 - 6} y={labelY + 12}
              textAnchor="middle" fontSize={11} fontWeight={600}
              fill="hsl(var(--muted-foreground))">
              {`"${tokLabel}"`}
            </text>
            {activeFocusToken > 0 && (
              <g onClick={() => setFocusedFFNToken(activeFocusToken - 1)} style={{ cursor: "pointer" }}>
                <rect x={(L.mlpLeft + L.mlpRight) / 2 - 6 - 74} y={labelY + 2} width={15} height={rh} rx={3}
                  fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={0.5} />
                <text x={(L.mlpLeft + L.mlpRight) / 2 - 6 - 66} y={labelY + 12}
                  textAnchor="middle" fontSize={12} fill="hsl(var(--muted-foreground))">‹</text>
              </g>
            )}
            {activeFocusToken < n - 1 && (
              <g onClick={() => setFocusedFFNToken(activeFocusToken + 1)} style={{ cursor: "pointer" }}>
                <rect x={(L.mlpLeft + L.mlpRight) / 2 - 6 + 59} y={labelY + 2} width={15} height={rh} rx={3}
                  fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={0.5} />
                <text x={(L.mlpLeft + L.mlpRight) / 2 - 6 + 67} y={labelY + 12}
                  textAnchor="middle" fontSize={12} fill="hsl(var(--muted-foreground))">›</text>
              </g>
            )}
          </>
        );
      })()}

      <g onClick={() => setShowFFNExplainer(v => !v)} style={{ cursor: "pointer" }}>
        <circle cx={L.mlpRight - 5} cy={ffnBoxTop - 10} r={8}
          fill={showFFNExplainer ? "rgb(99,102,241)" : "hsl(var(--secondary))"}
          stroke="rgb(99,102,241)" strokeWidth={1} />
        <text x={L.mlpRight - 5} y={ffnBoxTop - 6}
          textAnchor="middle" fontSize={10} fontWeight={700}
          fill={showFFNExplainer ? "white" : "rgb(99,102,241)"}>?</text>
      </g>

      {/* Column headers */}
      {[
        { cx: ffnInX + ffnInW / 2,               top: "in",   bot: "(768)"      },
        { cx: ffnArrowInX + ffnArrowRegionW / 2,  top: "W₁·x", bot: "(768→3072)" },
        { cx: ffnGeluX + FFN_GELU_STRIP_W / 2,   top: "GELU", bot: "(3072)"     },
        { cx: ffnArrowOutX + ffnArrowRegionW / 2, top: "W₂·h", bot: "(3072→768)" },
        { cx: ffnOutX + ffnOutW / 2,              top: "out",  bot: "(768)"      },
      ].map(({ cx, top, bot }, idx) => (
        <g key={`fh-${idx}`}>
          <text x={cx} y={ffnBoxTop + 10} textAnchor="middle"
            fontSize={9} fontWeight={600} fill="hsl(var(--muted-foreground))">{top}</text>
          <text x={cx} y={ffnBoxTop + 19} textAnchor="middle"
            fontSize={7.5} fill="hsl(var(--muted-foreground))">{bot}</text>
        </g>
      ))}
      {/* Single separator — below column headers */}
      <line x1={L.mlpLeft + 6} y1={ffnBoxTop + ffnHeaderH}
        x2={L.mlpRight - 6} y2={ffnBoxTop + ffnHeaderH}
        stroke="hsl(var(--border))" strokeWidth={0.6} strokeOpacity={0.6} />

      {/* W1: 10 straight lines from focused token input strip → evenly spaced GELU positions */}
      {Array.from({ length: FFN_ARROW_N }, (_, k) => {
        const t    = (k + 0.5) / FFN_ARROW_N;
        const srcY = L.tokenY(activeFocusToken);
        const dstY = ffnContentY + t * ffnContentH;
        return (
          <line key={`fa1-${k}`}
            x1={ffnInX + ffnInW} y1={srcY} x2={ffnGeluX} y2={dstY}
            stroke="rgba(99,102,241,0.38)" strokeWidth={0.9}
          />
        );
      })}

      {/* GELU activation strip — real data, last token, sampled from 3072 dims */}
      {Array.from({ length: FFN_GELU_ROWS }, (_, row) => {
        const val  = geluVec[row * geluStride] ?? 0;
        const t    = Math.min(Math.abs(val) / geluScale, 1);
        const fill = val > 0.001
          ? `rgba(34,197,94,${(0.15 + t * 0.78).toFixed(2)})`
          : `rgba(100,116,139,${(0.06 + t * 0.28).toFixed(2)})`;
        return (
          <rect key={`gelu-${row}`}
            x={ffnGeluX} y={ffnContentY + row * ffnGeluCellH}
            width={FFN_GELU_STRIP_W} height={Math.max(ffnGeluCellH - 0.3, 0.5)}
            fill={fill}
          />
        );
      })}

      {/* W2: 10 straight lines from evenly spaced GELU positions → focused token output strip */}
      {Array.from({ length: FFN_ARROW_N }, (_, k) => {
        const t    = (k + 0.5) / FFN_ARROW_N;
        const srcY = ffnContentY + t * ffnContentH;
        const dstY = L.tokenY(activeFocusToken);
        return (
          <line key={`fa2-${k}`}
            x1={ffnGeluX + FFN_GELU_STRIP_W} y1={srcY} x2={ffnOutX} y2={dstY}
            stroke="rgba(139,92,246,0.38)" strokeWidth={0.9}
          />
        );
      })}

      {/* Per-token input / output strips — real FFN data */}
      {tokens.map((_, i) => {
        const isFocused = i === activeFocusToken;
        const ry        = L.tokenY(i) - C.rowH / 2;
        const inVec     = layer.ffnIn?.[i]  ?? attnOutVecs[i];
        const outVec    = layer.ffnOut?.[i] ?? attnOutVecs[i];
        return (
          <g key={`ft-${i}`}
            opacity={isFocused ? 1 : 0.3}
            onClick={() => setFocusedFFNToken(i)}
            style={{ cursor: "pointer" }}
          >
            <text x={ffnInX + ffnInW / 2} y={ry - 3}
              textAnchor="middle" fontSize={9} fontFamily="monospace"
              fill="rgba(139,92,246,0.8)">{tokens[i]}</text>
            <rect x={ffnInX} y={ry} width={ffnInW} height={C.rowH} rx={2}
              fill="rgba(99,102,241,0.07)" stroke="rgba(99,102,241,0.20)" strokeWidth={0.5} />
            {renderStrip(inVec, ffnInX, ry, ffnInW, C.rowH, [99, 102, 241], FFN_DIM_VIS)}
            <text x={ffnOutX + ffnOutW / 2} y={ry - 3}
              textAnchor="middle" fontSize={9} fontFamily="monospace"
              fill="rgba(99,102,241,0.8)">{tokens[i]}</text>
            <rect x={ffnOutX} y={ry} width={ffnOutW} height={C.rowH} rx={2}
              fill="rgba(99,102,241,0.07)" stroke="rgba(99,102,241,0.2)" strokeWidth={0.5} />
            {renderStrip(outVec, ffnOutX, ry, ffnOutW, C.rowH, [99, 102, 241], FFN_DIM_VIS)}
          </g>
        );
      })}

      {/* Output lines — bezier fans mirroring the left-side K/Q/V flows */}
      {tokens.map((_, i) => {
        const cy          = L.tokenY(i);
        const startX      = ffnOutX + ffnOutW + 2;
        const endX        = expLeft + expW - 2;
        const isLastBlock = activeBlock === numLayers - 1;

        if (isLastBlock) {
          if (i !== lastIdx) return null;
          return (
            <line key={`ol-${i}`}
              x1={startX} y1={cy} x2={endX} y2={cy}
              stroke="rgba(139,92,246,0.65)" strokeWidth={2}
            />
          );
        }

        // Fan targets mirror the left-side span (flowLeft→stripLeft ≈ 76px).
        // The SVG viewBox clips anything past endX, so only the start of the
        // spread is visible — matching the visual weight of the left-side flows.
        const fanX      = endX - 20;
        const fanTarget = fanX + (L.stripLeft - L.flowLeft);
        return (
          <g key={`ol-${i}`}>
            <line x1={startX} y1={cy} x2={fanX} y2={cy}
              stroke="rgb(139,92,246)" strokeWidth={1.5} strokeOpacity={0.28} />
            <path d={bezierH(fanX, cy, fanTarget, L.kRowY(i))}
              fill="none" stroke="rgb(239,68,68)" strokeWidth={1.5} strokeOpacity={0.28} />
            <path d={bezierH(fanX, cy, fanTarget, L.qRowY(i))}
              fill="none" stroke="rgb(59,130,246)" strokeWidth={1.5} strokeOpacity={0.28} />
            <path d={bezierH(fanX, cy, fanTarget, L.vRowY(i))}
              fill="none" stroke="rgb(34,197,94)" strokeWidth={1.5} strokeOpacity={0.28} />
          </g>
        );
      })}

      {/* Head navigation */}
      {(() => {
        const navY  = L.cardTop - 8;
        const navCx = (L.cardLeft + L.cardRight) / 2;
        const ry    = navY - 11;
        const rh    = 14;
        return (
          <>
            {/* Section label */}
            <text x={navCx} y={navY - 24}
              textAnchor="middle" fontSize={9} fontWeight={700}
              letterSpacing={0.8}
              fill="hsl(var(--muted-foreground))" opacity={0.6}>
              SELF-ATTENTION
            </text>
            <text x={navCx} y={navY}
              textAnchor="middle" fontSize={11} fontWeight={600}
              fill="hsl(var(--muted-foreground))">
              Head {activeHead + 1} of {numHeads}
            </text>
            {activeHead > 0 && (
              <g onClick={() => setActiveHead(h => h - 1)} style={{ cursor: "pointer" }}>
                <rect x={navCx - 74} y={ry} width={15} height={rh} rx={3}
                  fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={0.5} />
                <text x={navCx - 66} y={ry + 10}
                  textAnchor="middle" fontSize={12} fill="hsl(var(--muted-foreground))">‹</text>
              </g>
            )}
            {activeHead < numHeads - 1 && (
              <g onClick={() => setActiveHead(h => h + 1)} style={{ cursor: "pointer" }}>
                <rect x={navCx + 59} y={ry} width={15} height={rh} rx={3}
                  fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={0.5} />
                <text x={navCx + 67} y={ry + 10}
                  textAnchor="middle" fontSize={12} fill="hsl(var(--muted-foreground))">›</text>
              </g>
            )}
          </>
        );
      })()}
    </svg>
  );

  // ── LM Head SVG (right column) ────────────────────────────────────────────
  // Layout constants
  const lmPad    = 80;
  const lmLeft   = L.probLeft + lmPad;
  const lmContW  = 246;   // total content width
  const lmStripH    = C.rowH;                   // same height as all other strips
  const lmStripDims = Math.floor(lmContW / 5); // ~5px/cell — similar density to FFN out

  // Per-row column positions
  const lmTokW  = 40;   // token label col
  const lmLogW  = 58;   // logit bar col
  const lmArrW  = 14;   // "→" col
  const lmProbW = 78;   // prob bar col
  // 40+6+58+6+14+6+78+6+32 = lmContW ✓

  const lmLogX  = lmLeft + lmTokW + 6;
  const lmArrCx = lmLogX + lmLogW + 3 + lmArrW / 2;
  const lmProbX = lmLogX + lmLogW + 6 + lmArrW + 6;
  const lmPctX  = lmProbX + lmProbW + 4;

  // Vertical layout (scales with svgH)
  const lmYTitle  = 14;
  const lmYSep1   = 28;
  const lmYSLbl   = 35;
  const lmYStrip  = 45;
  const lmYA1     = lmYStrip + lmStripH + 6;
  const lmYWUT    = lmYA1 + 12;
  const lmYWUB    = lmYWUT + 34;
  const lmYA2     = lmYWUB + 4;
  const lmYSMT    = lmYA2 + 12;
  const lmYSMB    = lmYSMT + 140;
  const lmYSep2   = lmYSMB + 8;
  const lmYColHdr = lmYSep2 + 10;
  const lmYRows   = lmYColHdr + 16;
  const lmTopRowH = 48;
  const lmRowH    = Math.min(44, Math.max(14, (L.svgH - lmYRows - 10 - lmTopRowH) / Math.max(topProbs.length - 1, 1)));
  const lmRowStartY = (rank: number) => rank === 0 ? lmYRows : lmYRows + lmTopRowH + (rank - 1) * lmRowH;

  const probBarsSVG = (
    <svg
      viewBox={`${probSVGLeft} 0 ${probSVGW} ${L.svgH}`}
      width={probSVGW}
      height={L.svgH}
      style={{ display: "block" }}
    >
      <defs>
        <marker id="lm-arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="hsl(var(--border))" />
        </marker>
        <marker id="lm-arr-indigo" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="rgba(139,92,246,0.70)" />
        </marker>
      </defs>

      {/* ── Section title ── */}
      <text x={lmLeft} y={lmYTitle + 10}
        fontSize={9} fontWeight={700} letterSpacing={1}
        fill="hsl(var(--muted-foreground))" opacity={0.65}>
        LM HEAD
      </text>
      <line x1={lmLeft} y1={lmYSep1}
        x2={lmLeft + lmContW} y2={lmYSep1}
        stroke="hsl(var(--border))" strokeWidth={0.6} />

      {/* ── Entry arrow: right edge of blocks → hidden-state strip ── */}
      {/* +32 compensates for the block accordion's top:32 offset vs probBarsSVG's y=0 origin */}
      <path
        d={bezierH(
          probSVGLeft + 1, L.tokenY(n - 1) + 32,
          lmLeft, lmYStrip + lmStripH / 2
        )}
        fill="none"
        stroke="rgba(139,92,246,0.65)" strokeWidth={2}
        markerEnd="url(#lm-arr-indigo)"
      />

      {/* ── Hidden-state strip (last token, last layer) ── */}
      <text x={lmLeft} y={lmYSLbl}
        fontSize={8.5} fill="hsl(var(--muted-foreground))">
        last token hidden state · 768-dim
      </text>
      <rect x={lmLeft} y={lmYStrip}
        width={lmContW} height={lmStripH} rx={2}
        fill="rgba(99,102,241,0.07)" stroke="rgba(99,102,241,0.20)" strokeWidth={0.5} />
      {renderStrip(lastTokenHiddenVec, lmLeft, lmYStrip, lmContW, lmStripH, [99, 102, 241], lmStripDims)}

      {/* ── Arrow + W_unembed box (2-line) ── */}
      <line x1={lmLeft + lmContW / 2} y1={lmYA1}
        x2={lmLeft + lmContW / 2} y2={lmYWUT - 1}
        stroke="hsl(var(--border))" strokeWidth={1.2} markerEnd="url(#lm-arr)" />
      <rect x={lmLeft} y={lmYWUT}
        width={lmContW} height={34} rx={3}
        fill="rgba(99,102,241,0.10)" stroke="rgba(99,102,241,0.40)" strokeWidth={1} />
      <text x={lmLeft + lmContW / 2} y={lmYWUT + 13}
        textAnchor="middle" fontSize={10} fontWeight={700}
        fill="rgb(99,102,241)">
        Score Every Word
      </text>
      <text x={lmLeft + lmContW / 2} y={lmYWUT + 27}
        textAnchor="middle" fontSize={8.5}
        fill="rgba(99,102,241,0.65)">
        dot product · hidden state vs. 50,257 words
      </text>

      {/* ── Arrow + Softmax card (title + formula + sliders) ── */}
      <line x1={lmLeft + lmContW / 2} y1={lmYA2}
        x2={lmLeft + lmContW / 2} y2={lmYSMT - 1}
        stroke="hsl(var(--border))" strokeWidth={1.2} markerEnd="url(#lm-arr)" />
      <rect x={lmLeft} y={lmYSMT}
        width={lmContW} height={130} rx={3}
        fill="rgba(34,197,94,0.06)" stroke="rgba(34,197,94,0.35)" strokeWidth={1} />
      <foreignObject x={lmLeft + 6} y={lmYSMT + 4} width={lmContW - 12} height={126}>
        <div
          // @ts-expect-error xmlns required in SVG foreignObject
          xmlns="http://www.w3.org/1999/xhtml"
          style={{ display: "flex", flexDirection: "column", gap: 0 }}
        >
          {/* Title */}
          <div style={{ textAlign: "center", fontWeight: 700, fontSize: 10, color: "rgb(34,197,94)", marginBottom: 6 }}>
            Softmax
          </div>

          {/* Fraction */}
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 5 }}>
            <span style={{ fontSize: 8.5, color: "hsl(var(--muted-foreground))", fontFamily: "monospace" }}>prob =</span>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <span style={{ fontSize: 9, fontFamily: "monospace", color: "hsl(var(--foreground))" }}>
                e^(score{temperature !== 1.0 ? ` ÷ ${temperature.toFixed(2)}` : ""})
              </span>
              <div style={{ width: "100%", height: 1, background: "rgba(34,197,94,0.5)" }} />
              <span style={{ fontSize: 9, fontFamily: "monospace", color: "hsl(var(--foreground))" }}>
                Σ e^(score{temperature !== 1.0 ? ` ÷ ${temperature.toFixed(2)}` : ""})
              </span>
              <span style={{ fontSize: 7.5, color: "hsl(var(--muted-foreground))", marginTop: 2 }}>
                summed over top-{topK} candidate tokens
              </span>
            </div>
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: "rgba(34,197,94,0.18)", margin: "7px 0" }} />

          {/* Sliders */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
            <span style={{ fontSize: 9, fontWeight: 600, color: "rgb(251,146,60)", width: 68, flexShrink: 0 }}>
              Temperature
            </span>
            <input
              type="range" min={0.1} max={2.0} step={0.05}
              value={temperature}
              onChange={e => setTemperature(Number(e.target.value))}
              style={{ flex: 1, height: 4, accentColor: "rgb(251,146,60)", cursor: "pointer" }}
            />
            <span style={{ fontSize: 10, fontFamily: "monospace", color: "rgb(251,146,60)", width: 28, textAlign: "right", flexShrink: 0 }}>
              {temperature.toFixed(2)}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ fontSize: 9, fontWeight: 600, color: "rgb(34,197,94)", width: 68, flexShrink: 0 }}>
              Top-k
            </span>
            <input
              type="range" min={1} max={MAX_K} step={1}
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              style={{ flex: 1, height: 4, accentColor: "rgb(34,197,94)", cursor: "pointer" }}
            />
            <span style={{ fontSize: 10, fontFamily: "monospace", color: "rgb(34,197,94)", width: 28, textAlign: "right", flexShrink: 0 }}>
              {topK}
            </span>
          </div>
        </div>
      </foreignObject>

      {/* ── Separator + column headers ── */}
      <line x1={lmLeft} y1={lmYSep2}
        x2={lmLeft + lmContW} y2={lmYSep2}
        stroke="hsl(var(--border))" strokeWidth={0.6} />
      <text x={lmLeft + lmTokW + 6 + lmLogW / 2} y={lmYColHdr + 10}
        textAnchor="middle" fontSize={8} fill="hsl(var(--muted-foreground))">
        score
      </text>
      <text x={lmLeft + lmTokW + 6 + lmLogW + 6 + lmArrW + 6 + lmProbW / 2} y={lmYColHdr + 10}
        textAnchor="middle" fontSize={8} fill="hsl(var(--muted-foreground))">
        probability
      </text>

      {/* ── Per-token rows: logit bar + → + prob bar + % ── */}
      {topProbs.map(({ token, prob }, rank) => {
        const isTop   = rank === 0;
        const rowH    = isTop ? lmTopRowH : lmRowH;
        const py      = lmRowStartY(rank);
        const cy      = py + rowH / 2;
        const barH    = isTop ? Math.max(18, rowH * 0.45) : Math.max(7, rowH * 0.40);
        const barY    = cy - barH / 2;

        const logNorm  = (rawLogits[rank] - minLogit) / logitSpan;
        const logBarW  = Math.max(3, logNorm * lmLogW);
        const probNorm = prob / maxProb;
        const probBarW = Math.max(3, probNorm * lmProbW);
        const pctText  = `${(prob * 100).toFixed(prob >= 0.01 ? 1 : 2)}%`;
        const pctInside = probBarW > 28;
        const tokFontSize = isTop ? 17 : 11;

        return (
          <g key={`lm-row-${rank}`}>
            {/* Top-token highlight background */}
            {isTop && (
              <rect x={lmLeft} y={py} width={lmContW} height={rowH} rx={4}
                fill="rgba(139,92,246,0.08)" stroke="rgba(139,92,246,0.30)" strokeWidth={1} />
            )}

            {/* Token label */}
            <text x={lmLeft + lmTokW} y={cy + tokFontSize * 0.35}
              textAnchor="end" fontSize={tokFontSize} fontFamily="monospace"
              fill={isTop ? "rgb(167,139,250)" : "hsl(var(--foreground))"}
              fontWeight={isTop ? 700 : 400}>
              {token || '""'}
            </text>

            {/* Score bar background */}
            <rect x={lmLogX} y={barY} width={lmLogW} height={barH} rx={2}
              fill="hsl(var(--muted))" fillOpacity={0.3} />
            {/* Score bar fill */}
            <rect x={lmLogX} y={barY} width={logBarW} height={barH} rx={2}
              fill="rgb(99,102,241)" fillOpacity={0.20 + logNorm * 0.60} />
            {/* Score value — right-aligned inside the bar background */}
            <text x={lmLogX + lmLogW - 3} y={cy + 3}
              textAnchor="end" fontSize={7.5} fontFamily="monospace"
              fill="rgba(180,185,255,0.75)">
              {rawLogits[rank].toFixed(1)}
            </text>

            {/* → softmax arrow */}
            <text x={lmArrCx} y={cy + 4} textAnchor="middle" fontSize={9}
              fill="rgba(34,197,94,0.75)">→</text>

            {/* Prob bar background */}
            <rect x={lmProbX} y={barY} width={lmProbW} height={barH} rx={2}
              fill="hsl(var(--muted))" fillOpacity={0.3} />
            {/* Prob bar fill */}
            <rect x={lmProbX} y={barY} width={probBarW} height={barH} rx={2}
              fill={isTop ? "rgb(139,92,246)" : "rgb(99,102,241)"}
              fillOpacity={isTop ? 0.85 : 0.22 + probNorm * 0.50} />

            {/* % label */}
            <text
              x={pctInside ? lmProbX + probBarW - 3 : lmPctX}
              y={cy + 4}
              textAnchor={pctInside ? "end" : "start"}
              fontSize={isTop ? 11 : 9}
              fontFamily="monospace"
              fill={pctInside ? "white" : (isTop ? "rgb(167,139,250)" : "hsl(var(--muted-foreground))")}
              fontWeight={isTop ? 700 : 400}>
              {pctText}
            </text>
          </g>
        );
      })}
    </svg>
  );

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="rounded-xl border border-border bg-card overflow-hidden relative">
      {leftHidden > 0 && (
        <div
          aria-hidden="true"
          className="absolute text-[10px] font-semibold tracking-wide text-muted-foreground"
          style={{ left: leftLabelX, top: labelY, transform: "translateX(-50%)", pointerEvents: "none", zIndex: 12 }}
        >
          {leftHidden} transformer {leftHidden === 1 ? "block" : "blocks"}
        </div>
      )}
      {rightHidden > 0 && (
        <div
          aria-hidden="true"
          className="absolute text-[10px] font-semibold tracking-wide text-muted-foreground"
          style={{ left: rightLabelX, top: labelY, transform: "translateX(-50%)", pointerEvents: "none", zIndex: 12 }}
        >
          {rightHidden} transformer {rightHidden === 1 ? "block" : "blocks"}
        </div>
      )}
      {showHoverLabel && hoverPos && (
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            zIndex: 20,
            left: hoverPos.x,
            top: hoverLabelY,
            transform: "translate(-50%, -50%)",
            padding: "2px 6px",
            borderRadius: 6,
            background: "rgba(139,92,246,0.16)",
            color: "rgb(139,92,246)",
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: 0.2,
            pointerEvents: "none",
          }}
        >
          Block {(hoveredBlock ?? 0) + 1}
        </div>
      )}

      {/* Attention Math Explainer — absolute overlay */}
      {showAttnMathExplainer && (
        <div
          className="absolute z-50 bg-card border border-border rounded-xl shadow-xl p-4"
          style={{ top: 12, left: 12, width: 600, maxHeight: "70%", overflow: "auto" }}
        >
          <button
            onClick={() => setShowAttnMathExplainer(false)}
            className="absolute top-2 right-2 text-xs text-muted-foreground hover:text-foreground leading-none"
          >✕</button>
          <p className="font-semibold text-foreground text-sm leading-snug mb-2">
            Context Vector = Weighted Sum of Values
          </p>
          <code className="block text-xs bg-secondary rounded px-2 py-1.5 font-mono text-indigo-400 mb-3 leading-relaxed">
            {"context[i] = softmax(QKᵀ/√d) · V"}
          </code>
          <div className="space-y-1.5 text-xs text-muted-foreground leading-relaxed">
            <p>Each row of the attention matrix is a probability distribution over token positions. Multiplying it by the <span className="text-green-400 font-medium">Value matrix</span> produces a weighted sum — the <span className="text-violet-400 font-medium">context vector</span> for that token.</p>
            <p>The <span className="text-violet-400 font-medium">Σ box</span> performs this per head for every token in parallel.</p>
            <p className="font-semibold text-foreground text-xs mt-2">Multi-Head Output (Concat + W_O)</p>
            <code className="block text-[11px] bg-secondary rounded px-2 py-1.5 font-mono text-violet-400 leading-relaxed">
              {"MultiHead = Concat(head₁,…,headₕ) · W_O"}
            </code>
            <p>Each attention head computes its own weighted sum of Values independently — the stacked cards behind the active one represent the other heads.</p>
            <p>All {numHeads} outputs are concatenated into a single 768-dim vector per token ({numHeads} × 64 = 768), then multiplied by <span className="text-violet-400 font-medium">W_O</span> (768×768) to mix information across heads before passing to the feed-forward network.</p>
            <p>The <span className="text-violet-400 font-medium">purple curves</span> show the active head&apos;s slice landing in its portion of the FFN input vector for each token.</p>
            <div className="space-y-1 text-[10px] pt-1 border-t border-border">
              <p><span className="text-indigo-400 font-medium">Indigo curves (→ Σ):</span> each token&apos;s attention row output flowing into the weighted sum.</p>
              <p><span className="text-green-400 font-medium">Green curves (↑ Σ):</span> Value vectors entering from below — thickness reflects how much attention that position receives on average across all query tokens.</p>
              <p><span className="text-violet-400 font-medium">Purple curves (Σ → FFN):</span> concat + W_O are implied between the box and the input strip.</p>
            </div>
          </div>
        </div>
      )}


      {/* FFN Explainer — absolute overlay */}
      {showFFNExplainer && (
        <div
          className="absolute z-50 bg-card border border-border rounded-xl shadow-xl p-4"
          style={{ top: 12, right: 12, width: 340, maxHeight: "70%", overflow: "auto" }}
        >
          <button
            onClick={() => setShowFFNExplainer(false)}
            className="absolute top-2 right-2 text-xs text-muted-foreground hover:text-foreground leading-none"
          >✕</button>
          <div className="flex items-start gap-4">
            <MiniFFNDiagram />
            <div className="space-y-2 text-sm min-w-0">
              <p className="font-semibold text-foreground leading-snug">
                A neural network applied to every token independently.
              </p>
              <code className="block text-xs bg-secondary rounded px-2 py-1 font-mono text-indigo-400">
                out = W₂ · GELU(W₁ · x)
              </code>
            </div>
          </div>
          <p className="text-muted-foreground text-xs leading-relaxed mt-3">
            W₁ expands 768 → 3072 dims (the fan arrows). GELU activates each hidden neuron — the colored strip shows real activation magnitudes for the last token. W₂ projects back 3072 → 768. Each block learns its own W₁ and W₂.
          </p>
        </div>
      )}

      {/* Horizontal accordion */}
      <div className="overflow-x-auto">
        <div
          className="flex items-stretch"
          style={{ minWidth: totalW, height: L.svgH }}
        >
          {/* Token column — sticky left */}
          <div
            style={{
              position: "sticky",
              left: 0,
              zIndex: 10,
              flexShrink: 0,
              background: "hsl(var(--card))",
            }}
          >
            <svg
              viewBox={`0 -32 ${tokColW} ${L.svgH}`}
              width={tokColW}
              height={L.svgH}
              style={{ display: "block", overflow: "visible" }}
            >
              <defs>
                {/* Clip to only the 20px fan-start strip between embedRight and expLeft */}
                <clipPath id="fan-start-clip">
                  <rect x={L.flowLeft} y={-32} width={20} height={L.svgH + 32} />
                </clipPath>
              </defs>

              {/* Fan start — same beziers as expanded block, clipped to first 20px */}
              <g clipPath="url(#fan-start-clip)">
                {tokens.map((_, i) => (
                  <path key={`kfs-${i}`}
                    d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.kRowY(i))}
                    fill="none" stroke="rgb(239,68,68)" strokeWidth={1.5} strokeOpacity={0.28} />
                ))}
                {tokens.map((_, i) => (
                  <path key={`qfs-${i}`}
                    d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.qRowY(i))}
                    fill="none" stroke="rgb(59,130,246)" strokeWidth={1.5} strokeOpacity={0.28} />
                ))}
                {tokens.map((_, i) => (
                  <path key={`vfs-${i}`}
                    d={bezierH(L.flowLeft, L.tokenY(i), L.stripLeft, L.vRowY(i))}
                    fill="none" stroke="rgb(34,197,94)" strokeWidth={1.5} strokeOpacity={0.28} />
                ))}
              </g>

              {tokens.map((tok, i) => {
                const cy = L.tokenY(i);
                const embedDims = FFN_DIM_VIS;
                const embedVec = Array.from({ length: embedDims }, (_, d) =>
                  Math.sin(i * 3.71 + d * 1.37)
                );
                const embedH = C.rowH;
                const embedY = cy - embedH / 2;
                return (
                  <g key={`tc-${i}`}>
                    <text x={L.embedLeft + C.embedW - 1} y={embedY - 4}
                      textAnchor="end" fontSize={10} fontFamily="monospace"
                      fill="hsl(var(--foreground))">
                      {tok}
                    </text>
                    <rect x={L.embedLeft} y={embedY}
                      width={C.embedW} height={embedH} rx={2}
                      fill="rgba(99,102,241,0.07)" stroke="rgba(99,102,241,0.20)" strokeWidth={0.5} />
                    {renderStrip(embedVec, L.embedLeft, embedY, C.embedW, embedH, [99, 102, 241], embedDims)}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Block accordion */}
          {layers.map((layerData, b) => {
            const isActive   = b === activeBlock;
            const isHovered  = b === hoveredBlock;
            const showHover  = isHovered && !isActive;
            const blockW     = isActive ? expW : COLLAPSED_W;
            const isFirst    = b === 0;
            const isLast     = b === numLayers - 1;
            const showDivider = !isFirst && b !== activeBlock + 1;

            // Collapsed blocks form a dashed rail; outer ends are rounded
            const railStyle = isActive ? {
              borderBottom: "2px dashed hsl(var(--foreground) / 0.15)",
              borderBottomLeftRadius:  isFirst ? 6 : 0,
              borderBottomRightRadius: isLast  ? 6 : 0,
            } : {
              borderTop:    "2px dashed hsl(var(--foreground) / 0.15)",
              borderBottom: "2px dashed hsl(var(--foreground) / 0.15)",
              borderLeft:   isFirst ? "2px dashed hsl(var(--foreground) / 0.15)" : undefined,
              borderRight:  isLast  ? "2px dashed hsl(var(--foreground) / 0.15)" : undefined,
              borderTopLeftRadius:     isFirst ? 6 : 0,
              borderBottomLeftRadius:  isFirst ? 6 : 0,
              borderTopRightRadius:    isLast  ? 6 : 0,
              borderBottomRightRadius: isLast  ? 6 : 0,
            };

            return (
              <div
                key={b}
                style={{
                  top: 32,
                  width: blockW,
                  height: L.svgH - 38,
                  flexShrink: 0,
                  overflow: "hidden",
                  transition: "width 220ms ease-in-out",
                  position: "relative",
                  ...railStyle,
                }}
                onMouseEnter={(event) => {
                  if (isActive) return;
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (rect) {
                    setHoverPos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
                  }
                  setHoveredBlock(b);
                }}
                onMouseMove={(event) => {
                  if (isActive) return;
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (rect) {
                    setHoverPos({ x: event.clientX - rect.left, y: event.clientY - rect.top });
                  }
                  setHoveredBlock(b);
                }}
                onMouseLeave={() => {
                  setHoveredBlock(null);
                  setHoverPos(null);
                }}
              >
                {showHover && (
                  <div
                    aria-hidden="true"
                    style={{
                      position: "absolute",
                      left: 0,
                      right: 0,
                      top: 0,
                      bottom: 0,
                      borderRadius: 6,
                      background: "rgba(139,92,246,0.18)",
                      pointerEvents: "none",
                    }}
                  />
                )}
                {isActive ? expandedBlockSVG : (
                  <>
                    {showDivider && (
                      <div
                        aria-hidden="true"
                        style={{
                          position: "absolute",
                          left: 0,
                          top: 0,
                          bottom: 0,
                          borderLeft: "2px dashed hsl(var(--foreground) / 0.15)",
                          pointerEvents: "none",
                        }}
                      />
                    )}
                    <button
                      onClick={() => setActiveBlock(b)}
                      className="bg-card w-full h-full"
                      style={{ display: "block", padding: 0, flexShrink: 0, background: "hsl(var(--card))" }}
                    >
                    </button>
                  </>
                )}
              </div>
            );
          })}
          <div
            style={{
              position: "relative",
              flexShrink: 0,
              width: probSVGW,
              height: L.svgH,
              background: "hsl(var(--card))",
            }}
          >
            {probBarsSVG}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="border-t border-border px-4 py-2.5 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-medium">Block</span>
          <div className="flex gap-0.5 flex-wrap" role="group">
            {layers.map((_, b) => (
              <button key={b} onClick={() => setActiveBlock(b)}
                className={[
                  "px-1.5 py-0.5 rounded text-xs font-medium transition-colors",
                  activeBlock === b
                    ? "bg-indigo-600 text-white"
                    : "bg-secondary text-muted-foreground hover:text-foreground",
                ].join(" ")}>
                {b + 1}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground font-medium">Head</span>
          <div className="flex gap-0.5 flex-wrap" role="group">
            {Array.from({ length: numHeads }).map((_, h) => (
              <button key={h} onClick={() => setActiveHead(h)}
                className={[
                  "px-1.5 py-0.5 rounded text-xs font-medium transition-colors",
                  activeHead === h
                    ? "bg-violet-500 text-white"
                    : "bg-secondary text-muted-foreground hover:text-foreground",
                ].join(" ")}>
                {h + 1}
              </button>
            ))}
          </div>
        </div>

        <div className="ml-auto flex gap-3 text-[10px] text-muted-foreground">
          {(["Key", "Query", "Value"] as const).map((label, idx) => {
            const c = ["rgb(239,68,68)", "rgb(59,130,246)", "rgb(34,197,94)"];
            return (
              <span key={label} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: c[idx] }} />
                {label}
              </span>
            );
          })}
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: "rgb(139,92,246)" }} />
            Context / Residual
          </span>
          <span className="flex items-center gap-1">
            <svg width={16} height={8} style={{ display: "inline-block", flexShrink: 0 }}>
              <line x1={0} y1={1} x2={16} y2={7} stroke="rgb(99,102,241)" strokeWidth={0.9} strokeOpacity={0.6} />
              <line x1={0} y1={4} x2={16} y2={4} stroke="rgb(99,102,241)" strokeWidth={0.9} strokeOpacity={0.6} />
              <line x1={0} y1={7} x2={16} y2={1} stroke="rgb(99,102,241)" strokeWidth={0.9} strokeOpacity={0.6} />
            </svg>
            Weights
          </span>
        </div>
      </div>
    </div>
  );
}
