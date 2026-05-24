"use client";
import { useState, useRef } from "react";
import Link from "next/link";
import { TransformerArchitectureViz, type VizExample } from "@/components/transformer/TransformerArchitectureViz";
import { runGPT2Inference } from "@/lib/gpt2-live-inference";
import vizExamplesData from "./gpt2-viz-examples.json";
import ContactInfo from "../contact";

const DEFAULT_EXAMPLE = (vizExamplesData as unknown as VizExample[])[0]; // "to be or not to"

// ─── Static SVG: single transformer block internals ───────────────────────────

function BlockDiagram() {
  return (
    <svg
      viewBox="0 0 220 260"
      width={220}
      height={260}
      aria-label="Diagram of a single transformer block"
      role="img"
      className="shrink-0"
    >
      {/* Background */}
      <rect x={10} y={10} width={200} height={250} rx={10} ry={10}
        fill="none" stroke="hsl(var(--border))" strokeWidth={1.5} strokeDasharray="5,3" />

      {/* Input residual */}
      <rect x={85} y={18} width={50} height={22} rx={4} fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={1} />
      <text x={110} y={33} textAnchor="middle" fontSize={9} fill="hsl(var(--muted-foreground))">Input xₙ</text>

      {/* Arrow down */}
      <line x1={110} y1={40} x2={110} y2={58} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Layer norm 1 */}
      <rect x={72} y={58} width={76} height={20} rx={4} fill="#fef3c7" stroke="#fbbf24" strokeWidth={1} />
      <text x={110} y={72} textAnchor="middle" fontSize={9} fill="#92400e">LayerNorm</text>

      {/* Arrow */}
      <line x1={110} y1={78} x2={110} y2={96} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Multi-head attention */}
      <rect x={44} y={96} width={132} height={32} rx={6} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
      <text x={110} y={113} textAnchor="middle" fontSize={10} fontWeight="600" fill="#5b21b6">Multi-Head Attention</text>
      <text x={110} y={123} textAnchor="middle" fontSize={8} fill="#7c3aed">Q · Kᵀ / √d → softmax → · V</text>

      {/* Arrow */}
      <line x1={110} y1={128} x2={110} y2={146} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Add & Norm */}
      <rect x={72} y={146} width={76} height={20} rx={4} fill="#fef3c7" stroke="#fbbf24" strokeWidth={1} />
      <text x={110} y={160} textAnchor="middle" fontSize={9} fill="#92400e">Add &amp; Norm</text>

      {/* Residual bypass line around attention */}
      <path d="M 40 40 Q 25 120 40 166" fill="none" stroke="hsl(var(--border))" strokeWidth={1} strokeDasharray="3,2" />
      <line x1={40} y1={166} x2={72} y2={166} stroke="hsl(var(--border))" strokeWidth={1} />

      {/* Arrow */}
      <line x1={110} y1={166} x2={110} y2={184} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Feed-forward */}
      <rect x={56} y={184} width={108} height={28} rx={6} fill="#dcfce7" stroke="#22c55e" strokeWidth={1.5} />
      <text x={110} y={200} textAnchor="middle" fontSize={10} fontWeight="600" fill="#166534">Feed-Forward</text>
      <text x={110} y={210} textAnchor="middle" fontSize={8} fill="#15803d">Linear → ReLU → Linear</text>

      {/* Arrow */}
      <line x1={110} y1={212} x2={110} y2={230} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Output */}
      <rect x={78} y={230} width={64} height={22} rx={4} fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={1} />
      <text x={110} y={245} textAnchor="middle" fontSize={9} fill="hsl(var(--muted-foreground))">Output xₙ₊₁</text>

      {/* Residual bypass line around FFN */}
      <path d="M 180 166 Q 195 200 180 230" fill="none" stroke="hsl(var(--border))" strokeWidth={1} strokeDasharray="3,2" />
      <line x1={142} y1={166} x2={180} y2={166} stroke="hsl(var(--border))" strokeWidth={1} />
      <line x1={142} y1={230} x2={180} y2={230} stroke="hsl(var(--border))" strokeWidth={1} />

      <defs>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 Z" fill="hsl(var(--border))" />
        </marker>
      </defs>
    </svg>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

type InferenceStatus = "idle" | "loading" | "error";

export default function TransformersClient() {
  const [activeExample, setActiveExample] = useState<VizExample>(DEFAULT_EXAMPLE);
  const [inputText, setInputText]         = useState("");
  const [status, setStatus]               = useState<InferenceStatus>("idle");
  const [loadProgress, setLoadProgress]   = useState(0);
  const [errorMsg, setErrorMsg]           = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const tokenCount = inputText.trim().split(/\s+/).filter(Boolean).length;
  const tooShort   = tokenCount < 3;

  async function handleAnalyze() {
    const text = inputText.trim();
    if (!text || tooShort || status === "loading") return;
    setStatus("loading");
    setLoadProgress(0);
    setErrorMsg("");
    try {
      const result = await runGPT2Inference(text, setLoadProgress);
      setActiveExample(result);
      setStatus("idle");
    } catch (e) {
      console.error(e);
      setErrorMsg("Inference failed. Please try again.");
      setStatus("error");
    }
  }

  return (
    <div className="space-y-8 py-4">
      {/* ── Header ── */}
      <header className="max-w-6xl mx-auto px-4 space-y-3 py-2 border-b border-border">
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-indigo-500" />
          Interactive Visualization
        </span>
        <h1 className="text-3xl font-bold tracking-tight">How Transformers Work</h1>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          The transformer is the architecture behind GPT, Claude, BERT, and every major language model.
          The interactive diagram below shows the full pipeline — from input tokens through stacked attention
          blocks to an output distribution. The sections below explain each component.
        </p>
      </header>

      {/* ── Architecture viz ── */}
      <section className="space-y-4">
        {/* ── Sentence input ── */}
        <div className="max-w-6xl mx-auto px-4 space-y-4">
          <p className="text-sm text-muted-foreground max-w-2xl">
            Enter any incomplete sentence to see how GPT-2 processes it
          </p>
          <div className="flex gap-2 max-w-lg">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleAnalyze()}
              placeholder={`${DEFAULT_EXAMPLE.label}`}
              className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              disabled={status === "loading"}
            />
            <button
              onClick={handleAnalyze}
              disabled={status === "loading" || !inputText.trim() || tooShort}
              className="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
            >
              {status === "loading" ? "Running…" : "Generate →"}
            </button>
          </div>
          {inputText.trim() && tooShort && (
            <p className="text-xs text-muted-foreground">Enter at least 3 words to visualize.</p>
          )}

          {/* Loading / error state */}
          {status === "loading" && (
            <div className="space-y-1 max-w-lg">
              <p className="text-xs text-muted-foreground">
                {loadProgress < 100
                  ? `Downloading GPT-2 model… ${loadProgress}%`
                  : "Running inference…"}
              </p>
              <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-all duration-300 rounded-full"
                  style={{ width: `${loadProgress < 100 ? loadProgress : 100}%` }}
                />
              </div>
            </div>
          )}
          {status === "error" && (
            <p className="text-xs text-red-500">{errorMsg}</p>
          )}
        </div>

        <div className="px-4">
          <TransformerArchitectureViz example={activeExample} />
        </div>

        <p className="max-w-6xl mx-auto px-4 text-sm text-muted-foreground leading-relaxed">
          Each block in the diagram runs the same two operations: <strong>multi-head attention</strong> (letting tokens
          communicate) and a <strong>feed-forward network</strong> (processing each token independently). The purple
          Context strips are the actual attention outputs for each token: a weighted sum of the Value vectors
          (softmax(QK^T) * V). Those context vectors feed the feed-forward network token-by-token, and the last
          token&apos;s output flows to the language-model head to predict the next word.
        </p>
      </section>

      {/* ── Remaining content — constrained to max-w-6xl ── */}
      <div className="max-w-6xl mx-auto px-4 space-y-8">

      {/* ── What is a Transformer? ── */}
      <section className="space-y-6">
        <h2 className="text-xl font-semibold">What is a Transformer?</h2>
        <div className="flex flex-col md:flex-row gap-8 items-start">
          <div className="space-y-4 max-w-xl">
            <p className="text-muted-foreground leading-relaxed">
              A transformer is a neural network that processes sequences — sentences, code, audio frames — by letting
              every element attend directly to every other element. Unlike earlier RNNs that processed tokens one
              at a time, a transformer processes the entire sequence in parallel.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              The architecture has three stages:
            </p>
            <ol className="space-y-3 text-sm text-muted-foreground">
              <li className="flex gap-3">
                <span className="inline-flex h-6 w-6 rounded-full bg-indigo-100 dark:bg-indigo-950 items-center justify-center text-indigo-700 dark:text-indigo-300 font-bold shrink-0 text-xs">1</span>
                <span>
                  <strong className="text-foreground">Embedding.</strong> Each input token is converted to a dense vector
                  (typically 768 or 1024 dimensions). A positional encoding is added so the model knows each token&apos;s position.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="inline-flex h-6 w-6 rounded-full bg-indigo-100 dark:bg-indigo-950 items-center justify-center text-indigo-700 dark:text-indigo-300 font-bold shrink-0 text-xs">2</span>
                <div className="space-y-3 min-w-0">
                  <span className="text-sm text-muted-foreground">
                    <strong className="text-foreground">Transformer blocks.</strong> The embedding passes through N stacked blocks
                    (12 for BERT-base, 96 for GPT-4). Each block runs two operations in sequence — first multi-head attention (letting tokens communicate), then a feed-forward network (refining each token independently) — with residual connections and layer normalization around each:
                  </span>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">First</p>
                      <p className="text-sm font-semibold text-foreground">Multi-Head Attention</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">Every token attends to every other token simultaneously, weighted by relevance.</p>
                      <Link href="/attention" className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline">
                        How attention works →
                      </Link>
                    </div>
                    <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Then</p>
                      <p className="text-sm font-semibold text-foreground">Feed-Forward Network</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">A small neural network applied independently to each token to refine its representation.</p>
                      <Link href="/" className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline">
                        See it in the neural network visualizer →
                      </Link>
                    </div>
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="inline-flex h-6 w-6 rounded-full bg-indigo-100 dark:bg-indigo-950 items-center justify-center text-indigo-700 dark:text-indigo-300 font-bold shrink-0 text-xs">3</span>
                <span>
                  <strong className="text-foreground">Output head.</strong> For language modelling, a final linear layer maps
                  the last hidden state to a probability distribution over the vocabulary. The next token is sampled from this distribution.
                </span>
              </li>
            </ol>
          </div>
          <BlockDiagram />
        </div>
      </section>

      {/* ── Why Stack Blocks? ── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Why Stack Blocks?</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          Each block refines the token representations it receives. Early layers tend to capture surface-level
          patterns — punctuation, adjacent word relationships. Later layers encode higher-level structure:
          syntactic roles, coreference, semantic relationships.
        </p>
        <div className="flex gap-4 overflow-x-auto pb-2">
          {[
            { layer: "Layer 1–2", color: "#bfdbfe", desc: "Punctuation,\nadjacent words" },
            { layer: "Layer 3–5", color: "#93c5fd", desc: "Syntax,\npart-of-speech" },
            { layer: "Layer 6–9", color: "#6366f1", desc: "Semantics,\ncoreference" },
            { layer: "Layer 10–12", color: "#4338ca", desc: "Task-specific\nfeatures" },
          ].map(({ layer, color, desc }) => (
            <div
              key={layer}
              className="flex flex-col items-center gap-2 shrink-0"
            >
              <div
                className="h-16 w-20 rounded-lg flex items-center justify-center text-white text-[10px] font-semibold text-center leading-tight"
                style={{ background: color }}
              >
                {desc}
              </div>
              <span className="text-[10px] text-muted-foreground text-center">{layer}</span>
            </div>
          ))}
        </div>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          This hierarchical processing — from surface to deep semantics — is why stacking many blocks is so powerful.
          Each block has its own set of learned attention weights, so different blocks can specialize in different
          types of relationships.
        </p>
      </section>

      {/* ── Residual Connections & Layer Norm ── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Residual Connections &amp; Layer Norm</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          Each <strong className="text-foreground">block</strong> in the visualizer above contains two
          sub-operations called <strong className="text-foreground">sublayers</strong>: first multi-head
          attention, then the feed-forward network. Both are individually wrapped in the same two
          stabilizing operations. The input is added directly to the sublayer&apos;s output, then layer
          normalization rescales the result before it moves on.
        </p>
        <div className="flex flex-col sm:flex-row gap-8 items-start">
          {/* Flow diagram */}
          <svg viewBox="0 0 200 248" width={200} height={248} style={{ flexShrink: 0 }}>
            <defs>
              <marker id="rc-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M 0 0 L 6 3 L 0 6 Z" fill="hsl(var(--border))" />
              </marker>
            </defs>
            {/* Input x */}
            <rect x={60} y={8} width={60} height={22} rx={4} fill="hsl(var(--secondary))" stroke="hsl(var(--border))" strokeWidth={1} />
            <text x={90} y={23} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="hsl(var(--foreground))">x</text>
            {/* Down to sublayer */}
            <line x1={90} y1={30} x2={90} y2={55} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#rc-arrow)" />
            {/* Sublayer box */}
            <rect x={22} y={55} width={136} height={28} rx={6} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
            <text x={90} y={73} textAnchor="middle" fontSize={10} fontWeight="600" fill="#5b21b6">sublayer(x)</text>
            {/* Down to addition */}
            <line x1={90} y1={83} x2={90} y2={126} stroke="hsl(var(--border))" strokeWidth={1.5} />
            {/* Bypass line — right side */}
            <path d="M 90 19 L 160 19 L 160 136 L 103 136" fill="none" stroke="rgb(99,102,241)" strokeWidth={1.5} strokeDasharray="4,2" />
            <text x={168} y={82} textAnchor="middle" fontSize={8} fill="rgb(99,102,241)" transform="rotate(90,168,82)">bypass</text>
            {/* Addition circle */}
            <circle cx={90} cy={136} r={11} fill="hsl(var(--card))" stroke="hsl(var(--border))" strokeWidth={1.5} />
            <text x={90} y={141} textAnchor="middle" fontSize={15} fill="hsl(var(--foreground))">+</text>
            {/* Down to LayerNorm */}
            <line x1={90} y1={147} x2={90} y2={168} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#rc-arrow)" />
            {/* LayerNorm box */}
            <rect x={32} y={168} width={116} height={24} rx={4} fill="#fef3c7" stroke="#fbbf24" strokeWidth={1} />
            <text x={90} y={184} textAnchor="middle" fontSize={10} fill="#92400e">LayerNorm</text>
            {/* Down to output */}
            <line x1={90} y1={192} x2={90} y2={212} stroke="hsl(var(--border))" strokeWidth={1.5} markerEnd="url(#rc-arrow)" />
            {/* Output */}
            <text x={90} y={230} textAnchor="middle" fontSize={9} fill="hsl(var(--muted-foreground))">next sublayer</text>
          </svg>
          {/* Explanatory cards */}
          <div className="space-y-3 flex-1">
            <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
              <p className="text-sm font-semibold">Residual connection</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Adding x to the sublayer&apos;s output creates a gradient highway: during backpropagation,
                error signals flow directly through the addition node, bypassing the sublayer entirely.
                This lets gradients reach the earliest layers without vanishing — the key reason
                transformers can stack 96+ blocks while older architectures struggled past 10.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
              <p className="text-sm font-semibold">Layer normalization</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Normalizes each token&apos;s activation vector to zero mean and unit variance, then applies
                learned scale and shift parameters. Keeps values in a stable range as they pass through
                many layers, preventing activations from exploding or collapsing to zero.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── How Transformers Learn ── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">How Transformers Learn</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          All the weights — W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub>, the FFN
          matrices, the embeddings — start random and are learned by gradient descent on a
          self-supervised objective. No human-labeled data required: the training signal comes
          from the text itself.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl">
          <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
            <p className="text-sm font-semibold">Decoder — next-token prediction</p>
            <code className="block text-xs bg-secondary rounded px-2 py-1 font-mono text-muted-foreground">
              &ldquo;The cat sat on the ___&rdquo;
            </code>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Predict the next token from all previous tokens. Every position in every sentence is a
              training example — one document generates as many training signals as it has tokens.
              GPT, Claude, Llama all use this objective.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-card p-3 space-y-1.5">
            <p className="text-sm font-semibold">Encoder — masked token prediction</p>
            <code className="block text-xs bg-secondary rounded px-2 py-1 font-mono text-muted-foreground">
              &ldquo;The cat [MASK] on the mat&rdquo;
            </code>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Randomly mask 15% of tokens and predict them from full bidirectional context.
              Produces richer per-token representations but can&apos;t generate text autoregressively.
              BERT uses this objective.
            </p>
          </div>
        </div>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          Both objectives scale with data: more text means more training examples with zero annotation cost.
          This is why pretraining on internet-scale corpora produces models that generalize to tasks the
          designers never explicitly targeted.
        </p>
      </section>

      {/* ── Scale ── */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">What Changes With Scale</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          The architecture stays constant. What changes is size: more layers, wider dimensions, more
          attention heads, larger context windows. The same training objective applied to more compute
          and more data consistently produces better models.
        </p>

        <div className="flex flex-col lg:flex-row gap-8 items-start">
          {/* Table */}
          <div className="overflow-x-auto shrink-0">
            <table className="text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  {["Model", "Params", "Layers", "Heads", "Context"].map((h, i) => (
                    <th key={h} className={`py-2 pr-6 text-muted-foreground font-medium text-xs ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { model: "GPT-2",      params: "117M",  layers: "12",   heads: "12", ctx: "1,024" },
                  { model: "GPT-3",      params: "175B",  layers: "96",   heads: "96", ctx: "2,048" },
                  { model: "Llama 3 8B", params: "8B",    layers: "32",   heads: "32", ctx: "8,192" },
                  { model: "GPT-4",      params: "~1.8T", layers: "~120", heads: "—",  ctx: "128k"  },
                ].map(({ model, params, layers, heads, ctx }) => (
                  <tr key={model} className="border-b border-border/50">
                    <td className="py-2 pr-6 font-medium text-foreground text-sm">{model}</td>
                    <td className="py-2 pr-6 text-right font-mono text-xs text-muted-foreground">{params}</td>
                    <td className="py-2 pr-6 text-right font-mono text-xs text-muted-foreground">{layers}</td>
                    <td className="py-2 pr-6 text-right font-mono text-xs text-muted-foreground">{heads}</td>
                    <td className="py-2 text-right font-mono text-xs text-muted-foreground">{ctx}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Training tokens bar chart (log scale) */}
          <div className="space-y-2 min-w-0 flex-1 max-w-sm">
            <p className="text-xs font-medium text-muted-foreground">Training data · tokens (log scale)</p>
            {[
              { model: "GPT-2",      tokens: "~40B",  logPct: 38,  est: false },
              { model: "GPT-3",      tokens: "~300B", logPct: 59,  est: false },
              { model: "Llama 3 8B", tokens: "~15T",  logPct: 100, est: false },
              { model: "GPT-4",      tokens: "~13T",  logPct: 98,  est: true  },
            ].map(({ model, tokens, logPct, est }) => (
              <div key={model} className="flex items-center gap-2">
                <span className="w-20 text-xs text-muted-foreground shrink-0 text-right">{model}</span>
                <div className="flex-1 h-5 bg-muted/40 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-indigo-400/80 rounded-sm transition-all"
                    style={{ width: `${logPct}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-muted-foreground w-12 shrink-0">
                  {tokens}{est ? "*" : ""}
                </span>
              </div>
            ))}
            <p className="text-[10px] text-muted-foreground pt-1">* estimated · bars show log₁₀ scale</p>
          </div>
        </div>

        <p className="text-xs text-muted-foreground max-w-2xl">
          GPT-4 architecture details are not publicly confirmed — figures are estimates from public research.
          Llama 3 uses grouped-query attention (GQA), which reduces the KV cache at the cost of some expressiveness.
        </p>
      </section>

      {/* ── Further Reading ── */}
      <section className="space-y-3 border-t border-border pt-8">
        <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">Further reading</h2>
        <ul className="space-y-1.5 text-sm">
          <li>
            <a
              href="https://arxiv.org/abs/1706.03762"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              Vaswani et al. — Attention Is All You Need (2017)
            </a>
            <span className="text-muted-foreground ml-2">The original transformer paper.</span>
          </li>
          <li>
            <a
              href="https://jalammar.github.io/illustrated-transformer/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              Jay Alammar — The Illustrated Transformer
            </a>
            <span className="text-muted-foreground ml-2">An excellent visual walkthrough.</span>
          </li>
          <li>
            <Link href="/attention" className="text-indigo-600 dark:text-indigo-400 hover:underline">
              Attention deep dive →
            </Link>
            <span className="text-muted-foreground ml-2">Real BERT weights, QKV breakdown, multi-head analysis.</span>
          </li>
        </ul>
      </section>

      <section className="border-t border-border pt-6">
        <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">
          Continue learning
        </h2>
        <p className="mt-2 text-muted-foreground">
          Return to the{" "}
          <Link href="/" className="underline underline-offset-4 hover:text-foreground">
            neural network visualizer
          </Link>
          {" "}or explore{" "}
          <Link href="/attention" className="underline underline-offset-4 hover:text-foreground">
            how attention works in detail
          </Link>
          .
        </p>
      </section>

      <ContactInfo />
      </div>{/* end max-w-6xl constrained content */}
    </div>
  );
}
