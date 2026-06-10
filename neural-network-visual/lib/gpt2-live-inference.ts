/**
 * In-browser GPT-2 inference using the custom ONNX model at /models/gpt2-viz-q8.onnx.
 *
 * The model was exported with all intermediate tensors as named outputs:
 *   logits          [1, seq, 50257]
 *   l{N}.q          [1, 12, seq, 64]   real query vectors
 *   l{N}.k          [1, 12, seq, 64]   real key vectors
 *   l{N}.v          [1, 12, seq, 64]   real value vectors
 *   l{N}.attn       [1, 12, seq, seq]  real softmax attention weights
 *   l{N}.ffn_in     [1, seq, 768]      LayerNorm'd residual entering FFN
 *   l{N}.ffn_hidden [1, seq, 3072]     post-GELU hidden activations
 *   l{N}.ffn_out    [1, seq, 768]      FFN output before residual add
 *
 * Tokenisation reuses @xenova/transformers (already cached by the attention page).
 */

import type { LayerData, VizExample } from "@/components/transformer/TransformerArchitectureViz";

const MODEL_URL = "https://nn-models-public.s3.us-east-2.amazonaws.com/gpt2-viz-q8.onnx"; // bump when model file changes
const MAX_TOKENS = 8;
const N_LAYERS   = 12;
const N_HEADS    = 12;
const HEAD_DIM   = 64;

// ─── Singletons ───────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _tokenizer: any     = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _session:   any     = null;
let _loadPromise: Promise<void> | null = null;

async function loadAll(onProgress?: (p: number) => void): Promise<void> {
  if (_tokenizer && _session) return;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    // ── 1. ORT must be configured before @xenova/transformers loads it ─────
    const ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = "/ort-wasm/";  // served from public/ort-wasm/
    ort.env.wasm.numThreads = 1;            // no SharedArrayBuffer in dev (no COEP/COOP)

    // ── 2. Tokenizer via @xenova/transformers ──────────────────────────────
    const { AutoTokenizer, env } = await import("@xenova/transformers");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (env as any).allowLocalModels = false;
    _tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt2");

    // ── 3. Fetch model with progress ───────────────────────────────────────
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Failed to fetch model: ${response.status}`);

    const contentLength = Number(response.headers.get("content-length") ?? 0);
    const reader = response.body!.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (contentLength > 0) onProgress?.(Math.round(received / contentLength * 95));
    }

    const total  = chunks.reduce((s, c) => s + c.length, 0);
    const buffer = new Uint8Array(total);
    let offset   = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }

    onProgress?.(98);
    _session = await ort.InferenceSession.create(buffer.buffer.slice(0), {
      executionProviders: ["wasm"],
    });
    onProgress?.(100);
  })();

  return _loadPromise;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function r4(x: number) { return Math.round(x * 1e4) / 1e4; }

/** Extract [N_HEADS, n, HEAD_DIM] from flat Float32Array with shape [1, N_HEADS, n, HEAD_DIM]. */
function extractHeads(data: Float32Array, n: number): number[][][] {
  return Array.from({ length: N_HEADS }, (_, h) =>
    Array.from({ length: n }, (_, s) =>
      Array.from({ length: HEAD_DIM }, (_, d) =>
        r4(data[h * n * HEAD_DIM + s * HEAD_DIM + d])
      )
    )
  );
}

/** Extract [N_HEADS, n, n] from flat Float32Array with shape [1, N_HEADS, n, n]. */
function extractAttn(data: Float32Array, n: number): number[][][] {
  return Array.from({ length: N_HEADS }, (_, h) =>
    Array.from({ length: n }, (_, row) =>
      Array.from({ length: n }, (_, col) =>
        r4(data[h * n * n + row * n + col])
      )
    )
  );
}

/** Extract [n, dim] from flat Float32Array with shape [1, n, dim]. */
function extract2D(data: Float32Array, n: number, dim: number): number[][] {
  return Array.from({ length: n }, (_, s) =>
    Array.from({ length: dim }, (_, d) => r4(data[s * dim + d]))
  );
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function runGPT2Inference(
  sentence: string,
  onProgress?: (p: number) => void,
): Promise<VizExample> {
  await loadAll(onProgress);

  // Tokenize and truncate
  const { input_ids } = _tokenizer(sentence);
  const rawData  = input_ids.data as BigInt64Array | Int32Array;
  const allIds: number[] = [];
  for (let i = 0; i < rawData.length; i++) allIds.push(Number(rawData[i]));
  const tokenIds = allIds.slice(0, MAX_TOKENS);
  const n        = tokenIds.length;
  const tokens   = tokenIds.map((id) => (_tokenizer.decode([id]) as string).trim() || `[${id}]`);

  // Run ONNX model
  const ort   = await import("onnxruntime-web");
  const input = new ort.Tensor("int64", BigInt64Array.from(tokenIds.map(BigInt)), [1, n]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const results: Record<string, any> = await _session.run({ input_ids: input });

  // ── Next-word probabilities ──────────────────────────────────────────────
  const logitsData = results["logits"].data as Float32Array;
  const vocabSize  = results["logits"].dims[2] as number;
  const lastOff    = (n - 1) * vocabSize;

  let maxL = -Infinity;
  for (let i = lastOff; i < lastOff + vocabSize; i++) if (logitsData[i] > maxL) maxL = logitsData[i];
  const exps = new Float32Array(vocabSize);
  let sumE = 0;
  for (let i = 0; i < vocabSize; i++) { exps[i] = Math.exp(logitsData[lastOff + i] - maxL); sumE += exps[i]; }
  const order = Array.from({ length: vocabSize }, (_, i) => i).sort((a, b) => exps[b] - exps[a]);
  const nextWordProbs = order.slice(0, 20).map((id) => ({
    token: (_tokenizer.decode([id]) as string).trim() || `[${id}]`,
    prob:  r4(exps[id] / sumE),
  }));

  // ── Per-layer real Q / K / V / attention / FFN activations ─────────────
  const FFN_HIDDEN_DIM = 3072;
  const EMBED_DIM      = 768;

  const layers: LayerData[] = Array.from({ length: N_LAYERS }, (_, layerIdx) => {
    const qVecs = extractHeads(results[`l${layerIdx}.q`].data as Float32Array, n);
    const kVecs = extractHeads(results[`l${layerIdx}.k`].data as Float32Array, n);
    const vVecs = extractHeads(results[`l${layerIdx}.v`].data as Float32Array, n);
    const multiHeadAttention = extractAttn(results[`l${layerIdx}.attn`].data as Float32Array, n);

    const attentionMatrix = Array.from({ length: n }, (_, row) =>
      Array.from({ length: n }, (_, col) => {
        let s = 0;
        for (let h = 0; h < N_HEADS; h++) s += multiHeadAttention[h][row][col];
        return r4(s / N_HEADS);
      })
    );

    const ffnIn     = results[`l${layerIdx}.ffn_in`]
      ? extract2D(results[`l${layerIdx}.ffn_in`].data     as Float32Array, n, EMBED_DIM)
      : Array.from({ length: n }, () => [] as number[]);
    const ffnHidden = results[`l${layerIdx}.ffn_hidden`]
      ? extract2D(results[`l${layerIdx}.ffn_hidden`].data  as Float32Array, n, FFN_HIDDEN_DIM)
      : Array.from({ length: n }, () => [] as number[]);
    const ffnOut    = results[`l${layerIdx}.ffn_out`]
      ? extract2D(results[`l${layerIdx}.ffn_out`].data     as Float32Array, n, EMBED_DIM)
      : Array.from({ length: n }, () => [] as number[]);

    return {
      layerIdx, multiHeadAttention, attentionMatrix,
      queryVectors: qVecs, keyVectors: kVecs, valueVectors: vVecs,
      ffnIn, ffnHidden, ffnOut,
    };
  });

  return { id: "custom", label: `"${sentence}"`, tokens, layers, nextWordProbs };
}
