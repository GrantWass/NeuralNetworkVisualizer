import type { NeuronLayer } from "@/components/network/static/types";

export function applyActivation(z: number[], activation: string): number[] {
  switch (activation) {
    case "relu":    return z.map((v) => Math.max(0, v));
    case "tanh":    return z.map((v) => Math.tanh(v));
    case "sigmoid": return z.map((v) => 1 / (1 + Math.exp(-v)));
    case "softmax": {
      const mx = Math.max(...z);
      const e = z.map((v) => Math.exp(v - mx));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }
    default: return z;
  }
}

export function forwardPassSingle(
  layers: NeuronLayer[],
  input: number[],
): { Z: number[][]; A: number[][] } {
  const zAll: number[][] = [];
  const aAll: number[][] = [];
  let prev = input;
  for (let li = 0; li < layers.length - 1; li++) {
    const { weights: W, biases: b, activation } = layers[li];
    if (!W?.length || !W[0]?.length) break;
    const nOut = W[0].length;
    const z = Array.from({ length: nOut }, (_, j) => {
      let sum = b?.[j] ?? 0;
      for (let i = 0; i < prev.length; i++) sum += (prev[i] ?? 0) * (W[i]?.[j] ?? 0);
      return sum;
    });
    const a = applyActivation(z, activation);
    zAll.push(z);
    aAll.push(a);
    prev = a;
  }
  return { Z: zAll, A: aAll };
}

export function computeStratifiedIndices(data: number[][], dataset: string): number[] {
  const n = data.length;
  const DISPLAY_COUNT = 26;
  if (dataset === "xor") return [0, 1, 2, 3];
  if (n <= DISPLAY_COUNT) return Array.from({ length: n }, (_, i) => i);

  if (dataset === "iris") {
    const buckets: number[][] = [[], [], []];
    data.forEach((row, i) => {
      const scores = [row[4] ?? 0, row[5] ?? 0, row[6] ?? 0];
      const cls = scores.indexOf(Math.max(...scores));
      if (cls >= 0) buckets[cls].push(i);
    });
    const perBucket = buckets.map((b) => {
      const want = Math.ceil(DISPLAY_COUNT / 3);
      return Array.from({ length: Math.min(want, b.length) }, (_, j) =>
        b[Math.round((j / (Math.min(want, b.length) - 1 || 1)) * (b.length - 1))]
      ).filter((v): v is number => v !== undefined);
    });
    const result: number[] = [];
    const ptrs = [0, 0, 0];
    while (result.length < DISPLAY_COUNT) {
      let added = false;
      for (let c = 0; c < 3; c++) {
        if (ptrs[c] < (perBucket[c]?.length ?? 0)) {
          result.push(perBucket[c][ptrs[c]++]!);
          added = true;
          if (result.length >= DISPLAY_COUNT) break;
        }
      }
      if (!added) break;
    }
    return result;
  }

  if (dataset === "mnist") {
    const buckets: number[][] = Array.from({ length: 10 }, () => []);
    data.forEach((row, i) => {
      const oneHot = row.slice(784, 794);
      if (oneHot.length < 10) return;
      const cls = oneHot.indexOf(Math.max(...oneHot));
      if (cls >= 0) buckets[cls].push(i);
    });
    const perClass = Math.ceil(DISPLAY_COUNT / 10);
    const perBucket = buckets.map((b) =>
      Array.from({ length: Math.min(perClass, b.length) }, (_, j) =>
        b[Math.round((j / (Math.min(perClass, b.length) - 1 || 1)) * (b.length - 1))]
      ).filter((v): v is number => v !== undefined)
    );
    const result: number[] = [];
    const ptrs = new Array(10).fill(0);
    while (result.length < DISPLAY_COUNT) {
      let added = false;
      for (let c = 0; c < 10; c++) {
        if (ptrs[c] < (perBucket[c]?.length ?? 0)) {
          result.push(perBucket[c][ptrs[c]++]!);
          added = true;
          if (result.length >= DISPLAY_COUNT) break;
        }
      }
      if (!added) break;
    }
    return result.slice(0, DISPLAY_COUNT);
  }

  // auto_mpg or unknown: spread evenly through dataset
  const step = (n - 1) / (DISPLAY_COUNT - 1);
  return Array.from({ length: DISPLAY_COUNT }, (_, i) => Math.round(i * step));
}
