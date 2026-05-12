export type TransformerExample = {
  id: string;
  label: string;
  sentence: string;
  tokens: string[];
  attentionMatrix: number[][];
  multiHeadAttention: number[][][];
  headLayer?: number;
  headIndices?: number[];
  rawScoresMatrix?: number[][];        // pre-softmax scores averaged across all heads
  multiHeadRawScores?: number[][][];   // pre-softmax scores for each of the 4 displayed heads
};

export type AttentionResult = Omit<TransformerExample, "id" | "label" | "sentence">;
