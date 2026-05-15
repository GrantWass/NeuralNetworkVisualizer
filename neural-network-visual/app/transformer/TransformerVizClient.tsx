"use client";
import { useState, useRef, useEffect } from "react";
import encDecTrace from "./enc_dec_trace.json";
import Link from "next/link";
import ContactInfo from "../contact";
import { HeatmapSVG } from "@/components/transformer/HeatmapSVG";
import { QKVBreakdown } from "@/components/transformer/QKVBreakdown";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
// import { Button } from "@/components/ui/button";   // live inference disabled
// import { Input } from "@/components/ui/input";      // live inference disabled
import { Gloss } from "@/components/transformer/Gloss";
import { EmbeddingExplorer } from "@/components/transformer/EmbeddingExplorer";
import type { TransformerExample, AttentionResult } from "./types";
import examplesData from "./data.json";

const examples = examplesData as TransformerExample[];

// Sentence used in sections 1 and 2
const SECTION1_SENTENCE = "The animal didn't cross the street because it was too tired";
const SECTION1_TOKENS = SECTION1_SENTENCE.split(/\s+/);

// ─── Section 1 ────────────────────────────────────────────────────────────────

function Section1() {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">The Problem Attention Solves</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        Read this sentence: <em>"{SECTION1_SENTENCE}."</em> What does "it" refer
        to — the animal or the street? You resolved that instantly, but how?
        Your brain didn't scan every word equally. It attended selectively,
        weighting <strong>animal</strong> and <strong>tired</strong> heavily
        when interpreting <strong>it</strong>.
      </p>
      <div
        className="flex flex-wrap gap-2 p-4 rounded-lg border border-border bg-card"
        aria-label="Sentence tokens"
      >
        {SECTION1_TOKENS.map((token, i) => {
          const isIt = token.toLowerCase() === "it";
          return (
            <span
              key={i}
              className={[
                "px-2.5 py-1 rounded-md text-sm font-mono border",
                isIt
                  ? "bg-indigo-50 border-indigo-400 ring-2 ring-indigo-400 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-200 font-semibold"
                  : "bg-secondary border-border text-secondary-foreground",
              ].join(" ")}
            >
              {token}
            </span>
          );
        })}
      </div>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        Older architectures like RNNs read words one at a time, left to right.
        By the time the model reached "it," the signal from "animal" had passed
        through many intermediate steps and faded. The transformer fixes this
        with <strong>self-attention</strong>: every token computes a direct
        connection to every other token simultaneously, so "it" can attend to
        "animal" in a single step regardless of how far apart they are.
      </p>
    </section>
  );
}

// ─── Section 2 ────────────────────────────────────────────────────────────────

function Section2({
  tokens,
  attentionMatrix,
  rawScoresMatrix,
  multiHeadAttention,
  multiHeadRawScores,
  queryVectors,
  keyVectors,
  headIndices,
  selectedIdx,
}: {
  tokens: string[];
  attentionMatrix: number[][];
  rawScoresMatrix?: number[][];
  multiHeadAttention?: number[][][];
  multiHeadRawScores?: number[][][];
  queryVectors?: number[][][];
  keyVectors?: number[][][];
  headIndices?: number[];
  selectedIdx: number;
}) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">How Attention Works Mechanically</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        For each token, the model learns three vectors:{" "}
        <Gloss term="Query">
          A learned linear projection of each token into a 64-dimensional vector encoding what
          this token is searching for in its context. Think of it as a question broadcast to
          every other token. At training time, the projection weights are optimized so that
          tokens needing related information produce aligned Query and Key vectors.
        </Gloss>{" "}
        ("what am I looking for?"),{" "}
        <Gloss term="Key">
          A learned projection encoding what a token contains and is willing to share. Tokens
          whose Keys align with a given Query receive more attention weight from that token.
          Q and K are independent projections — what a token is "asking for" and what it
          "offers" are separate learned representations that serve different roles.
        </Gloss>{" "}
        ("what do I contain?"), and{" "}
        <Gloss term="Value">
          A third projection, independent of Q and K. Once attention weights are decided by
          Q·K matching, the Value vector is what actually gets mixed into the output. Think of
          it this way: K determines whether "animal" gets attended to; V determines what
          information "animal" contributes when it does. The same token can have a very
          different Key and Value — matching and contributing are separate jobs.
        </Gloss>{" "}
        ("what do I pass along?"). To score how much token A should attend to token B, we
        compute the{" "}
        <Gloss term="dot product">
          Measures geometric alignment between two vectors: the sum of element-wise products.
          If Q and K have large values in the same dimensions (same sign), the dot product is
          large and positive. Dimensions where they disagree (opposite signs) subtract from the
          score. It collapses two 64-d vectors into a single number summarizing how much they
          point in the same direction.
        </Gloss>{" "}
        of A's Query and B's Key.{" "}
        We divide by{" "}
        <Gloss term="√d">
          Without this scaling, dot products grow proportionally to the vector dimension d.
          In high-dimensional spaces, large scores push softmax toward 0/1 extremes, which
          kills gradients during training. Dividing by √d (here √64 = 8) keeps scores in a
          stable range regardless of model size — a fix introduced in the original
          "Attention Is All You Need" paper.
        </Gloss>{" "}
        to stop scores from growing too large, then apply{" "}
        <Gloss term="softmax">
          Converts a vector of real numbers into probabilities summing to 1 using
          eˣⁱ / Σeˣ. The exponential amplifies differences non-linearly: a score advantage
          of 2.0 gives ~7× more weight, not 2×. This means the highest-scoring token
          captures a disproportionately large fraction of the attention budget — attention
          concentrates rather than spreading evenly.
        </Gloss>{" "}
        to convert scores into probabilities that sum to 1. Finally, each token's output is a{" "}
        <Gloss term="weighted sum">
          output = Σᵢ wᵢ · Vᵢ. Before this step every token has a fixed, context-free
          embedding. After it, each token's representation is a blend of all Value vectors
          weighted by attention. The same word produces a different output in every sentence
          because the weights change with context — this is how "it" comes to encode that
          it refers to an animal rather than a street.
        </Gloss>{" "}
        of all Value vectors, where the weights are those probabilities.
      </p>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        Click any token below to step through the computation using real BERT attention
        weights. Use the head selector to see how different heads specialize in different
        relationships.
      </p>
      <QKVBreakdown
        key={selectedIdx}
        tokens={tokens}
        attentionMatrix={attentionMatrix}
        rawScoresMatrix={rawScoresMatrix}
        multiHeadAttention={multiHeadAttention}
        multiHeadRawScores={multiHeadRawScores}
        queryVectors={queryVectors}
        keyVectors={keyVectors}
        headIndices={headIndices}
      />
    </section>
  );
}

// ─── Per-example heatmap interpretation ──────────────────────────────────────

const EXAMPLE_INTERPRETATION: Record<string, string> = {
  anaphora:
    '"it" attends most strongly to "animal" — the model correctly identifies what the pronoun refers to, even though the words "didn\'t cross the street because" sit between them. ',
  modifier:
    '"with" attends most strongly to "man," telling us the model reads "with the telescope" as describing the man — not the act of seeing. This is the kind of structural ambiguity ("did she use the telescope, or did the man have it?") that trips up simpler models.',
  ditransitive:
    '"gave" attends most strongly to "Mary," picking out who did the giving. The word "a" splits its attention between "book" and "gave," linking the object to both its own noun and the verb that governs it.',
  agreement:
    '"is" and "sleeping" attend strongly to each other, the model linking the helper verb to the main verb it pairs with. "on" points back to "cat" — connecting the preposition to the noun it modifies rather than the nearby "mats."',
  reflexive:
    '"himself" attends most strongly to "hurt" — the verb it is the object of — rather than back to "John." The model is tracking the grammatical role here: "himself" belongs to "hurt." The word "while" also points to "hurt," linking the two parts of the sentence.',
  coordination:
    '"and" attends to "cats" and "dogs" attends back to "and," forming the two-way link between the joined nouns. "wonderful" points to "make," connecting the adjective to the verb it modifies.',
  winograd:
    'Multiple tokens converge on "large" — the adjective that explains why the trophy doesn\'t fit. The subject "trophy" itself draws broad attention as the sentence anchor. The pronoun "it" does not strongly resolve to "trophy" in the averaged view; that coreference signal lives in individual heads rather than the average.',
  negation:
    '"not" attends strongly to "did" and "chase" attends back to "not," capturing the three-word negation unit as a single semantic block. Almost every token also pulls toward "cat" — the model treats the direct object as the focal point of the sentence, even when the action it undergoes is negated.',
};

// ─── Per-example, per-head hand-written analyses ─────────────────────────────
// Heads [0, 3, 7, 11] from BERT layer 6. Each array has 4 entries matching
// multiHeadAttention[0..3]. Used in Section 4; live sentences fall back to
// detectHeadPattern.

type HeadAnalysis = { label: string; description: string };

const HEAD_ANALYSIS: Record<string, HeadAnalysis[]> = {
  anaphora: [
    {
      label: "Coreference",
      description:
        '"it" attends strongly to "animal." Across examples this head consistently links pronouns and repeated nouns back to their antecedents — it is the closest thing BERT has to a dedicated reference-tracking head.',
    },
    {
      label: "Verb dependency",
      description:
        '"the" and "street" both point strongly to "cross." This head pulls article and noun tokens toward the main verb of their clause — mapping out which words belong to the same predicate.',
    },
    {
      label: "Clause bridging",
      description:
        '"was" attends back to "didn\'t" across the "because" boundary. This head often links the two verb phrases of a complex clause, building long-range structural connections rather than local ones.',
    },
    {
      label: "Local chain",
      description:
        'Strong backward attention throughout — each token attends heavily to its left neighbor. This head is tracking local word order rather than meaning. It is the most positional of the four.',
    },
  ],
  modifier: [
    {
      label: "Determiner tracking",
      description:
        '"man" attends to "the." This head traces articles back to the nouns they modify. Across examples it consistently links determiners to their noun heads rather than making longer-range connections.',
    },
    {
      label: "PP attachment",
      description:
        '"telescope" and "the" both attend strongly to "with," and "with" then points to "man." This head is resolving the classic prepositional-phrase ambiguity — deciding the instrument belongs to the man, not to the act of seeing.',
    },
    {
      label: "Semantic focus",
      description:
        'Almost every token attends to "telescope" — the concrete object at the end of the sentence. This head tends to fixate on the most salient content noun, treating it as the semantic anchor of the phrase.',
    },
    {
      label: "Noun–modifier bond",
      description:
        '"with" attends almost exclusively to "man." This head captures the direct syntactic bond between a prepositional head and its governing noun. It mostly follows the immediately preceding token, but here that happens to encode the PP attachment perfectly.',
    },
  ],
  ditransitive: [
    {
      label: "Determiner–noun",
      description:
        '"a" attends strongly to "book." This head links every article to the noun it determines. Across examples it is the most reliable tracker of determiner–head noun pairs.',
    },
    {
      label: "Object–verb",
      description:
        '"book" and "a" both attend back to "gave." This head maps objects and their determiners back to the governing verb — identifying the predicate that assigns their grammatical role.',
    },
    {
      label: "Argument pairing",
      description:
        '"mary" and "john" attend strongly to each other. This head links the two noun arguments of the clause — subject and indirect object. Across examples it consistently pairs the main participants in a sentence.',
    },
    {
      label: "Local chain",
      description:
        'Strong backward attention following the left-to-right syntactic chain — "a" to "john," "john" to "gave." This head builds a sequential backbone rather than capturing semantic roles.',
    },
  ],
  agreement: [
    {
      label: "Function-word spread",
      description:
        'Attention is diffuse here — "is" toward "on," "mats" toward "the." This head does not show a single dominant pattern for this sentence; it may be distributing context broadly across the prepositional phrase.',
    },
    {
      label: "Verb–auxiliary link",
      description:
        '"sleeping" attends strongly to "is" and "on" points to "cat." This head connects the main verb to its auxiliary and the preposition to the noun head it modifies — both are exactly the dependencies needed to parse the subject-verb structure correctly.',
    },
    {
      label: "Agreement ambiguity",
      description:
        '"mats" and "cat" attend strongly to each other. These are the two nouns in "the cat on the mats" — the source of the number-agreement ambiguity. This head has captured the key noun pair that determines whether the verb should be singular or plural.',
    },
    {
      label: "Local chain",
      description:
        'Backward attention dominates — each token attends to its predecessor. In this sentence that happens to encode the preposition pointing to the noun it follows, but the pattern is structural rather than semantic.',
    },
  ],
  reflexive: [
    {
      label: "Antecedent resolution",
      description:
        '"himself" attends strongly to "john" across four intervening tokens. This head also pulls "cooking" and "while" back toward "john," treating the subject as the sentence anchor. It is the closest thing BERT has to a dedicated coreference head.',
    },
    {
      label: "Reflexive–verb link",
      description:
        '"himself" attends almost exclusively to "hurt." This head connects the reflexive to the verb it is the object of — identifying the grammatical role rather than the coreference. The difference from Head 1 is subtle but real: role vs. reference.',
    },
    {
      label: "Clause-final anchor",
      description:
        'Most tokens attend heavily to "cooking." This head has anchored on the sentence-final verb, likely encoding the participial clause boundary. Across examples it tends to fix on the last content word as a structural endpoint.',
    },
    {
      label: "Local chain",
      description:
        '"himself" attends to "hurt" and "cooking" attends to "while." The pattern is mostly backward attention — left-neighbor tracking — though in this sentence the neighbors happen to be meaningful pairs (reflexive after its verb, adverbial after its conjunction).',
    },
  ],
  coordination: [
    {
      label: "Conjunct linking",
      description:
        '"dogs" attends strongly to "cats" and "and" attends back to "cats." This head links the second conjunct back to the first, encoding the symmetric relationship between the two nouns in the coordinate structure.',
    },
    {
      label: "Predicate–modifier",
      description:
        '"wonderful" attends almost exclusively to "make." Predicate adjectives and conjuncts both point to their governing word. Across examples this head reliably maps dependent words — objects, adjectives, conjuncts — back to the verb or coordinator that governs them.',
    },
    {
      label: "Broad context",
      description:
        'Attention is spread fairly evenly across all tokens with no dominant pair. This head is building a general sentence-level representation rather than encoding a specific syntactic link — useful for downstream tasks that need a global summary.',
    },
    {
      label: "Coordination chain",
      description:
        '"dogs," "and," and "cats" attend almost exclusively to each other. This head has nearly entirely captured the conjunction structure, treating "and" as the hinge between the two nouns. It is the clearest example of a positional head accidentally encoding something linguistically meaningful.',
    },
  ],
  winograd: [
    {
      label: "Subject anchoring",
      description:
        'Most tokens across the sentence — "the," "fit," "in," "suitcase" — attend toward "trophy." This head treats the sentence subject as the primary anchor, pulling in article, verb, and preposition alike. Notably, "it" does not strongly resolve to "trophy" here — that role falls to a different head. This one maps the structural gravity of the subject rather than resolving the pronoun.',
    },
    {
      label: "Clause-internal chain",
      description:
        '"is" attends almost exclusively to "it," and "in" points strongly to "fit." This head tracks the syntactic backbone within each clause — predicate to subject pronoun, preposition to verb — making one direct dependency step at a time rather than bridging across clause boundaries.',
    },
    {
      label: "Predicate anchor",
      description:
        'Nearly every token attends toward "large" — the adjective that answers why the trophy doesn\'t fit. "trophy," "doesn\'t," "in," and "suitcase" all point there. This head has identified the key predicate adjective as the semantic endpoint of the sentence, effectively converging on the stated cause regardless of position.',
    },
    {
      label: "Sequential backbone",
      description:
        'Extremely strong consecutive links: "is" attends to "it," "it" attends to "because," "in" points to "fit," "too" points to "is." This head builds the left-to-right syntactic chain of the sentence. The pronoun "it" bridges the two clauses by linking to "because" rather than to "trophy" — encoding clause structure rather than coreference.',
    },
  ],
  negation: [
    {
      label: "Self-preservation",
      description:
        'Most tokens attend primarily to themselves — "chase," "did," and the second "the" all show high self-attention. The pair "did not" shows modest mutual linking. This head is largely preserving each token\'s own representation rather than redistributing information. Across examples this pattern appears in heads that serve as identity residuals rather than dependency-tracking heads.',
    },
    {
      label: "Negation structure",
      description:
        '"not" attends almost entirely to "did," and "chase" splits its attention between "not" and "did." This head has captured the three-word negation sequence "did not chase" as a unit — the main verb traces back through the negation to the auxiliary. The second "the" also points to "chase," linking the article of the object to the governing verb.',
    },
    {
      label: "Object fixation",
      description:
        'Every token in the sentence attends heavily toward "cat" — subject, auxiliary, negation, verb, and determiners all converge there. Only "cat" itself breaks the pattern, pointing back to "dog." This head has latched on to the direct object as the semantic endpoint. Across examples this type of fixation appears in heads that anchor on the most goal-oriented or concrete noun in the sentence.',
    },
    {
      label: "Subject–auxiliary chain",
      description:
        '"did" attends almost entirely to "dog," and "not" also strongly attends to "did" and "dog." The second "the" points almost entirely to "chase." This head is encoding the subject–auxiliary dependency and the determiner–verb relationship — the structural skeleton of the verb phrase. The strength of "did" → "dog" is especially notable: the auxiliary has locked on to its subject.',
    },
  ],
};

// ─── Head pattern detector ────────────────────────────────────────────────────

type HeadPattern = { label: string; description: string };

function detectHeadPattern(rawMatrix: number[][], rawTokens: string[]): HeadPattern {
  // Strip [CLS], [SEP], etc. before scoring — same filter as HeatmapSVG
  const keep = rawTokens.map((t) => !/^\[.*\]$/.test(t));
  const indices = keep.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
  const matrix = indices.map((i) => indices.map((j) => rawMatrix[i][j]));

  const n = matrix.length;
  if (n < 2) return { label: "Attention pattern", description: "" };

  let diagSum = 0;
  let nextSum = 0;
  let prevSum = 0;
  let entropySum = 0;

  for (let i = 0; i < n; i++) {
    const row = matrix[i];
    const rowSum = row.reduce((a, b) => a + b, 0) || 1;

    diagSum += row[i] / rowSum;
    if (i + 1 < n) nextSum += row[i + 1] / rowSum;
    if (i - 1 >= 0) prevSum += row[i - 1] / rowSum;

    // Per-row Shannon entropy (normalized to [0,1])
    let h = 0;
    for (const v of row) {
      const p = v / rowSum;
      if (p > 1e-9) h -= p * Math.log2(p);
    }
    entropySum += h / Math.log2(n);
  }

  const diag = diagSum / n;
  const next = nextSum / Math.max(n - 1, 1);
  const prev = prevSum / Math.max(n - 1, 1);
  const entropy = entropySum / n;

  // Uniform baseline: 1/n
  const uniform = 1 / n;
  const threshold = 3 * uniform;

  const scores: [number, HeadPattern][] = [
    [next, { label: "Forward attention", description: "Each token mostly attends to the next token — this head is picking up left-to-right word order." }],
    [prev, { label: "Backward attention", description: "Each token mostly attends to the previous token — this head reads right-to-left, often tracking grammatical links." }],
    [diag, { label: "Self-attention", description: "Each token attends most strongly to itself — this head is largely keeping each word's own information intact." }],
  ];

  // Pick the highest-scoring structural pattern if it clears the threshold
  const best = scores.reduce((a, b) => (b[0] > a[0] ? b : a));
  if (best[0] > threshold) return best[1];

  // High entropy → broadly distributed
  if (entropy > 0.75) {
    return { label: "Broad attention", description: "Attention is spread roughly evenly — this head is building a general summary of the whole sentence rather than focusing on specific word pairs." };
  }

  return { label: "Focused attention", description: "Attention is concentrated on a small number of words — this head has learned a specific relationship, like which words refer to the same thing." };
}


// ─── Section 3 + 4 ────────────────────────────────────────────────────────────

function Sections3And4({
  selectedIdx,
  liveResult,
  setLiveResult,
}: {
  selectedIdx: number;
  liveResult: AttentionResult | null;
  setLiveResult: (r: AttentionResult | null) => void;
}) {
  const activeExample = liveResult
    ? { ...examples[selectedIdx], ...liveResult }
    : examples[selectedIdx];

  return (
    <>
      {/* Section 3 */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Attention as a Heatmap</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          We can lay out every token-to-token attention score as a grid. Each
          row is a token asking a question; each column is a token being
          considered as an answer. A bright cell means the row token is paying
          strong attention to the column token. Patterns in this grid reveal
          what relationships the model has learned to track.
        </p>

        {liveResult && (
          <p className="text-xs text-muted-foreground">
            Showing live inference result.{" "}
            <button
              className="underline hover:no-underline"
              onClick={() => setLiveResult(null)}
            >
              Clear
            </button>
          </p>
        )}

        <div className="space-y-2">
          <p className="text-sm font-medium">
            Average
            <span className="ml-2 text-xs font-normal text-muted-foreground">all heads combined</span>
          </p>
          <HeatmapSVG tokens={activeExample.tokens} matrix={activeExample.attentionMatrix} cellSize={26} />
        </div>

        {!liveResult && EXAMPLE_INTERPRETATION[examples[selectedIdx].id] && (
          <p className="text-muted-foreground leading-relaxed max-w-2xl">
            {EXAMPLE_INTERPRETATION[examples[selectedIdx].id]}
          </p>
        )}
      </section>

      {/* Section 4 */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Multi-Head Attention</h2>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          A single attention head can only focus on one type of relationship at
          a time. Transformers run many heads in parallel — each with its own
          independent Query, Key, and Value weights — then concatenate all their
          outputs. This lets different heads specialize: one might track which
          words refer to the same thing, another might focus on which verb a
          noun belongs to, another on nearby word order.
        </p>
        <p className="text-muted-foreground leading-relaxed max-w-2xl">
          The four grids below show individual heads from BERT layer 6. Each is
          automatically labeled by the pattern its attention forms. Compare them
          to the averaged grid above — averaging mixes all the heads together
          and hides the distinct roles each one plays.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {activeExample.multiHeadAttention.map((headMatrix, hi) => {
            const headNumber = activeExample.headIndices
              ? (activeExample.headIndices![hi] ?? hi) + 1
              : hi + 1;
            const curated = !liveResult && HEAD_ANALYSIS[examples[selectedIdx].id]?.[hi];
            const { label, description } = curated || detectHeadPattern(headMatrix, activeExample.tokens);
            return (
              <div key={hi} className="space-y-2">
                <div>
                  <p className="text-sm font-medium">
                    Head {headNumber}
                    <span className="ml-2 text-xs font-normal text-indigo-500">{label}</span>
                  </p>
                  {description && (
                    <p className="text-xs text-muted-foreground mt-0.5 max-w-xs">{description}</p>
                  )}
                </div>
                <HeatmapSVG
                  tokens={activeExample.tokens}
                  matrix={headMatrix}
                  cellSize={18}
                />
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}

// ─── Section 5: Encoder vs. Decoder ──────────────────────────────────────────

function CausalMaskViz() {
  const tokens = ["The", "cat", "sat", "on", "mat"];
  const n = tokens.length;
  const cellSize = 32;
  const labelPad = 36;
  const topPad = 48;
  const width = labelPad + n * cellSize;
  const height = topPad + n * cellSize;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      style={{ fontFamily: "var(--font-geist-mono, monospace)" }}
      aria-label="Causal attention mask: each token can only attend to itself and earlier tokens"
    >
      {/* Top labels */}
      {tokens.map((t, j) => (
        <text
          key={`col-${j}`}
          x={labelPad + j * cellSize + cellSize / 2}
          y={topPad - 6}
          fontSize={9}
          textAnchor="start"
          fill="currentColor"
          className="text-foreground"
          transform={`rotate(-45, ${labelPad + j * cellSize + cellSize / 2}, ${topPad - 6})`}
        >
          {t}
        </text>
      ))}
      {/* Left labels */}
      {tokens.map((t, i) => (
        <text
          key={`row-${i}`}
          x={labelPad - 4}
          y={topPad + i * cellSize + cellSize / 2 + 4}
          fontSize={9}
          textAnchor="end"
          fill="currentColor"
          className="text-foreground"
        >
          {t}
        </text>
      ))}
      {/* Cells */}
      {Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
          const allowed = j <= i;
          const intensity = allowed ? (i === j ? 1 : 0.45 - (i - j) * 0.07) : 0;
          const r = Math.round(255 - intensity * (255 - 99));
          const g = Math.round(255 - intensity * (255 - 102));
          const b = Math.round(255 - intensity * (255 - 241));
          const fill = allowed ? `rgb(${r},${g},${b})` : "hsl(var(--muted))";
          return (
            <rect
              key={`${i}-${j}`}
              x={labelPad + j * cellSize}
              y={topPad + i * cellSize}
              width={cellSize}
              height={cellSize}
              fill={fill}
              stroke="rgba(0,0,0,0.06)"
              strokeWidth={0.5}
            />
          );
        })
      )}
      {/* "masked" label in upper-right */}
      <text
        x={labelPad + 2.8 * cellSize}
        y={topPad + 0.6 * cellSize}
        fontSize={8}
        fill="hsl(var(--muted-foreground))"
        textAnchor="middle"
      >
        masked
      </text>
    </svg>
  );
}

function Section5() {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Encoder vs. Decoder</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        BERT, the model shown on this page, is an <strong>encoder-only</strong>{" "}
        transformer. When processing a sentence, every token can attend to every
        other token simultaneously — left, right, and across any distance. That
        bidirectional view makes encoders excellent at understanding tasks: named
        entity recognition, question answering, sentence classification.
      </p>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        The models you interact with day-to-day such as GPT, Claude, Llama, and Mistral 
        are <strong>decoder-only</strong>. They generate text one token at a time,
        left to right. Because each token is produced before the ones that follow
        it, future tokens don't exist yet and must be masked out. The attention
        matrix becomes lower-triangular: a token can only attend to itself and
        everything to its left.
      </p>

      <div className="flex flex-col sm:flex-row gap-8 items-start">
        <div className="space-y-2">
          <p className="text-sm font-medium">
            Causal mask
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              decoder attention pattern
            </span>
          </p>
          <CausalMaskViz />
          <p className="text-xs text-muted-foreground max-w-xs">
            Shaded cells are allowed; muted cells are masked to −∞ before
            softmax, making their attention weights effectively zero.
          </p>
        </div>
        <div className="space-y-3 max-w-sm text-sm text-muted-foreground leading-relaxed">
          <p>
            This constraint is a feature, not a limitation. Training a decoder
            is self-supervised: given any text, the model learns by predicting
            the next token, using only what came before as context. No labels
            required. This objective scales to internet-scale corpora, which is
            why decoder-only models have dominated the scaling curve.
          </p>
          <p>
            There is a third family: <strong>encoder-decoder</strong>{" "}
            transformers (T5, BART, the original translation model from
            "Attention Is All You Need"). The encoder reads the full input; the
            decoder generates the output one token at a time, cross-attending
            to the encoder's representations. Translation and summarization are
            natural fits. In practice, very large decoder-only models have
            matched or exceeded encoder-decoder models on most benchmarks,
            which is why the industry has largely converged on decoder-only
            architectures.
          </p>
        </div>
      </div>
    </section>
  );
}

// ─── Inference Trace: shared heatmap ─────────────────────────────────────────

function traceColor(v: number): string {
  const c = Math.max(0, Math.min(1, v));
  return `rgb(${Math.round(255 - c * 156)},${Math.round(255 - c * 153)},${Math.round(255 - c * 14)})`;
}

// Derive encoder data from real BERT weights — strip [CLS] / [SEP]
const _encRaw = encDecTrace.encoder;
const _encKeep = _encRaw.tokens
  .map((t, i) => (!/^\[.*\]$/.test(t) ? i : -1))
  .filter((i) => i >= 0);
const ENCODER_TOKENS = _encKeep.map((i) => _encRaw.tokens[i]);
const ENCODER_MATRIX = _encKeep.map((i) =>
  _encKeep.map((j) => _encRaw.attentionMatrix[i][j])
);

// Decoder data from real GPT-2 weights — already clean, upper triangle is 0
const DECODER_TOKENS = encDecTrace.decoder.tokens as string[];
const DECODER_MATRIX = encDecTrace.decoder.attentionMatrix as number[][];

// Generation step distributions from real GPT-2 forward passes
const GENERATION_PROMPT = encDecTrace.generationPrompt;

function InferenceHeatmap({
  tokens,
  matrix,
  hoveredRow,
  onHoverRow,
  maskUpperTriangle = false,
  normalizeRows = false,
  cellSize = 28,
}: {
  tokens: string[];
  matrix: number[][];
  hoveredRow: number | null;
  onHoverRow: (i: number | null) => void;
  maskUpperTriangle?: boolean;
  normalizeRows?: boolean;
  cellSize?: number;
}) {
  const n = tokens.length;
  const lp = 44, tp = 48;
  const w = lp + n * cellSize, h = tp + n * cellSize;

  // Per-row normalization: scale each row by its own max over the visible cells.
  // Needed for the decoder where the attention-sink token absorbs ~80% of every
  // row's budget, washing out the linguistically interesting pairs.
  const displayMatrix = normalizeRows
    ? matrix.map((row, i) => {
        const visibleMax = Math.max(
          ...row.filter((_, j) => !(maskUpperTriangle && j > i)),
          1e-9
        );
        return row.map((v) => v / visibleMax);
      })
    : matrix;

  const globalMax = normalizeRows
    ? 1
    : Math.max(...matrix.flatMap((row, i) => row.filter((_, j) => !(maskUpperTriangle && j > i))), 1e-9);

  const d = (t: string) => (t.length > 6 ? t.slice(0, 5) + "…" : t);

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width={w}
      height={h}
      style={{ fontFamily: "var(--font-geist-mono,monospace)", display: "block" }}
      onMouseLeave={() => onHoverRow(null)}
    >
      {tokens.map((t, j) => (
        <text
          key={j}
          x={lp + j * cellSize + cellSize / 2}
          y={tp - 6}
          fontSize={9}
          textAnchor="start"
          fill="currentColor"
          transform={`rotate(-45,${lp + j * cellSize + cellSize / 2},${tp - 6})`}
        >
          {d(t)}
        </text>
      ))}
      {hoveredRow !== null && (
        <rect
          x={0}
          y={tp + hoveredRow * cellSize}
          width={w}
          height={cellSize}
          fill="rgba(99,102,241,0.10)"
          style={{ pointerEvents: "none" }}
        />
      )}
      {tokens.map((t, i) => (
        <g key={i} onMouseEnter={() => onHoverRow(i)}>
          <rect x={0} y={tp + i * cellSize} width={lp} height={cellSize} fill="transparent" />
          <text
            x={lp - 4}
            y={tp + i * cellSize + cellSize / 2 + 3}
            fontSize={9}
            textAnchor="end"
            fontWeight={hoveredRow === i ? "bold" : "normal"}
            fill="currentColor"
          >
            {d(t)}
          </text>
        </g>
      ))}
      {matrix.map((row, i) =>
        row.map((_, j) => {
          const muted = maskUpperTriangle && j > i;
          return (
            <rect
              key={`${i}-${j}`}
              x={lp + j * cellSize}
              y={tp + i * cellSize}
              width={cellSize}
              height={cellSize}
              fill={muted ? "hsl(var(--muted))" : traceColor(displayMatrix[i][j] / globalMax)}
              stroke={hoveredRow === i ? "rgba(99,102,241,0.35)" : "rgba(0,0,0,0.06)"}
              strokeWidth={hoveredRow === i ? 1 : 0.5}
              onMouseEnter={() => onHoverRow(i)}
            />
          );
        })
      )}
    </svg>
  );
}

// ─── Section 6: Interactive Enc/Dec Trace ─────────────────────────────────────

function Section6() {
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Encoder vs. Decoder: Side-by-Side</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        Both architectures run the same sentence through attention — but the matrices look
        fundamentally different. The encoder (BERT) sees the full sequence bidirectionally;
        the decoder (GPT-2) is lower-triangular because future tokens don't exist yet. Hover
        any token row to see the same row highlighted in both panels simultaneously.
      </p>
      <div className="flex flex-col sm:flex-row gap-8 items-start flex-wrap">
        <div className="space-y-1.5">
          <p className="text-sm font-medium">
            Encoder · BERT
            <span className="ml-2 text-xs font-normal text-indigo-500">
              bidirectional · all tokens visible
            </span>
          </p>
          <p className="text-xs text-muted-foreground">
            Row token can attend anywhere — left <em>or</em> right
          </p>
          <InferenceHeatmap
            tokens={ENCODER_TOKENS}
            matrix={ENCODER_MATRIX}
            hoveredRow={hoveredRow}
            onHoverRow={setHoveredRow}
          />
        </div>
        <div className="space-y-1.5">
          <p className="text-sm font-medium">
            Decoder · GPT-2
            <span className="ml-2 text-xs font-normal text-indigo-500">
              causal mask · left context only
            </span>
          </p>
          <p className="text-xs text-muted-foreground">
            Single head (layer 11, head 8) · muted cells masked to −∞
          </p>
          <InferenceHeatmap
            tokens={DECODER_TOKENS}
            matrix={DECODER_MATRIX}
            hoveredRow={hoveredRow}
            onHoverRow={setHoveredRow}
            maskUpperTriangle
          />
        </div>
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed max-w-2xl">
        Consider "sat" (row 2). In BERT its attention reaches forward to "on" and "mat." In
        GPT-2, "sat" is blind to its right — it only knows "the" and "cat," and attends
        mostly to "cat" (its subject). The GPT-2 panel shows a single head (layer 11,
        head 8) chosen because it encodes clear syntactic links; the 12-head average is
        swamped by an <em>attention sink</em> where every token dumps most of its budget
        onto position 0 — a known GPT-2 artifact, not a linguistic feature.
      </p>
    </section>
  );
}

// ─── Section 7: Token-by-Token Generation ────────────────────────────────────

type VocabEntry = { token: string; prob: number };
type SamplerStatus = "idle" | "loading" | "ready" | "running";

function TokenSampler() {
  const [selected, setSelected] = useState<string[]>([]);
  const [dist, setDist] = useState<VocabEntry[] | null>(null);
  const [status, setStatus] = useState<SamplerStatus>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tokenizerRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);

  const promptTokens = GENERATION_PROMPT.split(" ");
  const done = selected.length >= 5;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { init(); }, []);
  const maxProb = dist ? Math.max(...dist.map((e) => e.prob)) : 1;
  const totalShown = dist ? dist.reduce((s, e) => s + e.prob, 0) : 0;

  async function infer(tokens: string[]) {
    setStatus("running");
    const ctx = [...promptTokens, ...tokens].join(" ");
    const { input_ids } = tokenizerRef.current(ctx);
    const { logits } = await modelRef.current({ input_ids });
    const [, seqLen, vocabSize]: number[] = logits.dims;
    const data: Float32Array = logits.data;
    const off = (seqLen - 1) * vocabSize;

    // Numerically stable softmax over last-position logits
    let mx = -Infinity;
    for (let i = off; i < off + vocabSize; i++) if (data[i] > mx) mx = data[i];
    const exps = new Float32Array(vocabSize);
    let sum = 0;
    for (let i = 0; i < vocabSize; i++) { exps[i] = Math.exp(data[off + i] - mx); sum += exps[i]; }

    const ids = Array.from({ length: vocabSize }, (_, i) => i);
    ids.sort((a, b) => exps[b] - exps[a]);

    setDist(ids.slice(0, 12).map((id) => ({
      token: tokenizerRef.current.decode([id]).trim() || `[${id}]`,
      prob: exps[id] / sum,
    })));
    setStatus("ready");
  }

  async function init() {
    setStatus("loading");
    try {
      const { AutoTokenizer, AutoModelForCausalLM, env } =
        await import("@xenova/transformers");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (env as any).allowLocalModels = false;
      tokenizerRef.current = await AutoTokenizer.from_pretrained("Xenova/gpt2");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      modelRef.current = await (AutoModelForCausalLM as any).from_pretrained("Xenova/gpt2", {
        dtype: "q4",
        progress_callback: (d: any) => {
          if (typeof d.progress === "number") setLoadProgress(Math.round(d.progress));
        },
      });
      await infer([]);
    } catch (e) {
      console.error(e);
      setStatus("idle");
    }
  }

  async function pick(token: string) {
    const next = [...selected, token];
    setSelected(next);
    if (next.length < 5) await infer(next);
    else setDist(null);
  }

  return (
    <div className="space-y-4 max-w-xl">
      {/* Sequence display */}
      <div className="flex flex-wrap items-center gap-1.5 p-3 rounded-lg border border-border bg-card font-mono text-sm min-h-[46px]">
        {promptTokens.map((t, i) => (
          <span key={i} className="px-2 py-0.5 rounded bg-secondary border border-border text-secondary-foreground">
            {t}
          </span>
        ))}
        {selected.map((t, i) => (
          <span key={i} className="px-2 py-0.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-800 dark:bg-indigo-950 dark:border-indigo-800 dark:text-indigo-200">
            {t}
          </span>
        ))}
        {status === "ready" && !done && (
          <span className="inline-block w-0.5 h-4 bg-foreground/60 animate-pulse" />
        )}
      </div>

      {/* Download progress */}
      {status === "loading" && (
        <div className="space-y-1.5">
          <p className="text-xs text-muted-foreground">
            Downloading GPT-2 (~40 MB, cached after first load)…
          </p>
          <div className="h-1.5 w-64 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-300 rounded-full"
              style={{ width: `${loadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Inference spinner */}
      {status === "running" && (
        <p className="text-xs text-muted-foreground animate-pulse">Running forward pass…</p>
      )}

      {/* Distribution */}
      {status === "ready" && dist && !done && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground mb-2">
            Step {selected.length + 1} of 5 — click any bar to sample
          </p>
          {dist.map(({ token, prob }) => (
            <button
              key={token}
              className="flex items-center gap-2 w-full group"
              onClick={() => pick(token)}
            >
              <span className="w-16 text-xs font-mono text-right text-muted-foreground group-hover:text-foreground transition-colors">
                {token}
              </span>
              <div className="flex-1 h-4 bg-muted/50 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-indigo-400/80 group-hover:bg-indigo-500 transition-colors duration-100 rounded-sm"
                  style={{ width: `${((prob / maxProb) * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="w-10 text-xs text-muted-foreground text-right">
                {(prob * 100).toFixed(1)}%
              </span>
            </button>
          ))}
          <p className="text-xs text-muted-foreground pt-1.5 border-t border-border mt-2">
            Top 12 of ~50,000 tokens · {(totalShown * 100).toFixed(0)}% of probability mass shown
          </p>
        </div>
      )}

      {done && (
        <p className="text-sm text-muted-foreground italic">
          Generation complete — restart and make different choices to see how sequences diverge.
        </p>
      )}

      {(status === "ready" || done) && (
        <button
          className="text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground transition-colors"
          onClick={() => { setSelected([]); infer([]); }}
        >
          ↺ Restart
        </button>
      )}
    </div>
  );
}

function Section7() {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Generating One Token at a Time</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        A decoder doesn't output a whole sentence at once. At every step it runs a real{" "}
        <strong>forward pass over the full context</strong> and produces a probability
        distribution over the entire vocabulary. Click any bar to sample that token — the
        next distribution is computed live from GPT-2 running in your browser.
      </p>
      <TokenSampler />
      <p className="text-muted-foreground text-sm leading-relaxed max-w-2xl">
        Pick different tokens at each step and watch the distributions shift. This is why{" "}
        <strong>temperature</strong> matters: lowering it concentrates probability on the
        tallest bars, making output more deterministic; raising it spreads mass across the
        distribution, introducing more variety.
      </p>
    </section>
  );
}

// ─── Further Reading ──────────────────────────────────────────────────────────

function FurtherReading() {
  const links = [
    {
      href: "https://arxiv.org/abs/1706.03762",
      text: "Attention Is All You Need (Vaswani et al., 2017)",
    },
    {
      href: "https://jalammar.github.io/illustrated-transformer/",
      text: "The Illustrated Transformer — Jay Alammar",
    },
    {
      href: "https://arxiv.org/abs/1810.04805",
      text: "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
    },
    {
      href: "https://nlp.seas.harvard.edu/2018/04/03/attention.html",
      text: "The Annotated Transformer — Harvard NLP",
    },
  ];
  return (
    <section className="space-y-3 border-t border-border pt-6">
      <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">
        Further reading
      </h2>
      <ul className="space-y-1">
        {links.map((l) => (
          <li key={l.href}>
            <a
              href={l.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-foreground underline underline-offset-4 hover:text-muted-foreground"
            >
              {l.text}
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function TransformerVizClient() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  // live inference disabled — re-enable when BERT is back in the build
  // const [liveResult, setLiveResult] = useState<AttentionResult | null>(null);
  // const [userInput, setUserInput] = useState("");
  // const [inferenceStatus, setInferenceStatus] = useState<"idle" | "loading" | "error">("idle");
  // async function handleCompute() { ... }

  const rawTokens = examples[selectedIdx].tokens;
  const keepIndices = rawTokens
    .map((t, i) => (!/^\[.*\]$/.test(t) ? i : -1))
    .filter((i) => i >= 0);
  const tokens = keepIndices.map((i) => rawTokens[i]);
  const ex = examples[selectedIdx];

  // Strip [CLS]/[SEP] from 2-D matrices
  const strip2 = (m: number[][]) => keepIndices.map((i) => keepIndices.map((j) => m[i][j]));
  // Strip [CLS]/[SEP] from per-head vector arrays: [heads × seq × dim] → [heads × stripped_seq × dim]
  const stripVecs = (v: number[][][]) => v.map((head) => keepIndices.map((i) => head[i]));

  const attentionMatrix     = strip2(ex.attentionMatrix);
  const rawScoresMatrix     = ex.rawScoresMatrix     ? strip2(ex.rawScoresMatrix)     : undefined;
  const multiHeadAttention  = ex.multiHeadAttention  ? ex.multiHeadAttention.map((h) => strip2(h)) : undefined;
  const multiHeadRawScores  = ex.multiHeadRawScores  ? ex.multiHeadRawScores.map((h) => strip2(h)) : undefined;
  const queryVectors        = ex.queryVectors        ? stripVecs(ex.queryVectors)        : undefined;
  const keyVectors          = ex.keyVectors          ? stripVecs(ex.keyVectors)          : undefined;

  return (
    <div className="p-4 max-w-9xl mx-auto space-y-12">
      <header className="space-y-4 py-6 border-b border-border">
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-indigo-500" />
          Interactive Visualization
        </span>
        <h1 className="text-3xl font-bold tracking-tight">Transformer &amp; Self-Attention</h1>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          The transformer is the architecture behind GPT, BERT, and almost every
          modern language model. Its core idea — <strong>self-attention</strong> —
          lets every word in a sentence directly look at every other word to
          decide what context it needs. This page walks through that mechanism
          step by step, using real attention weights extracted from BERT (Bidirectional
          Encoder Representations from Transformers).”
        </p>
      </header>

      <EmbeddingExplorer />

      <Section1 />

      {/* Sentence chooser — example picker + live input side by side */}
      <div className="rounded-lg border border-border bg-card p-4 space-y-3 max-w-sm">
        <p className="text-sm font-medium">Choose a sentence</p>
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Example picker */}
          <div className="flex flex-col gap-1.5 flex-1">
            <label className="text-xs text-muted-foreground" htmlFor="example-select">
              Pick an example
            </label>
            <Select
              value={String(selectedIdx)}
              onValueChange={(v) => { setSelectedIdx(Number(v)); }}
            >
              <SelectTrigger id="example-select" className="w-full">
                <SelectValue placeholder="Choose an example" />
              </SelectTrigger>
              <SelectContent>
                {examples.map((ex, i) => (
                  <SelectItem key={ex.id} value={String(i)}>
                    {ex.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Live input disabled — re-enable when BERT is back in the build */}
          {/* <div className="hidden sm:flex items-center">
            <span className="text-xs text-muted-foreground px-2">or</span>
          </div>
          <div className="flex flex-col gap-1.5 flex-1">
            <label className="text-xs text-muted-foreground">Try your own</label>
            <div className="flex gap-2">
              <Input value={userInput} onChange={(e) => setUserInput(e.target.value)}
                placeholder="The cat sat on the mat"
                onKeyDown={(e) => e.key === "Enter" && handleCompute()} className="flex-1" />
              <Button onClick={handleCompute}
                disabled={inferenceStatus === "loading" || !userInput.trim()}>
                {inferenceStatus === "loading" ? "Loading…" : "Compute"}
              </Button>
            </div>
            {inferenceStatus === "error" && (
              <p className="text-xs text-destructive">Couldn&apos;t reach the model — try one of the examples above.</p>
            )}
          </div> */}
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed border-t border-border pt-3">
          These examples are hand-picked to surface clear patterns. As you explore, look for pronouns and what they refer to, verbs and their subjects, or a modifier and the noun it belongs to.
        </p>
      </div>

      <Section2
        tokens={tokens}
        attentionMatrix={attentionMatrix}
        rawScoresMatrix={rawScoresMatrix}
        multiHeadAttention={multiHeadAttention}
        multiHeadRawScores={multiHeadRawScores}
        queryVectors={queryVectors}
        keyVectors={keyVectors}
        headIndices={ex.headIndices}
        selectedIdx={selectedIdx}
      />
      <Sections3And4
        selectedIdx={selectedIdx}
        liveResult={null}
        setLiveResult={() => {}}
      />
      <Section5 />
      <Section6 />
      <Section7 />
      <FurtherReading />

      <section className="border-t border-border pt-6">
        <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">
          Continue learning
        </h2>
        <p className="mt-2 text-muted-foreground">
          Return to the{" "}
          <Link href="/" className="underline underline-offset-4 hover:text-foreground">
            neural network visualizer
          </Link>
          .
        </p>
      </section>

      <ContactInfo />
    </div>
  );
}
