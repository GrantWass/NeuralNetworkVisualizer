import type { Metadata } from "next";
import type { ReactNode } from "react";
import fs from "fs";
import path from "path";
import Link from "next/link";
import Config from "@/components/network/config";
import Graph from "@/components/network/neural";
import Explain from "@/components/network/explain";
import Walkthrough from "@/components/network/walkthrough";
import { JsonLd } from "@/components/JsonLd";
import ContactInfo from "./contact";
import {
  BrainCircuit,
  Network,
  TrendingUp,
  ScanEye,
  Compass,
  Layers,
} from "lucide-react";

type MarkdownBlock =
  | { type: "h2"; text: string }
  | { type: "h3"; text: string }
  | { type: "h4"; text: string }
  | { type: "p"; text: string }
  | { type: "blockquote"; text: string }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] };

const markdownPath = path.join(
  process.cwd(),
  "content",
  "neural-network-explanation.md"
);
const markdown = fs.readFileSync(markdownPath, "utf8");

function parseMarkdown(content: string): MarkdownBlock[] {
  const lines = content.split(/\r?\n/);
  const blocks: MarkdownBlock[] = [];
  let paragraph: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let listItems: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ type: "p", text: paragraph.join(" ") });
      paragraph = [];
    }
  };

  const flushList = () => {
    if (listType && listItems.length > 0) {
      blocks.push({ type: listType, items: listItems });
      listItems = [];
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (trimmed.startsWith("## ")) {
      flushParagraph();
      flushList();
      blocks.push({ type: "h2", text: trimmed.slice(3).trim() });
      continue;
    }

    if (trimmed.startsWith("#### ")) {
      flushParagraph();
      flushList();
      blocks.push({ type: "h4", text: trimmed.slice(5).trim() });
      continue;
    }

    if (trimmed.startsWith("### ")) {
      flushParagraph();
      flushList();
      blocks.push({ type: "h3", text: trimmed.slice(4).trim() });
      continue;
    }

    if (trimmed.startsWith("> ")) {
      flushParagraph();
      flushList();
      blocks.push({ type: "blockquote", text: trimmed.slice(2).trim() });
      continue;
    }

    const unorderedMatch = /^-\s+(.+)$/.exec(trimmed);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(unorderedMatch[1].trim());
      continue;
    }

    const orderedMatch = /^\d+\.\s+(.+)$/.exec(trimmed);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(orderedMatch[1].trim());
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  return blocks;
}

function renderInline(text: string) {
  const parts: ReactNode[] = [];
  const pattern = /`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|(\*[^*]+\*)|(_[^_]+_)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    if (match[1]) {
      parts.push(
        <code
          key={`code-${parts.length}`}
          className="rounded bg-muted px-1 py-0.5 font-mono text-xs"
        >
          {match[1]}
        </code>
      );
    } else if (match[2] && match[3]) {
      parts.push(
        <a
          key={`link-${parts.length}`}
          href={match[3]}
          className="underline underline-offset-4 hover:text-muted-foreground"
        >
          {match[2]}
        </a>
      );
    } else if (match[4]) {
      parts.push(<em key={`em-${parts.length}`}>{match[4].slice(1, -1)}</em>);
    } else if (match[5]) {
      parts.push(<em key={`em-${parts.length}`}>{match[5].slice(1, -1)}</em>);
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

const markdownBlocks = parseMarkdown(markdown);

const SECTION_ICONS: Record<string, ReactNode> = {
  "The problem neural networks solve": (
    <BrainCircuit className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
  ),
  "How a neural network represents information": (
    <Network className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
  ),
  "How a neural network learns": (
    <TrendingUp className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
  ),
  "Reading the visualization": (
    <ScanEye className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
  ),
  "What's next": (
    <Compass className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
  ),
};

const SUBSECTION_ICONS: Record<string, ReactNode> = {
  "Weights, biases, and activations": (
    <Layers className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
  ),
};

// Activation function names get pill styling
const ACTIVATION_FN_NAMES = new Set(["ReLU", "Sigmoid", "Tanh", "Softmax"]);

// The 5-step training loop gets stepper treatment
const TRAINING_STEPS = [
  "Make a prediction",
  "Measure error",
  "Compute responsibility",
  "Update parameters",
  "Repeat",
];

function isTrainingStepList(items: string[]) {
  return (
    items.length === TRAINING_STEPS.length &&
    items.every((item, i) => item === TRAINING_STEPS[i])
  );
}

function isActivationFnList(items: string[]) {
  return items.length > 0 && items.every((item) => ACTIVATION_FN_NAMES.has(item));
}

function TrainingStepper() {
  return (
    <div className="my-2 flex flex-wrap items-center gap-1">
      {TRAINING_STEPS.map((step, i) => (
        <div key={step} className="flex items-center gap-1">
          <div className="flex items-center gap-1.5 rounded-md border border-border bg-muted/50 px-2.5 py-1.5">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-indigo-500 text-[10px] font-bold text-white">
              {i + 1}
            </span>
            <span className="text-xs font-medium text-foreground">{step}</span>
          </div>
          {i < TRAINING_STEPS.length - 1 && (
            <span className="text-muted-foreground/50 text-xs">→</span>
          )}
        </div>
      ))}
    </div>
  );
}

function ActivationChips({ items }: { items: string[] }) {
  const colors: Record<string, string> = {
    ReLU: "bg-orange-100 text-orange-700 border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800",
    Sigmoid: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-800",
    Tanh: "bg-violet-100 text-violet-700 border-violet-200 dark:bg-violet-950/40 dark:text-violet-300 dark:border-violet-800",
    Softmax: "bg-green-100 text-green-700 border-green-200 dark:bg-green-950/40 dark:text-green-300 dark:border-green-800",
  };
  return (
    <div className="my-1 flex flex-wrap gap-2">
      {items.map((name) => (
        <span
          key={name}
          className={`rounded-md border px-2.5 py-1 text-xs font-mono font-semibold ${colors[name] ?? "bg-muted text-foreground border-border"}`}
        >
          {name}
        </span>
      ))}
    </div>
  );
}

export const metadata: Metadata = {
  title: "Interactive Neural Network Visualizer — See How Each Layer Activates",
  description:
    "An interactive tool for understanding how neural networks learn. Configure hidden layers, train on real data, and watch forward and backpropagation step by step.",
  alternates: { canonical: "https://nn-visual.com" },
  openGraph: {
    title: "Interactive Neural Network Visualizer — See How Each Layer Activates",
    description:
      "An interactive tool for understanding how neural networks learn. Configure hidden layers, train on real data, and watch forward and backpropagation step by step.",
    url: "https://nn-visual.com",
  },
};

const learningResourceLd = {
  "@context": "https://schema.org",
  "@type": "LearningResource",
  name: "Interactive Neural Network Visualization",
  description:
    "An interactive tool for understanding how neural networks learn. Configure hidden layers, train on real data, and watch forward and backpropagation step by step.",
  url: "https://nn-visual.com",
  learningResourceType: "Interactive simulation",
  educationalLevel: "Undergraduate",
  teaches: [
    "Neural networks",
    "Backpropagation",
    "Forward propagation",
    "Gradient descent",
    "Activation functions",
  ],
  audience: {
    "@type": "EducationalAudience",
    educationalRole: "student",
  },
  isAccessibleForFree: true,
  inLanguage: "en",
  author: {
    "@type": "Person",
    name: "Grant Wasserman",
    url: "https://grantwasserman.com",
  },
};

const articleLd = {
  "@context": "https://schema.org",
  "@type": "Article",
  headline: "How Neural Networks Learn: An Interactive Explanation",
  description:
    "A walkthrough of how neural networks represent and learn from data, paired with an interactive visualization.",
  url: "https://nn-visual.com",
  author: {
    "@type": "Person",
    name: "Grant Wasserman",
    url: "https://grantwasserman.com",
  },
  publisher: {
    "@type": "Person",
    name: "Grant Wasserman",
  },
  datePublished: "2026-05-11",
  dateModified: "2026-05-11",
};

export default function NeuralNetworkViz() {
  return (
    <div className="p-4 max-w-6xl mx-auto relative">
      <JsonLd data={learningResourceLd} />
      <JsonLd data={articleLd} />

      <header className="mt-4 mb-6 space-y-4 pb-6 border-b border-border">
        <div className="flex items-center justify-between gap-4">
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-indigo-500" />
            Interactive Visualization
          </span>
          <Walkthrough />
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Neural Network Visualizer</h1>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          Configure a network, train it on real data, and watch forward and
          backpropagation step by step. Below the visualizer, a written
          explanation walks through what&apos;s happening and why.
        </p>
        <p className="text-muted-foreground max-w-2xl leading-relaxed text-sm">
          Build intuition for deep learning — from gradient descent and
          loss functions to weights, biases, and activation functions. No
          code required.
        </p>
      </header>

      <Config />
      <Graph />
      <Explain />

      <article className="mt-8 w-full space-y-3 border-t border-border pt-6">
        {markdownBlocks.map((block, index) => {
          if (block.type === "h2") {
            const icon = SECTION_ICONS[block.text];
            return (
              <h2 key={index} className="flex items-start gap-2 text-lg font-semibold pt-2 first:pt-0">
                {icon}
                <span>{renderInline(block.text)}</span>
              </h2>
            );
          }

          if (block.type === "h3") {
            const icon = SUBSECTION_ICONS[block.text];
            return (
              <h3 key={index} className="flex items-start gap-1.5 text-base font-semibold text-foreground">
                {icon}
                <span>{renderInline(block.text)}</span>
              </h3>
            );
          }

          if (block.type === "h4") {
            return (
              <h4 key={index} className="text-base font-semibold text-muted-foreground">
                {renderInline(block.text)}
              </h4>
            );
          }

          if (block.type === "blockquote") {
            return (
              <blockquote
                key={index}
                className="my-1 border-l-2 border-indigo-400 pl-4 italic text-muted-foreground"
              >
                {renderInline(block.text)}
              </blockquote>
            );
          }

          if (block.type === "ul") {
            if (isActivationFnList(block.items)) {
              return <ActivationChips key={index} items={block.items} />;
            }
            return (
              <ul key={index} className="list-disc pl-5 space-y-0.5 text-muted-foreground">
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInline(item)}</li>
                ))}
              </ul>
            );
          }

          if (block.type === "ol") {
            if (isTrainingStepList(block.items)) {
              return <TrainingStepper key={index} />;
            }
            return (
              <ol key={index} className="list-decimal pl-5 space-y-0.5 text-muted-foreground">
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInline(item)}</li>
                ))}
              </ol>
            );
          }

          return (
            <p key={index} className="text-muted-foreground leading-relaxed">
              {renderInline(block.text)}
            </p>
          );
        })}
      </article>

      <section className="mt-10 w-full border-t border-border pt-6">
        <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">
          Continue learning
        </h2>
        <p className="mt-2 text-muted-foreground">
          Explore{" "}
          <Link href="/attention" className="underline underline-offset-4 hover:text-foreground">
            how attention works
          </Link>
          , see the{" "}
          <Link href="/transformers" className="underline underline-offset-4 hover:text-foreground">
            full transformer architecture
          </Link>
          , or read{" "}
          <Link href="/about" className="underline underline-offset-4 hover:text-foreground">
            how this project started
          </Link>
          .
        </p>
      </section>

      <ContactInfo />
    </div>
  );
}