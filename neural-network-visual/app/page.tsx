import type { Metadata } from "next";
import type { ReactNode } from "react";
import fs from "fs";
import path from "path";
import Link from "next/link";
import Config from "@/components/network/config";
import Graph from "@/components/network/neural";
import Explain from "@/components/network/explain";
import { JsonLd } from "@/components/JsonLd";
import ContactInfo from "./contact";

type MarkdownBlock =
  | { type: "h2"; text: string }
  | { type: "h3"; text: string }
  | { type: "h4"; text: string }
  | { type: "p"; text: string }
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

export const metadata: Metadata = {
  title: "Interactive Neural Network Visualization",
  description:
    "An interactive tool for understanding how neural networks learn. Configure hidden layers, train on real data, and watch forward and backpropagation step by step.",
  alternates: { canonical: "https://nn-visual.com" },
  openGraph: {
    title: "Interactive Neural Network Visualization",
    description:
      "An interactive tool for understanding how neural networks learn. Configure hidden layers, train on real data, and watch forward and backpropagation step by step.",
    url: "https://nn-visual.com",
    images: ["/og/home.png"],
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
    <div className="p-4 max-w-9xl mx-auto relative">
      <JsonLd data={learningResourceLd} />
      <JsonLd data={articleLd} />

      <header className="mt-4 mb-6 space-y-3">
        <h1 className="text-3xl font-bold">Interactive Neural Network Visualization</h1>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          Configure a network, train it on real data, and watch forward and
          backpropagation step by step. Below the visualizer, a written
          explanation walks through what&apos;s happening and why.
        </p>
      </header>

      <Config />
      <Graph />
      <Explain />

      <article className="mt-8 max-w-[65ch] space-y-3 border-t border-border pt-6">
        {markdownBlocks.map((block, index) => {
          if (block.type === "h2") {
            return (
              <h2 key={index} className="text-lg font-semibold">
                {renderInline(block.text)}
              </h2>
            );
          }

          if (block.type === "h3") {
            return (
              <h3 key={index} className="text-base font-semibold text-foreground">
                {renderInline(block.text)}
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

          if (block.type === "ul") {
            return (
              <ul key={index} className="list-disc pl-5 space-y-0.5 text-muted-foreground">
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInline(item)}</li>
                ))}
              </ul>
            );
          }

          if (block.type === "ol") {
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

      <section className="mt-10 max-w-[65ch] border-t border-border pt-6">
        <h2 className="text-base font-semibold text-muted-foreground uppercase tracking-wide">
          Continue learning
        </h2>
        <p className="mt-2 text-muted-foreground">
          Explore the{" "}
          <Link href="/transformer" className="underline underline-offset-4 hover:text-foreground">
            interactive transformer visualization
          </Link>
          .
        </p>
      </section>

      <ContactInfo />
    </div>
  );
}