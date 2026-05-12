import type { Metadata } from "next";
import { JsonLd } from "@/components/JsonLd";
import TransformerVizClient from "./TransformerVizClient";

export const metadata: Metadata = {
  title: "Interactive Transformer Visualization — How Attention Works",
  description:
    "Explore self-attention, the mechanism behind GPT, Claude, BERT, and every modern language model. Click through real attention patterns and see how transformers handle pronouns, agreement, and long-range dependencies.",
  alternates: { canonical: "https://nn-visual.com/transformer" },
  openGraph: {
    title: "Interactive Transformer Visualization — How Attention Works",
    description:
      "Explore self-attention, the mechanism behind GPT, Claude, BERT, and every modern language model. Click through real attention patterns and see how transformers handle pronouns, agreement, and long-range dependencies.",
    url: "https://nn-visual.com/transformer",
  },
};

const learningResourceLd = {
  "@context": "https://schema.org",
  "@type": "LearningResource",
  name: "Interactive Transformer Visualization",
  description:
    "Explore self-attention, the mechanism behind every major language model. Click through attention patterns and see how transformers handle pronouns, agreement, and long-range dependencies.",
  url: "https://nn-visual.com/transformer",
  learningResourceType: "Interactive simulation",
  educationalLevel: "Undergraduate",
  teaches: [
    "Self-attention",
    "Transformers",
    "Multi-head attention",
    "Query Key Value vectors",
    "BERT",
    "Anaphora resolution",
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
  headline: "How Attention Works: An Interactive Transformer Visualization",
  description:
    "A hands-on explanation of self-attention, the core mechanism behind transformer models like GPT, Claude, and BERT.",
  url: "https://nn-visual.com/transformer",
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
  dateModified: "2026-05-12",
};

export default function TransformerPage() {
  return (
    <>
      <JsonLd data={learningResourceLd} />
      <JsonLd data={articleLd} />
      <TransformerVizClient />
    </>
  );
}
