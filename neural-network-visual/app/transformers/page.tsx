import type { Metadata } from "next";
import { JsonLd } from "@/components/JsonLd";
import TransformersClient from "./TransformersClient";

export const metadata: Metadata = {
  title: {
    absolute: "Transformer Architecture Visualizer — See How Transformers Work",
  },
  description:
    "Explore transformer architecture step by step with an interactive GPT-2 visualizer. See embeddings, attention blocks, and output predictions free, with no install.",
  alternates: { canonical: "https://nn-visual.com/transformers" },
  openGraph: {
    title: "Transformer Architecture Visualizer — See How Transformers Work",
    description:
      "Explore transformer architecture step by step with an interactive GPT-2 visualizer. See embeddings, attention blocks, and output predictions free, with no install.",
    url: "https://nn-visual.com/transformers",
  },
};

const learningResourceLd = {
  "@context": "https://schema.org",
  "@type": "LearningResource",
  name: "How Transformers Work — Interactive Architecture Diagram",
  description:
    "An interactive diagram of the transformer architecture with real attention weights. Explore embeddings, attention blocks, and output distributions for GPT and BERT.",
  url: "https://nn-visual.com/transformers",
  learningResourceType: "Interactive simulation",
  educationalLevel: "Undergraduate",
  teaches: [
    "Transformer architecture",
    "Self-attention",
    "Multi-head attention",
    "Encoder vs Decoder",
    "GPT",
    "BERT",
    "Embeddings",
    "Feed-forward networks",
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
  headline: "How Transformers Work: An Interactive Architecture Visualization",
  description:
    "A visual walkthrough of the transformer — the architecture behind GPT, Claude, and BERT. Includes an interactive pipeline diagram with real attention weights.",
  url: "https://nn-visual.com/transformers",
  author: {
    "@type": "Person",
    name: "Grant Wasserman",
    url: "https://grantwasserman.com",
  },
  publisher: {
    "@type": "Person",
    name: "Grant Wasserman",
  },
  datePublished: "2026-05-19",
  dateModified: "2026-05-19",
};

export default function TransformersPage() {
  return (
    <>
      <JsonLd data={learningResourceLd} />
      <JsonLd data={articleLd} />
      <TransformersClient />
    </>
  );
}
