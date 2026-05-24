import type { Metadata } from "next";
import Link from "next/link";
import { JsonLd } from "@/components/JsonLd";

export const metadata: Metadata = {
  title: "About NN-Visual",
  description:
    "Learn how NN-Visual started as a deep learning journey and became a tool for teaching neural networks and transformers.",
  alternates: { canonical: "https://nn-visual.com/about" },
  openGraph: {
    title: "About NN-Visual",
    description:
      "Learn how NN-Visual started as a deep learning journey and became a tool for teaching neural networks and transformers.",
    url: "https://nn-visual.com/about",
  },
};

const personLd = {
  "@context": "https://schema.org",
  "@type": "Person",
  name: "Grant Wasserman",
  url: "https://grantwasserman.com",
  sameAs: [
    "https://www.linkedin.com/in/grant-wass/",
    "https://grantwasserman.com",
  ],
  creator: [
    {
      "@type": "WebApplication",
      name: "Interactive Neural Network Visualization",
      url: "https://nn-visual.com",
    },
    {
      "@type": "WebApplication",
      name: "How Attention Works — Interactive Visualization",
      url: "https://nn-visual.com/attention",
    },
    {
      "@type": "WebApplication",
      name: "How Transformers Work — Interactive Architecture Diagram",
      url: "https://nn-visual.com/transformers",
    },
  ],
};

const articleLd = {
  "@context": "https://schema.org",
  "@type": "Article",
  headline:
    "About NN-Visual: How an Interactive Neural Network Visualizer Came to Be",
  description:
    "The origin story of NN-Visual — from building a neural network from scratch to creating a teaching tool used in university classrooms.",
  url: "https://nn-visual.com/about",
  author: {
    "@type": "Person",
    name: "Grant Wasserman",
    url: "https://grantwasserman.com",
  },
  publisher: {
    "@type": "Person",
    name: "Grant Wasserman",
  },
  datePublished: "2026-05-14",
  dateModified: "2026-05-21",
};

export default function AboutPage() {
  return (
    <div className="p-4 max-w-6xl mx-auto">
      <JsonLd data={personLd} />
      <JsonLd data={articleLd} />

      <header className="mt-4 mb-6 space-y-4 pb-6 border-b border-border">
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-indigo-500" />
          About
        </span>
        <h1 className="text-3xl font-bold tracking-tight">About NN-Visual</h1>
        <p className="text-muted-foreground max-w-2xl leading-relaxed">
          I&apos;m Grant — a developer who built this site because a professor
          asked a question that sparked an idea.
        </p>
      </header>

      <article className="max-w-2xl space-y-10">

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Where it started</h2>
          <p className="text-muted-foreground leading-relaxed">
            A few years ago I decided to build a neural network entirely from scratch in
            Python — no PyTorch, no Keras, just NumPy and the math. The goal was to understand
            exactly what happens at each step: how the weights update, what the gradients
            represent, why activation functions change a network&apos;s behavior. Working at that
            level gives you access to everything the higher-level libraries abstract away:
            intermediate activations, pre-activation values, model weights and biases, per-layer deltas during backprop.
          </p>
          <p className="text-muted-foreground leading-relaxed">
            Around the same time, a professor I respect mentioned he wished he had a better way
            to show students how neural networks actually work — not the equations in isolation,
            not a high-level diagram, but something that bridged the two. The gap between the
            underlying math and the overall picture is where most students get lost.
          </p>
          <p className="text-muted-foreground leading-relaxed">
            I realized I already had what was needed to close that gap. The scratch implementation
            surfaced every intermediate value at every layer. All that was left was building a
            way to show it that allowed other students to explore and build intuition for how it all fit together.
          </p>
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">How it&apos;s evolved</h2>
          <p className="text-muted-foreground leading-relaxed">
            The first version was simple — a graphical visualization of a small network training on toy
            data and the underlying math that powered it. Over the past couple of years the site has grown to cover more ideas,
            realistic datasets, and advanced visualizations that enable deeper exploration. You can now step through both forward
            propagation and backpropagation, tracing how individual weights and biases update while training on real-world datasets
            such as handwritten digits or car fuel economy data.
          </p>
          <p className="text-muted-foreground leading-relaxed">
            The transformer visualizer came later, when the same professor challanged me to 
            create an intuitive visualization of attention mechanisms as they became the
            concept everyone wanted to understand but few could explain concretely. The same
            principle applied: take the intermediate state of query and key vectors, raw attention
            scores, per-head weights and make it something you can actually click through. It helps students
            draw conclusions as to how an attention mechanism draws connections and semantic understanding.
          </p>
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Who uses it</h2>
          <p className="text-muted-foreground leading-relaxed">
            That professor has been showing it to his students consistently ever since. It has
            become a regular part of how he introduces deep learning in his courses — something
            I genuinely didn&apos;t expect when I started. Knowing it works as a teaching tool
            is the most useful and rewarding feedback I could get.
          </p>
          <p className="text-muted-foreground leading-relaxed">
            Beyond the classroom, the site is free and open to anyone building intuition for
            how these models work — students, developers who are new to ML, or people who just
            want to understand what a gradient descent step actually does to the weights.
          </p>
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Try it</h2>
          <p className="text-muted-foreground leading-relaxed">
            All three visualizers are live and require no setup.
          </p>
          <ul className="space-y-3 text-muted-foreground">
            <li>
              <Link
                href="/"
                className="underline underline-offset-4 hover:text-foreground font-medium text-foreground"
              >
                Neural Network Visualizer
              </Link>
              {" "}— configure a network, pick a dataset, and train it. Step through forward
              and backpropagation to see exactly how each activation and weight changes at
              every layer.
            </li>
            <li>
              <Link
                href="/attention"
                className="underline underline-offset-4 hover:text-foreground font-medium text-foreground"
              >
                Attention Visualizer
              </Link>
              {" "}— explore self-attention using real BERT weights. Click any token to step
              through the Q·K scoring, see multi-head patterns, and compare encoder vs. decoder masking.
            </li>
            <li>
              <Link
                href="/transformers"
                className="underline underline-offset-4 hover:text-foreground font-medium text-foreground"
              >
                Transformer Architecture Visualizer
              </Link>
              {" "}— see the full transformer pipeline with a live GPT-2 inference diagram.
              Explore how embeddings, stacked attention blocks, and feed-forward layers
              work together to produce an output distribution.
            </li>
          </ul>
        </section>

        <section className="space-y-3 border-t border-border pt-6">
          <h2 className="text-lg font-semibold">Get in touch</h2>
          <p className="text-muted-foreground leading-relaxed">
            If you&apos;re using this in a course, have feedback, or just want to say hello —
            I&apos;d like to hear from you and connect!
          </p>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
            <a
              href="https://grantwasserman.com"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-4 hover:text-foreground"
            >
              grantwasserman.com
            </a>
            <a
              href="https://www.linkedin.com/in/grant-wass/"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-4 hover:text-foreground"
            >
              LinkedIn
            </a>
            <a
              href="mailto:grantmwasserman@gmail.com"
              className="underline underline-offset-4 hover:text-foreground"
            >
              grantmwasserman@gmail.com
            </a>
          </div>
        </section>

      </article>
    </div>
  );
}
