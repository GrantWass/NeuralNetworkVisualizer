"use client";
import React, { useState } from "react";
import ContactInfo from "../contact";
import { EmbeddingsSection } from "@/components/transformer/embedding";
import { AttentionSection } from "@/components/transformer/attention";
import { InferenceSection } from "@/components/transformer/inference";

// Tokenize helper
const tokenize = (text: string) => text.split(/\s+/).filter(Boolean);

const TransformerVizPage: React.FC = () => {
  const [promptInput, setPromptInput] = useState("cat dog king queen");
  const [submittedPrompt, setSubmittedPrompt] = useState(promptInput);

  const tokens = tokenize(submittedPrompt);

  const handleSubmit = () => {
    setSubmittedPrompt(promptInput);
  };

  return (
    <div className="p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Transformer Visualization *IN PROGRESS*</h1>
        <p className="text-sm text-slate-500">Interactive demo: embeddings → attention → inference</p>
      </header>

      <div className="flex items-center gap-2">
        <input
          className="border p-2 rounded flex-1"
          value={promptInput}
          onChange={(e) => setPromptInput(e.target.value)}
        />
        <button
          className="px-4 py-2 bg-black text-white rounded hover:bg-gray-800"
          onClick={handleSubmit}
        >
          Submit
        </button>
      </div>

      <EmbeddingsSection tokens={tokens} />
      {/* <AttentionSection tokens={tokens} dim={32} heads={3} seed={2.1} />
      <InferenceSection prompt={submittedPrompt} dim={16} seed={3.7} /> */}

      <ContactInfo />
    </div>
  );
};

export default TransformerVizPage;
