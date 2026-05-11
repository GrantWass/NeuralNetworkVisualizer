"use client";

import { useState } from "react";
import { X, BookOpen } from "lucide-react";

const TERMS = [
  {
    term: "Weight",
    definition:
      "A learnable number on each connection between neurons. During training, weights are adjusted so the network makes better predictions. Thicker lines in the diagram = larger weight magnitude.",
  },
  {
    term: "Bias",
    definition:
      "A learnable offset added to each neuron's input before the activation function. Allows the network to shift its output even when all inputs are zero.",
  },
  {
    term: "Activation Function",
    definition:
      "A non-linear function applied to a neuron's weighted sum. Without it, stacking layers would still be linear. Common choices: ReLU (clamps negatives to 0), Sigmoid (squishes to 0–1), Tanh (squishes to −1–1).",
  },
  {
    term: "Forward Pass",
    definition:
      "Computing a prediction by passing inputs through the network layer by layer: input → hidden layers → output. Each layer multiplies by weights, adds biases, then applies an activation function.",
  },
  {
    term: "Loss",
    definition:
      "A number measuring how wrong the network's predictions are. Lower is better. For classification: Cross-Entropy Loss. For regression: Mean Squared Error (MSE). Training tries to minimize this.",
  },
  {
    term: "Gradient",
    definition:
      "The direction and magnitude of steepest increase in loss with respect to a weight. By moving in the opposite direction (gradient descent), the network reduces its loss.",
  },
  {
    term: "Backpropagation",
    definition:
      "The algorithm that computes gradients efficiently by propagating the error signal backwards through the network, layer by layer, using the chain rule of calculus.",
  },
  {
    term: "Learning Rate (η)",
    definition:
      "Controls how large each weight update step is. Too high → weights overshoot and loss diverges. Too low → training is very slow. Typical values: 0.001–0.1.",
  },
  {
    term: "Epoch",
    definition:
      "One full pass through the training dataset. After each epoch, weights are updated. Running many epochs lets the network refine its parameters over time.",
  },
  {
    term: "Pre-activation (Z)",
    definition:
      "The raw weighted sum before the activation function: Z = W·x + b. This intermediate value is shown in the Forward Pass tab.",
  },
  {
    term: "Activation (A)",
    definition:
      "The output of a neuron after applying its activation function: A = σ(Z). Node brightness in the diagram represents this value.",
  },
  {
    term: "Softmax",
    definition:
      "An activation function used in the output layer for classification. Converts raw scores into a probability distribution that sums to 1 (e.g., 87% Setosa, 9% Versicolor, 4% Virginica).",
  },
];

const Glossary = () => {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors border border-gray-300 rounded-md px-3 py-1.5 bg-white hover:bg-gray-50"
        title="Open ML glossary"
      >
        <BookOpen className="w-4 h-4" />
        <span>Glossary</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30"
            onClick={() => setOpen(false)}
          />

          {/* Drawer */}
          <div className="relative w-full max-w-sm bg-white h-full shadow-2xl flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-gray-700" />
                <h2 className="text-lg font-semibold text-gray-900">ML Glossary</h2>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-gray-400 hover:text-gray-700 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
              {TERMS.map(({ term, definition }) => (
                <div key={term}>
                  <p className="font-semibold text-gray-900 text-sm mb-1">{term}</p>
                  <p className="text-sm text-gray-600 leading-relaxed">{definition}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Glossary;
