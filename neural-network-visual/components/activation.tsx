"use client";

import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Line } from "react-chartjs-2";
import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from "chart.js";
Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale);

const activationFunctions = {
  relu: (x: number) => Math.max(0, x),
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  tanh: (x: number) => Math.tanh(x),
  linear: (x: number) => x,
};

function getChartData(fn: (x: number) => number) {
  const xs = Array.from({ length: 101 }, (_, i) => -5 + i * 0.1);
  return {
    labels: xs.map(x => Number(x.toFixed(1))), // round to 1 decimal place
    datasets: [
      {
        label: "y = f(x)",
        data: xs.map(fn),
        borderColor: "#0070f3",
        fill: false,
        pointRadius: 0,
      },
    ],
  };
}

function ModalPortal({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  if (typeof document === "undefined") return null;

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />

      <div
        role="dialog"
        aria-modal="true"
        className="relative z-10 w-[90%] max-w-md rounded-lg bg-white p-6 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body
  );
}

export function ActivationInfoPopup() {
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState<"relu" | "sigmoid" | "tanh" | "linear">("relu");
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isOpen]);

  // Focus the close button when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => closeBtnRef.current?.focus(), 0);
    }
  }, [isOpen]);

  return (
    <span className="inline-block ml-2">
      <button
        aria-label="Show activation function info"
        onClick={() => setIsOpen(true)}
        className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gray-100 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-primary"
      >
        <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
          <circle cx="10" cy="10" r="9" stroke="#0070f3" strokeWidth="1.5" fill="#fff" />
          <path d="M10 7.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm1 2.5h-2v5h2v-5z" fill="#0070f3" />
        </svg>
      </button>

      {isOpen && (
        <ModalPortal onClose={() => setIsOpen(false)}>
          <button
            ref={closeBtnRef}
            aria-label="Close activation info"
            onClick={() => setIsOpen(false)}
            className="absolute right-3 top-3 text-lg leading-none bg-transparent border-0 cursor-pointer"
          >
            ×
          </button>

          <h3 className="mb-2 text-lg font-semibold">Select Activation Function</h3>

          <label htmlFor="activation-select" className="block mb-2 text-sm font-medium">
            Activation
          </label>
          <select
            id="activation-select"
            value={selected}
            onChange={(e) => setSelected(e.target.value as typeof selected)}
            className="mb-4 w-full rounded border px-2 py-1"
          >
            <option value="relu">relu</option>
            <option value="sigmoid">sigmoid</option>
            <option value="tanh">tanh</option>
            <option value="linear">linear</option>
          </select>

          <p className="mb-4 text-sm">
            {selected === "relu" &&
              "Returns max(0, x). Commonly used for hidden layers in deep neural networks due to its simplicity and effectiveness at reducing vanishing gradients."}
            {selected === "sigmoid" &&
              "S-shaped curve, outputs between 0 and 1. Often used for binary classification problems and output layers where probabilities are needed."}
            {selected === "tanh" &&
              "Outputs between -1 and 1. Frequently used in hidden layers, especially in recurrent neural networks, as it centers data and can help with convergence."}
            {selected === "linear" &&
              "Identity function, outputs x. Typically used in the output layer for regression tasks where the output is a continuous value."}
          </p>

          <div className="h-44 w-full rounded border border-gray-200 bg-gray-50 flex items-center justify-center text-sm text-gray-500">
             <Line
  data={getChartData(activationFunctions[selected])}
  options={{
    responsive: true,
    plugins: { legend: { display: false } },
  }}
/>
          </div>
        </ModalPortal>
      )}
    </span>
  );
}
