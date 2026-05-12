"use client";

import useStore from "@/components/network/lib/store";
import { NeuronLayer } from "@/components/network/static/types";
import { useState, useEffect, useRef } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { InputInfo, OutputInfo } from "@/components/network/tooltips";
import { reshapeTo2D, transpose, multiplyMatrices } from "@/components/network/lib/utils"
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import { useMemo } from "react";
import Glossary from "@/components/network/glossary";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

type PropagationView = 'forward' | 'backward' | 'calculation';

// ----------------------------------------------------------------
// Prediction confidence display
// ----------------------------------------------------------------
const PredictionSummary = ({
  dataset,
  network,
  sampleIndex,
  originalData,
}: {
  dataset: string;
  network: import("@/components/network/static/types").NetworkState | null;
  sampleIndex: number;
  originalData: number[][];
}) => {
  if (!network) return null;
  const outputLayer = network.layers[network.layers.length - 2];
  const prediction = outputLayer?.A?.[sampleIndex];
  const sample = originalData[sampleIndex];
  if (!prediction || prediction.length === 0 || !sample) return null;

  if (dataset === "iris") {
    const classNames = ["Setosa", "Versicolor", "Virginica"];
    const colors = ["bg-blue-500", "bg-green-500", "bg-purple-500"];
    const actualRaw = sample.slice(-3);
    const actualIdx = actualRaw.findIndex((v) => v === 1);
    const predIdx = prediction.indexOf(Math.max(...prediction));
    const correct = actualIdx === predIdx;

    return (
      <div className="w-full bg-white border border-gray-200 rounded-lg p-3 shadow-sm h-full">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Prediction</p>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
            {correct ? "✓ Correct" : "✗ Wrong"}
          </span>
        </div>

        {/* Predicted / Actual summary when wrong */}
        {!correct ? (
          <div className="flex gap-2 mb-2.5 text-xs">
            <div className="flex-1 rounded bg-gray-50 border border-gray-200 px-2 py-1">
              <p className="text-gray-400 mb-0.5">Predicted</p>
              <p className="font-semibold text-gray-900">{classNames[predIdx]}</p>
            </div>
            <div className="flex-1 rounded bg-gray-50 border border-gray-200 px-2 py-1">
              <p className="text-gray-400 mb-0.5">Actual</p>
              <p className="font-semibold text-gray-700">{classNames[actualIdx] ?? "?"}</p>
            </div>
          </div>
        ) : (
          <p className="text-xs text-green-700 font-medium mb-2">
            {classNames[predIdx]} — {(prediction[predIdx] * 100).toFixed(1)}% confidence
          </p>
        )}

        <div className="space-y-1.5">
          {classNames.map((name, i) => {
            const isPred = i === predIdx;
            const isActual = i === actualIdx;
            const opacity = isPred ? "opacity-100" : isActual && !correct ? "opacity-60" : "opacity-20";
            return (
              <div key={name} className="flex items-center gap-2">
                <span className={`text-xs w-20 text-right ${isPred || isActual ? "font-semibold text-gray-800" : "text-gray-400"}`}>
                  {name}
                </span>
                <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${colors[i]} ${opacity}`}
                    style={{ width: `${(prediction[i] * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className={`text-xs w-10 font-mono ${isPred ? "font-bold text-gray-900" : "text-gray-400"}`}>
                  {(prediction[i] * 100).toFixed(1)}%
                </span>
                <span className="w-10 text-xs">
                  {isPred && isActual && <span className="text-green-600 font-medium">✓</span>}
                  {isPred && !isActual && <span className="text-blue-500">pred</span>}
                  {!isPred && isActual && <span className="text-orange-500">actual</span>}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  if (dataset === "auto_mpg") {
    const actualMPG = sample[sample.length - 1];
    const predMPG = prediction[0];
    const error = Math.abs(predMPG - actualMPG);
    return (
      <div className="w-full bg-white border border-gray-200 rounded-lg p-3 shadow-sm h-full">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Prediction</p>
        <div className="flex items-center justify-around text-center">
          <div>
            <p className="text-xl font-bold text-gray-900">{predMPG.toFixed(1)}</p>
            <p className="text-xs text-gray-500">Predicted MPG</p>
          </div>
          <div className="h-10 w-px bg-gray-200" />
          <div>
            <p className="text-xl font-bold text-gray-600">{actualMPG.toFixed(1)}</p>
            <p className="text-xs text-gray-500">Actual MPG</p>
          </div>
          <div className="h-10 w-px bg-gray-200" />
          <div>
            <p className={`text-xl font-bold ${error < 2 ? "text-green-600" : error < 5 ? "text-yellow-600" : "text-red-600"}`}>
              {error.toFixed(1)}
            </p>
            <p className="text-xs text-gray-500">Error (MPG)</p>
          </div>
        </div>
      </div>
    );
  }

  if (dataset === "xor") {
    const prob = prediction[0];
    const predLabel = prob >= 0.5 ? 1 : 0;
    const actualLabel = sample[sample.length - 1];
    const correct = predLabel === actualLabel;
    return (
      <div className="w-full bg-white border border-gray-200 rounded-lg p-3 shadow-sm h-full">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Prediction</p>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
            {correct ? "✓ Correct" : "✗ Wrong"}
          </span>
        </div>
        <div className="space-y-1.5">
          {[0, 1].map((label) => (
            <div key={label} className="flex items-center gap-2">
              <span className={`text-xs w-6 text-right font-medium ${predLabel === label ? "text-gray-900" : "text-gray-400"}`}>{label}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-3 rounded-full transition-all duration-300 ${label === 1 ? "bg-blue-500" : "bg-gray-400"} ${predLabel === label ? "opacity-100" : "opacity-30"}`}
                  style={{ width: `${((label === 1 ? prob : 1 - prob) * 100).toFixed(1)}%` }}
                />
              </div>
              <span className={`text-xs w-10 font-mono ${predLabel === label ? "font-bold text-gray-900" : "text-gray-400"}`}>
                {((label === 1 ? prob : 1 - prob) * 100).toFixed(1)}%
              </span>
              {actualLabel === label && <span className="text-xs text-gray-400 italic">← actual</span>}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400 mt-2">
          A={sample[0]}, B={sample[1]} → XOR={actualLabel}
        </p>
      </div>
    );
  }

  return null;
};

// ----------------------------------------------------------------
// Main Explain component
// ----------------------------------------------------------------
const Explain = () => {
    const {
      network,
      getExplanation,
      dataset,
      losses,
      accuracies,
      learningRate,
      sampleIndex,
      originalData,
      name,
      setStepLayerHighlight,
      sessionId,
      hoveredConnection,
      hoveredNode,
      setWeight,
    } = useStore();

    const [view, setView] = useState<PropagationView>('forward');
    const [fontSize, setFontSize] = useState("md");

    // Step-by-step forward pass mode
    const [stepMode, setStepMode] = useState(false);
    const [stepIndex, setStepIndex] = useState(0);

    // Weight editing state (for connection panel)
    const [editingWeight, setEditingWeight] = useState(false);
    const [weightInput, setWeightInput] = useState("");
    const weightInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
      setEditingWeight(false);
      if (hoveredConnection) setWeightInput(hoveredConnection.weight.toFixed(4));
    }, [hoveredConnection]);

    useEffect(() => {
      if (editingWeight) weightInputRef.current?.focus();
    }, [editingWeight]);

    const handleWeightSubmit = async () => {
      if (!hoveredConnection) return;
      const val = parseFloat(weightInput);
      if (isNaN(val)) return;
      await setWeight(hoveredConnection.layerIndex, hoveredConnection.fromIndex, hoveredConnection.toIndex, val);
      setEditingWeight(false);
    };

    const layerComputationCount = network
      ? network.layers.filter((_, i) => i < network.layers.length - 1).length
      : 0;

    useEffect(() => {
      setFontSize(window.innerWidth < 1000 ? "sm" : "md");
    }, []);

    // Sync step highlight to store so SVG can dim non-active layers
    useEffect(() => {
      if (stepMode && network) {
        setStepLayerHighlight(stepIndex);
      } else {
        setStepLayerHighlight(null);
      }
    }, [stepMode, stepIndex, network, setStepLayerHighlight]);

    // Exit step mode when switching views
    useEffect(() => {
      if (view !== 'forward') {
        setStepMode(false);
      }
    }, [view]);

    const enterStepMode = () => {
      setStepMode(true);
      setStepIndex(0);
    };

    const exitStepMode = () => {
      setStepMode(false);
      setStepLayerHighlight(null);
    };

    const pointSettings = { pointRadius: 0, pointHoverRadius: 4, pointHitRadius: 12 };

    const lossData = {
        labels: losses.map((_, i) => i + 1),
        datasets: [
            {
                label: `η=${learningRate}`,
                data: losses,
                borderColor: 'rgb(99, 102, 241)',
                backgroundColor: 'rgba(99, 102, 241, 0.08)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                ...pointSettings,
            },
        ]
    };

    const accuracyData = {
        labels: accuracies.map((_, i) => i + 1),
        datasets: [{
            label: name === "accuracy" ? 'Accuracy (%)' : 'MAE',
            data: accuracies,
            borderColor: 'rgb(16, 185, 129)',
            backgroundColor: 'rgba(16, 185, 129, 0.08)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            ...pointSettings,
        }]
    };

    const baseChartOptions = useMemo(() => {
        if (typeof window === "undefined") return {};
        const sm = window.innerWidth < 640;
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 } as const,
            plugins: {
                legend: {
                    position: "top" as const,
                    labels: { font: { size: sm ? 10 : 11 }, boxWidth: 12, padding: 12 },
                },
                tooltip: {
                    callbacks: {
                        title: (items: { label: string }[]) => `Epoch ${items[0]?.label}`,
                    },
                },
            },
            scales: {
                x: {
                    display: true,
                    title: { display: !sm, text: "Epoch", font: { size: 10 }, color: '#9ca3af' },
                    ticks: {
                        font: { size: sm ? 9 : 10 },
                        maxTicksLimit: 8,
                        color: '#9ca3af',
                    },
                    grid: { color: 'rgba(0,0,0,0.04)' },
                },
                y: {
                    ticks: { font: { size: sm ? 9 : 10 }, color: '#9ca3af' },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                },
            },
        };
    }, []);

    const lossChartOptions = useMemo(() => ({
        ...baseChartOptions,
        scales: {
            ...baseChartOptions.scales,
            y: {
                ...baseChartOptions.scales?.y,
                beginAtZero: false,
            },
        },
    }), [baseChartOptions]);

    const metricChartOptions = useMemo(() => ({
        ...baseChartOptions,
        scales: {
            ...baseChartOptions.scales,
            y: {
                ...baseChartOptions.scales?.y,
                beginAtZero: name === "accuracy",
                ...(name === "accuracy" ? {
                    max: 100,
                    ticks: {
                        ...baseChartOptions.scales?.y?.ticks,
                        callback: (v: number | string) => `${v}%`,
                    },
                } : {}),
            },
        },
    }), [baseChartOptions, name]);

    const renderMatrix = (matrix: number[][] | undefined | null, label: string, subLabel?: string, extendDecimal?: boolean) => (
        matrix && matrix.length > 0 ? (
            <div className="inline-block p-1 mx-1">
                <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="grid border border-gray-400 rounded bg-white shadow-sm overflow-x-auto" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, auto)` }}>
                    {matrix.map((row, rowIndex) => (
                        row.map((val, colIndex) => (
                            <span key={`${rowIndex}-${colIndex}`} className={`px-1 sm:px-1.5 py-0.5 text-xs sm:text-${fontSize}`}>
                                {val.toFixed(extendDecimal ? 3 : 2)}
                            </span>
                        ))
                    ))}
                </div>
            </div>
        ) : null
    );

    const renderVector = (vector: number[] | undefined | null, label: string, subLabel?: string, extendDecimal?: boolean, tooltip?: React.ReactNode) => (
        vector && vector.length > 0 ? (
            <div className="inline-block px-1 mx-1">
                <div className="flex flex-row items-center justify-center gap-1 relative group">
                    <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">{label}</p>
                    {tooltip as React.ReactNode}
                </div>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="flex flex-row justify-center border border-gray-400 px-1 py-0.5 rounded bg-white shadow-sm overflow-x-auto">
                    {vector.map((val, index) => (
                        <span key={index} className={`px-1 py-0.5 text-xs sm:text-${fontSize}`}>{val.toFixed(extendDecimal ? 3 : 2)}</span>
                    ))}
                </div>
            </div>
        ) : null
    );

    const fullExplanation = getExplanation() || '';
    const hasTrained = network?.layers && network.layers[0].A?.length > 0;

    return (
        <>
            {/* Connection panel + Prediction side by side */}
            <div className="flex flex-col sm:flex-row gap-3 mx-2 mt-4 mb-2">
                {/* Left: connection / node details */}
                <div className="flex-1 min-w-0 bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
                    {hoveredConnection ? (
                        <>
                            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Connection</p>
                            <p className="text-sm font-medium text-gray-800 mb-2">
                                Layer {hoveredConnection.layerIndex} → {hoveredConnection.layerIndex + 1}
                                <span className="text-gray-400 mx-1.5">·</span>
                                Neuron {hoveredConnection.fromIndex + 1} → {hoveredConnection.toIndex + 1}
                            </p>
                            <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-xs text-gray-500">Weight</span>
                                {!editingWeight ? (
                                    <>
                                        <span className="font-mono text-sm font-semibold text-gray-900">{hoveredConnection.weight.toFixed(4)}</span>
                                        {sessionId && (
                                            <button onClick={() => setEditingWeight(true)} className="text-xs border border-gray-300 rounded px-2 py-0.5 hover:bg-gray-50 text-gray-600">Edit</button>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <input
                                            ref={weightInputRef}
                                            type="number"
                                            value={weightInput}
                                            onChange={(e) => setWeightInput(e.target.value)}
                                            onKeyDown={(e) => { if (e.key === 'Enter') handleWeightSubmit(); if (e.key === 'Escape') setEditingWeight(false); }}
                                            step="0.01"
                                            className="w-24 text-sm border border-gray-400 rounded px-2 py-0.5 font-mono"
                                        />
                                        <button onClick={handleWeightSubmit} className="text-xs bg-black text-white rounded px-2 py-1">Set</button>
                                        <button onClick={() => setEditingWeight(false)} className="text-xs border border-gray-300 rounded px-2 py-1 hover:bg-gray-50">Cancel</button>
                                    </>
                                )}
                            </div>
                            {!sessionId && <p className="text-xs text-gray-400 italic mt-1">Initialize the model to edit weights.</p>}
                        </>
                    ) : hoveredNode && network ? (
                        <>
                            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Node</p>
                            {hoveredNode.layerIndex > 0 ? (
                                <div className="space-y-1.5">
                                    {network.layers[hoveredNode.layerIndex - 1]?.biases?.[hoveredNode.nodeIndex] !== undefined && (
                                        <div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Bias</span>
                                            <span className="font-mono font-medium text-gray-900">{network.layers[hoveredNode.layerIndex - 1].biases[hoveredNode.nodeIndex].toFixed(4)}</span>
                                        </div>
                                    )}
                                    {network.layers[hoveredNode.layerIndex - 1]?.Z?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                        <div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Pre-activation (Z)</span>
                                            <span className="font-mono font-medium text-gray-900">{network.layers[hoveredNode.layerIndex - 1].Z[sampleIndex][hoveredNode.nodeIndex].toFixed(4)}</span>
                                        </div>
                                    )}
                                    {network.layers[hoveredNode.layerIndex - 1]?.A?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                        <div className="flex justify-between text-sm">
                                            <span className="text-gray-500">Post-activation (A)</span>
                                            <span className="font-mono font-medium text-gray-900">{network.layers[hoveredNode.layerIndex - 1].A[sampleIndex][hoveredNode.nodeIndex].toFixed(4)}</span>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <p className="text-sm text-gray-500">Input node — values come from the dataset.</p>
                            )}
                        </>
                    ) : (
                        <div className="flex items-center justify-center h-full min-h-[60px]">
                            <p className="text-sm text-gray-400">Click a node or connection in the graph</p>
                        </div>
                    )}
                </div>

                {/* Right: prediction */}
                {hasTrained && (
                    <div className="flex-1 min-w-0">
                        <PredictionSummary
                            dataset={dataset}
                            network={network}
                            sampleIndex={sampleIndex}
                            originalData={originalData}
                        />
                    </div>
                )}
            </div>
        <div className="mt-2 p-4 bg-gray-100 rounded-lg mx-2 shadow-md">

            {hasTrained && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3">
                <div className="flex flex-col sm:flex-row rounded-md shadow-sm" role="group">
                    <button
                        type="button"
                        onClick={() => setView('forward')}
                        className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-t-lg sm:rounded-l-lg sm:rounded-t-none ${
                            view === 'forward' ? 'bg-black text-white' : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        1 · Forward Pass
                    </button>
                    <button
                        type="button"
                        onClick={() => setView('calculation')}
                        className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium ${
                            view === 'calculation' ? 'bg-black text-white' : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        2 · Compute Gradients
                    </button>
                    <button
                        type="button"
                        onClick={() => setView('backward')}
                        className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-b-lg sm:rounded-r-lg sm:rounded-b-none ${
                            view === 'backward' ? 'bg-black text-white' : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        3 · Update Weights
                    </button>
                </div>
                <Glossary />
            </div>)}

            {hasTrained && (
                <div className="text-center">
                    {view === 'forward' ? (
                        <>

                            <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                                <div>
                                    <p className="text-base sm:text-lg font-bold">Layer-by-layer computation</p>
                                    <p className="text-xs sm:text-sm text-gray-600">Each neuron: input × weight + bias → activation function → output</p>
                                </div>
                                {/* Step mode controls */}
                                {!stepMode ? (
                                    <button
                                        onClick={enterStepMode}
                                        className="text-xs border border-gray-300 rounded-md px-3 py-1.5 bg-white hover:bg-gray-50 text-gray-700"
                                    >
                                        Step Through ▶
                                    </button>
                                ) : (
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setStepIndex(Math.max(0, stepIndex - 1))}
                                            disabled={stepIndex === 0}
                                            className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white hover:bg-gray-50 disabled:opacity-40"
                                        >
                                            ← Prev
                                        </button>
                                        <span className="text-xs font-medium text-gray-600">
                                            Layer {stepIndex + 1} of {layerComputationCount}
                                        </span>
                                        <button
                                            onClick={() => setStepIndex(Math.min(layerComputationCount - 1, stepIndex + 1))}
                                            disabled={stepIndex === layerComputationCount - 1}
                                            className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white hover:bg-gray-50 disabled:opacity-40"
                                        >
                                            Next →
                                        </button>
                                        <button
                                            onClick={exitStepMode}
                                            className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white hover:bg-gray-50 text-gray-500"
                                        >
                                            Show All
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div className="flex flex-col justify-center items-center flex-wrap gap-2 sm:gap-4">
                                {network?.layers.map((layer: NeuronLayer, layerIndex: number) => {
                                    if (layerIndex >= network.layers.length - 1) return null;
                                    // In step mode, only show the active layer
                                    if (stepMode && layerIndex !== stepIndex) return null;

                                    const isOutputComputation = layerIndex === network.layers.length - 2;
                                    const layerLabel = isOutputComputation
                                      ? "Output Layer Computation"
                                      : `Layer ${layerIndex + 1} Computation`;

                                    // Plain-English header per layer
                                    const inputDesc = layerIndex === 0
                                      ? (dataset === "xor" ? "the two binary inputs (A, B)"
                                        : dataset === "iris" ? "the four flower measurements"
                                        : "the four car features")
                                      : "the previous layer's activations";
                                    const outputDesc = isOutputComputation
                                      ? (dataset === "iris" ? `a ${layer.activation} probability over 3 classes`
                                        : dataset === "xor" ? `a ${layer.activation} probability (0–1) for the XOR output`
                                        : `a linear value representing predicted MPG`)
                                      : `${layer.activation} activations passed to the next layer`;
                                    const layerHint = `Takes ${inputDesc}, multiplies by weights, adds biases, then applies ${layer.activation} → ${outputDesc}.`;

                                    return (
                                        <div key={layerIndex} className="flex flex-col items-center border-t border-gray-300 pt-4 w-full">
                                            <h4 className="text-md font-semibold mb-1 text-gray-700">{layerLabel}</h4>
                                            <p className="text-xs text-gray-500 italic mb-3 max-w-lg">{layerHint}</p>

                                            <div className="flex flex-row justify-center items-center flex-wrap gap-1 sm:gap-2">
                                                <div className="flex flex-col items-center">
                                                    {renderVector(
                                                        layerIndex === 0
                                                            ? network?.input[sampleIndex]
                                                            : network?.layers[layerIndex - 1].A[sampleIndex],
                                                        (layerIndex === 0 ? "Input Features" : `Layer ${layerIndex} Activations`),
                                                        layerIndex === 0 ? "Original input data" : "Output from previous layer",
                                                        false,
                                                        layerIndex === 0
                                                            ? <InputInfo dataset={dataset} input={network?.input[sampleIndex]} originalInput={originalData[sampleIndex]} />
                                                            : null
                                                    )}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">×</span>

                                                <div className="flex flex-col items-center">
                                                    {renderMatrix(layer.prevWeights, `Weights`, "Connection strengths between neurons")}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">+</span>

                                                <div className="flex flex-col items-center">
                                                    {renderVector(layer.prevBias, `Biases`)}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">=</span>

                                                <div className="flex flex-col items-center">
                                                    <div className="flex flex-col items-center">
                                                        {layer.activation !== "linear" && (<>
                                                            {renderVector(
                                                                layer.Z[sampleIndex],
                                                                isOutputComputation ? "Raw Output" : `Pre-activations`,
                                                            )}
                                                            <div className="relative flex flex-row justify-center items-center py-4">
                                                                <span className="absolute left-1/2 transform -translate-x-1/2 text-lg">↓</span>
                                                                <p className="absolute left-1/2 transform translate-x-2 text-sm text-gray-500 whitespace-nowrap">
                                                                    {layer.activation}
                                                                </p>
                                                            </div>
                                                        </>)}
                                                        {renderVector(
                                                            layer.A[sampleIndex],
                                                            isOutputComputation ? "Final Output" : `Activations`,
                                                            "",
                                                            isOutputComputation ? true : false,
                                                            isOutputComputation
                                                                ? <OutputInfo dataset={dataset} output={layer.A[sampleIndex]} actual={originalData[sampleIndex]} />
                                                                : null
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </>
                    ) : view === 'backward' ? (
                        <>
                            <p className="mt-4 sm:mt-6 mb-1 text-base sm:text-lg font-bold">Updating weight and bias values</p>
                            <p className="text-xs sm:text-sm text-gray-600 mb-1">Adjusting parameters to minimize loss</p>
                            <p className="text-xs text-gray-500 italic mb-3">
                                Each weight moves a small step opposite to its gradient. The step size is controlled by the learning rate η = {learningRate}.
                            </p>
                            <p className="text-xs sm:text-sm text-gray-600 mb-2">
                                <strong className="text-lg sm:text-xl text-gray-600">η</strong>
                                {` represents learning rate (`}
                                <strong className="text-sm sm:text-l text-gray-600">{learningRate}</strong>
                                {`)`}
                            </p>
                            <div className="flex flex-col justify-center items-center flex-wrap gap-2 sm:gap-4">
                            {network?.layers.slice().reverse().map((layer: NeuronLayer, reversedIndex: number) => {
                                const layerIndex = network.layers.length - reversedIndex;
                                if (reversedIndex === 0) return null;

                                return (
                                    <div key={`layer-${reversedIndex}-backprop`} className="flex flex-col gap-4 items-center border-t border-gray-300 pt-4">
                                        {layerIndex === network.layers.length - 1
                                            ? <h2 className="text-md font-semibold text-gray-700">Output Layer</h2>
                                            : <h2 className="text-md font-semibold text-gray-700">Layer {layerIndex}</h2>}

                                        {/* Weights Equation */}
                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                            {renderMatrix(layer.prevWeights, `Previous Weights`, "", true)}
                                            <span className="text-lg sm:text-xxl mt-2 sm:mt-8">-</span>
                                            <span className="text-xs sm:text-sm mt-2 sm:mt-8 text-gray-600"><strong className="text-lg sm:text-xl text-gray-600">η</strong> ×</span>
                                            {renderMatrix(layer.dW, `Change in Weights`, "dW (∇Weights)", true)}
                                            <span className="text-lg sm:text-xl mt-2 sm:mt-8">=</span>
                                            {renderMatrix(layer.weights, `Current Weights`, "", true)}
                                        </div>

                                        {/* Biases Equation */}
                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                            {renderVector(layer.prevBias, `Previous Biases`, "", true)}
                                            <span className="text-lg sm:text-xxl mt-2 sm:mt-8">-</span>
                                            <span className="text-xs sm:text-sm mt-2 sm:mt-8 text-gray-600"><strong className="text-lg sm:text-xl text-gray-600">η</strong> ×</span>
                                            {renderVector(layer.db, `Change in Biases`, "dB (∇Biases)", true)}
                                            <span className="text-lg sm:text-xl mt-2 sm:mt-8">=</span>
                                            {renderVector(layer.biases, `Current Biases`, "", true)}
                                        </div>
                                    </div>
                                );
                            })}
                            </div>
                        </>
                    ) : (
                        <div className="flex flex-col justify-center items-center flex-wrap">
                            <p className="mt-4 sm:mt-6 mb-1 text-base sm:text-lg font-bold">Propagating error backwards through model</p>
                            <p className="text-xs sm:text-sm text-gray-600 mb-1">Calculating how each neuron contributed to the error</p>
                            <p className="text-xs text-gray-500 italic mb-3">
                                The gradient tells us: if we nudge this weight slightly, how much does the loss change? We use the chain rule to compute this efficiently layer by layer.
                            </p>
                            <div className="flex flex-col justify-center items-center flex-wrap gap-2 sm:gap-4">
                            {network?.layers.slice().reverse().map((layer: NeuronLayer, index: number) => {
                                const initialContent = index === 0;
                                const outputLayerIndex = network.layers.length - 2;
                                const actual = originalData[sampleIndex].slice(dataset === "iris" ? -3 : -1);
                                const dA = multiplyMatrices([layer.dZ[sampleIndex]], transpose(network.layers[outputLayerIndex - index + 1]?.weights));

                                return (
                                    <div key={`layer-${index}-update`} className="flex flex-col gap-2 items-center border-t border-gray-300 pt-6">
                                        <h2 className="text-md font-semibold text-gray-700">
                                            {initialContent
                                                ? "Error from Result"
                                                : `Backpropagation — ${
                                                    1 === index
                                                        ? `Output Layer → Layer ${outputLayerIndex - index + 1}`
                                                        : `Layer ${outputLayerIndex - index + 2} → ${(outputLayerIndex - index + 1) === 0 ? "Input" : ""} Layer ${(outputLayerIndex - index + 1) === 0 ? "" : outputLayerIndex - index + 1}`
                                                    }`}
                                        </h2>

                                        {initialContent ? (
                                            <div className="flex flex-col items-center gap-3">
                                                <p className="text-sm text-gray-600">Calculate initial error (dZ) at the output:</p>
                                                <p className="text-xs text-gray-600 italic">*Note: this is done with all samples at once, unlike the single sample shown here*</p>
                                                <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                    {renderVector(network.layers[outputLayerIndex].A[sampleIndex], `Prediction`, "", true)}
                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">-</span>
                                                    {renderVector(actual, `Actual`, "", false)}
                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                    {renderVector(network.layers[outputLayerIndex].dZ[sampleIndex], `Error`, "", true)}
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="flex flex-col gap-4 items-center">
                                                <div className="flex flex-col items-center gap-2">
                                                    {(outputLayerIndex - index + 1) !== 0 ? (
                                                        <>
                                                            <p className="text-sm text-gray-600">
                                                                To propagate the error backward, compute <code>dZ</code> by taking the dot product of the forward layer&apos;s <code>dZ</code> and transposed weights, then apply the activation derivative.
                                                            </p>
                                                            <p className="text-xs text-gray-600 italic">*Note: this is done with all samples at once, unlike the single sample shown here*</p>
                                                        </>
                                                    ) : (
                                                        <p className="text-sm text-gray-600 italic">No backward calculation needed here — this is the input layer</p>
                                                    )}
                                                    <div className="flex flex-col items-center gap-2 flex-wrap justify-center">
                                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                            {renderVector(layer.dZ[sampleIndex], `dZ`, "Calculated above", true)}
                                                            {(outputLayerIndex - index + 1) !== 0 ? <>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                {renderMatrix(transpose(network.layers[outputLayerIndex - index + 1]?.prevWeights), `Wᵀ`, "Transposed weights from backwards layer", true)}
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                {renderMatrix(dA, `dA`, "∇ Activations", true)}
                                                            </> : null}
                                                        </div>
                                                        {(outputLayerIndex - index + 1) !== 0 ? (
                                                            <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                {renderMatrix(dA, `dA`, "∇ Activations", true)}
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                <div className="inline-block px-1 mx-1 text-center">
                                                                    <p className="text-xs sm:text-sm text-gray-600 mb-1">{"Activation Derivative (σ′(Z))"}</p>
                                                                    <p className="text-xs text-gray-500 mb-1">Derivative of the activation function at this layer</p>
                                                                    <div className="border border-gray-400 rounded px-2 py-1 bg-white shadow-sm">
                                                                        <span className="text-sm sm:text-l">{`σ′(Z) `}</span>
                                                                        <span className="text-sm sm:text-l text-gray-700 italic">{`(${network.layers[outputLayerIndex - index].activation})`}</span>
                                                                    </div>
                                                                </div>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                {renderVector(network.layers[outputLayerIndex - index]?.dZ[sampleIndex], `dZ`, "Error to pass to backwards layer", true)}
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                </div>
                                                <h2 className="text-md font-semibold mt-6 text-gray-700">Calculating change in weight and biases</h2>

                                                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-12 flex-wrap">
                                                    <div className="flex flex-col items-center gap-2">
                                                        {dataset === "xor" ? (
                                                            <>
                                                                <p className="text-sm text-gray-600">dW = (1/4) × A<sub>prev</sub><sup>T</sup> · dZ — all 4 XOR patterns:</p>
                                                                <p className="text-xs text-gray-600 italic">*Gradients from Y=0 patterns and Y=1 patterns partially cancel, giving small but non-zero dW*</p>
                                                                <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                    {renderMatrix(
                                                                        transpose(outputLayerIndex - index === -1 ? network.input : network.layers[outputLayerIndex - index].A) ?? [],
                                                                        `A_prev.T`, "neurons × samples", true
                                                                    )}
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                    {renderMatrix(layer.dZ, `dZ`, "samples × outputs", true)}
                                                                    <span className="text-sm sm:text-base mt-2 sm:mt-6 text-gray-600">× (1/4) =</span>
                                                                    {renderMatrix(layer.dW, `dW`, "∇ Weights", true)}
                                                                </div>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <p className="text-sm text-gray-600">dW = (1/m) &Sigma; A<sub>prev</sub><sup>T</sup> &middot; dZ averaged over all samples:</p>
                                                                <p className="text-xs text-gray-600 italic">*One sample shown for illustration — the actual dW is the average across all training samples*</p>
                                                                <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                    {renderMatrix(reshapeTo2D(outputLayerIndex - index === -1 ? network.input[sampleIndex] : network.layers[outputLayerIndex - index].A[sampleIndex]), `Sample A_prev`, "one sample, for reference", false)}
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                    {renderVector(layer.dZ[sampleIndex], `dZ`, "one sample", false)}
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">≠</span>
                                                                    {renderMatrix(layer.dW, `dW (batch avg)`, "∇ Weights", true)}
                                                                </div>
                                                            </>
                                                        )}
                                                    </div>

                                                    <div className="flex flex-col items-center gap-2">
                                                        {dataset === "xor" ? (
                                                            <>
                                                                <p className="text-sm text-gray-600">Gradient of biases (dB) = (1/4) Σ dZ across all 4 patterns:</p>
                                                                <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                    <span className="text-sm sm:text-base mt-2 sm:mt-6 text-gray-600">(1/4)</span>
                                                                    {renderMatrix(layer.dZ, `dZ`, "all samples", true)}
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                    {renderVector(layer.db, `dB`, "∇ Biases", true)}
                                                                </div>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <p className="text-sm text-gray-600">Gradient of biases (dB) is the sum of dZ across samples:</p>
                                                                <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">Σ</span>
                                                                    <div className="flex flex-col gap-1">
                                                                        {renderVector(layer.dZ[sampleIndex], `dZ`, "Different sample's dZ shown", false)}
                                                                        {renderVector(layer.dZ[sampleIndex + 1], ``, "", false)}
                                                                        ⋮
                                                                        {renderVector(layer.dZ[sampleIndex + 2], ``, "", false)}
                                                                    </div>
                                                                    <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                    {renderVector(layer.db, `dB`, "∇ Biases", true)}
                                                                </div>
                                                            </>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Loss and Accuracy Charts */}
            <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-3 sm:gap-4 lg:gap-6">
                <div className="bg-white p-3 sm:p-4 rounded-lg border border-gray-200">
                    <div className="flex items-baseline gap-2 mb-0.5">
                        <h3 className="text-sm font-semibold text-gray-800">Training Loss</h3>
                        {losses.length > 0 && (
                            <span className="text-xs font-mono text-gray-500">{losses[losses.length - 1].toFixed(4)}</span>
                        )}
                    </div>
                    <p className="text-xs text-gray-400 mt-0.5 mb-3">Decreasing = network is learning</p>
                    <div className="h-40 sm:h-48 lg:h-56">
                        <Line data={lossData} options={lossChartOptions} />
                    </div>
                </div>
                <div className="bg-white p-3 sm:p-4 rounded-lg border border-gray-200">
                    <div className="flex items-baseline gap-2 mb-0.5">
                        <h3 className="text-sm font-semibold text-gray-800">
                            {name === "accuracy" ? "Accuracy" : "Mean Absolute Error"}
                        </h3>
                        {accuracies.length > 0 && (
                            <span className="text-xs font-mono text-gray-500">
                                {name === "accuracy"
                                    ? `${accuracies[accuracies.length - 1].toFixed(1)}%`
                                    : accuracies[accuracies.length - 1].toFixed(3)}
                            </span>
                        )}
                    </div>
                    <p className="text-xs text-gray-400 mb-3">
                        {name === "accuracy" ? "Higher = more correct classifications" : "Lower = predictions closer to actual values"}
                    </p>
                    <div className="h-40 sm:h-48 lg:h-56">
                        <Line data={accuracyData} options={metricChartOptions} />
                    </div>
                </div>
            </div>

            {/* Explanation panel */}
            <div className="mt-4 sm:mt-6 bg-white border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
                <h3 className="font-semibold text-gray-900 mb-3">About this training run</h3>
                <ReactMarkdown
                    rehypePlugins={[rehypeRaw]}
                    components={{
                        a: ({...props }) => (
                            <a {...props} className="underline font-medium hover:text-blue-600" target="_blank" rel="noopener noreferrer" />
                        ),
                        p: ({...props}) => <p {...props} className="mb-2 leading-relaxed" />,
                        ul: ({...props }) => <ul {...props} className="list-disc pl-5 space-y-0.5 mb-2" />,
                        li: ({...props }) => <li {...props} />,
                        strong: ({...props}) => <strong {...props} className="font-semibold text-gray-900" />,
                    }}
                >
                    {fullExplanation}
                </ReactMarkdown>
            </div>
        </div>
        </>
    );
};

export default Explain;
