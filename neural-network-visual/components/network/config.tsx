"use client"
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import InfoPopup from "@/components/network/popup";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import useStore from "@/components/network/lib/store";
import { DATASETS, ACTIVATION_FUNCTIONS, DATASET_INPUT_FEATURES } from "@/components/network/static/constants";
import { HIDDEN_LAYER_LEARN_MORE } from "@/components/network/static/explanation";
import { useEffect, useState } from "react";
import { ActivationInfoPopup } from "./activation";

// ─── Dataset metadata for Step 1 ──────────────────────────────────────────────
const DATASET_DETAILS: Record<string, {
  task: string;
  taskType: string;
  inputs: string[];
  output: string;
  loss: string;
  samples: string;
}> = {
  iris: {
    task: "Classification",
    taskType: "Which species of flower is this?",
    inputs: ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"],
    output: "Setosa, Versicolor, or Virginica",
    loss: "Cross-Entropy Loss",
    samples: "150 samples · 3 classes",
  },
  auto_mpg: {
    task: "Regression",
    taskType: "How fuel-efficient is this car?",
    inputs: ["Displacement (cu. in.)", "Horsepower (hp)", "Weight (lbs)", "Acceleration (s)"],
    output: "Miles per gallon — a continuous number",
    loss: "Mean Squared Error (MSE)",
    samples: "392 samples",
  },
  xor: {
    task: "Binary Classification",
    taskType: "Can a hidden layer learn a non-linear rule?",
    inputs: ["Input A (0 or 1)", "Input B (0 or 1)"],
    output: "1 if inputs differ, 0 if they match",
    loss: "Binary Cross-Entropy",
    samples: "4 patterns (the full XOR truth table)",
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
const getLRFeedback = (lr: number, dataset: string): { text: string; color: string } => {
  if (dataset === "xor") {
    // XOR uses sigmoid activations and only 4 samples — higher LRs are normal and expected
    if (lr < 0.05) return { text: "Very slow — XOR will take many cycles to converge.", color: "text-blue-600" };
    if (lr < 0.15) return { text: "Conservative — reliable but slow for XOR.", color: "text-green-600" };
    if (lr < 0.55) return { text: "Good range for XOR — should converge in 50–200 cycles.", color: "text-green-600" };
    if (lr < 0.8)  return { text: "Aggressive for XOR — may oscillate on some initializations.", color: "text-yellow-600" };
    return { text: "Very high — likely to overshoot for XOR.", color: "text-red-600" };
  }
  if (lr < 0.01) return { text: "Very slow — may take many epochs to converge.", color: "text-blue-600" };
  if (lr < 0.05) return { text: "Conservative — stable, reliable learning.", color: "text-green-600" };
  if (lr < 0.2)  return { text: "Moderate — good default starting point.", color: "text-green-600" };
  if (lr < 0.5)  return { text: "Aggressive — updates are large; watch for overshooting.", color: "text-yellow-600" };
  return { text: "Very high — high risk of divergence. Loss may increase.", color: "text-red-600" };
};

const countParams = (hiddenLayers: number[], dataset: string) => {
  const inputSize = dataset === "xor" ? 2 : 4;
  const outputSize = dataset === "iris" ? 3 : 1;
  const sizes = [inputSize, ...hiddenLayers, outputSize];
  let total = 0;
  for (let i = 0; i < sizes.length - 1; i++) {
    total += sizes[i] * sizes[i + 1] + sizes[i + 1];
  }
  return total;
};

// ─── Step 1: Dataset ──────────────────────────────────────────────────────────
const StepDataset = ({
  dataset,
  onDatasetChange,
  onNext,
}: {
  dataset: string;
  onDatasetChange: (d: string) => void;
  onNext: () => void;
}) => {
  const details = DATASET_DETAILS[dataset];
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Choose a dataset</h2>
        <p className="text-sm text-gray-500 mt-0.5">The dataset defines what problem the network learns to solve.</p>
      </div>

      {/* Dataset cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {DATASETS.map((ds) => {
          const d = DATASET_DETAILS[ds];
          return (
            <button
              key={ds}
              onClick={() => onDatasetChange(ds)}
              className={`text-left p-4 rounded-xl border-2 transition-all ${
                dataset === ds
                  ? "border-indigo-500 bg-indigo-500 text-white"
                  : "border-border bg-card hover:border-indigo-300 hover:bg-indigo-50/50 dark:hover:bg-indigo-950/30"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="font-semibold text-sm">{ds}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ml-auto ${
                  dataset === ds ? "bg-white/20 text-white" : "bg-muted text-muted-foreground"
                }`}>{d.task}</span>
              </div>
              <p className={`text-xs mt-1 ${dataset === ds ? "text-white/80" : "text-muted-foreground"}`}>{d.taskType}</p>
            </button>
          );
        })}
      </div>

      {/* Selected dataset info */}
      {details && (
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Inputs (features)</p>
              <ul className="space-y-0.5">
                {details.inputs.map((inp) => (
                  <li key={inp} className="flex items-center gap-1.5 text-gray-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" />
                    {inp}
                  </li>
                ))}
              </ul>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Output</p>
                <p className="text-gray-700">{details.output}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Loss function</p>
                <p className="text-gray-700">{details.loss}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Dataset size</p>
                <p className="text-gray-700">{details.samples}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-end">
        <Button onClick={onNext} className="px-6">
          Next: Configure Network →
        </Button>
      </div>
    </div>
  );
};

// ─── Step 2: Configure Network ────────────────────────────────────────────────
const StepConfigure = ({
  dataset,
  hiddenLayers,
  activations,
  onAddLayer,
  onRemoveLayer,
  onUpdateLayer,
  onUpdateActivation,
  onBack,
  onNext,
}: {
  dataset: string;
  hiddenLayers: number[];
  activations: string[];
  onAddLayer: () => void;
  onRemoveLayer: () => void;
  onUpdateLayer: (i: number, v: number) => void;
  onUpdateActivation: (i: number, v: string) => void;
  onBack: () => void;
  onNext: () => void;
}) => {
  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const inputSize = dataset === "xor" ? 2 : 4;
  const outputSize = dataset === "iris" ? 3 : 1;

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Design your network</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Hidden layers let the network learn non-linear patterns. More layers = more capacity (but also more risk of overfitting).
        </p>
      </div>

      {/* Architecture preview bar */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <div className="flex items-center gap-1">
          <span className="text-xs px-2.5 py-1 rounded-full bg-blue-100 text-blue-700 font-medium">Input ({inputSize})</span>
        </div>
        {hiddenLayers.map((size, i) => (
          <div key={i} className="flex items-center gap-1">
            <span className="text-gray-300 text-sm">→</span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-gray-100 text-gray-700 font-medium">
              Hidden {i + 1} ({size}, {activations[i]})
            </span>
          </div>
        ))}
        <span className="text-gray-300 text-sm">→</span>
        <span className="text-xs px-2.5 py-1 rounded-full bg-red-100 text-red-700 font-medium">
          Output ({outputSize})
        </span>
      </div>

      {/* Layer controls */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">Hidden Layers</span>
            <ActivationInfoPopup />
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onRemoveLayer}
              disabled={hiddenLayers.length <= 1}
              className="text-xs"
            >
              − Remove
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onAddLayer}
              disabled={hiddenLayers.length >= 3}
              className="text-xs"
            >
              + Add Layer
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          {/* Column headers */}
          <div className="grid grid-cols-[80px_1fr_1fr] gap-3 text-xs font-semibold text-gray-400 uppercase tracking-wide px-1">
            <span>Nodes</span>
            <span>Layer</span>
            <span>Activation</span>
          </div>
          {hiddenLayers.map((nodes, index) => (
            <div key={index} className="grid grid-cols-[80px_1fr_1fr] items-center gap-3">
              <Input
                type="number"
                value={nodes}
                onChange={(e) => onUpdateLayer(index, Math.max(1, Math.min(6, Number(e.target.value))))}
                min={1}
                max={6}
                className="text-center"
              />
              <span className="text-sm text-gray-600 font-medium">Layer {index + 1}</span>
              <Select value={activations[index]} onValueChange={(v) => onUpdateActivation(index, v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ACTIVATION_FUNCTIONS.map((af) => (
                    <SelectItem key={af} value={af}>{af}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
        </div>

        {/* Tip */}
        <div className="pt-1 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            <span className="font-medium">Tip:</span> Start with 1–2 hidden layers and 4–6 nodes. The network diagram on the right updates live as you change these.
          </p>
          <button onClick={() => setIsPopupOpen(true)} className="text-xs font-semibold text-gray-600 hover:text-black mt-1">
            Learn more about hidden layers →
          </button>
        </div>
      </div>

      {isPopupOpen && (
        <InfoPopup title="Hidden Layer Information" message={HIDDEN_LAYER_LEARN_MORE} onClose={() => setIsPopupOpen(false)} />
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>← Back</Button>
        <Button onClick={onNext} className="px-6">Next: Review & Initialize →</Button>
      </div>
    </div>
  );
};

// ─── Step 3: Review & Initialize ──────────────────────────────────────────────
const StepInitialize = ({
  dataset,
  hiddenLayers,
  activations,
  onBack,
  onInitialize,
}: {
  dataset: string;
  hiddenLayers: number[];
  activations: string[];
  onBack: () => void;
  onInitialize: () => void;
}) => {
  const inputSize = dataset === "xor" ? 2 : 4;
  const outputSize = dataset === "iris" ? 3 : 1;
  const totalParams = countParams(hiddenLayers, dataset);
  const allLayers = [
    { label: "Input Layer", size: inputSize, activation: null, color: "bg-blue-50 border-blue-200 text-blue-800" },
    ...hiddenLayers.map((size, i) => ({
      label: `Hidden Layer ${i + 1}`,
      size,
      activation: activations[i],
      color: "bg-gray-50 border-gray-200 text-gray-700",
    })),
    {
      label: "Output Layer",
      size: outputSize,
      activation: dataset === "iris" ? "softmax" : dataset === "xor" ? "sigmoid" : "linear",
      color: "bg-red-50 border-red-200 text-red-800",
    },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Review & initialize</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Initialization sets all weights and biases to small random values. Training will adjust them to fit the data.
        </p>
      </div>

      {/* Architecture summary */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-semibold text-gray-700">Network Architecture</p>
          <span className="text-xs text-gray-400">{totalParams} total parameters</span>
        </div>
        <div className="space-y-1.5">
          {allLayers.map((layer, i) => (
            <div key={i} className="flex flex-col items-center">
              <div className={`flex items-center justify-between w-full max-w-xs px-3 py-2 rounded-lg border text-sm ${layer.color}`}>
                <span className="font-medium">{layer.label}</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs">{layer.size} nodes</span>
                  {layer.activation && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-white/60 font-mono">{layer.activation}</span>
                  )}
                </div>
              </div>
              {i < allLayers.length - 1 && (
                <div className="flex flex-col items-center my-0.5">
                  <div className="w-px h-3 bg-gray-300" />
                  <span className="text-gray-400 text-xs leading-none">↓</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Dataset summary */}
      <div className="flex items-center gap-3 bg-white border border-gray-200 rounded-xl px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-gray-800">{dataset}</p>
          <p className="text-xs text-gray-500">{DATASET_DETAILS[dataset]?.taskType}</p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-xs text-gray-400">Loss function</p>
          <p className="text-xs font-medium text-gray-700">{DATASET_DETAILS[dataset]?.loss}</p>
        </div>
      </div>

      <div className="flex justify-between items-center">
        <Button variant="outline" onClick={onBack}>← Back</Button>
        <Button onClick={onInitialize} className="px-8 py-2 text-base font-semibold">
          Initialize Model ▶
        </Button>
      </div>
    </div>
  );
};

// ─── Step 4: Train ────────────────────────────────────────────────────────────
const SampleStory = ({
  sampleIndex,
  dataset,
  originalData,
  network,
}: {
  sampleIndex: number;
  dataset: string;
  originalData: number[][];
  network: import("@/components/network/static/types").NetworkState | null;
}) => {
  const sample = originalData[sampleIndex];
  if (!sample || !network) return null;
  const features = DATASET_INPUT_FEATURES[dataset] || [];
  const outputLayer = network.layers[network.layers.length - 2];
  const prediction = outputLayer?.A?.[sampleIndex];
  if (!prediction || prediction.length === 0) return null;

  if (dataset === "iris") {
    const classNames = ["Setosa", "Versicolor", "Virginica"];
    const actualRaw = sample.slice(-3);
    const actualIdx = actualRaw.findIndex((v) => v === 1);
    const predIdx = prediction.indexOf(Math.max(...prediction));
    const correct = actualIdx === predIdx;
    return (
      <div className="text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2">
        <span className="font-medium">Sample #{sampleIndex}: </span>
        {features.map((f, i) => `${f.split(" ")[0]} ${sample[i]?.toFixed(1)}`).join(" / ")}
        <span className="mx-1">→</span>
        <span className="font-semibold text-gray-800">
          {classNames[predIdx]} ({(prediction[predIdx] * 100).toFixed(0)}%)
        </span>
        <span className={`ml-1 font-bold ${correct ? "text-green-600" : "text-red-500"}`}>
          {correct ? "✓" : "✗"} Actual: {classNames[actualIdx] ?? "?"}
        </span>
      </div>
    );
  }

  if (dataset === "auto_mpg") {
    const actualMPG = sample[sample.length - 1];
    const predMPG = prediction[0];
    const err = Math.abs(predMPG - actualMPG).toFixed(1);
    return (
      <div className="text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2">
        <span className="font-medium">Sample #{sampleIndex + 1}: </span>
        {features.map((f, i) => `${f.split(" ")[0]} ${sample[i]?.toFixed(0)}`).join(" / ")}
        <span className="mx-1">→</span>
        <span className="font-semibold text-gray-800">Predicted: {predMPG.toFixed(1)} MPG</span>
        <span className="ml-1 text-gray-500">| Actual: {actualMPG.toFixed(1)} MPG | Error: {err} MPG</span>
      </div>
    );
  }

  if (dataset === "xor") {
    const prob = prediction[0];
    const predLabel = prob >= 0.5 ? 1 : 0;
    const actualLabel = sample[sample.length - 1];
    const correct = predLabel === actualLabel;
    return (
      <div className="text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2">
        <span className="font-medium">A={sample[0]}, B={sample[1]}</span>
        <span className="mx-1">→</span>
        <span className="font-semibold text-gray-800">Predicted: {predLabel} ({(prob * 100).toFixed(0)}%)</span>
        <span className={`ml-1 font-bold ${correct ? "text-green-600" : "text-red-500"}`}>
          {correct ? "✓" : "✗"} Actual: {actualLabel}
        </span>
      </div>
    );
  }

  return null;
};

const StepTrain = ({
  dataset,
  hiddenLayers,
  activations,
  epoch,
  loss,
  metric,
  name,
  learningRate,
  sampleIndex,
  originalData,
  network,
  sessionId,
  onSetLR,
  onSetSample,
  onTrain,
  onChangeModel,
}: {
  dataset: string;
  hiddenLayers: number[];
  activations: string[];
  epoch: number;
  loss: number;
  metric: number;
  name: string;
  learningRate: number;
  sampleIndex: number;
  originalData: number[][];
  network: import("@/components/network/static/types").NetworkState | null;
  sessionId: string | null;
  onSetLR: (v: number) => void;
  onSetSample: (v: number) => void;
  onTrain: () => void;
  onChangeModel: () => void;
}) => {
  const lrFeedback = getLRFeedback(learningRate, dataset);
  const inputSize = dataset === "xor" ? 2 : 4;
  const outputSize = dataset === "iris" ? 3 : 1;

  return (
    <div className="space-y-4">
      {/* Compact config summary */}
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="px-2 py-1 rounded-full bg-black text-white font-medium">{dataset}</span>
        <span className="text-gray-300">|</span>
        <span className="text-gray-500">Input({inputSize})</span>
        {hiddenLayers.map((s, i) => (
          <span key={i} className="text-gray-500">→ H{i + 1}({s}, {activations[i]})</span>
        ))}
        <span className="text-gray-500">→ Output({outputSize})</span>
      </div>

      {/* Training metrics */}
      {epoch > 0 && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-gray-900">{epoch}</p>
            <p className="text-xs text-gray-500 mt-0.5">Epochs run</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-gray-900">{loss.toFixed(3)}</p>
            <p className="text-xs text-gray-500 mt-0.5">Loss</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-gray-900">
              {name === "accuracy" ? `${metric.toFixed(1)}%` : metric.toFixed(2)}
            </p>
            <p className="text-xs text-gray-500 mt-0.5">{name === "accuracy" ? "Accuracy" : "MAE"}</p>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-4">
        {/* Learning rate */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <Label className="text-sm font-semibold">Learning Rate (η)</Label>
            <span className="text-sm font-mono font-bold">{learningRate.toFixed(2)}</span>
          </div>
          <Slider
            value={[learningRate]}
            onValueChange={(v) => onSetLR(v[0])}
            max={1}
            step={0.01}
            className="w-full"
          />
          <p className={`text-xs mt-1.5 ${lrFeedback.color}`}>{lrFeedback.text}</p>
        </div>

        {/* Sample picker */}
        <div>
          <div className="flex items-center gap-3">
            <Label className="text-sm font-semibold whitespace-nowrap">Sample #</Label>
            <Input
              type="number"
              value={sampleIndex}
              onChange={(e) => onSetSample(Math.max(0, Math.min(25, Number(e.target.value))))}
              min={0}
              max={25}
              className="w-20 text-center"
            />
            <span className="text-xs text-gray-400">0 – 25 from the test set</span>
          </div>
          <div className="mt-2">
            <SampleStory
              sampleIndex={sampleIndex}
              dataset={dataset}
              originalData={originalData}
              network={network}
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <Button onClick={onTrain} disabled={!sessionId} className="w-full text-base py-5 font-semibold">
        ▶ Run Training Cycle
      </Button>

      <div className="flex justify-center">
        <button
          onClick={onChangeModel}
          className="text-xs text-gray-400 hover:text-gray-700 underline underline-offset-2"
        >
          ← Change model / dataset
        </button>
      </div>
    </div>
  );
};

// ─── Step indicator ───────────────────────────────────────────────────────────
const STEP_LABELS = ["Pick Dataset", "Configure Network", "Initialize", "Train"];

const StepIndicator = ({
  currentStep,
  onClickStep,
}: {
  currentStep: number;
  onClickStep: (step: number) => void;
}) => (
  <div className="flex items-center gap-0 w-full overflow-x-auto p-1">
    {STEP_LABELS.map((label, i) => {
      const step = i + 1;
      const done = step < currentStep;
      const active = step === currentStep;
      const clickable = done;
      return (
        <div key={step} className="flex items-center min-w-0">
          <button
            disabled={!clickable}
            onClick={() => clickable && onClickStep(step)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
              done
                ? "bg-black text-white hover:bg-gray-800 cursor-pointer"
                : active
                ? "bg-gray-800 text-white ring-2 ring-offset-1 ring-gray-400 cursor-default"
                : "bg-gray-100 text-gray-400 cursor-default"
            }`}
          >
            <span className={`w-4 h-4 rounded-full flex items-center justify-center text-xs flex-shrink-0 ${
              done ? "bg-white text-black" : active ? "bg-white/20" : "bg-gray-300"
            }`}>
              {done ? "✓" : step}
            </span>
            {label}
          </button>
          {i < STEP_LABELS.length - 1 && (
            <div className={`h-px w-5 mx-0.5 flex-shrink-0 ${step < currentStep ? "bg-black" : "bg-gray-300"}`} />
          )}
        </div>
      );
    })}
  </div>
);

// ─── Root Config ──────────────────────────────────────────────────────────────
const Config = () => {
  const {
    sessionId,
    hiddenLayers,
    learningRate,
    dataset,
    activations,
    epoch,
    loss,
    metric,
    name,
    network,
    originalData,
    sampleIndex,
    setLearningRate,
    initModel,
    clearSessionAndReset,
    runTrainingCycle,
    addHiddenLayer,
    removeHiddenLayer,
    updateHiddenLayer,
    updateActivation,
    handleDatasetChange,
    initModelFrontend,
    setSampleIndex,
  } = useStore();

  // Wizard step: 1=dataset, 2=configure, 3=initialize, 4=train
  const [wizardStep, setWizardStep] = useState(1);

  // Keep frontend network preview in sync with config
  useEffect(() => {
    initModelFrontend();
  }, [initModelFrontend]);

  // Advance to step 4 once the model is initialized (session exists)
  useEffect(() => {
    if (sessionId) setWizardStep(4);
  }, [sessionId]);

  const handleInitialize = async () => {
    await initModel();
    // setWizardStep(4) is handled by the sessionId effect above
  };

  const handleChangeModel = () => {
    clearSessionAndReset();
    setWizardStep(1);
  };

  return (
    <div className="mb-6 max-w-4xl mx-auto px-2 sm:px-4">
      {/* Step indicator */}
      <div className="mb-6">
        <StepIndicator currentStep={wizardStep} onClickStep={setWizardStep} />
      </div>

      {/* Step content */}
      <div className="bg-white border border-gray-200 rounded-2xl p-5 sm:p-6 shadow-sm">
        {wizardStep === 1 && (
          <StepDataset
            dataset={dataset}
            onDatasetChange={handleDatasetChange}
            onNext={() => setWizardStep(2)}
          />
        )}

        {wizardStep === 2 && (
          <StepConfigure
            dataset={dataset}
            hiddenLayers={hiddenLayers}
            activations={activations}
            onAddLayer={addHiddenLayer}
            onRemoveLayer={removeHiddenLayer}
            onUpdateLayer={updateHiddenLayer}
            onUpdateActivation={updateActivation}
            onBack={() => setWizardStep(1)}
            onNext={() => setWizardStep(3)}
          />
        )}

        {wizardStep === 3 && (
          <StepInitialize
            dataset={dataset}
            hiddenLayers={hiddenLayers}
            activations={activations}
            onBack={() => setWizardStep(2)}
            onInitialize={handleInitialize}
          />
        )}

        {wizardStep === 4 && (
          <StepTrain
            dataset={dataset}
            hiddenLayers={hiddenLayers}
            activations={activations}
            epoch={epoch}
            loss={loss}
            metric={metric}
            name={name}
            learningRate={learningRate}
            sampleIndex={sampleIndex}
            originalData={originalData}
            network={network}
            sessionId={sessionId}
            onSetLR={setLearningRate}
            onSetSample={setSampleIndex}
            onTrain={runTrainingCycle}
            onChangeModel={handleChangeModel}
          />
        )}
      </div>
    </div>
  );
};

export default Config;
