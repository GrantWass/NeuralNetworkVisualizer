"use client"
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import InfoPopup from "@/components/network/popup";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import useStore from "@/components/network/lib/store";
import { DATASETS, ACTIVATION_FUNCTIONS, DATASET_INPUT_FEATURES } from "@/components/network/static/constants";
import { HIDDEN_LAYER_LEARN_MORE } from "@/components/network/static/explanation";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Play } from "lucide-react";
import { ActivationInfoPopup } from "./activation";
import LeaderboardPanel from "@/components/network/leaderboard";
import { Trophy } from "lucide-react";

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
  mnist: {
    task: "Classification",
    taskType: "Which handwritten digit is this?",
    inputs: ["784 pixel values (28×28 grayscale image)"],
    output: "Digit class 0–9",
    loss: "Cross-Entropy Loss",
    samples: "~2,500 samples · 10 classes (25% stratified subset)",
  },
};

const getInputSize = (dataset: string) =>
  dataset === "xor" ? 2 : dataset === "mnist" ? 784 : 4;

const getOutputSize = (dataset: string) =>
  dataset === "iris" || dataset === "mnist" ? (dataset === "mnist" ? 10 : 3) : 1;

const getOutputActivation = (dataset: string) =>
  dataset === "iris" || dataset === "mnist" ? "softmax" : dataset === "xor" ? "sigmoid" : "linear";

const countParams = (hiddenLayers: number[], dataset: string) => {
  const sizes = [getInputSize(dataset), ...hiddenLayers, getOutputSize(dataset)];
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
      <div className="grid grid-cols-2 gap-3">
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
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  dataset === ds ? "bg-white/20 text-white" : "bg-muted text-muted-foreground"
                }`}>{d.task}</span>
                {ds === "mnist" && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                    dataset === ds ? "bg-white/25 text-white" : "bg-indigo-100 text-indigo-700"
                  }`}>✏ draw</span>
                )}
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
  const inputSize = getInputSize(dataset);
  const outputSize = getOutputSize(dataset);

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
                onChange={(e) => onUpdateLayer(index, Math.max(1, Math.min(dataset === "mnist" ? 128 : 6, Number(e.target.value))))}
                min={1}
                max={dataset === "mnist" ? 128 : 6}
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
  isInitializing,
  onBack,
  onInitialize,
}: {
  dataset: string;
  hiddenLayers: number[];
  activations: string[];
  isInitializing: boolean;
  onBack: () => void;
  onInitialize: () => void;
}) => {
  const inputSize = getInputSize(dataset);
  const outputSize = getOutputSize(dataset);
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
      activation: getOutputActivation(dataset),
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
        <Button variant="outline" onClick={onBack} disabled={isInitializing}>← Back</Button>
        <Button onClick={onInitialize} disabled={isInitializing} className="px-8 py-2 text-base font-semibold">
          {isInitializing ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              Initializing…
            </span>
          ) : <span className="flex items-center gap-1.5">Initialize Model <Play size={13} /></span>}
        </Button>
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
    trainingEpochs,
    dataset,
    activations,
    epoch,
    loss,
    metric,
    name,
    sampleIndex,
    setLearningRate,
    setTrainingEpochs,
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
    isInitializing,
    runModel,
    leaderboardOpen,
    setLeaderboardOpen,
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

      {/* Floating training widget — portalled to body, always fixed top-right once a session exists */}
      {sessionId && typeof document !== "undefined" && createPortal(
        <div style={{ position: "fixed", top: 56, right: 16, zIndex: 9999 }}
          className="bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden w-[320px]"
        >
          {/* Row 1: epochs · loss · metric · Train · Leaderboard */}
          <div className="flex items-stretch divide-x divide-gray-100">
            <div className="flex flex-col items-center justify-center px-3 py-2 flex-1">
              <span className="text-base font-bold text-gray-900 leading-none tabular-nums">{epoch > 0 ? epoch : "—"}</span>
              <span className="text-[9px] text-gray-400 mt-0.5">epochs</span>
            </div>
            <div className="flex flex-col items-center justify-center px-3 py-2 flex-1">
              <span className="text-base font-bold text-gray-900 leading-none tabular-nums">{epoch > 0 ? loss.toFixed(3) : "—"}</span>
              <span className="text-[9px] text-gray-400 mt-0.5">loss</span>
            </div>
            <div className="flex flex-col items-center justify-center px-3 py-2 flex-1">
              <span className="text-base font-bold text-gray-900 leading-none tabular-nums">
                {epoch > 0 ? (name === "accuracy" ? `${metric.toFixed(1)}%` : metric.toFixed(2)) : "—"}
              </span>
              <span className="text-[9px] text-gray-400 mt-0.5">{name === "accuracy" ? "accuracy" : "MAE"}</span>
            </div>
            <button
              onClick={runTrainingCycle}
              disabled={runModel}
              className="flex items-center justify-center gap-1 bg-gray-900 hover:bg-gray-700 disabled:opacity-50 text-white text-xs font-semibold w-[56px] shrink-0 transition-colors"
            >
              {runModel ? "…" : <><Play size={11} /> Train</>}
            </button>
            <button
              onClick={() => setLeaderboardOpen(true)}
              title="Leaderboard"
              className="flex items-center justify-center w-9 shrink-0 text-amber-400 hover:text-amber-500 transition-colors"
            >
              <Trophy size={13} />
            </button>
          </div>

          {/* Row 2: Learning rate */}
          <div className="flex items-center gap-3 px-4 py-2 border-t border-gray-100 bg-gray-50">
            <span className="text-[9px] text-gray-500 font-medium shrink-0 w-20">Learning rate</span>
            <Slider value={[learningRate]} onValueChange={(v) => setLearningRate(v[0])} max={1} step={0.01} className="flex-1" />
            <span className="text-[9px] font-mono text-gray-700 w-7 text-right shrink-0">{learningRate.toFixed(2)}</span>
          </div>

          {/* Row 3: Sample stepper · epochs per click */}
          <div className="flex items-center gap-3 px-4 py-2 border-t border-gray-100 bg-gray-50">
            <span className="text-[9px] text-gray-500 font-medium shrink-0">Sample</span>
            <button onClick={() => setSampleIndex(Math.max(0, sampleIndex - 1))} disabled={sampleIndex === 0} className="w-5 h-5 flex items-center justify-center rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-25 text-gray-700 text-xs font-bold leading-none shrink-0">‹</button>
            <span className="font-mono text-xs font-semibold text-gray-900 w-4 text-center shrink-0">{sampleIndex}</span>
            <button onClick={() => setSampleIndex(Math.min(dataset === "xor" ? 3 : 25, sampleIndex + 1))} disabled={sampleIndex === (dataset === "xor" ? 3 : 25)} className="w-5 h-5 flex items-center justify-center rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-25 text-gray-700 text-xs font-bold leading-none shrink-0">›</button>
            <div className="w-px h-3 bg-gray-200 shrink-0" />
            <span className="text-[9px] text-gray-500 font-medium shrink-0">Epochs / click</span>
            <input type="number" min={1} max={999} value={trainingEpochs} onChange={(e) => setTrainingEpochs(Math.max(1, Math.min(999, parseInt(e.target.value) || 1)))} className="w-12 text-[9px] border border-gray-200 rounded px-1.5 py-0.5 font-mono outline-none focus:ring-1 focus:ring-gray-300 text-center bg-white ml-auto" />
          </div>
        </div>,
        document.body
      )}

      {leaderboardOpen && <LeaderboardPanel />}

      {/* Step content */}
      {wizardStep < 4 && (
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
              isInitializing={isInitializing}
              onBack={() => setWizardStep(2)}
              onInitialize={handleInitialize}
            />
          )}
        </div>
      )}

      {wizardStep === 4 && (
        <div className="flex justify-center mt-1">
          <button
            onClick={handleChangeModel}
            className="text-xs text-gray-400 hover:text-gray-700 underline underline-offset-2"
          >
            ← Change model / dataset
          </button>
        </div>
      )}
    </div>
  );
};

export default Config;
