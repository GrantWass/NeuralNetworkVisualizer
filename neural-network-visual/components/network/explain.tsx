"use client";

import useStore from "@/components/network/lib/store";
import { NeuronLayer, NetworkState } from "@/components/network/static/types";
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
import DigitCanvas from "@/components/network/digit-canvas";
import { DecisionBoundary } from "@/components/network/decision-boundary";
import { SampleVisual } from "@/components/network/sample-visual";

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
// Value tracer — types, styles, and connection map
// ----------------------------------------------------------------
type ValueConnection = {
  view: PropagationView;
  description: string;
  relatedValueId?: string;
};

type ValueInfo = {
  label: string;
  type: string;
  connections: ValueConnection[];
};

const VALUE_TYPE_STYLES: Record<string, {
  primaryBorder: string;
  relatedBorder: string;
  bg: string;
  badge: string;
  text: string;
  dot: string;
}> = {
  W:     { primaryBorder: 'border-indigo-500', relatedBorder: 'border-indigo-300', bg: 'bg-indigo-50',  badge: 'bg-indigo-100 text-indigo-800', text: 'text-indigo-600', dot: 'bg-indigo-500' },
  A:     { primaryBorder: 'border-blue-500',   relatedBorder: 'border-blue-300',   bg: 'bg-blue-50',    badge: 'bg-blue-100 text-blue-800',    text: 'text-blue-600',   dot: 'bg-blue-500'   },
  Z:     { primaryBorder: 'border-teal-500',   relatedBorder: 'border-teal-300',   bg: 'bg-teal-50',    badge: 'bg-teal-100 text-teal-800',    text: 'text-teal-600',   dot: 'bg-teal-500'   },
  dZ:    { primaryBorder: 'border-amber-500',  relatedBorder: 'border-amber-300',  bg: 'bg-amber-50',   badge: 'bg-amber-100 text-amber-800',  text: 'text-amber-600',  dot: 'bg-amber-500'  },
  dW:    { primaryBorder: 'border-rose-500',   relatedBorder: 'border-rose-300',   bg: 'bg-rose-50',    badge: 'bg-rose-100 text-rose-800',    text: 'text-rose-600',   dot: 'bg-rose-500'   },
  dB:    { primaryBorder: 'border-pink-500',   relatedBorder: 'border-pink-300',   bg: 'bg-pink-50',    badge: 'bg-pink-100 text-pink-800',    text: 'text-pink-600',   dot: 'bg-pink-500'   },
  B:     { primaryBorder: 'border-violet-500', relatedBorder: 'border-violet-300', bg: 'bg-violet-50',  badge: 'bg-violet-100 text-violet-800', text: 'text-violet-600', dot: 'bg-violet-500' },
  input: { primaryBorder: 'border-green-500',  relatedBorder: 'border-green-300',  bg: 'bg-green-50',   badge: 'bg-green-100 text-green-800',  text: 'text-green-600',  dot: 'bg-green-500'  },
};

const VIEW_LABELS: Record<PropagationView, string> = {
  forward: '1 · Forward Pass',
  calculation: '2 · Compute Gradients',
  backward: '3 · Update Weights',
};

function buildValueMap(network: NetworkState): Map<string, ValueInfo> {
  const map = new Map<string, ValueInfo>();
  const N = network.layers.length;

  map.set('input', {
    label: 'Network input features',
    type: 'input',
    connections: [
      { view: 'forward', description: 'Raw features fed into the first layer', relatedValueId: 'Z:0' },
      { view: 'backward', description: 'A_prev for first layer gradient: dW = Input.T × dZ / m', relatedValueId: 'dW:0' },
    ],
  });

  for (let i = 0; i < N - 1; i++) {
    const isOutput = i === N - 2;
    const lbl = isOutput ? 'Output Layer' : i === 0 ? 'Layer 1' : `Layer ${i + 1}`;
    const act = network.layers[i].activation || 'activation';

    map.set(`W:${i}`, {
      label: `${lbl} weight matrix`,
      type: 'W',
      connections: [
        { view: 'forward', description: 'Multiplied with the previous layer\'s activations (A) to produce Z: A × W + b', relatedValueId: `Z:${i}` },
        { view: 'calculation', description: 'Transposed (Wᵀ) to propagate error backward', relatedValueId: `dZ:${i}` },
        { view: 'backward', description: 'Updated: W_new = W_prev − η · dW', relatedValueId: `dW:${i}` },
      ],
    });

    const aConns: ValueConnection[] = [
      { view: 'forward', description: `Output after applying ${act}: A = σ(Z)`, relatedValueId: `Z:${i}` },
    ];
    if (!isOutput) {
      aConns.push({ view: 'forward', description: 'Passed as input into the next layer', relatedValueId: `Z:${i + 1}` });
      aConns.push({ view: 'backward', description: 'A_prev for next layer weight gradient dW', relatedValueId: `dW:${i + 1}` });
    } else {
      aConns.push({ view: 'calculation', description: 'Used as Prediction in the output error: dZ = 2(Ŷ−Y) or Ŷ−Y', relatedValueId: `dZ:${i}` });
    }
    map.set(`A:${i}`, { label: `${lbl} activations`, type: 'A', connections: aConns });

    map.set(`Z:${i}`, {
      label: `${lbl} pre-activation (Z)`,
      type: 'Z',
      connections: [
        { view: 'forward', description: `Weighted sum of the previous layer's activations (A) × W + b, before the activation function. Passed through ${act} → A.`, relatedValueId: `A:${i}` },
        { view: 'calculation', description: 'Activation derivative σ′(Z) applied during backpropagation' },
      ],
    });

    const dzConns: ValueConnection[] = [];
    if (isOutput) {
      dzConns.push({ view: 'calculation', description: 'Initial output error: dZ = 2(Ŷ−Y) for MSE, or Ŷ−Y for cross-entropy', relatedValueId: `A:${i}` });
    } else {
      dzConns.push({ view: 'calculation', description: `Propagated backward: dZ = (dZ_next × Wᵀ) × σ′(Z)`, relatedValueId: `W:${i + 1}` });
    }
    dzConns.push({ view: 'backward', description: 'Used to compute weight gradient: dW = A_prev.T × dZ / m', relatedValueId: `dW:${i}` });
    dzConns.push({ view: 'backward', description: 'Used to compute bias gradient: dB = Σ dZ / m', relatedValueId: `dB:${i}` });
    map.set(`dZ:${i}`, { label: `${lbl} error gradient (dZ)`, type: 'dZ', connections: dzConns });

    map.set(`dW:${i}`, {
      label: `${lbl} weight gradient (dW)`,
      type: 'dW',
      connections: [
        { view: 'backward', description: 'Computed from A_prev.T × dZ / m', relatedValueId: `dZ:${i}` },
        { view: 'backward', description: 'Applied in weight update: W_new = W_prev − η · dW', relatedValueId: `W:${i}` },
      ],
    });

    map.set(`dB:${i}`, {
      label: `${lbl} bias gradient (dB)`,
      type: 'dB',
      connections: [
        { view: 'backward', description: 'Computed as Σ dZ / m across all samples', relatedValueId: `dZ:${i}` },
        { view: 'backward', description: 'Applied in bias update: b_new = b_prev − η · dB', relatedValueId: `B:${i}` },
      ],
    });

    map.set(`B:${i}`, {
      label: `${lbl} biases`,
      type: 'B',
      connections: [
        { view: 'forward', description: 'Added to the weighted sum (A × W + b) to produce Z', relatedValueId: `Z:${i}` },
        { view: 'backward', description: 'Previous value used in update equation: b_new = b_prev − η · dB', relatedValueId: `dB:${i}` },
      ],
    });

    map.set(`currentW:${i}`, {
      label: `${lbl} current weights (after update)`,
      type: 'W',
      connections: [
        { view: 'backward', description: 'Result of this step — W_new = W_prev − η · dW. These weights are now stored in the model and used in the next forward pass.' },
      ],
    });

    map.set(`currentB:${i}`, {
      label: `${lbl} current biases (after update)`,
      type: 'B',
      connections: [
        { view: 'backward', description: 'Result of this step — b_new = b_prev − η · dB. These biases are now stored in the model and used in the next forward pass.' },
      ],
    });
  }

  return map;
}

// ----------------------------------------------------------------
// Activation chart (Chart.js Line) shown when a node is hovered
// ----------------------------------------------------------------
const ACTIVATION_FNS: Record<string, (x: number) => number> = {
  relu:    (x) => Math.max(0, x),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh:    (x) => Math.tanh(x),
  linear:  (x) => x,
};

const ACTIVATION_DESCRIPTIONS: Record<string, string> = {
  relu:    "Returns max(0, x). Commonly used for hidden layers — simple, effective, and reduces vanishing gradients.",
  sigmoid: "S-shaped curve, outputs 0–1. Used in binary classification output layers where probabilities are needed.",
  tanh:    "Outputs −1 to 1. Centers data and often converges faster than sigmoid in hidden layers.",
  linear:  "Identity function: outputs x unchanged. Typically used for regression output layers.",
};

function ActivationMiniChart({ activation, zVal }: { activation: string; zVal: number; aVal: number }) {
  const fn = ACTIVATION_FNS[activation] ?? ((x: number) => x);
  const xMin = Math.min(-5, zVal - 1);
  const xMax = Math.max(5, zVal + 1);
  const N = 100;
  const xs = Array.from({ length: N + 1 }, (_, i) => xMin + (i / N) * (xMax - xMin));

  // Index closest to zVal gets the red dot
  const dotIdx = xs.reduce((best, x, i) => Math.abs(x - zVal) < Math.abs(xs[best] - zVal) ? i : best, 0);

  const data = {
    labels: xs.map(x => x.toFixed(2)),
    datasets: [{
      label: activation,
      data: xs.map(fn),
      borderColor: '#6366f1',
      backgroundColor: 'transparent',
      borderWidth: 2,
      tension: 0.3,
      fill: false,
      pointRadius: xs.map((_, i) => i === dotIdx ? 5 : 0),
      pointBackgroundColor: xs.map((_, i) => i === dotIdx ? '#ef4444' : 'transparent'),
      pointBorderColor: xs.map((_, i) => i === dotIdx ? 'white' : 'transparent'),
      pointBorderWidth: xs.map((_, i) => i === dotIdx ? 1.5 : 0),
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 } as const,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false as const },
    },
    scales: {
      x: {
        title: { display: true, text: 'Pre-activation (Z)', font: { size: 9 as const }, color: '#9ca3af' },
        ticks: { display: false as const },
        grid: { color: 'rgba(0,0,0,0.06)' },
      },
      y: {
        title: { display: true, text: 'Post-activation (A)', font: { size: 9 as const }, color: '#9ca3af' },
        ticks: { font: { size: 9 as const }, color: '#9ca3af' },
        grid: { color: 'rgba(0,0,0,0.06)' },
      },
    },
  };

  const description = ACTIVATION_DESCRIPTIONS[activation];

  return (
    <div className="mt-2">
      {description && <p className="text-[11px] text-gray-500 mb-2 leading-relaxed">{description}</p>}
      <div className="h-48 w-full">
        <Line data={data} options={options} />
      </div>
    </div>
  );
}

function NodeStat({ label, sub, value, dim }: { label: string; sub?: string; value: number | undefined; dim?: boolean }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div className="flex flex-col items-center leading-tight">
        <span className="text-[9px] uppercase tracking-wide text-gray-500 text-center">{label}</span>
        {sub && <span className="text-[8px] text-gray-300 text-center">{sub}</span>}
      </div>
      <span className={`font-mono text-xs font-medium ${dim ? "text-gray-400" : "text-gray-900"}`}>
        {value !== undefined ? value.toFixed(3) : "—"}
      </span>
    </div>
  );
}

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
      yMean,
      yStd,
      drawnDigitPrediction,
      setSampleIndex,
    } = useStore();

    const [view, setView] = useState<PropagationView>('forward');
    const [fontSize, setFontSize] = useState("md");

    // Value tracer state
    const [highlightedValueId, setHighlightedValueId] = useState<string | null>(null);
    const [lockedValueId, setLockedValueId] = useState<string | null>(null);
    const lockedValueIdRef = useRef<string | null>(null);
    // Flag set by clickValue so the document listener knows not to clear the lock
    // (React fires before document in the bubble chain, so the flag is always set first)
    const clickConsumedRef = useRef(false);
    const hoverClearTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const jumpLockRef = useRef(false);
    const viewSectionRef = useRef<HTMLDivElement>(null);

    const hoverValue = (id: string) => {
        if (jumpLockRef.current) return;
        if (lockedValueIdRef.current) return;
        clearTimeout(hoverClearTimer.current);
        setHighlightedValueId(id);
    };
    const unhoverValue = () => {
        if (lockedValueIdRef.current) return;
        hoverClearTimer.current = setTimeout(() => setHighlightedValueId(null), 1000);
    };
    const clickValue = (id: string) => {
        clickConsumedRef.current = true; // tell document listener to skip this click
        clearTimeout(hoverClearTimer.current);
        jumpLockRef.current = false;
        if (lockedValueIdRef.current === id) {
            // Toggle off: clicking the same locked value unlocks
            lockedValueIdRef.current = null;
            setLockedValueId(null);
            hoverClearTimer.current = setTimeout(() => setHighlightedValueId(null), 1000);
        } else {
            lockedValueIdRef.current = id;
            setLockedValueId(id);
            setHighlightedValueId(id);
        }
    };
    const clearLock = () => {
        if (!lockedValueIdRef.current) return;
        lockedValueIdRef.current = null;
        setLockedValueId(null);
        clearTimeout(hoverClearTimer.current);
        hoverClearTimer.current = setTimeout(() => setHighlightedValueId(null), 1000);
    };

    // Clear lock when user clicks outside a tracked value
    useEffect(() => {
        const handleDocClick = () => {
            if (clickConsumedRef.current) {
                clickConsumedRef.current = false;
                return;
            }
            if (!lockedValueIdRef.current) return;
            lockedValueIdRef.current = null;
            setLockedValueId(null);
            clearTimeout(hoverClearTimer.current);
            hoverClearTimer.current = setTimeout(() => setHighlightedValueId(null), 1000);
        };
        document.addEventListener('click', handleDocClick);
        return () => document.removeEventListener('click', handleDocClick);
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const jumpToView = (targetView: PropagationView) => {
        jumpLockRef.current = true;
        clearTimeout(hoverClearTimer.current);
        const targetValueId = highlightedValueId;
        setView(targetView);
        setTimeout(() => {
            // Scroll to the specific highlighted element if possible
            if (targetValueId) {
                const valueEl = document.querySelector(`[data-value-id="${targetValueId}"]`);
                if (valueEl) {
                    valueEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    return;
                }
            }
            // Fallback: scroll to top of view section
            const el = viewSectionRef.current;
            if (!el) return;
            const top = el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.18;
            window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
        }, 100);
        // Hold highlight for 2s, then release lock and start normal fade
        setTimeout(() => {
            jumpLockRef.current = false;
            if (!lockedValueIdRef.current) {
                hoverClearTimer.current = setTimeout(() => setHighlightedValueId(null), 1200);
            }
        }, 2000);
    };

    const valueMap = useMemo(() => network ? buildValueMap(network) : new Map<string, ValueInfo>(), [network]);

    const connectedViews = useMemo(() => {
        if (!highlightedValueId) return new Set<PropagationView>();
        const info = valueMap.get(highlightedValueId);
        if (!info) return new Set<PropagationView>();
        return new Set(info.connections.map(c => c.view));
    }, [highlightedValueId, valueMap]);

    const getHighlightStyle = (valueId?: string): { border: string; bg: string } | null => {
        if (!valueId || valueId !== highlightedValueId) return null;
        const type = valueId.split(':')[0];
        const styles = VALUE_TYPE_STYLES[type];
        if (!styles) return null;
        return { border: `border-2 ${styles.primaryBorder}`, bg: styles.bg };
    };

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

    const renderMatrix = (matrix: number[][] | undefined | null, label: string, subLabel?: string, valueId?: string) => {
        if (!matrix || matrix.length === 0) return null;
        const hl = getHighlightStyle(valueId);
        const isLocked = valueId ? valueId === lockedValueId : false;
        return (
            <div
                data-value-id={valueId}
                className={`inline-block p-1 mx-1 rounded transition-colors ${valueId ? 'cursor-pointer' : ''}`}
                onMouseEnter={valueId ? () => hoverValue(valueId) : undefined}
                onMouseLeave={valueId ? unhoverValue : undefined}
                onClick={valueId ? () => clickValue(valueId) : undefined}
            >
                <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div
                    className={`grid rounded overflow-x-auto shadow-sm transition-all duration-150 ${hl ? `${hl.border} ${hl.bg}${isLocked ? ' ring-2 ring-offset-1 ring-current/20' : ''}` : 'border border-gray-400 bg-white'}`}
                    style={{ gridTemplateColumns: `repeat(${matrix[0].length}, auto)` }}
                >
                    {matrix.map((row, rowIndex) => (
                        row.map((val, colIndex) => (
                            <span key={`${rowIndex}-${colIndex}`} className={`px-1 sm:px-1.5 py-0.5 text-xs sm:text-${fontSize}`}>
                                {val.toFixed(dataset === 'xor' ? 3 : 2)}
                            </span>
                        ))
                    ))}
                </div>
            </div>
        );
    };

    const renderVector = (vector: number[] | undefined | null, label: string, subLabel?: string, tooltip?: React.ReactNode, valueId?: string) => {
        if (!vector || vector.length === 0) return null;
        const hl = getHighlightStyle(valueId);
        const isLocked = valueId ? valueId === lockedValueId : false;
        return (
            <div
                data-value-id={valueId}
                className={`inline-block px-1 mx-1 ${valueId ? 'cursor-pointer' : ''}`}
                onMouseEnter={valueId ? () => hoverValue(valueId) : undefined}
                onMouseLeave={valueId ? unhoverValue : undefined}
                onClick={valueId ? () => clickValue(valueId) : undefined}
            >
                <div className="flex flex-row items-center justify-center gap-1 relative group">
                    <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">{label}</p>
                    {tooltip as React.ReactNode}
                </div>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className={`flex flex-row justify-center px-1 py-0.5 rounded shadow-sm overflow-x-auto transition-all duration-150 ${hl ? `${hl.border} ${hl.bg}${isLocked ? ' ring-2 ring-offset-1 ring-current/20' : ''}` : 'border border-gray-400 bg-white'}`}>
                    {vector.map((val, index) => (
                        <span key={index} className={`px-1 py-0.5 text-xs sm:text-${fontSize}`}>{val.toFixed(dataset === 'xor' ? 3 : 2)}</span>
                    ))}
                </div>
            </div>
        );
    };

    const fullExplanation = getExplanation() || '';
    const hasTrained = network?.layers && network.layers[0].A?.length > 0;

    return (
        <>
            {/* Connection panel + Decision Boundary (XOR/Iris) + Prediction side by side */}
            <div className="flex gap-3 mx-2 mt-4 mb-2">
                {/* Left: connection / node details */}
                <div className={`${dataset === "mnist" || dataset === "xor" || dataset === "iris" ? "w-1/3 flex-shrink-0" : "flex-1"} min-w-0 bg-white border border-gray-200 rounded-lg p-3 shadow-sm`}>
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
                            <div className="flex items-center justify-between mb-2">
                                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Node</p>
                                {hoveredNode.layerIndex > 0 && (
                                    <span className="text-[10px] font-mono text-indigo-500">
                                        {network.layers[hoveredNode.layerIndex - 1]?.activation ?? ""}
                                    </span>
                                )}
                            </div>
                            {hoveredNode.layerIndex > 0 ? (() => {
                                const layerObj = network.layers[hoveredNode.layerIndex - 1];
                                const ni = hoveredNode.nodeIndex;
                                const bias = layerObj?.biases?.[ni];
                                const zVal = layerObj?.Z?.[sampleIndex]?.[ni];
                                const aRaw = layerObj?.A?.[sampleIndex]?.[ni];
                                const inputVal = zVal !== undefined && bias !== undefined ? zVal - bias : undefined;
                                const isOutputLayer = hoveredNode.layerIndex === network.layers.length - 1;
                                const aDisplay = isOutputLayer && dataset === "auto_mpg" && yMean !== null && yStd !== null && aRaw !== undefined
                                    ? aRaw * yStd + yMean : aRaw;
                                const aLabel = isOutputLayer && dataset === "auto_mpg" ? "MPG" : "A";
                                const activation = layerObj?.activation ?? "linear";
                                const showChart = zVal !== undefined && aRaw !== undefined && activation !== "softmax";
                                return (
                                    <>
                                        <div className="grid grid-cols-4 gap-1 mb-2 px-0.5">
                                            <NodeStat label="Weighted sum" value={inputVal} />
                                            <NodeStat label="Bias" value={bias} />
                                            <NodeStat label="Pre-activation" sub="(Z)" value={zVal} />
                                            <NodeStat label={aLabel === "MPG" ? "MPG" : "Post-activation"} sub={aLabel === "MPG" ? undefined : "(A)"} value={aDisplay} />
                                        </div>
                                        {showChart && <ActivationMiniChart activation={activation} zVal={zVal!} aVal={aRaw!} />}
                                        {activation === "softmax" && zVal !== undefined && (
                                            <p className="text-[10px] text-gray-400 text-center mt-1">softmax — output depends on all logits</p>
                                        )}
                                    </>
                                );
                            })() : (
                                <p className="text-sm text-gray-500">Input node — values come from the dataset.</p>
                            )}
                        </>
                    ) : (
                        <div className="flex items-center justify-center h-full min-h-[60px]">
                            <p className="text-sm text-gray-400">Click a node or connection in the graph</p>
                        </div>
                    )}
                </div>

                {/* Middle: decision boundary for XOR and Iris */}
                {(dataset === "xor" || dataset === "iris") && network && (
                    <div className="w-1/3 flex-shrink-0 min-w-0 bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
                        <DecisionBoundary
                            layers={network.layers}
                            dataset={dataset}
                            originalData={originalData}
                        />
                    </div>
                )}

                {/* Right: for MNIST — draw canvas + prediction; otherwise sample preview */}
                {dataset === "mnist" ? (
                    <>
                        <div className="flex-1 min-w-0 bg-white border border-gray-200 rounded-lg p-4 shadow-sm flex flex-col items-center gap-2">
                            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide self-start">Draw a Digit</p>
                            <DigitCanvas hidePrediction />
                        </div>
                        <div className="flex-1 min-w-0 bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
                            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Prediction</p>
                            {drawnDigitPrediction ? (
                                <div className="space-y-1">
                                    {drawnDigitPrediction.confidences.map((conf, digit) => {
                                        const isPredicted = digit === drawnDigitPrediction.predictedClass;
                                        const pct = (conf * 100).toFixed(1);
                                        return (
                                            <div key={digit} className="flex items-center gap-2">
                                                <span className={`w-4 text-xs font-mono text-right flex-shrink-0 ${isPredicted ? "font-bold text-indigo-700" : "text-gray-500"}`}>
                                                    {digit}
                                                </span>
                                                <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all duration-300 ${isPredicted ? "bg-indigo-500" : "bg-gray-300"}`}
                                                        style={{ width: `${pct}%` }}
                                                    />
                                                </div>
                                                <span className={`w-10 text-xs text-right flex-shrink-0 ${isPredicted ? "font-bold text-indigo-700" : "text-gray-400"}`}>
                                                    {pct}%
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="flex items-center justify-center h-full min-h-[60px]">
                                    <p className="text-sm text-gray-400">Draw a digit and click Predict</p>
                                </div>
                            )}
                        </div>
                    </>
                ) : hasTrained && originalData[sampleIndex] && (
                    <div className={`${dataset === "xor" || dataset === "iris" ? "w-1/3 flex-shrink-0" : "flex-1"} min-w-0 flex flex-col gap-2 bg-white border border-gray-200 rounded-lg p-3 shadow-sm`}>
                        <SampleVisual
                            dataset={dataset}
                            original={originalData[sampleIndex]}
                            network={network}
                            sampleIndex={sampleIndex}
                            yMean={yMean}
                            yStd={yStd}
                        />
                        <div className="flex items-center gap-2 px-1">
                            <span className="text-xs text-gray-500 whitespace-nowrap">Sample #</span>
                            <input
                                type="number"
                                value={sampleIndex}
                                onChange={(e) => {
                                    const max = dataset === "xor" ? 3 : 25;
                                    setSampleIndex(Math.max(0, Math.min(max, Number(e.target.value))));
                                }}
                                min={0}
                                max={dataset === "xor" ? 3 : 25}
                                className="w-14 text-center text-xs border border-gray-200 rounded px-1 py-0.5"
                            />
                            <span className="text-xs text-gray-400">0 – {dataset === "xor" ? 3 : 25}</span>
                        </div>
                    </div>
                )}
            </div>
        <div className="mt-2 p-4 bg-gray-100 rounded-lg mx-2 shadow-md">

            {hasTrained && dataset !== "mnist" && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3">
                <div className="flex flex-col sm:flex-row rounded-md shadow-sm" role="group">
                    {(['forward', 'calculation', 'backward'] as PropagationView[]).map((v, vi) => {
                        const labels = { forward: '1 · Forward Pass', calculation: '2 · Compute Gradients', backward: '3 · Update Weights' };
                        const isActive = view === v;
                        const hasConnection = !isActive && highlightedValueId && connectedViews.has(v);
                        const dotColor = hasConnection ? VALUE_TYPE_STYLES[valueMap.get(highlightedValueId!)?.type || '']?.dot : '';
                        const roundClass = vi === 0
                            ? 'rounded-t-lg sm:rounded-l-lg sm:rounded-t-none'
                            : vi === 2
                            ? 'rounded-b-lg sm:rounded-r-lg sm:rounded-b-none'
                            : '';
                        return (
                            <button
                                key={v}
                                type="button"
                                onClick={() => setView(v)}
                                className={`relative px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium ${roundClass} ${
                                    isActive ? 'bg-black text-white' : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                                }`}
                            >
                                {labels[v]}
                                {hasConnection && dotColor && (
                                    <span className={`absolute top-1 right-1 w-1.5 h-1.5 rounded-full ${dotColor}`} />
                                )}
                            </button>
                        );
                    })}
                </div>
                <Glossary />
            </div>)}

            {hasTrained && dataset !== "mnist" && (
                <>
                {/* ── Main view content ── */}
                <div className="text-center" ref={viewSectionRef}>
                    {view === 'forward' ? (
                        <>

                            <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                                <div className="flex-1 text-center">
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
                                                        layerIndex === 0
                                                            ? <InputInfo dataset={dataset} input={network?.input[sampleIndex]} originalInput={originalData[sampleIndex]} />
                                                            : null,
                                                        layerIndex === 0 ? "input" : `A:${layerIndex - 1}`
                                                    )}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">×</span>

                                                <div className="flex flex-col items-center">
                                                    {renderMatrix(layer.prevWeights, `Weights`, "Connection strengths between neurons", `W:${layerIndex}`)}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">+</span>

                                                <div className="flex flex-col items-center">
                                                    {renderVector(layer.prevBias, `Biases`, undefined, undefined, `B:${layerIndex}`)}
                                                </div>

                                                <span className="text-sm sm:text-lg mt-2 sm:mt-5 mx-1">=</span>

                                                <div className="flex flex-col items-center">
                                                    <div className="flex flex-col items-center">
                                                        {layer.activation !== "linear" && (<>
                                                            {renderVector(
                                                                layer.Z[sampleIndex],
                                                                isOutputComputation ? "Raw Output" : `Pre-activations`,
                                                                undefined, undefined,
                                                                `Z:${layerIndex}`
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
                                                            isOutputComputation
                                                                ? <OutputInfo dataset={dataset} output={layer.A[sampleIndex]} actual={originalData[sampleIndex]} />
                                                                : null,
                                                            `A:${layerIndex}`
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

                                // value-tracer ids for this layer
                                const li = network.layers.length - 1 - reversedIndex;
                                const aPrevIdx = li - 1;
                                const aPrevId = aPrevIdx < 0 ? 'input' : `A:${aPrevIdx}`;

                                return (
                                    <div key={`layer-${reversedIndex}-backprop`} className="flex flex-col gap-4 items-center border-t border-gray-300 pt-4">
                                        {layerIndex === network.layers.length - 1
                                            ? <h2 className="text-md font-semibold text-gray-700">Output Layer</h2>
                                            : <h2 className="text-md font-semibold text-gray-700">Layer {layerIndex}</h2>}

                                        {/* dW/dB Derivation */}
                                        <h2 className="text-md font-semibold mt-2 text-gray-700">Calculating change in weights and biases</h2>
                                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-12 flex-wrap">
                                            <div className="flex flex-col items-center gap-2">
                                                {dataset === "xor" ? (
                                                    <>
                                                        <p className="text-sm text-gray-600">dW = (1/4) × A<sub>prev</sub><sup>T</sup> · dZ — all 4 XOR patterns:</p>
                                                        <p className="text-xs text-gray-600 italic">*Gradients from Y=0 patterns and Y=1 patterns partially cancel, giving small but non-zero dW*</p>
                                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                            {renderMatrix(
                                                                transpose(aPrevIdx < 0 ? network.input : network.layers[aPrevIdx].A) ?? [],
                                                                `A_prev.T`, "neurons × samples", aPrevId
                                                            )}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                            {renderMatrix(layer.dZ, `dZ`, "samples × outputs", `dZ:${li}`)}
                                                            <span className="text-sm sm:text-base mt-2 sm:mt-6 text-gray-600">× (1/4) =</span>
                                                            {renderMatrix(layer.dW, `dW`, "∇ Weights", `dW:${li}`)}
                                                        </div>
                                                    </>
                                                ) : (() => {
                                                    const aPrev = aPrevIdx < 0 ? network.input : network.layers[aPrevIdx].A;
                                                    const last = aPrev.length - 1;
                                                    return (
                                                        <>
                                                            <p className="text-sm text-gray-600">dW = (1/m) &Sigma; A<sub>prev</sub><sup>T</sup> &middot; dZ averaged over all samples:</p>
                                                            <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                <div className="flex flex-col gap-1 items-center">
                                                                    <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">A_prev rows</p>
                                                                    {renderVector(aPrev[0], ``, "", undefined, aPrevId)}
                                                                    {renderVector(aPrev[1], ``, "", undefined, aPrevId)}
                                                                    <span className="text-gray-400 text-center leading-none">⋮</span>
                                                                    {renderVector(aPrev[last], ``, "", undefined, aPrevId)}
                                                                </div>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                <div className="flex flex-col gap-1 items-center">
                                                                    <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">dZ rows</p>
                                                                    {renderVector(layer.dZ[0], ``, "", undefined, `dZ:${li}`)}
                                                                    {renderVector(layer.dZ[1], ``, "", undefined, `dZ:${li}`)}
                                                                    <span className="text-gray-400 text-center leading-none">⋮</span>
                                                                    {renderVector(layer.dZ[last], ``, "", undefined, `dZ:${li}`)}
                                                                </div>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                {renderMatrix(layer.dW, `dW`, "∇ Weights avg", `dW:${li}`)}
                                                            </div>
                                                        </>
                                                    );
                                                })()}
                                            </div>

                                            <div className="flex flex-col items-center gap-2">
                                                {dataset === "xor" ? (
                                                    <>
                                                        <p className="text-sm text-gray-600">Gradient of biases (dB) = (1/4) Σ dZ across all 4 patterns:</p>
                                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                            <span className="text-sm sm:text-base mt-2 sm:mt-6 text-gray-600">(1/4)</span>
                                                            {renderMatrix(layer.dZ, `dZ`, "all samples", `dZ:${li}`)}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                            {renderVector(layer.db, `dB`, "∇ Biases", undefined, `dB:${li}`)}
                                                        </div>
                                                    </>
                                                ) : (() => {
                                                    const last = layer.dZ.length - 1;
                                                    return (
                                                        <>
                                                            <p className="text-sm text-gray-600">Gradient of biases (dB) is the sum of dZ across samples:</p>
                                                            <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">Σ</span>
                                                                <div className="flex flex-col gap-1 items-center">
                                                                    <p className="text-xs sm:text-sm text-gray-600 text-center mb-1">dZ rows</p>
                                                                    {renderVector(layer.dZ[0], ``, "", undefined, `dZ:${li}`)}
                                                                    {renderVector(layer.dZ[1], ``, "", undefined, `dZ:${li}`)}
                                                                    <span className="text-gray-400 text-center leading-none">⋮</span>
                                                                    {renderVector(layer.dZ[last], ``, "", undefined, `dZ:${li}`)}
                                                                </div>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                {renderVector(layer.db, `dB`, "∇ Biases", undefined, `dB:${li}`)}
                                                            </div>
                                                        </>
                                                    );
                                                })()}
                                            </div>
                                        </div>

                                        {/* Weights Equation */}
                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                            {renderMatrix(layer.prevWeights, `Previous Weights`, "", `W:${li}`)}
                                            <span className="text-lg sm:text-xxl mt-2 sm:mt-8">-</span>
                                            <span className="text-xs sm:text-sm mt-2 sm:mt-8 text-gray-600"><strong className="text-lg sm:text-xl text-gray-600">η</strong> ×</span>
                                            {renderMatrix(layer.dW, `Change in Weights`, "dW (∇Weights)", `dW:${li}`)}
                                            <span className="text-lg sm:text-xl mt-2 sm:mt-8">=</span>
                                            {renderMatrix(layer.weights, `Current Weights`, "", `currentW:${li}`)}
                                        </div>

                                        {/* Biases Equation */}
                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                            {renderVector(layer.prevBias, `Previous Biases`, "", undefined, `B:${li}`)}
                                            <span className="text-lg sm:text-xxl mt-2 sm:mt-8">-</span>
                                            <span className="text-xs sm:text-sm mt-2 sm:mt-8 text-gray-600"><strong className="text-lg sm:text-xl text-gray-600">η</strong> ×</span>
                                            {renderVector(layer.db, `Change in Biases`, "dB (∇Biases)", undefined, `dB:${li}`)}
                                            <span className="text-lg sm:text-xl mt-2 sm:mt-8">=</span>
                                            {renderVector(layer.biases, `Current Biases`, "", undefined, `currentB:${li}`)}
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
                                const actualRaw = originalData[sampleIndex].slice(dataset === "iris" ? -3 : -1);
                                const actual = dataset === "auto_mpg" && yMean !== null && yStd !== null
                                    ? actualRaw.map(v => (v - yMean) / yStd)
                                    : actualRaw;
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
                                                {dataset === "auto_mpg" ? (
                                                    <>
                                                        <p className="text-sm text-gray-600">Calculate initial error (dZ) at the output — MSE derivative: dZ = 2(Ŷ − Y):</p>
                                                        <p className="text-xs text-gray-600 italic">*Note: this is done with all samples at once, unlike the single sample shown here*</p>
                                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                            <span className="text-sm sm:text-base mt-2 sm:mt-6 text-gray-600">2 ×</span>
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">(</span>
                                                            {renderVector(network.layers[outputLayerIndex].A[sampleIndex], `Prediction`, "(normalized)", undefined, `A:${outputLayerIndex}`)}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">-</span>
                                                            {renderVector(actual, `Actual`, "(normalized)")}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">)</span>
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                            {renderVector(network.layers[outputLayerIndex].dZ[sampleIndex], `dZ`, "", undefined, `dZ:${outputLayerIndex}`)}
                                                        </div>
                                                    </>
                                                ) : (
                                                    <>
                                                        <p className="text-sm text-gray-600">Calculate initial error (dZ) at the output:</p>
                                                        <p className="text-xs text-gray-600 italic">*Note: this is done with all samples at once, unlike the single sample shown here*</p>
                                                        <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                            {renderVector(network.layers[outputLayerIndex].A[sampleIndex], `Prediction`, "", undefined, `A:${outputLayerIndex}`)}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">-</span>
                                                            {renderVector(actual, `Actual`, "")}
                                                            <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                            {renderVector(network.layers[outputLayerIndex].dZ[sampleIndex], `Error`, "", undefined, `dZ:${outputLayerIndex}`)}
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        ) : (() => {
                                            const li = network.layers.length - 1 - index;
                                            return (
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
                                                            {renderVector(layer.dZ[sampleIndex], `dZ`, "Calculated above", undefined, `dZ:${li}`)}
                                                            {(outputLayerIndex - index + 1) !== 0 ? <>
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">×</span>
                                                                {renderMatrix(transpose(network.layers[outputLayerIndex - index + 1]?.prevWeights), `Wᵀ`, "Transposed weights from backwards layer", `W:${li}`)}
                                                                <span className="text-lg sm:text-2xl mt-2 sm:mt-6">=</span>
                                                                {renderMatrix(dA, `dA`, "∇ Activations")}
                                                            </> : null}
                                                        </div>
                                                        {(outputLayerIndex - index + 1) !== 0 ? (
                                                            <div className="flex flex-row items-center gap-1 sm:gap-2 flex-wrap justify-center">
                                                                {renderMatrix(dA, `dA`, "∇ Activations")}
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
                                                                {renderVector(network.layers[outputLayerIndex - index]?.dZ[sampleIndex], `dZ`, "Error to pass to backwards layer", undefined, li > 0 ? `dZ:${li - 1}` : undefined)}
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                </div>
                                            </div>
                                            );
                                        })()}
                                    </div>
                                );
                            })}
                            </div>
                        </div>
                    )}
                </div>

                {/* Idle hint */}
                <p className="mt-3 text-xs text-gray-400 text-center">Hover any labeled value to trace where it comes from and where it&apos;s used ↓</p>
                </>
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

        {/* ── Fixed bottom value tracer panel ── */}
        {highlightedValueId && valueMap.get(highlightedValueId) && (() => {
            const info = valueMap.get(highlightedValueId)!;
            const styles = VALUE_TYPE_STYLES[info.type];
            return (
                <div
                    className={`fixed bottom-0 left-0 right-0 z-50 border-t-2 shadow-2xl px-4 py-3 ${styles.bg}`}
                    style={{ borderColor: undefined }}
                    onMouseEnter={() => clearTimeout(hoverClearTimer.current)}
                    onMouseLeave={unhoverValue}
                    onClick={() => { clickConsumedRef.current = true; }}
                >
                    <div className={`max-w-5xl mx-auto`}>
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                            <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${styles.badge}`}>{info.type}</span>
                            <span className="text-xs font-semibold text-gray-800">{info.label}</span>
                            {lockedValueId === highlightedValueId ? (
                                <button
                                    onClick={clearLock}
                                    className="text-xs text-gray-500 ml-auto hover:text-gray-700"
                                >
                                    pinned · click to unpin
                                </button>
                            ) : (
                                <span className="text-xs text-gray-400 ml-auto">click to pin · Value Tracer</span>
                            )}
                        </div>
                        <div className="flex flex-row flex-wrap gap-2">
                            {info.connections.map((conn, ci) => {
                                const isCurrent = conn.view === view;
                                return (
                                    <div key={ci} className="flex flex-col gap-0.5 bg-white/80 border border-gray-200 rounded p-2 min-w-[180px] flex-1">
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="text-xs font-semibold text-gray-600">{VIEW_LABELS[conn.view]}</span>
                                            {isCurrent ? (
                                                <span className="text-xs text-gray-400 shrink-0">here</span>
                                            ) : (
                                                <button
                                                    onClick={() => jumpToView(conn.view)}
                                                    className={`text-xs font-bold ${styles.text} hover:underline shrink-0`}
                                                >
                                                    jump →
                                                </button>
                                            )}
                                        </div>
                                        <p className="text-xs text-gray-600 leading-snug">{conn.description}</p>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            );
        })()}
        </>
    );
};

export default Explain;
