"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { X, ChevronRight, ChevronLeft, Map } from "lucide-react";
import useStore from "@/components/network/lib/store";

const STORAGE_KEY = "nn-viz-tour-v3";

type Placement = "top" | "bottom" | "left" | "right" | "center";

type Step = {
  title: string;
  body: React.ReactNode;
  target?: string;
  placement?: Placement;
  padding?: number;
};

// ─── Step definitions ─────────────────────────────────────────────────────────

const STEPS: Step[] = [
  // 0 — Welcome
  {
    title: "Welcome to the Neural Network Visualizer",
    body: (
      <div className="space-y-2.5">
        <p>Build, train, and explore neural networks — no code required.</p>
        <p className="text-gray-500 text-xs">
          This 2-minute tour covers every major feature. Hit <strong>Next</strong> to
          start, or <strong>Skip tour</strong> if you&apos;d rather dive straight in.
        </p>
        <div className="grid grid-cols-3 gap-2 text-xs text-center pt-1">
          {["Pick dataset", "Design network", "Train & explore"].map((s, i) => (
            <div key={s} className="bg-indigo-50 rounded-lg px-2 py-2 border border-indigo-100">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mx-auto mb-1 text-[10px] font-bold">
                {i + 1}
              </div>
              <span className="text-indigo-700 font-medium">{s}</span>
            </div>
          ))}
        </div>
      </div>
    ),
    placement: "center",
  },

  // 1 — Dataset selection
  {
    title: "Step 1 · Choose a Dataset",
    body: (
      <div className="space-y-2">
        <p>The dataset defines what problem the network will learn to solve.</p>
        <ul className="text-sm space-y-1.5">
          {[
            ["Iris",     "Classify flower species from 4 measurements"],
            ["Auto MPG", "Predict fuel efficiency — a regression task"],
            ["XOR",      "Learn a non-linear boolean rule from scratch"],
            ["MNIST",    "Recognize handwritten digits 0–9, draw your own"],
          ].map(([name, desc]) => (
            <li key={name} className="flex gap-2">
              <span className="font-semibold text-indigo-600 shrink-0 w-20">{name}</span>
              <span className="text-gray-600">{desc}</span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-gray-400">
          Click a card to see its inputs, output, loss function, and sample count. Each dataset switches the network architecture and loss function automatically.
        </p>
      </div>
    ),
    target: "[data-tour='wizard-container']",
    placement: "bottom",
    padding: 6,
  },

  // 2 — Network configuration
  {
    title: "Step 2 · Design Your Network",
    body: (
      <div className="space-y-2">
        <p>
          Choose the number of hidden layers (1–3), nodes per layer, and the{" "}
          <strong>activation function</strong> for each layer.
        </p>
        <div className="bg-gray-50 rounded-lg p-2 text-xs font-mono text-gray-600 border border-gray-200">
          Input (4) → Hidden 1 (4, relu) → Hidden 2 (4, relu) → Output (3, softmax)
        </div>
        <ul className="text-sm space-y-1 text-gray-600">
          <li>The architecture preview updates live as you configure</li>
          <li>Total parameter count shown so you can gauge model size</li>
          <li>Try <strong>relu</strong> for hidden layers — it usually trains fastest</li>
        </ul>
        <p className="text-xs text-gray-400">
          Hit <strong>Initialize Model</strong> when you&apos;re happy with the design.
        </p>
      </div>
    ),
    target: "[data-tour='wizard-container']",
    placement: "bottom",
    padding: 6,
  },

  // 3 — Network diagram
  {
    title: "The Network Diagram",
    body: (
      <div className="space-y-2">
        <p>
          Each circle is a neuron; each line is a weight. The diagram updates live
          as you train — <strong>brightness</strong> reflects activation strength.
        </p>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs mt-1">
          {[
            [<span key="b" className="w-3 h-3 rounded-full bg-blue-400 border border-blue-600 shrink-0 inline-block"/>, "Input node"],
            [<span key="g" className="w-3 h-3 rounded-full bg-gray-400 border border-gray-600 shrink-0 inline-block"/>, "Hidden node"],
            [<span key="r" className="w-3 h-3 rounded-full bg-red-400 border border-red-700 shrink-0 inline-block"/>,  "Output node"],
            [<span key="i" className="w-4 h-1 bg-indigo-400 rounded shrink-0 inline-block"/>, "Positive weight"],
            [<span key="o" className="w-4 h-1 bg-orange-400 rounded shrink-0 inline-block"/>, "Negative weight"],
            [<span key="flash" className="flex gap-0.5"><span className="w-2 h-2 rounded-full bg-green-400 inline-block"/><span className="w-2 h-2 rounded-full bg-orange-400 inline-block"/></span>, "Weight changed"],
          ].map(([icon, label], i) => (
            <div key={i} className="flex items-center gap-1.5">{icon}<span className="text-gray-600">{label}</span></div>
          ))}
        </div>
        <p className="text-xs text-indigo-700 font-medium">
          Click any node or weight line to inspect its value, gradient, and contribution to the output.
        </p>
      </div>
    ),
    target: "[data-tour='network-svg']",
    placement: "bottom",
    padding: 8,
  },

  // 4 — Training widget
  {
    title: "Training Controls",
    body: (
      <div className="space-y-2">
        <p>
          The floating widget in the top-right corner is your main training panel.
          It shows live stats and controls everything:
        </p>
        <ul className="text-xs space-y-1.5 text-gray-600">
          <li><strong className="text-gray-800">Epochs · Loss · Accuracy</strong> — updated after every training run</li>
          <li><strong className="text-gray-800">Train</strong> — runs the next batch of epochs</li>
          <li><strong className="text-gray-800">Learning rate slider</strong> — controls how large each gradient step is. Too high and loss oscillates; too low and training crawls</li>
          <li><strong className="text-gray-800">Sample navigator</strong> — cycles through training examples, highlighting that data point in the diagram</li>
          <li><strong className="text-gray-800">Trophy</strong> — opens the full leaderboard panel</li>
        </ul>
      </div>
    ),
    target: "[data-tour='training-widget']",
    placement: "left",
    padding: 8,
  },

  // 5 — Leaderboard
  {
    title: "The Leaderboard",
    body: (
      <div className="space-y-2">
        <p>Compete with other users. Each dataset uses a unique scoring rule:</p>
        <div className="space-y-1.5 text-sm">
          {[
            ["XOR",      "Fewest epochs to reach 100% accuracy"],
            ["Iris",     "Best accuracy at exactly epoch 100"],
            ["Auto MPG", "Lowest MAE at exactly epoch 200"],
            ["MNIST",    "Best accuracy at exactly epoch 300"],
          ].map(([ds, rule]) => (
            <div key={ds} className="flex gap-2 items-start">
              <span className="font-semibold text-indigo-600 shrink-0 w-20">{ds}</span>
              <span className="text-gray-600 text-xs">{rule}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400">
          Once you hit the epoch target, a username input appears — enter a name to
          claim your rank.
        </p>
      </div>
    ),
    target: "[data-tour='leaderboard-inline']",
    placement: "right",
    padding: 8,
  },

  // 6 — Dataset-specific visuals
  {
    title: "Dataset-Specific Visualizations",
    body: (
      <div className="space-y-2">
        <p>Each dataset gets a visualization tailored to the task:</p>
        <ul className="text-sm space-y-1.5">
          <li>
            <strong>XOR / Iris</strong> — decision boundary plot showing which regions map
            to each class. Watch the boundary sharpen as training progresses
          </li>
          <li>
            <strong>Auto MPG</strong> — scatter of predicted vs. actual MPG across all
            samples. A diagonal line means perfect predictions
          </li>
          <li>
            <strong>MNIST</strong> — draw your own digit on a canvas and watch the network
            classify it in real time
          </li>
        </ul>
      </div>
    ),
    target: "[data-tour='dataset-viz']",
    placement: "right",
    padding: 8,
  },

  // 7 — The math section
  {
    title: "The Underlying Math",
    body: (
      <div className="space-y-2.5">
        <p>
          This section shows the <strong>actual matrix math</strong> inside your
          network — with real values from the last training step.
        </p>
        <div className="grid grid-cols-3 gap-1.5 text-xs text-center">
          <div className="bg-black text-white rounded-lg px-2 py-1.5 font-medium">1 · Forward Pass</div>
          <div className="bg-white border border-gray-300 rounded-lg px-2 py-1.5 text-gray-600">2 · Gradients</div>
          <div className="bg-white border border-gray-300 rounded-lg px-2 py-1.5 text-gray-600">3 · Update Weights</div>
        </div>
        <ul className="text-sm space-y-1 text-gray-600">
          <li><strong>Forward</strong>: Input × W + b → activation → A</li>
          <li><strong>Gradients</strong>: Chain-rule backprop — dZ, dW, dB</li>
          <li><strong>Update</strong>: W_new = W_old − η · dW</li>
        </ul>
        <p className="text-xs text-indigo-700 font-medium">
          Use &quot;Step Through&quot; to walk layer by layer. Hover any matrix cell to activate the
          Value Tracer — it shows exactly where that value flows through the network.
        </p>
      </div>
    ),
    target: "[data-tour='math-panel']",
    placement: "top",
    padding: 4,
  },

  // 8 — Charts
  {
    title: "Loss and Accuracy Charts",
    body: (
      <div className="space-y-2">
        <p>Two live charts track learning progress across every epoch:</p>
        <div className="space-y-1.5 text-sm mt-1">
          <div className="flex gap-2 items-start">
            <div className="w-3 h-3 rounded-full bg-indigo-500 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-gray-800">Training Loss</span>
              <span className="text-gray-500 text-xs ml-1">— should trend downward as the network improves</span>
            </div>
          </div>
          <div className="flex gap-2 items-start">
            <div className="w-3 h-3 rounded-full bg-emerald-500 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-gray-800">Accuracy / MAE</span>
              <span className="text-gray-500 text-xs ml-1">— rises for classifiers, falls for regression (lower MAE = better)</span>
            </div>
          </div>
        </div>
        <p className="text-xs text-gray-400">
          Crank the learning rate up and watch the loss curve oscillate instead of
          descending smoothly — a classic sign of overshooting the minimum.
        </p>
      </div>
    ),
    target: "[data-tour='charts']",
    placement: "top",
    padding: 6,
  },

  // 9 — Wrap-up / other pages
  {
    title: "That's it — keep exploring",
    body: (
      <div className="space-y-3">
        <p className="text-gray-700">
          You&apos;ve seen everything this page has to offer. Two more interactive tools are a
          click away:
        </p>
        <a
          href="/attention"
          className="block bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 hover:bg-indigo-100 transition-colors group"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-indigo-900 text-sm group-hover:underline">
                Attention Visualizer
              </p>
              <p className="text-xs text-indigo-600 mt-0.5">
                See how transformers compute attention weights across tokens — step through
                encoder/decoder layers
              </p>
            </div>
            <ChevronRight size={16} className="text-indigo-400 shrink-0 ml-2" />
          </div>
        </a>
        <a
          href="/transformers"
          className="block bg-violet-50 border border-violet-200 rounded-xl px-4 py-3 hover:bg-violet-100 transition-colors group"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="font-semibold text-violet-900 text-sm group-hover:underline">
                Transformer Architecture
              </p>
              <p className="text-xs text-violet-600 mt-0.5">
                Explore the full GPT-2 model — embeddings, QKV attention, multi-head
                attention, MLP blocks, and the LM head
              </p>
            </div>
            <ChevronRight size={16} className="text-violet-400 shrink-0 ml-2" />
          </div>
        </a>
      </div>
    ),
    placement: "center",
  },
];

// ─── Types ────────────────────────────────────────────────────────────────────

type SpotlightRect = { x: number; y: number; width: number; height: number };

// ─── Scroll lock ──────────────────────────────────────────────────────────────

function lockScroll()   { document.documentElement.style.overflow = "hidden"; }
function unlockScroll() { document.documentElement.style.overflow = ""; }

// ─── Layout constants ─────────────────────────────────────────────────────────

const TOOLTIP_W = 400;
const GAP       = 12;   // gap between spotlight edge and tooltip
const MARGIN    = 8;    // min distance from viewport edge
const EST_H     = 440;  // estimated tooltip height for space checks

// ─── Component ────────────────────────────────────────────────────────────────

export default function Walkthrough() {
  const [open,          setOpen]          = useState(false);
  const [step,          setStep]          = useState(0);
  const [spotlight,     setSpotlight]     = useState<SpotlightRect | null>(null);
  const [tooltipStyle,  setTooltipStyle]  = useState<React.CSSProperties>({});
  const [mounted,       setMounted]       = useState(false);
  // Only visible/active when ?tour is in the URL — lets you test in prod without
  // exposing the button or auto-show to regular users.
  const [previewMode,   setPreviewMode]   = useState(false);
  const positionTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Store actions for auto-training
  const {
    sessionId,
    initModel,
    runTrainingCycle,
    setTrainingEpochs,
    setTourActive,
    setTourStep,
  } = useStore();
  const tourInitRef = useRef(false);

  // ── Mount: check for ?tour param ──────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    const params = new URLSearchParams(window.location.search);
    if (params.has("tour")) {
      setPreviewMode(true);
      // Auto-open immediately in preview mode (don't gate on localStorage)
      const t = setTimeout(() => setOpen(true), 400);
      return () => clearTimeout(t);
    }
  }, []);

  // ── Auto-init + train when user advances past the config steps ───────────
  // Fires once (guarded by ref) the moment step 3 is reached, so the network,
  // training widget, and charts all show real data by the time the user gets there.
  useEffect(() => {
    if (!open || step !== 3) return;
    if (tourInitRef.current) return;
    tourInitRef.current = true;

    if (sessionId) return; // user already has a trained model — use it as-is

    // Store defaults: dataset="iris", hiddenLayers=[4,4], activations=["relu","relu"]
    initModel().then(() => {
      setTrainingEpochs(20);
      return runTrainingCycle();
    }).then(() => {
      setTrainingEpochs(1);
    }).catch(() => {
      setTrainingEpochs(1);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, open]);

  // Reset the init guard when the tour is closed so it can fire again next open
  useEffect(() => {
    if (!open) tourInitRef.current = false;
  }, [open]);

  // ── Position computation ──────────────────────────────────────────────────
  const updatePositions = useCallback(() => {
    const s  = STEPS[step];
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w  = Math.min(TOOLTIP_W, vw - MARGIN * 2);

    const centeredStyle: React.CSSProperties = {
      position: "fixed",
      top:  "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      width: w,
      maxHeight: `calc(100vh - ${MARGIN * 4}px)`,
    };

    if (!s.target) { setSpotlight(null); setTooltipStyle(centeredStyle); return; }

    const el = document.querySelector(s.target);
    if (!el)       { setSpotlight(null); setTooltipStyle(centeredStyle); return; }

    const r   = el.getBoundingClientRect();
    const pad = s.padding ?? 12;

    setSpotlight({ x: r.left - pad, y: r.top - pad, width: r.width + pad*2, height: r.height + pad*2 });

    const spaceBelow = vh - r.bottom - pad - GAP - MARGIN;
    const spaceAbove = r.top  - pad - GAP - MARGIN;
    const spaceRight = vw - r.right - pad - GAP - MARGIN;
    const spaceLeft  = r.left - pad - GAP - MARGIN;

    const hCenter = Math.max(MARGIN, Math.min(vw - w - MARGIN, r.left + r.width/2 - w/2));

    const placeBelow = () => ({
      top:  r.bottom + pad + GAP,
      left: hCenter,
      maxHeight: Math.max(80, spaceBelow),
    });
    const placeAbove = () => {
      const mh  = Math.max(80, spaceAbove);
      const top = Math.max(MARGIN, r.top - pad - GAP - Math.min(EST_H, mh));
      return { top, left: hCenter, maxHeight: mh };
    };
    const placeRight = () => {
      const top = Math.max(MARGIN, Math.min(vh - EST_H - MARGIN, r.top + r.height/2 - EST_H/2));
      return { top, left: r.right + pad + GAP, maxHeight: vh - top - MARGIN };
    };
    const placeLeft = () => {
      const top  = Math.max(MARGIN, Math.min(vh - EST_H - MARGIN, r.top + r.height/2 - EST_H/2));
      const left = Math.max(MARGIN, r.left - pad - GAP - w);
      return { top, left, maxHeight: vh - top - MARGIN };
    };

    let pos: ReturnType<typeof placeBelow> | ReturnType<typeof placeAbove>;
    switch (s.placement ?? "bottom") {
      case "bottom":
        pos = (spaceBelow >= EST_H || spaceBelow >= spaceAbove) ? placeBelow()
            : spaceAbove >= EST_H * 0.5                         ? placeAbove()
            : null!;
        break;
      case "top":
        pos = (spaceAbove >= EST_H || spaceAbove >= spaceBelow) ? placeAbove()
            : spaceBelow >= EST_H * 0.5                         ? placeBelow()
            : null!;
        break;
      case "right":
        if (spaceRight >= w) { setTooltipStyle({ position:"fixed", width:w, ...placeRight() }); return; }
        if (spaceLeft  >= w) { setTooltipStyle({ position:"fixed", width:w, ...placeLeft()  }); return; }
        pos = spaceBelow >= spaceAbove ? placeBelow() : placeAbove();
        break;
      case "left":
        if (spaceLeft  >= w) { setTooltipStyle({ position:"fixed", width:w, ...placeLeft()  }); return; }
        if (spaceRight >= w) { setTooltipStyle({ position:"fixed", width:w, ...placeRight() }); return; }
        pos = spaceBelow >= spaceAbove ? placeBelow() : placeAbove();
        break;
      default:
        setTooltipStyle(centeredStyle); return;
    }

    setTooltipStyle(pos ? { position: "fixed", width: w, ...pos } : centeredStyle);
  }, [step]);

  // ── Scroll to target then lock scroll ─────────────────────────────────────
  useEffect(() => {
    if (!open) return;
    clearTimeout(positionTimer.current);

    const s = STEPS[step];
    if (s.target) {
      const el = document.querySelector(s.target);
      if (el) {
        unlockScroll();
        const r       = el.getBoundingClientRect();
        const targetY = window.scrollY + r.top - window.innerHeight / 2 + r.height / 2;
        window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
        positionTimer.current = setTimeout(() => { lockScroll(); updatePositions(); }, 650);
        return;
      }
    }

    lockScroll();
    updatePositions();
    return () => clearTimeout(positionTimer.current);
  }, [step, open, updatePositions]);

  // Unlock + deactivate tour when closed; activate when opened
  useEffect(() => {
    if (open) { setTourActive(true); } else { unlockScroll(); setTourActive(false); }
  }, [open, setTourActive]);

  // Keep store in sync with current step so other components can react
  useEffect(() => { setTourStep(step); }, [step, setTourStep]);
  useEffect(() => () => { unlockScroll(); setTourActive(false); }, [setTourActive]);

  // Recompute on resize
  useEffect(() => {
    if (!open) return;
    const h = () => { clearTimeout(positionTimer.current); positionTimer.current = setTimeout(updatePositions, 80); };
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [open, updatePositions]);

  // ── Keyboard navigation ───────────────────────────────────────────────────
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape")     handleClose();
      if (e.key === "ArrowRight") handleNext();
      if (e.key === "ArrowLeft")  handlePrev();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, step]);

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleNext  = () => step < STEPS.length - 1 ? setStep(s => s + 1) : handleClose();
  const handlePrev  = () => { if (step > 0) setStep(s => s - 1); };
  const handleClose = () => {
    unlockScroll();
    setOpen(false);
    setSpotlight(null);
    // Don't persist in preview mode — ?tour should always re-trigger on reload
    if (!previewMode) {
      try { localStorage.setItem(STORAGE_KEY, "1"); } catch { /* ignore */ }
    }
  };
  const startTour   = () => { setStep(0); setOpen(true); };

  if (!mounted) return null;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Tour trigger button — only shown in ?tour preview mode */}
      {previewMode && (
        <button
          onClick={startTour}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-600 hover:text-indigo-800 border border-indigo-200 hover:border-indigo-400 rounded-full px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 transition-all"
          title="Start interactive tour"
        >
          <Map size={12} />
          Tour
        </button>
      )}

      {/* Portal overlay */}
      {open && typeof document !== "undefined" && createPortal(
        <>
          {/* Dimmed SVG backdrop with spotlight cutout */}
          <svg
            className="fixed inset-0 pointer-events-none"
            style={{ width: "100vw", height: "100vh", zIndex: 9990 }}
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <mask id="tour-spotlight-mask">
                <rect width="100%" height="100%" fill="white" />
                {spotlight && (
                  <rect x={spotlight.x} y={spotlight.y}
                    width={spotlight.width} height={spotlight.height}
                    fill="black" rx={10} />
                )}
              </mask>
            </defs>
            <rect width="100%" height="100%" fill="rgba(0,0,0,0.56)" mask="url(#tour-spotlight-mask)" />
            {spotlight && (
              <rect x={spotlight.x} y={spotlight.y}
                width={spotlight.width} height={spotlight.height}
                fill="none" stroke="rgba(99,102,241,0.75)" strokeWidth={2} rx={10} />
            )}
          </svg>

          {/* Tooltip card */}
          <div
            style={{ ...tooltipStyle, zIndex: 9991 }}
            className="bg-white rounded-2xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden"
          >
            {/* Progress bar */}
            <div className="h-1 bg-gray-100 shrink-0">
              <div
                className="h-full bg-indigo-500 transition-all duration-300 ease-in-out"
                style={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
              />
            </div>

            {/* Scrollable inner content */}
            <div className="overflow-y-auto flex-1 p-5">
              {/* Dot nav + close */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-1.5">
                  {STEPS.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setStep(i)}
                      className={`transition-all rounded-full ${
                        i === step
                          ? "w-4 h-1.5 bg-indigo-500"
                          : i < step
                          ? "w-1.5 h-1.5 bg-indigo-300 hover:bg-indigo-400"
                          : "w-1.5 h-1.5 bg-gray-200 hover:bg-gray-300"
                      }`}
                      title={`Step ${i + 1}: ${STEPS[i].title}`}
                    />
                  ))}
                </div>
                <button
                  onClick={handleClose}
                  className="w-6 h-6 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 hover:text-gray-700 transition-colors"
                  title="Close (Esc)"
                >
                  <X size={13} />
                </button>
              </div>

              {/* Title */}
              <h3 className="text-base font-semibold text-gray-900 mb-2.5">
                {STEPS[step].title}
              </h3>

              {/* Body */}
              <div className="text-sm text-gray-700">{STEPS[step].body}</div>

              {/* Footer nav */}
              <div className="flex items-center justify-between mt-5 pt-4 border-t border-gray-100">
                <button
                  onClick={handleClose}
                  className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                >
                  Skip tour
                </button>
                <div className="flex items-center gap-2">
                  {step > 0 && (
                    <button
                      onClick={handlePrev}
                      className="flex items-center gap-1 text-xs border border-gray-200 rounded-lg px-3 py-1.5 hover:bg-gray-50 text-gray-600 transition-colors"
                      title="Previous (←)"
                    >
                      <ChevronLeft size={13} /> Back
                    </button>
                  )}
                  <button
                    onClick={handleNext}
                    className="flex items-center gap-1 text-xs bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg px-4 py-1.5 font-medium transition-colors shadow-sm"
                    title={step === STEPS.length - 1 ? "Finish" : "Next (→)"}
                  >
                    {step === STEPS.length - 1
                      ? "Done"
                      : <> Next <ChevronRight size={13} /></>}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>,
        document.body
      )}
    </>
  );
}
