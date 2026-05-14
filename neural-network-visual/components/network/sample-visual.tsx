"use client";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { NetworkState } from "@/components/network/static/types";

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const norm = (v: number, lo: number, hi: number) => clamp01((v - lo) / (hi - lo));

// ─── Iris Flower ──────────────────────────────────────────────────────────────
const IRIS_CLASSES = ["Setosa", "Versicolor", "Virginica"] as const;
type IrisClass = (typeof IRIS_CLASSES)[number];

const IRIS_PETAL: Record<IrisClass, string> = {
  Setosa: "#fda4af",
  Versicolor: "#c084fc",
  Virginica: "#818cf8",
};
const IRIS_BADGE: Record<IrisClass, string> = {
  Setosa: "#f43f5e",
  Versicolor: "#a855f7",
  Virginica: "#6366f1",
};

const IrisFlower = ({
  original,
  outputActivations,
}: {
  original: number[];
  outputActivations: number[];
}) => {
  const [hovered, setHovered] = useState(false);
  const [sepalLen, sepalWid, petalLen, petalWid] = original;

  const predIdx =
    outputActivations.length > 0
      ? outputActivations.indexOf(Math.max(...outputActivations))
      : -1;
  const predClass = IRIS_CLASSES[predIdx] as IrisClass | undefined;
  const predConf = predIdx >= 0 ? (outputActivations[predIdx] ?? 0) : 0;
  const predColor = predClass ? IRIS_PETAL[predClass] : "#d1d5db";
  const predBadge = predClass ? IRIS_BADGE[predClass] : "#6b7280";

  const labelIdx = original.slice(4, 7).indexOf(1);
  const actualClass = IRIS_CLASSES[labelIdx] as IrisClass | undefined;
  const isCorrect = predIdx >= 0 && predIdx === labelIdx;

  const petalRY = 9 + norm(petalLen, 1.0, 6.9) * 21;
  const petalRX = 3 + norm(petalWid, 0.1, 2.5) * 9;
  const sepalRY = 5 + norm(sepalLen, 4.3, 7.9) * 11;
  const sepalRX = 2.5 + norm(sepalWid, 2.0, 4.4) * 6;

  const cx = 60;
  const cy = 58;
  const ringR = Math.max(petalRY, sepalRY) + 9;

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="relative cursor-default"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <svg viewBox="0 0 120 116" className="w-[96px] h-[96px]">
          {/* Correctness ring */}
          <circle
            cx={cx}
            cy={cy}
            r={ringR}
            fill="none"
            stroke={
              isCorrect ? "#22c55e" : predIdx >= 0 ? "#ef4444" : "#d1d5db"
            }
            strokeWidth={1.5}
            strokeDasharray={isCorrect ? undefined : "4 3"}
            opacity={0.55}
          />
          {/* Sepals */}
          {[45, 135, 225, 315].map((angle) => (
            <ellipse
              key={angle}
              cx={cx}
              cy={cy - sepalRY - 2}
              rx={sepalRX}
              ry={sepalRY}
              fill={predColor}
              opacity={0.35}
              transform={`rotate(${angle}, ${cx}, ${cy})`}
            />
          ))}
          {/* Petals */}
          {[0, 90, 180, 270].map((angle) => (
            <ellipse
              key={angle}
              cx={cx}
              cy={cy - petalRY - 3}
              rx={petalRX}
              ry={petalRY}
              fill={predColor}
              opacity={0.85}
              transform={`rotate(${angle}, ${cx}, ${cy})`}
            />
          ))}
          {/* Stamen */}
          <circle
            cx={cx}
            cy={cy}
            r={5}
            fill="#fef08a"
            stroke="#ca8a04"
            strokeWidth={1}
          />
        </svg>

        {hovered && (
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-10 bg-gray-900 text-white text-xs rounded-lg px-2.5 py-1.5 whitespace-nowrap shadow-lg pointer-events-none">
            <div>
              Sepal {sepalLen?.toFixed(1)} × {sepalWid?.toFixed(1)} cm
            </div>
            <div>
              Petal {petalLen?.toFixed(1)} × {petalWid?.toFixed(1)} cm
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        {predClass ? (
          <span
            className="px-2 py-0.5 rounded-full text-white text-[10px] font-semibold"
            style={{ backgroundColor: predBadge }}
          >
            {predClass} {(predConf * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 text-[10px]">
            Not trained
          </span>
        )}
        {isCorrect ? (
          <span className="text-green-600 font-bold text-[11px]">✓</span>
        ) : (
          actualClass && (
            <span className="text-[10px] text-gray-500">
              actual: {actualClass}
            </span>
          )
        )}
      </div>
    </div>
  );
};

// ─── XOR Gate ─────────────────────────────────────────────────────────────────
const XorGate = ({
  original,
  outputActivations,
}: {
  original: number[];
  outputActivations: number[];
}) => {
  const inputA = Math.round(original[0] ?? 0);
  const inputB = Math.round(original[1] ?? 0);
  const actual = Math.round(original[2] ?? 0);

  const rawOut = outputActivations[0] ?? -1;
  const hasPred = rawOut >= 0;
  const predicted = rawOut >= 0.5 ? 1 : 0;
  const confidence = predicted === 1 ? rawOut : 1 - rawOut;
  const isCorrect = hasPred && predicted === actual;

  const gateStroke = !hasPred ? "#9ca3af" : isCorrect ? "#22c55e" : "#ef4444";
  const inputFill = (v: number) => (v === 1 ? "#1e3a8a" : "#f9fafb");
  const inputText = (v: number) => (v === 1 ? "white" : "#374151");

  return (
    <div className="flex flex-col items-center gap-2">
      <svg viewBox="0 0 208 92" className="w-[172px] h-[76px]">
        {/* Input A */}
        <circle
          cx={18}
          cy={28}
          r={15}
          fill={inputFill(inputA)}
          stroke="#94a3b8"
          strokeWidth={1.5}
        />
        <text
          x={18}
          y={24}
          textAnchor="middle"
          fontSize={8}
          fill={inputText(inputA)}
          fontWeight="bold"
        >
          A
        </text>
        <text
          x={18}
          y={35}
          textAnchor="middle"
          fontSize={12}
          fill={inputText(inputA)}
          fontWeight="bold"
        >
          {inputA}
        </text>

        {/* Input B */}
        <circle
          cx={18}
          cy={64}
          r={15}
          fill={inputFill(inputB)}
          stroke="#94a3b8"
          strokeWidth={1.5}
        />
        <text
          x={18}
          y={60}
          textAnchor="middle"
          fontSize={8}
          fill={inputText(inputB)}
          fontWeight="bold"
        >
          B
        </text>
        <text
          x={18}
          y={71}
          textAnchor="middle"
          fontSize={12}
          fill={inputText(inputB)}
          fontWeight="bold"
        >
          {inputB}
        </text>

        {/* Input wires */}
        <line x1={33} y1={28} x2={53} y2={30} stroke="#94a3b8" strokeWidth={1.5} />
        <line x1={33} y1={64} x2={53} y2={62} stroke="#94a3b8" strokeWidth={1.5} />

        {/* XOR extra arch — drawn before gate body so fill covers the inner overlap */}
        <path
          d="M 56,14 Q 38,46 56,78"
          fill="none"
          stroke={gateStroke}
          strokeWidth={2}
        />
        {/* Gate body (OR shape) */}
        <path
          d="M 56,14 Q 46,46 56,78 Q 92,78 130,46 Q 92,14 56,14"
          fill="white"
          stroke={gateStroke}
          strokeWidth={2}
        />

        {/* Output wire */}
        <line x1={130} y1={46} x2={168} y2={46} stroke={gateStroke} strokeWidth={1.5} />

        {/* Output circle */}
        <circle
          cx={186}
          cy={46}
          r={16}
          fill={
            !hasPred ? "#f3f4f6" : isCorrect ? "#dcfce7" : "#fee2e2"
          }
          stroke={gateStroke}
          strokeWidth={2}
        />
        <text
          x={186}
          y={42}
          textAnchor="middle"
          fontSize={8}
          fill={gateStroke}
          fontWeight="bold"
        >
          out
        </text>
        <text
          x={186}
          y={54}
          textAnchor="middle"
          fontSize={13}
          fill={gateStroke}
          fontWeight="bold"
        >
          {hasPred ? predicted : "?"}
        </text>
      </svg>

      <div className="flex items-center gap-2">
        {hasPred ? (
          <span
            className={`px-2 py-0.5 rounded-full font-semibold text-[10px] ${
              isCorrect
                ? "bg-green-100 text-green-700"
                : "bg-red-100 text-red-700"
            }`}
          >
            {predicted} · {(confidence * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 text-[10px]">
            Not trained
          </span>
        )}
        {hasPred &&
          (isCorrect ? (
            <span className="text-green-600 text-[11px]">✓ correct</span>
          ) : (
            <span className="text-[10px] text-gray-500">actual: {actual}</span>
          ))}
      </div>
    </div>
  );
};

// ─── Auto MPG Car ─────────────────────────────────────────────────────────────
const mpgToColor = (mpg: number): string => {
  if (mpg < 18) return "#ef4444";
  if (mpg < 25) return "#f97316";
  if (mpg < 32) return "#eab308";
  if (mpg < 38) return "#84cc16";
  return "#22c55e";
};

const AutoMpgCar = ({
  original,
  outputActivations,
  yMean,
  yStd,
}: {
  original: number[];
  outputActivations: number[];
  yMean: number | null;
  yStd: number | null;
}) => {
  const disp  = original[0]; // 68–455 cu.in.
  const hp    = original[1]; // 46–230 hp
  const wt    = original[2]; // 1613–5140 lbs
  const accel = original[3]; // 8–24.8 s (0-60)
  const actualMpg = original[4];

  const dispN  = norm(disp,  68,   455);
  const hpN    = norm(hp,    46,   230);
  const wtN    = norm(wt,    1613, 5140);
  const accelN = norm(accel, 8,    24.8);

  const rawPred = outputActivations[0] ?? -1;
  const hasPred = rawPred >= 0;
  const predMpg =
    hasPred && yMean != null && yStd != null
      ? rawPred * yStd + yMean
      : hasPred
      ? rawPred
      : null;

  const bodyColor = predMpg != null ? mpgToColor(predMpg) : "#9ca3af";
  const delta = predMpg != null ? predMpg - actualMpg : null;

  // Car geometry (viewBox 0 0 230 100)
  const groundY     = 82;
  const wheelR      = 11 + hpN * 7;        // 11–18 → bigger wheels = more HP
  const rearWheelX  = 50;
  const frontWheelX = 158;

  const chassisBottom = groundY - wheelR * 0.7;
  const chassisH      = 11 + wtN * 11;     // 11–22 → heavier = taller body
  const chassisTop    = chassisBottom - chassisH;
  const chassisLeft   = rearWheelX - 20;
  const chassisRight  = frontWheelX + 18;

  const hoodLen  = 15 + dispN * 20;        // 15–35 → bigger engine = longer hood
  const hoodTopY = chassisTop + chassisH * 0.38;

  const trunkLen  = 12;
  const trunkTopY = chassisTop + chassisH * 0.55;

  const cabinLeft  = rearWheelX + 8;
  const cabinRight = chassisRight - hoodLen;
  const cabinH     = 16 + accelN * 12;     // 16–28 → slower car = taller boxy cabin
  const cabinTop   = chassisTop - cabinH;
  // Sporty (low accelN) → roof more inset at front; boxy (high accelN) → flat roof
  const roofInsetF = 8 + (1 - accelN) * 8;
  const roofInsetR = 5 + accelN * 3;
  const roofRight  = cabinRight - roofInsetF;
  const roofLeft   = cabinLeft + roofInsetR;

  const cabinPts = [
    `${cabinLeft},${chassisTop}`,
    `${cabinRight},${chassisTop}`,
    `${roofRight},${cabinTop}`,
    `${roofLeft},${cabinTop}`,
  ].join(" ");

  const features = [
    { label: "Displacement", val: disp,  n: dispN,  unit: "cu.in" },
    { label: "Horsepower",   val: hp,    n: hpN,    unit: "hp"    },
    { label: "Weight",       val: wt,    n: wtN,    unit: "lbs"   },
    { label: "Accel 0-60",   val: accel, n: accelN, unit: "s"     },
  ];

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex gap-3 items-center">
        <svg viewBox="0 0 230 100" className="w-[158px] h-[70px] flex-shrink-0">
          {/* Trunk */}
          <rect
            x={chassisLeft - trunkLen}
            y={trunkTopY}
            width={trunkLen}
            height={chassisBottom - trunkTopY}
            fill={bodyColor}
            opacity={0.65}
            rx={2}
          />
          {/* Chassis */}
          <rect
            x={chassisLeft}
            y={chassisTop}
            width={chassisRight - chassisLeft}
            height={chassisH}
            fill={bodyColor}
            rx={3}
          />
          {/* Hood */}
          <rect
            x={chassisRight}
            y={hoodTopY}
            width={hoodLen}
            height={chassisBottom - hoodTopY}
            fill={bodyColor}
            opacity={0.78}
            rx={2}
          />
          {/* Cabin */}
          <polygon points={cabinPts} fill={bodyColor} opacity={0.92} />
          {/* Windshield glint */}
          <line
            x1={cabinRight - 2}
            y1={chassisTop - 1}
            x2={roofRight - 2}
            y2={cabinTop + 3}
            stroke="rgba(255,255,255,0.45)"
            strokeWidth={3}
            strokeLinecap="round"
          />
          {/* Rear wheel */}
          <circle cx={rearWheelX}  cy={groundY} r={wheelR}          fill="#1f2937" />
          <circle cx={rearWheelX}  cy={groundY} r={wheelR * 0.42}   fill="#6b7280" />
          {/* Front wheel */}
          <circle cx={frontWheelX} cy={groundY} r={wheelR}          fill="#1f2937" />
          <circle cx={frontWheelX} cy={groundY} r={wheelR * 0.42}   fill="#6b7280" />
        </svg>

        {/* MPG panel */}
        <div className="flex flex-col gap-0.5 text-xs min-w-[72px]">
          <div className="text-[9px] text-gray-400 uppercase tracking-wide font-medium">
            Pred. MPG
          </div>
          <div
            className="px-1.5 py-0.5 rounded font-bold text-white text-sm leading-tight"
            style={{ backgroundColor: bodyColor }}
          >
            {predMpg != null ? predMpg.toFixed(1) : "—"}
          </div>
          <div className="text-[9px] text-gray-500 mt-0.5">
            Actual: {actualMpg?.toFixed(1)}
          </div>
          {delta != null && (
            <div
              className={`text-[9px] font-semibold ${
                Math.abs(delta) < 2.5 ? "text-green-600" : "text-red-500"
              }`}
            >
              {delta >= 0 ? "+" : ""}
              {delta.toFixed(1)} Δ
            </div>
          )}
        </div>
      </div>

      {/* Feature bars */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        {features.map(({ label, val, n, unit }) => (
          <div key={label} className="flex flex-col gap-0.5">
            <div className="flex justify-between text-[10px] text-gray-500">
              <span>{label}</span>
              <span className="font-mono text-gray-700">
                {val?.toFixed(0)}&thinsp;{unit}
              </span>
            </div>
            <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gray-400 rounded-full"
                style={{ width: `${n * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── MNIST Digit ──────────────────────────────────────────────────────────────
const MnistDigit = ({
  original,
  outputActivations,
  onDrawOwn,
}: {
  original: number[];
  outputActivations: number[];
  onDrawOwn?: () => void;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pixels = useMemo(() => original.slice(0, 784), [original]);
  const actualLabels = original.slice(784);
  const actual = actualLabels.length >= 10
    ? actualLabels.indexOf(Math.max(...actualLabels))
    : Math.round(original[original.length - 1]);

  const predIdx =
    outputActivations.length > 0
      ? outputActivations.indexOf(Math.max(...outputActivations))
      : -1;
  const predConf = predIdx >= 0 ? (outputActivations[predIdx] ?? 0) : 0;
  const isCorrect = predIdx >= 0 && predIdx === actual;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round((pixels[i] ?? 0) * 255);
      imageData.data[i * 4 + 0] = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pixels]);

  return (
    <div className="flex gap-3 items-center">
      <canvas
        ref={canvasRef}
        width={28}
        height={28}
        className="w-[84px] h-[84px] rounded border border-gray-200 flex-shrink-0"
        style={{ imageRendering: "pixelated", background: "#000" }}
      />

      <div className="flex flex-col gap-2 flex-1">
        <div className="flex flex-col gap-0.5">
          {predIdx >= 0 ? (
            <>
              <div className="flex items-end gap-1">
                <span
                  className={`text-3xl font-bold leading-none ${
                    isCorrect ? "text-green-600" : "text-red-500"
                  }`}
                >
                  {predIdx}
                </span>
                <span className="text-[10px] text-gray-400 pb-0.5">
                  predicted
                </span>
              </div>
              <div className="text-[10px] text-gray-500">
                {(predConf * 100).toFixed(1)}% confidence
              </div>
              <div
                className={`text-[10px] font-medium ${
                  isCorrect ? "text-green-600" : "text-red-500"
                }`}
              >
                {isCorrect ? "✓" : "✗"} actual: {actual}
              </div>
            </>
          ) : (
            <div className="text-[10px] text-gray-400">
              Not trained — actual: {actual}
            </div>
          )}
        </div>

        {onDrawOwn && (
          <button
            onClick={onDrawOwn}
            className="text-xs px-2.5 py-1.5 rounded-lg border border-gray-200 bg-white hover:bg-gray-50 hover:border-gray-300 transition-colors text-gray-700 font-medium flex items-center gap-1.5 w-fit"
          >
            ✏ Draw your own
          </button>
        )}
      </div>
    </div>
  );
};

// ─── Root export ──────────────────────────────────────────────────────────────
interface SampleVisualProps {
  dataset: string;
  original: number[];
  network: NetworkState | null;
  sampleIndex: number;
  yMean: number | null;
  yStd: number | null;
  onDrawOwn?: () => void;
}

export const SampleVisual: React.FC<SampleVisualProps> = ({
  dataset,
  original,
  network,
  sampleIndex,
  yMean,
  yStd,
  onDrawOwn,
}) => {
  if (!original || original.length === 0) return null;

  // Mirrors the output-node activation extraction used in network.tsx renderResults
  const outputLayerIdx = network ? network.layers.length - 2 : -1;
  const outputActivations: number[] =
    outputLayerIdx >= 0
      ? (network!.layers[outputLayerIdx]?.A?.[sampleIndex] ?? [])
      : [];

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3">
      <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-2.5">
        Sample Preview
      </p>

      {dataset === "iris" && (
        <IrisFlower original={original} outputActivations={outputActivations} />
      )}
      {dataset === "xor" && (
        <XorGate original={original} outputActivations={outputActivations} />
      )}
      {dataset === "auto_mpg" && (
        <AutoMpgCar
          original={original}
          outputActivations={outputActivations}
          yMean={yMean}
          yStd={yStd}
        />
      )}
      {dataset === "mnist" && (
        <MnistDigit
          original={original}
          outputActivations={outputActivations}
          onDrawOwn={onDrawOwn}
        />
      )}
    </div>
  );
};
