import React from "react";
import { NetworkState, HoveredConnection, HoveredNode } from "@/components/network/static/types";
import { DATASET_INPUT_FEATURES, DATASET_INPUT_FEATURES_SHORT } from "@/components/network/static/constants";
import { renderResults, renderInputValues } from "@/components/network/results";

interface FlashConnection {
  li: number;
  fi: number;
  ti: number;
  delta: number;
  positive: boolean;
}

// --------------------
// Type Definitions
// --------------------
interface NetworkSVGProps {
  SVGWIDTH: number;
  SVGHEIGHT: number;
  network: NetworkState | null;
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
  sampleIndex?: number;
  dataset?: string;
  original?: number[];
  flashConnections?: FlashConnection[];
  flashKey?: number;
  stepLayerHighlight?: number | null;
  yMean?: number | null;
  yStd?: number | null;
  drawnPixels?: number[] | null;
}

// --------------------
// Utility Functions
// --------------------
const computeLayout = (SVGWIDTH: number, layerCount: number) => {
  const layerSpacing = SVGWIDTH / (layerCount + 1) - (layerCount > 4 ? 15 : 0);
  const SHIFT = layerCount > 4 ? 40 : 0;
  return { layerSpacing, SHIFT };
};

const fontSizeForWidth = (w: number) => (w < 450 ? 8 : w < 600 ? 11 : 14);

const featureList = (dataset?: string, w?: number): string[] => {
  if (!dataset) return [];
  return (w ?? 800) < 800
    ? DATASET_INPUT_FEATURES_SHORT[dataset] || []
    : DATASET_INPUT_FEATURES[dataset] || [];
};

const nodeColors = (isInput: boolean, isOutput: boolean, val: number) => ({
  fill: isInput
    ? `hsl(210,100%,${85 - val * 50}%)`
    : isOutput
    ? `hsl(0,70%,${80 - val * 40}%)`
    : `hsl(215,30%,${85 - val * 50}%)`,
  stroke: isOutput ? "#7f1d1d" : "#475569",
});

// --------------------
// MNIST: pixel grid + single representative connections
// --------------------
const MnistInputGrid: React.FC<{
  pixels: number[];  // 784 values 0-1
  cx: number;        // x of the input column — grid right-aligns here
  SVGHEIGHT: number;
}> = ({ pixels, cx, SVGHEIGHT }) => {
  const GRID = 28;
  // Fill the available left space: right-align grid to cx, cell size fills the room
  const CELL = Math.max(2, Math.min(5, Math.floor((cx - 8) / GRID)));
  const SIZE = GRID * CELL;
  const x0 = cx - SIZE;                      // right edge flush with input column
  const y0 = SVGHEIGHT / 2 - SIZE / 2;

  return (
    <g>
      {pixels.map((v, i) => {
        const row = Math.floor(i / GRID);
        const col = i % GRID;
        const gray = Math.round((1 - v) * 255);
        return (
          <rect
            key={i}
            x={x0 + col * CELL}
            y={y0 + row * CELL}
            width={CELL}
            height={CELL}
            fill={`rgb(${gray},${gray},${gray})`}
          />
        );
      })}
      <rect x={x0} y={y0} width={SIZE} height={SIZE} fill="none" stroke="#475569" strokeWidth={1} rx={1} />
    </g>
  );
};

const MnistConnections: React.FC<{
  inputWeights: number[][];  // shape [784, hidden_size] — layer 0 weights
  inputCx: number;
  inputCy: number; // center y of the grid
  SVGWIDTH: number;
  SVGHEIGHT: number;
  layerSpacing: number;
  SHIFT: number;
  hiddenSize: number;
  stepLayerHighlight?: number | null;
}> = ({ inputWeights, inputCx, SVGWIDTH, SVGHEIGHT, layerSpacing, SHIFT, hiddenSize, stepLayerHighlight }) => {
  const GRID = 28;
  const CELL = Math.max(2, Math.min(5, Math.floor((inputCx - 8) / GRID)));
  const SIZE = GRID * CELL;
  const gridRight = inputCx; // right edge of grid is flush with input column
  const inStepMode = stepLayerHighlight !== null && stepLayerHighlight !== undefined;
  const isActive = !inStepMode || stepLayerHighlight === 0 || stepLayerHighlight === 1;

  return (
    <>
      {Array.from({ length: hiddenSize }, (_, ti) => {
        // mean absolute weight for this hidden node across all 784 inputs
        const col = inputWeights.map((row) => Math.abs(row[ti] ?? 0));
        const meanW = col.length > 0 ? col.reduce((a, b) => a + b, 0) / col.length : 0;
        const toX = 2 * layerSpacing + SHIFT;
        const toY = ((ti + 1) * SVGHEIGHT) / (hiddenSize + 1);
        const fromY = SVGHEIGHT / 2;
        const color = `rgba(99,102,241,${(0.15 + Math.min(meanW * 10, 1) * 0.7).toFixed(2)})`;
        return (
          <line
            key={`mnist-conn-${ti}`}
            x1={gridRight}
            y1={fromY}
            x2={toX}
            y2={toY}
            stroke={color}
            strokeWidth={Math.min(meanW * 20, 1) * (SVGWIDTH > 600 ? 2.5 : 1.8) + 1}
            opacity={inStepMode ? (isActive ? 1 : 0.08) : 0.7}
          />
        );
      })}
    </>
  );
};

// --------------------
// Subcomponents
// --------------------
const ConnectionLines: React.FC<{
  network: NetworkState;
  SVGWIDTH: number;
  SVGHEIGHT: number;
  onClick: (hovered: HoveredConnection | null) => void;
  flashConnections?: FlashConnection[];
  flashKey?: number;
  stepLayerHighlight?: number | null;
  dataset?: string;
}> = ({ network, SVGWIDTH, SVGHEIGHT, onClick, flashConnections = [], flashKey = 0, stepLayerHighlight, dataset }) => {
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const inStepMode = stepLayerHighlight !== null && stepLayerHighlight !== undefined;

  return (
    <>
      {network.layers.flatMap((layer, li) => {
        // MNIST: skip drawing the 784×hidden connections from layer 0 — MnistConnections handles it
        if (dataset === "mnist" && li === 0) return [];
        return layer.weights.flatMap((weights: number[], fi: number) =>
          weights.map((w: number, ti: number) => {
            const next = network.layers[li + 1];
            const fromX = (li + 1) * layerSpacing + SHIFT;
            const fromY = ((fi + 1) * SVGHEIGHT) / (layer.size + 1);
            const toX = next ? (li + 2) * layerSpacing + SHIFT : fromX;
            const toY = next ? ((ti + 1) * SVGHEIGHT) / (next.size + 1) : fromY;

            // Dim connection if in step mode and not connected to highlighted layer
            const isActiveConn = !inStepMode || li === stepLayerHighlight || li + 1 === stepLayerHighlight;
            const connOpacity = inStepMode ? (isActiveConn ? 1 : 0.08) : 1;

            // Check if this connection is in the flash list
            const flashEntry = flashConnections.find(
              (fc) => fc.li === li && fc.fi === fi && fc.ti === ti
            );

            const wAbs = Math.abs(w);
            const isMnist = dataset === "mnist";
            const strokeScale = isMnist ? (SVGWIDTH > 600 ? 1.2 : 0.8) : (SVGWIDTH > 600 ? 2.5 : 1.8);
            const strokeBase = isMnist ? 0.6 : 1.8;
            const baseColor = w >= 0
              ? `rgba(99,102,241,${(0.12 + Math.min(wAbs, 1) * 0.75).toFixed(2)})`
              : `rgba(249,115,22,${(0.12 + Math.min(wAbs, 1) * 0.75).toFixed(2)})`;

            return (
              <g key={`conn-${li}-${fi}-${ti}`} opacity={connOpacity}>
                {/* Base connection line */}
                <line
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke={baseColor}
                  strokeWidth={wAbs * strokeScale + strokeBase}
                />
                {/* Flash overlay for most-changed connections */}
                {flashEntry && (
                  <line
                    key={`flash-${li}-${fi}-${ti}-${flashKey}`}
                    x1={fromX}
                    y1={fromY}
                    x2={toX}
                    y2={toY}
                    stroke={flashEntry.positive ? "#22c55e" : "#ef4444"}
                    strokeWidth={Math.abs(w) * (SVGWIDTH > 600 ? 3 : 2) + 3}
                    strokeOpacity={0.85}
                    className="weight-flash"
                  />
                )}
                {/* Hit zone */}
                <line
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke="transparent"
                  strokeWidth={15}
                  cursor="pointer"
                  onClick={() => onClick({ layerIndex: li, fromIndex: fi, toIndex: ti, weight: w })}
                />
              </g>
            );
          })
        );
      })}
    </>
  );
};


const NodeCircles: React.FC<{
  network: NetworkState;
  SVGWIDTH: number;
  SVGHEIGHT: number;
  dataset?: string;
  sampleIndex?: number;
  original?: number[];
  onNodeClick: (hovered: HoveredNode | null) => void;
  stepLayerHighlight?: number | null;
  yMean?: number | null;
  yStd?: number | null;
}> = ({ network, SVGWIDTH, SVGHEIGHT, dataset, sampleIndex, original, onNodeClick, stepLayerHighlight, yMean, yStd }) => {
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const fontSize = fontSizeForWidth(SVGWIDTH);
  const features = featureList(dataset, SVGWIDTH);
  const isMnistDataset = dataset === "mnist";
  const nodeRadius = isMnistDataset ? 5 + SVGWIDTH / 180 : 10 + SVGWIDTH / 100;
  const INPUTLABELOFFSET = SVGWIDTH / 10 + 10;
  const inStepMode = stepLayerHighlight !== null && stepLayerHighlight !== undefined;

  return (
    <>
      {network.layers.flatMap((layer, li) => {
        const total = layer.size;
        const isInput = li === 0;
        const isOutput = li === network.layers.length - 1;
        const isActiveLayer = !inStepMode || li === stepLayerHighlight || li === stepLayerHighlight + 1;
        const layerOpacity = inStepMode ? (isActiveLayer ? 1 : 0.15) : 1;

        // For MNIST, skip rendering the 784 input nodes — shown as pixel grid instead
        if (isInput && dataset === "mnist") return [];

        return Array.from({ length: total }, (_, ni) => {
          const cx = (li + 1) * layerSpacing + SHIFT;
          const cy = ((ni + 1) * SVGHEIGHT) / (total + 1);
          const inputSample = network.input?.[sampleIndex ?? 0];
          const activationValue =
            (isInput
              ? inputSample?.[ni]
              : network.layers[li - 1]?.A?.[sampleIndex ?? 0]?.[ni]) ?? 0;

          const clamped = Math.min(1, Math.abs(activationValue));
          const { fill, stroke } = nodeColors(isInput, isOutput, clamped);
          const r = nodeRadius + clamped * 3;

          return (
            <g key={`node-${li}-${ni}`} opacity={layerOpacity}>
              <circle cx={cx} cy={cy} r={r} fill={fill} stroke={stroke} strokeWidth={2} />
              <circle
                cx={cx}
                cy={cy}
                r={r + 10}
                fill="transparent"
                cursor="pointer"
                onClick={() => onNodeClick({ layerIndex: li, nodeIndex: ni })}
              />
              {isInput && (
                <text
                  x={cx - INPUTLABELOFFSET}
                  y={cy}
                  fontSize={fontSize}
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {dataset === "xor" && inputSample
                    ? `${features[ni]} (${Math.round(inputSample[ni])})`
                    : features[ni]}
                </text>
              )}
              {isInput && dataset !== "xor" && original && activationValue && original.length > 0 &&
                renderInputValues(cx, cy, original, ni, activationValue, fontSize, INPUTLABELOFFSET, SVGWIDTH, SVGHEIGHT)
              }
              {isOutput && dataset && (() => {
                let isPredicted: boolean | undefined;
                if (dataset === "iris" || dataset === "mnist") {
                  const outputActivations = network.layers[li - 1]?.A?.[sampleIndex ?? 0] ?? [];
                  const predIdx = outputActivations.reduce((best, v, i) => v > outputActivations[best] ? i : best, 0);
                  isPredicted = ni === predIdx;
                }
                return renderResults({ SVGWIDTH, SVGHEIGHT, cx, cy, dataset, ni, activationValue, original, fontSize, yMean, yStd, isPredicted });
              })()}
            </g>
          );
        });
      })}
    </>
  );
};

const LayerLabels: React.FC<{ network: NetworkState; SVGWIDTH: number; SVGHEIGHT: number; dataset?: string }> = ({
  network,
  SVGWIDTH,
  SVGHEIGHT,
  dataset,
}) => {
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const fontSize = fontSizeForWidth(SVGWIDTH);

  return (
    <>
      {network.layers.map((layer, i) => (
        <text
          key={`label-${i}`}
          x={(i + 1) * layerSpacing + SHIFT}
          y={SVGHEIGHT * 0.97}
          fontSize={fontSize}
          fontWeight="bold"
          textAnchor="middle"
        >
          {layer.name}
        </text>
      ))}
      {dataset !== "mnist" && (
        <>
          <text
            key={`inputlabel-og`}
            x={(SVGWIDTH * 0.065 + 10)}
            y={SVGHEIGHT * 0.04}
            textAnchor="middle"
            fontSize={fontSize}
            fontWeight="bold"
          >
            {`Input Values`}
          </text>
          <text
            key={`inputlabel-normal`}
            x={(SVGWIDTH * 0.065 + 10) + (SVGWIDTH > 600 ? 90 : SVGWIDTH * 0.135)}
            y={SVGHEIGHT * 0.04}
            textAnchor="middle"
            fontSize={fontSize}
          >
            {`(Normalized)`}
          </text>
        </>
      )}
    </>
  );
};

// --------------------
// Main Component
// --------------------
export const Network: React.FC<NetworkSVGProps> = ({
  SVGWIDTH,
  SVGHEIGHT,
  network,
  setHoveredConnection,
  setHoveredNode,
  sampleIndex,
  dataset,
  original,
  flashConnections = [],
  flashKey = 0,
  stepLayerHighlight,
  yMean,
  yStd,
  drawnPixels,
}) => {
  if (!network) return null;

  const isMnist = dataset === "mnist";
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const inputCx = layerSpacing + SHIFT;

  // Pixel data to display: prefer drawn pixels, fall back to training sample
  const inputSample = network.input?.[sampleIndex ?? 0];
  const displayPixels: number[] =
    drawnPixels && drawnPixels.length === 784
      ? drawnPixels
      : inputSample?.length === 784
      ? Array.from(inputSample)
      : new Array(784).fill(0);

  return (
    <>
      <ConnectionLines
        SVGWIDTH={SVGWIDTH}
        SVGHEIGHT={SVGHEIGHT}
        network={network}
        onClick={setHoveredConnection}
        flashConnections={flashConnections}
        flashKey={flashKey}
        stepLayerHighlight={stepLayerHighlight}
        dataset={dataset}
      />
      {isMnist && (
        <>
          <MnistInputGrid pixels={displayPixels} cx={inputCx} SVGHEIGHT={SVGHEIGHT} />
          {network.layers[0]?.weights?.length > 0 && network.layers[1] && (
            <MnistConnections
              inputWeights={network.layers[0].weights}
              inputCx={inputCx}
              inputCy={SVGHEIGHT / 2}
              SVGWIDTH={SVGWIDTH}
              SVGHEIGHT={SVGHEIGHT}
              layerSpacing={layerSpacing}
              SHIFT={SHIFT}
              hiddenSize={network.layers[1].size}
              stepLayerHighlight={stepLayerHighlight}
            />
          )}
        </>
      )}
      <NodeCircles
        SVGWIDTH={SVGWIDTH}
        SVGHEIGHT={SVGHEIGHT}
        network={network}
        dataset={dataset}
        sampleIndex={sampleIndex}
        original={original}
        onNodeClick={setHoveredNode}
        stepLayerHighlight={stepLayerHighlight}
        yMean={yMean}
        yStd={yStd}
      />
      <LayerLabels SVGWIDTH={SVGWIDTH} SVGHEIGHT={SVGHEIGHT} network={network} dataset={dataset} />
    </>
  );
};
