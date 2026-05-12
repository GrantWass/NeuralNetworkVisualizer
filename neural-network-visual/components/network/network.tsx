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
}> = ({ network, SVGWIDTH, SVGHEIGHT, onClick, flashConnections = [], flashKey = 0, stepLayerHighlight }) => {
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const inStepMode = stepLayerHighlight !== null && stepLayerHighlight !== undefined;

  return (
    <>
      {network.layers.flatMap((layer, li) =>
        layer.weights.flatMap((weights: number[], fi: number) =>
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
                  strokeWidth={wAbs * (SVGWIDTH > 600 ? 2.5 : 1.8) + 1.8}
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
        )
      )}
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
}> = ({ network, SVGWIDTH, SVGHEIGHT, dataset, sampleIndex, original, onNodeClick, stepLayerHighlight }) => {
  const { layerSpacing, SHIFT } = computeLayout(SVGWIDTH, network.layers.length);
  const fontSize = fontSizeForWidth(SVGWIDTH);
  const features = featureList(dataset, SVGWIDTH);
  const nodeRadius = 10 + SVGWIDTH / 100;
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
                  {features[ni]}
                </text>
              )}
              {isInput && original && activationValue && original.length > 0 &&
                renderInputValues(cx, cy, original, ni, activationValue, fontSize, INPUTLABELOFFSET, SVGWIDTH, SVGHEIGHT)
              }
              {isOutput && dataset &&
                renderResults({ SVGWIDTH, SVGHEIGHT, cx, cy, dataset, ni, activationValue, original, fontSize })
              }
            </g>
          );
        });
      })}
    </>
  );
};

const LayerLabels: React.FC<{ network: NetworkState; SVGWIDTH: number; SVGHEIGHT: number }> = ({
  network,
  SVGWIDTH,
  SVGHEIGHT,
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
}) => {
  if (!network) return null;
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
      />
      <NodeCircles
        SVGWIDTH={SVGWIDTH}
        SVGHEIGHT={SVGHEIGHT}
        network={network}
        dataset={dataset}
        sampleIndex={sampleIndex}
        original={original}
        onNodeClick={setHoveredNode}
        stepLayerHighlight={stepLayerHighlight}
      />
      <LayerLabels SVGWIDTH={SVGWIDTH} SVGHEIGHT={SVGHEIGHT} network={network} />
    </>
  );
};
