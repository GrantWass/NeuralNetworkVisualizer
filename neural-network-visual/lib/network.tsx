import React from "react";
import { NetworkState, HoveredConnection, HoveredNode } from "@/static/types";
import { DATASET_INPUT_FEATURES, DATASET_INPUT_FEATURES_SHORT } from "@/static/constants";
import { formatValue, formatActual, outputMap } from "@/lib/utils";

interface RenderNetworkProps {
  SVGWIDTH: number
  SVGHEIGHT: number
  network: NetworkState | null;
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
  sampleIndex?: number;
  dataset?: string;
  original?: number[];
}

export const maxNodes = 6;
let SHIFT = 0
let layerSpacing = 0


export const renderConnections = ({SVGWIDTH, SVGHEIGHT, network, setHoveredConnection }: RenderNetworkProps) => {
  if (!network) return null;
  let svgHeight = SVGHEIGHT
  if (network.layers.some((layer) => layer.size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }
  layerSpacing = SVGWIDTH / (network.layers.length + 1) - (network.layers.length > 4 ? 15 : 0);
  SHIFT = (network.layers.length > 4 ? 40 : 0)

  return network.layers.flatMap((layer, layerIndex) => 
    layer.weights.flatMap((nodeWeights: number[], fromIndex: number) =>
      nodeWeights.map((weight: number, toIndex: number) => {
        const layerFromSize = layer.size;
        const layerToSize = network.layers[layerIndex + 1]?.size;
        let additionalFromSpace = 0;
        let additionalToSpace = 0;

        if (layerFromSize > maxNodes && fromIndex >= 3) additionalFromSpace = 30;
        if (layerToSize > maxNodes && toIndex >= 3) additionalToSpace = 30;
        if (fromIndex >= maxNodes || toIndex >= maxNodes) return null;

        const fromX = (layerIndex + 1) * layerSpacing + SHIFT;
        const fromY = ((fromIndex + 1) * svgHeight) / (Math.min(layerFromSize, 6) + 1) + additionalFromSpace;
        let toX = (layerIndex + 2) * layerSpacing + SHIFT;
        let toY = ((toIndex + 1) * svgHeight) / (Math.min(layerToSize ?? 6, 6) + 1) + additionalToSpace;

        if (isNaN(toY) || isNaN(toX)) {
          toX = fromX;
          toY = fromY;
        }

        const keyBase = `conn-${layerIndex}-${fromIndex}-${toIndex}`;
        return (
          <g key={keyBase}>
            <line
              x1={fromX}
              y1={fromY}
              x2={toX}
              y2={toY}
              stroke={Math.abs(weight) < 0.5 ? "#94a3b8" : "#475569"}
              strokeWidth={Math.abs(weight) * (SVGWIDTH > 600 ? 3 : 2) + 1}
            />
            <line
              x1={fromX}
              y1={fromY}
              x2={toX}
              y2={toY}
              stroke="transparent"
              strokeWidth={15}
              cursor={"pointer"}
              onClick={() => setHoveredConnection({ layerIndex, fromIndex, toIndex, weight })}
            />
          </g>
        );
      })
    ).filter(Boolean)
  );
};

// Helper function to calculate responsive font size
const getFontSize = (svgWidth: number) => {
  if (svgWidth < 450) return 8;
  if (svgWidth < 600) return 11;
  return 14;
};

// Helper function to get features based on screen size
const getFeatures = (dataset: string | undefined, svgWidth: number) => {
  const features = svgWidth < 800 
    ? DATASET_INPUT_FEATURES_SHORT[dataset ?? ""] || []
    : DATASET_INPUT_FEATURES[dataset ?? ""] || [];
  return features;
};

// Helper function to get node colors based on layer type and activation
const getNodeColors = (isInputLayer: boolean, isOutputLayer: boolean, clampedValue: number) => {
  const fillColor = isInputLayer
    ? `hsl(210, 100%, ${85 - clampedValue * 50}%)`  // light blue to dark blue
    : isOutputLayer
    ? `hsl(0, 70%, ${80 - clampedValue * 40}%)`     // light red to dark red
    : `hsl(215, 30%, ${85 - clampedValue * 50}%)`;  // gray to dark slate

  const strokeColor = isOutputLayer ? "#7f1d1d" : "#475569";
  
  return { fillColor, strokeColor };
};

// Helper function to render input layer labels
const renderInputLabels = (cx: number, cy: number, features: string[], nodeIndex: number, fontSize: number, INPUTLABELOFFSET: number) => (
  <text 
    x={cx - INPUTLABELOFFSET}
    y={cy}
    className=""
    textAnchor="middle"
    fontSize={fontSize}
    fontWeight="bold"
  >
    {features[nodeIndex]}
  </text>
);

// Helper function to render input values
const renderInputValues = (cx: number, cy: number, original: number[], nodeIndex: number, activationValue: number, fontSize: number, INPUTLABELOFFSET: number, SVGWIDTH: number, SVGHEIGHT: number) => (
  <>
    <text
      x={cx - INPUTLABELOFFSET - (10 + SVGWIDTH * .012)}
      y={cy + (SVGHEIGHT * 0.04)}
      className=""
      textAnchor="middle"
      fontSize={fontSize}
    >
      {original[nodeIndex].toFixed(2)}
    </text>
    <text
      x={cx - INPUTLABELOFFSET + (SVGWIDTH * .01) + (Math.abs(original[nodeIndex]) > 100 ? (SVGWIDTH * .02) : (SVGWIDTH * .01))}
      y={cy + (SVGHEIGHT * 0.04)}
      className=""
      textAnchor="middle"
      fontSize={fontSize}
    >
      {`(${activationValue.toFixed(2)})`}
    </text>
  </>
);

export const renderNodes = ({SVGWIDTH, SVGHEIGHT, network, setHoveredNode, sampleIndex, dataset, original}: RenderNetworkProps) => {
  if (!network) return null;
  
  const svgHeight = network.layers.some((layer) => layer.size > maxNodes) 
    ? SVGHEIGHT - 30 
    : SVGHEIGHT;

  const nodeRadius = 10 + SVGWIDTH/100;
  const INPUTLABELOFFSET = SVGWIDTH/10 + 10;
  const fontSize = getFontSize(SVGWIDTH);
  const features = getFeatures(dataset, SVGWIDTH);


  return network.layers.flatMap((layer, layerIndex) => {
    const totalNodes = layer.size;
    let additionalSpace = 0;
    const isInputLayer = layerIndex === 0;
    const isOutputLayer = layerIndex === network.layers.length - 1;

    const nodes = Array.from({ length: Math.min(totalNodes, maxNodes) }, (_, nodeIndex) => {
      if (totalNodes > maxNodes && nodeIndex >= 3) additionalSpace = 30;

      const cx = (layerIndex + 1) * layerSpacing + SHIFT;
      const cy = (((nodeIndex + 1) * svgHeight) / (Math.min(totalNodes, maxNodes) + 1)) + additionalSpace;

      const hasActivations = Array.isArray(network.layers[layerIndex - 1]?.A) && network.layers[layerIndex - 1]?.A.length > 0;
      const inputSample = network?.input?.[sampleIndex ?? 0];

      let activationValue = 0;
      if (hasActivations) {
        activationValue = network.layers[layerIndex - 1].A?.[sampleIndex ?? 0][nodeIndex] ?? 0;
      }
      if (isInputLayer && inputSample?.[nodeIndex] != null){
        activationValue = inputSample[nodeIndex] ?? 0;
      }

      const clampedValue = Math.min(1, Math.abs(activationValue));
      const dynamicRadius = nodeRadius + clampedValue * 3;
      const { fillColor, strokeColor } = getNodeColors(isInputLayer, isOutputLayer, clampedValue);
      const nodeKey = `node-${layerIndex}-${nodeIndex}`;

      return (
        <g key={nodeKey}>
          <circle
            cx={cx}
            cy={cy}
            r={dynamicRadius}
            fill={fillColor}
            stroke={strokeColor}
            strokeWidth="2"
          />
          <circle
            cx={cx}
            cy={cy}
            r={dynamicRadius + 10}
            fill="transparent"
            onClick={() => setHoveredNode({ layerIndex, nodeIndex })}
            cursor="pointer"
          />
          {isInputLayer && renderInputLabels(cx, cy, features, nodeIndex, fontSize, INPUTLABELOFFSET)}
          {isInputLayer && original && activationValue && original.length > 0 && 
            renderInputValues(cx, cy, original, nodeIndex, activationValue, fontSize, INPUTLABELOFFSET, SVGWIDTH, SVGHEIGHT)
          }
          {isOutputLayer && dataset && activationValue &&
          <>
            <text
              x={cx + (SVGWIDTH * .11)}
              y={dataset === "iris" ? cy : cy - (SVGHEIGHT * 0.12)}
              className=""
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight="bold"
            >
              {outputMap[dataset][nodeIndex]}
            </text>
            {dataset === "auto_mpg" &&
            <>
              <text
                x={cx + (SVGWIDTH * .11)}
                y={cy - (SVGHEIGHT * 0.06)}
                className=""
                textAnchor="middle"
                fontSize={fontSize}
              >
                Predicted Value:
              </text>
              <text
              x={cx + (SVGWIDTH * .11)}
              y={cy + (SVGHEIGHT * 0.04)}
              className=""
              textAnchor="middle"
              fontSize={fontSize}
              >
                Actual Value:
              </text>
              <text
                x={cx + (SVGWIDTH * .11)}
                y={cy + (SVGHEIGHT * 0.08)}
                className=""
                textAnchor="middle"
                fontSize={fontSize}
              >
                {formatActual(original || [], dataset)}
              </text>
            </>
            }
            {dataset === "iris" &&
              <text
                x={cx + (SVGWIDTH * .11)}
                y={cy + (SVGHEIGHT * 0.08)}
                className=""
                textAnchor="middle"
                fontSize={fontSize}
              >
                {formatActual(original || [], dataset) === outputMap[dataset][nodeIndex] ? "Actual Answer" : false}
              </text>
            }
            <text
              x={cx + (SVGWIDTH * .11)}
              y={(dataset === "iris" ? cy : cy - (SVGHEIGHT * 0.06)) + (SVGHEIGHT * 0.04)}
              className=""
              textAnchor="middle"
              fontSize={fontSize}
            >
              {formatValue(activationValue, dataset)}
            </text>
          </>
          }
        </g>
      );
    });

    const additionalIndicators =
      totalNodes > maxNodes
        ? [
            <text
              key={`more-${layerIndex}`}
              x={(layerIndex + 1) * layerSpacing + SHIFT - (layerIndex + 1 === network.layers.length ? -20 : 20)}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight="bold"
            >
              {`+${totalNodes - maxNodes}`}
            </text>,
            <text
              key={`dots-${layerIndex}`}
              x={(layerIndex + 1) * layerSpacing + SHIFT}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight="bold"
            >
              ⋮
            </text>,
          ]
        : [];

    return [...nodes, ...additionalIndicators];
  });
};

export const renderLayerLabels = ({SVGWIDTH, SVGHEIGHT, network }: RenderNetworkProps) => {
  if (!network) return null;

  let fontSize = 14
  if (SVGWIDTH < 600){
    fontSize = 11
  }
  if (SVGWIDTH < 450){
    fontSize = 9
  }

  return network.layers.flatMap((layer, index) => {
    const labelKey = `label-${index}`;
    const activationKey = `activation-${index}`;
    const elements = [
      <text
        key={labelKey}
        x={(index + 1) * layerSpacing + SHIFT}
        y={(SVGHEIGHT * 0.97)}
        textAnchor="middle"
        fontSize={fontSize}
        fontWeight="bold"
      >
        {layer.name}
      </text>,
    ];

    if (network.initialized && network.layers[index - 1]) {
      elements.push(
        <text
          key={activationKey}
          x={(index + 1) * layerSpacing + SHIFT}
          y={20}
          textAnchor="middle"
          fontSize={fontSize}
        >
          {network.layers[index - 1].activation}
        </text>
      );
    }

    if (network) {
      elements.push(
        <>
        <text
          key={`inputlabel-${index}`}
          x={(SVGWIDTH*.065 + 10)}
          y={SVGHEIGHT * .04}
          textAnchor="middle"
          fontSize={fontSize}
          fontWeight="bold"
        >
          {`Input Values`}
        </text>
        <text
          key={`inputlabel2-${index}`}
          x={(SVGWIDTH*.065 + 10) + (SVGWIDTH > 600 ? 90 : SVGWIDTH*.135)}
          y={SVGHEIGHT * .04}
          textAnchor="middle"
          fontSize={fontSize}
        >
          {`(Normalized)`}
        </text>
        </>
      );
    }

    return elements;
  });
};
