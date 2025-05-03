import React from "react";
import { NetworkState, HoveredConnection, HoveredNode } from "@/static/types";
import { DATASET_INPUT_FEATURES } from "@/static/constants";

interface RenderNetworkProps {
  network: NetworkState | null;
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
  sampleIndex?: number; // Optional prop for sample index
  dataset?: string;
  original?: number[];
}

export const maxNodes = 6;
let SHIFT = 0
const INPUTLABELOFFSET = 110
export const SVGHEIGHT = 500;
export const svgWidth = 1000;
const nodeRadius = 20;
const outputMap: { [key: string]: string[] } = {
  california_housing: ["Median House Value"],
  iris: ["Setosa", "Versicolor", "Virginica"],
};
let layerSpacing = 0


export const renderConnections = ({ network, setHoveredConnection }: RenderNetworkProps) => {
  if (!network) return null;
  let svgHeight = SVGHEIGHT
  if (network.layers.some((layer) => layer.size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }
  layerSpacing = svgWidth / (network.layers.length + 1) - (network.layers.length > 4 ? 15 : 0);
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
              strokeWidth={Math.abs(weight) * 3 + 1}
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

export const renderNodes = ({ network, setHoveredNode, sampleIndex, dataset, original}: RenderNetworkProps) => {
  if (!network) return null;
  let svgHeight = SVGHEIGHT
  if (network.layers.some((layer) => layer.size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }

  let features = DATASET_INPUT_FEATURES[dataset ?? ""] || [];

  //This is hardcoded for now for california datasets
  //This accounts for the fact not all nodes are shown
  if (dataset === "california_housing" && original != null) {
    features = features.slice(0,3).concat(features.slice(5))
    original = original.slice(0, 10)
    original = original.slice(0,3).concat(original.slice(5))
  } 

  //could move this in a separate file (same logic as in tooltip)
  const formatValue = (value: number, dataset: string) => {
    if (dataset === "iris") {
      return `${(value * 100).toFixed(1)}%`;
    }
    if (dataset === "california_housing") {
      const dollars = value * 100000; // adjust scale
      return dollars.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      });
    }
    return value.toFixed(3);
  };
  
  const formatActual = () => {
    if (!original || original.length < 3){
      return;
    }
    if (dataset === "iris") {
      const actualResults = original.slice(-3)
      const index = actualResults.findIndex((v) => v === 1);
      return outputMap[dataset]?.[index] ?? "Unknown";
    } else if (dataset === "california_housing") {
      const actualResults = original.slice(-1)
      return formatValue(actualResults[0], dataset);
    }
    return "N/A";
  };

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

      const dynamicRadius = nodeRadius + clampedValue * 3; // Slightly larger for high activation
      const fillColor = isInputLayer
      ? `hsl(210, 100%, ${85 - clampedValue * 50}%)`  // light blue to dark blue
      : isOutputLayer
      ? `hsl(0, 70%, ${80 - clampedValue * 40}%)`     // light red to dark red
      : `hsl(215, 30%, ${85 - clampedValue * 50}%)`;  // gray to dark slate
    

      const strokeColor = isOutputLayer ? "#7f1d1d" : "#475569";
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
          {isInputLayer &&
            <text 
              x={cx - INPUTLABELOFFSET}
              y={cy}
              className=""
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            >
              {features[nodeIndex]}
            </text>
          }
          {isInputLayer && original && activationValue && original.length > 0 &&
          <>
            <text
              x={cx - INPUTLABELOFFSET- 22}
              y={cy + 20}
              className=""
              textAnchor="middle"
              fontSize="14"
            >
              {original[nodeIndex].toFixed(2)}
            </text>
            <text
              x={cx - INPUTLABELOFFSET + 22}
              y={cy + 20}
              className=""
              textAnchor="middle"
              fontSize="14"
            >
              {`(${activationValue.toFixed(3)})`}
            </text>
          </>
          }
          {isInputLayer &&
            <text 
              x={cx - INPUTLABELOFFSET}
              y={cy}
              className=""
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            > 
              {features[nodeIndex]}
            </text>
          }
          {isOutputLayer && dataset && activationValue &&
          <>
            <text
              x={cx + 110}
              y={dataset === "iris" ? cy : cy - 60}
              className=""
              textAnchor="middle"
              fontSize="16"
              fontWeight="bold"
            >
              {outputMap[dataset][nodeIndex]}
            </text>
            {dataset === "california_housing" &&
            <>
              <text
                x={cx + 110}
                y={cy - 30}
                className=""
                textAnchor="middle"
                fontSize="16"
              >
                Predicted Value:
              </text>
              <text
              x={cx + 110}
              y={cy + 20}
              className=""
              textAnchor="middle"
              fontSize="16"
              >
                Actual Value:
              </text>
              <text
                x={cx + 110}
                y={cy + 40}
                className=""
                textAnchor="middle"
                fontSize="16"
              >
                {formatActual()}
              </text>
            </>
            }
            {dataset === "iris" &&
              <text
                x={cx + 110}
                y={cy + 40}
                className=""
                textAnchor="middle"
                fontSize="16"
              >
                {formatActual() === outputMap[dataset][nodeIndex] ? "Actual Answer" : false}
              </text>
            }
            <text
              x={cx + 110}
              y={(dataset === "iris" ? cy : cy - 30) + 20}
              className=""
              textAnchor="middle"
              fontSize="16"
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
              fontSize="14"
              fontWeight="bold"
            >
              {`+${totalNodes - maxNodes}`}
            </text>,
            <text
              key={`dots-${layerIndex}`}
              x={(layerIndex + 1) * layerSpacing + SHIFT}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            >
              ⋮
            </text>,
          ]
        : [];

    return [...nodes, ...additionalIndicators];
  });
};

export const renderLayerLabels = ({ network }: RenderNetworkProps) => {
  if (!network) return null;
  return network.layers.flatMap((layer, index) => {
    const labelKey = `label-${index}`;
    const activationKey = `activation-${index}`;
    const elements = [
      <text
        key={labelKey}
        x={(index + 1) * layerSpacing + SHIFT}
        y={SVGHEIGHT - 10}
        textAnchor="middle"
        fontSize="14"
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
          fontSize="14"
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
          x={65}
          y={20}
          textAnchor="middle"
          fontSize="14"
          fontWeight="bold"
        >
          {`Input Values`}
        </text>
        <text
          key={`inputlabel2-${index}`}
          x={150}
          y={20}
          textAnchor="middle"
          fontSize="14"
        >
          {`(Normalized)`}
        </text>
        </>
      );
    }

    return elements;
  });
};
