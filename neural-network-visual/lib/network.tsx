import React from "react";
import { NetworkState, HoveredConnection, HoveredNode } from "@/static/types";
import { motion } from "framer-motion";

interface RenderNetworkProps {
  network: NetworkState | null;
  svgWidth: number;
  svgHeight: number;
  nodeRadius: number; 
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
  sampleIndex?: number; // Optional prop for sample index
}

export const maxNodes = 6;

export const renderConnections = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  if (network.layers.some((layer) => layer.size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }
  const layerSpacing = svgWidth / (network.layers.length + 1);

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

        let fromX = (layerIndex + 1) * layerSpacing;
        let fromY = ((fromIndex + 1) * svgHeight) / (Math.min(layerFromSize, 6) + 1) + additionalFromSpace;
        let toX = (layerIndex + 2) * layerSpacing;
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

export const renderNodes = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredNode, sampleIndex}: RenderNetworkProps) => {
  if (!network) return null;
  if (network.layers.some((layer) => layer.size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }

  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, layerIndex) => {
    const totalNodes = layer.size;
    let additionalSpace = 0;
    const isInputLayer = layerIndex === 0;
    const isOutputLayer = layerIndex === network.layers.length - 1;

    const nodes = Array.from({ length: Math.min(totalNodes, maxNodes) }, (_, nodeIndex) => {
      if (totalNodes > maxNodes && nodeIndex >= 3) additionalSpace = 30;

      const cx = (layerIndex + 1) * layerSpacing;
      const cy = (((nodeIndex + 1) * svgHeight) / (Math.min(totalNodes, maxNodes) + 1)) + additionalSpace;

      const hasActivations = Array.isArray(network.layers[layerIndex - 1]?.A) && network.layers[layerIndex - 1]?.A.length > 0;
      const inputSample = network?.input?.[sampleIndex ?? 0];
      var activationValue = 0;
      if (hasActivations) {
        activationValue = network.layers[layerIndex - 1].A?.[sampleIndex ?? 0][nodeIndex] ?? 0;
      }
      if (isInputLayer && inputSample?.[nodeIndex] != null){
        activationValue = inputSample[nodeIndex] ?? 0;
      }

      const clampedValue = Math.max(0, Math.min(1, activationValue)); // Clamp to [0,1] (change to normalizing?)

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
          {/* <circle
            cx={cx}
            cy={cy}
            r={dynamicRadius + clampedValue * 5}
            fill="black"
            opacity={clampedValue * 0.3}
            stroke="none"
          /> */}
        </g>
      );
    });

    const additionalIndicators =
      totalNodes > maxNodes
        ? [
            <text
              key={`more-${layerIndex}`}
              x={(layerIndex + 1) * layerSpacing - (layerIndex + 1 === network.layers.length ? -30 : 30)}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            >
              {`+${totalNodes - maxNodes}`}
            </text>,
            <text
              key={`dots-${layerIndex}`}
              x={(layerIndex + 1) * layerSpacing}
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

export const renderLayerLabels = ({ network, svgWidth, svgHeight }: RenderNetworkProps) => {
  if (!network) return null;
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, index) => {
    const labelKey = `label-${index}`;
    const activationKey = `activation-${index}`;
    const elements = [
      <text
        key={labelKey}
        x={(index + 1) * layerSpacing}
        y={svgHeight - 10}
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
          x={(index + 1) * layerSpacing}
          y={20}
          textAnchor="middle"
          fontSize="14"
          fontWeight="bold"
        >
          {network.layers[index - 1].activation}
        </text>
      );
    }

    return elements;
  });
};
