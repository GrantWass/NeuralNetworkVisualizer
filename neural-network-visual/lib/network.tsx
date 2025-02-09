import React from "react";
import { NetworkState, HoveredConnection, HoveredNode } from "@/static/types";

interface RenderNetworkProps {
  network: NetworkState | null;
  svgWidth: number;
  svgHeight: number;
  nodeRadius: number; 
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
}

export const renderConnections = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, layerIndex) =>
    layer.weights.flatMap((nodeWeights: number[], fromIndex: number) =>
      nodeWeights.map((weight: number, toIndex: number) => {
        let fromX = (layerIndex + 1) * layerSpacing;
        let fromY = ((fromIndex + 1) * svgHeight) / (network.layers[layerIndex].output_size + 1);
        let toX = (layerIndex + 2) * layerSpacing;
        let toY = ((toIndex + 1) * svgHeight) / (network.layers[layerIndex + 1]?.output_size + 1);
        if (isNaN(toY) || isNaN(toX)) {
          toX = fromX;
          toY = fromY;
        }
        return (
          <line
            key={`${layerIndex}-${fromIndex}-${toIndex}`}
            x1={fromX}
            y1={fromY}
            x2={toX}
            y2={toY}
            stroke={Math.abs(weight) < 0.5 ? "#94a3b8" : "#475569"}
            strokeWidth={Math.abs(weight) * 3}
            onMouseEnter={() => setHoveredConnection({ layerIndex, fromIndex, toIndex, weight })}
            onMouseLeave={() => setHoveredConnection(null)}
          />
        );
      })
    )
  );
};

export const renderNodes = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, layerIndex) =>
    Array.from({ length: layer.output_size }, (_, nodeIndex) => {
      const cx = (layerIndex + 1) * layerSpacing;
      const cy = ((nodeIndex + 1) * svgHeight) / (layer.output_size + 1);
      return (
        <g key={`${layerIndex}-${nodeIndex}`}>
          <circle
            cx={cx}
            cy={cy}
            r={nodeRadius}
            fill="#e2e8f0"
            stroke="#475569"
            strokeWidth="2"
            onMouseEnter={() => setHoveredNode({ layerIndex, nodeIndex })}
            onMouseLeave={() => setHoveredNode(null)}
          />
          <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" fontSize="12">
            {/* {layer.activations[nodeIndex]?.toFixed(2) || "0.00"} */}
          </text>
        </g>
      );
    })
  );
};

export const renderLayerLabels = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.map((layer, index) => (
    <text
      key={index}
      x={(index + 1) * layerSpacing}
      y={svgHeight - 10}
      textAnchor="middle"
      fontSize="14"
      fontWeight="bold"
    >
      {layer.name}
    </text>
  ));
};
