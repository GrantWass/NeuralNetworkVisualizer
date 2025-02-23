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

const maxNodes = 6;

export const renderConnections = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  if (network.layers.some((layer) => layer.output_size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, layerIndex) => 
    layer.weights.flatMap((nodeWeights: number[], fromIndex: number) =>
      nodeWeights.map((weight: number, toIndex: number) => {
        const layerFromSize = layer.output_size;
        const layerToSize = network.layers[layerIndex + 1]?.output_size
        let additionalFromSpace = 0;
        let additionalToSpace = 0;

        if (layerFromSize > maxNodes && (fromIndex >=3)){
            additionalFromSpace = 30;
        }
        if (layerToSize > maxNodes && (toIndex >=3)){
            additionalToSpace = 30;
        }
        if (fromIndex >= 6 || toIndex >= 6) return null; 
        let fromX = (layerIndex + 1) * layerSpacing;
        let fromY = ((fromIndex + 1) * svgHeight) / (Math.min(layerFromSize, 6) + 1) + additionalFromSpace;
        let toX = (layerIndex + 2) * layerSpacing;
        let toY = ((toIndex + 1) * svgHeight) / (Math.min(network.layers[layerIndex + 1]?.output_size, 6) + 1) + additionalToSpace;
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
            strokeWidth={Math.abs(weight) * 3 + 1}
            onMouseEnter={() => setHoveredConnection({ layerIndex, fromIndex, toIndex, weight })}
            onMouseLeave={() => setHoveredConnection(null)}
          />
        );
      })
    ).filter(Boolean)
  );
};

export const renderNodes = ({ network, svgWidth, svgHeight, nodeRadius, setHoveredNode }: RenderNetworkProps) => {
  if (!network) return null;
  if (network.layers.some((layer) => layer.output_size > maxNodes)) {
    svgHeight = svgHeight - 30;
  }
  const layerSpacing = svgWidth / (network.layers.length + 1);
  return network.layers.flatMap((layer, layerIndex) => {
    const totalNodes = layer.output_size;
    let additionalSpace = 0;
    return Array.from({ length: Math.min(totalNodes, maxNodes) }, (_, nodeIndex) => {
      if (totalNodes > maxNodes && nodeIndex >= 3) {
        additionalSpace = 30;
      }
      const cx = (layerIndex + 1) * layerSpacing;
      const cy = (((nodeIndex + 1) * svgHeight) / (Math.min(totalNodes, maxNodes) + 1)) + additionalSpace;
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
        </g>
      );
    }).concat(
      totalNodes > maxNodes
        ? [
            <text
              key={`${layerIndex}-more`}
              x={(layerIndex + 1) * layerSpacing - (layerIndex + 1 === network.layers.length ? -30 : 30)}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            >
              {`+${totalNodes - maxNodes}`}
            </text>,
            <text
              key={`${layerIndex}-dots`}
              x={(layerIndex + 1) * layerSpacing}
              y={svgHeight / 2 + 20}
              textAnchor="middle"
              fontSize="14"
              fontWeight="bold"
            >
              ⋮
            </text>
          ]
        : []
    );
  });
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
