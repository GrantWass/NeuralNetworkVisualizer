"use client"

import useStore from "@/hooks/store";
import { renderConnections, renderNodes, renderLayerLabels } from "@/lib/network";


const Graph = () => {
    const svgWidth = 1000;
    const svgHeight = 400;
    const nodeRadius = 20;
    const {
        network,
        setHoveredConnection,
        setHoveredNode,
    } = useStore();

  return (
    <div className="grid place-items-center w-full mt-8 mb-6">
        <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
        {renderConnections({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderNodes({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderLayerLabels({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        </svg>
      </div>
  );
};

export default Graph;