"use client"

import useStore from "@/hooks/store";
import { renderConnections, renderNodes, renderLayerLabels, maxNodes } from "@/lib/network";


const Graph = () => {
    const svgWidth = 1000;
    const svgHeight = 400;
    const nodeRadius = 20;
    const {
        network,
        setHoveredConnection,
        setHoveredNode,
        hoveredConnection,
        hoveredNode,
    } = useStore();
    
  return (
    <div className="grid place-items-center w-full mt-8 mb-6">
        <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
        {renderConnections({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderNodes({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderLayerLabels({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        </svg>
        <div className="mt-4 h-50">
                {hoveredConnection ? (
                    <div>
                        <p>
                            Connection: Layer {hoveredConnection.layerIndex + 1}, Node {hoveredConnection.fromIndex + 1} to Node{" "}
                            {hoveredConnection.toIndex + 1}
                        </p>
                        <p>Weight: {hoveredConnection.weight.toFixed(4)}</p>
                    </div>
                ) : hoveredNode && network ? (
                    <div>
                        <p><strong>Node Details:</strong></p>
                        <p>
                            Node: Layer {hoveredNode.layerIndex + 1}, Node {hoveredNode.nodeIndex + 1}
                        </p>
                        {hoveredNode.layerIndex > 0 && (
                            <>
                                {network.layers[hoveredNode.layerIndex - 1]?.biases?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>
                                        Bias: {network.layers[hoveredNode.layerIndex - 1].biases[hoveredNode.nodeIndex].toFixed(4)}
                                    </p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.Z?.[hoveredNode.nodeIndex]?.[0] !== undefined && (
                                    <p>
                                        Preactivation Value: {network.layers[hoveredNode.layerIndex - 1].Z[0][hoveredNode.nodeIndex].toFixed(2)}
                                    </p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.A?.[hoveredNode.nodeIndex]?.[0] !== undefined && (
                                    <p>
                                        Postactivation Value: {network.layers[hoveredNode.layerIndex - 1].A[0][hoveredNode.nodeIndex].toFixed(2)}
                                    </p>
                                )}
                            </>
                        )}
                    </div>
                ) : (
                    <p>Click a node or connection to see details</p>
                )}
            </div>
      </div>
  );
};

export default Graph;