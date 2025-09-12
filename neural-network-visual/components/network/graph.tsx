"use client"

import useStore from "@/hooks/store";
import { renderConnections, renderNodes, renderLayerLabels } from "@/lib/network";


const Graph = () => {
    const {
        network,
        setHoveredConnection,
        setHoveredNode,
        hoveredConnection,
        hoveredNode,
        sampleIndex,
        dataset,
        originalData,
    } = useStore();

    
  return (
    <div className="grid place-items-center w-full mt-8 mb-6">
        <svg 
          className="w-[90%] max-w-[1000px] h-auto aspect-[2/1] border border-gray-300 rounded"
          viewBox="0 0 1000 500"
          preserveAspectRatio="xMidYMid meet"
        >
        {renderConnections({SVGWIDTH: 1000, SVGHEIGHT: 500, network, setHoveredConnection, setHoveredNode })}
        {renderNodes({SVGWIDTH: 1000, SVGHEIGHT: 500, network, setHoveredConnection, setHoveredNode, sampleIndex, dataset, original: originalData[sampleIndex] })}
        {renderLayerLabels({SVGWIDTH: 1000, SVGHEIGHT: 500, network, setHoveredConnection, setHoveredNode, dataset: dataset })}
        </svg>
        <div className="mt-4 h-50">
                {hoveredConnection ? (
                    <div>
                        <p><strong>Connection Details:</strong></p>
                        <p>Weight: {hoveredConnection.weight.toFixed(4)}</p>
                    </div>
                ) : hoveredNode && network ? (
                    <div>
                        <p><strong>Node Details:</strong></p>
                        {hoveredNode.layerIndex > 0 ? (
                            <>
                                {network.layers[hoveredNode.layerIndex - 1]?.biases?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>
                                        Bias: {network.layers[hoveredNode.layerIndex - 1].biases[hoveredNode.nodeIndex].toFixed(4)}
                                    </p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.Z?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>
                                        Preactivation Value: {network.layers[hoveredNode.layerIndex - 1].Z[sampleIndex][hoveredNode.nodeIndex].toFixed(2)}
                                    </p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.A?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>
                                        Postactivation Value: {network.layers[hoveredNode.layerIndex - 1].A[sampleIndex][hoveredNode.nodeIndex].toFixed(2)}
                                    </p>
                                )}
                            </>
                        ): <p>Input Node</p>}
                    </div>
                ) : (
                    <p>Click a node or connection to see details</p>
                )}
            </div>
      </div>
  );
};

export default Graph;