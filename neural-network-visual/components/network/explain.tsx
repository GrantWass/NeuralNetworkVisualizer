"use client"

import useStore from "@/hooks/store";

const Explain = () => {
    const {
        network,
        hoveredConnection,
        hoveredNode,
        getExplanation,
    } = useStore();

  return (
    <>
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
                <p>
                <strong>Node Details:</strong>
                </p>
                <p>
                Node: Layer {hoveredNode.layerIndex + 1}, Node {hoveredNode.nodeIndex + 1}
                </p>
                {/* <p>
                Activation (A): {network.layers[hoveredNode.layerIndex]?.A[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}
                </p>
                <p>
                Z (Linear Input): {network.layers[hoveredNode.layerIndex]?.Z[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}
                </p> */}
                {hoveredNode.layerIndex > 0 && (
                <>
                <p>
                    Bias: {network.layers[hoveredNode.layerIndex - 1]?.biases[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}
                </p>
                {/* <p>
                    db (Gradient for Bias): {network.layers[hoveredNode.layerIndex]?.db[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}
                </p>
                <p>
                    dZ (Gradient for Z): {network.layers[hoveredNode.layerIndex]?.dZ[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}
                </p> */}
                </>)}
            </div>
            ) : (
            <p>Hover over a node or connection to see details</p>
            )}
        </div>
        <div className="mt-8 p-4 bg-gray-100 rounded-lg">
            <h2 className="text-xl font-semibold mb-2">Explanation</h2>
            <p className="whitespace-pre-line">{getExplanation()}</p>
        </div>
    </>
  );
};

export default Explain;