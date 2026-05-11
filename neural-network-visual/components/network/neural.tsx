"use client"

import useStore from "@/components/network/lib/store";
import { Network } from "@/components/network/network";
import { useEffect, useState, useRef } from "react";

const Legend = () => (
  <div className="flex flex-wrap items-center justify-center gap-4 mt-3 text-xs text-gray-600 px-2">
    <div className="flex items-center gap-1.5">
      <div className="w-4 h-4 rounded-full bg-blue-400 border border-blue-600" />
      <span>Input node</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-4 h-4 rounded-full bg-gray-400 border border-gray-600" />
      <span>Hidden node</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-4 h-4 rounded-full bg-red-400 border border-red-800" />
      <span>Output node</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-5 h-0.5 bg-gray-400" />
      <span>Weak weight</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-5 h-1.5 bg-gray-700" />
      <span>Strong weight</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-3 h-3 rounded-full bg-green-400" />
      <span>Weight increased</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-3 h-3 rounded-full bg-orange-400" />
      <span>Weight decreased</span>
    </div>
    <span className="text-gray-400">| Node brightness = activation strength</span>
  </div>
);

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
        changedConnections,
        stepLayerHighlight,
        sessionId,
        setWeight,
    } = useStore();

    // Flash changed connections for 2s after each training cycle
    const [flashConnections, setFlashConnections] = useState<typeof changedConnections>([]);
    const [flashKey, setFlashKey] = useState(0);

    // Weight editing state
    const [editingWeight, setEditingWeight] = useState(false);
    const [weightInput, setWeightInput] = useState("");
    const weightInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (changedConnections.length > 0) {
            setFlashConnections(changedConnections);
            setFlashKey(k => k + 1);
            const timer = setTimeout(() => setFlashConnections([]), 2000);
            return () => clearTimeout(timer);
        }
    }, [changedConnections]);

    // Reset edit state when selection changes
    useEffect(() => {
        setEditingWeight(false);
        if (hoveredConnection) {
            setWeightInput(hoveredConnection.weight.toFixed(4));
        }
    }, [hoveredConnection]);

    useEffect(() => {
        if (editingWeight) weightInputRef.current?.focus();
    }, [editingWeight]);

    const handleWeightSubmit = async () => {
        if (!hoveredConnection) return;
        const val = parseFloat(weightInput);
        if (isNaN(val)) return;
        await setWeight(hoveredConnection.layerIndex, hoveredConnection.fromIndex, hoveredConnection.toIndex, val);
        setEditingWeight(false);
    };

    return (
        <div className="grid place-items-center w-full mt-8 mb-2">
            <svg
                className="w-[90%] max-w-[1000px] h-auto aspect-[2/1] border border-gray-300 rounded"
                viewBox="0 0 1000 500"
                preserveAspectRatio="xMidYMid meet"
            >
                <Network
                    SVGWIDTH={1000}
                    SVGHEIGHT={500}
                    network={network}
                    setHoveredConnection={setHoveredConnection}
                    setHoveredNode={setHoveredNode}
                    sampleIndex={sampleIndex}
                    dataset={dataset}
                    original={originalData[sampleIndex]}
                    flashConnections={flashConnections}
                    flashKey={flashKey}
                    stepLayerHighlight={stepLayerHighlight}
                />
            </svg>

            {/* Legend */}
            <Legend />

            <div className="mt-4 min-h-[60px]">
                {hoveredConnection ? (
                    <div className="text-sm space-y-1">
                        <p className="font-semibold">Connection: Layer {hoveredConnection.layerIndex} → {hoveredConnection.layerIndex + 1} &nbsp;|&nbsp; Neuron {hoveredConnection.fromIndex} → {hoveredConnection.toIndex}</p>
                        <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-gray-600">Weight:</span>
                            {!editingWeight ? (
                                <>
                                    <span className="font-mono font-medium">{hoveredConnection.weight.toFixed(4)}</span>
                                    {sessionId && (
                                        <button
                                            onClick={() => setEditingWeight(true)}
                                            className="text-xs border border-gray-300 rounded px-2 py-0.5 hover:bg-gray-50"
                                        >
                                            Edit
                                        </button>
                                    )}
                                </>
                            ) : (
                                <>
                                    <input
                                        ref={weightInputRef}
                                        type="number"
                                        value={weightInput}
                                        onChange={(e) => setWeightInput(e.target.value)}
                                        onKeyDown={(e) => { if (e.key === 'Enter') handleWeightSubmit(); if (e.key === 'Escape') setEditingWeight(false); }}
                                        step="0.01"
                                        className="w-24 text-sm border border-gray-400 rounded px-2 py-0.5 font-mono"
                                    />
                                    <button
                                        onClick={handleWeightSubmit}
                                        className="text-xs bg-black text-white rounded px-2 py-1"
                                    >
                                        Set
                                    </button>
                                    <button
                                        onClick={() => setEditingWeight(false)}
                                        className="text-xs border border-gray-300 rounded px-2 py-1 hover:bg-gray-50"
                                    >
                                        Cancel
                                    </button>
                                </>
                            )}
                        </div>
                        {!sessionId && <p className="text-xs text-gray-400 italic">Initialize the model to edit weights.</p>}
                    </div>
                ) : hoveredNode && network ? (
                    <div className="text-sm space-y-0.5">
                        <p className="font-semibold">Node Details</p>
                        {hoveredNode.layerIndex > 0 ? (
                            <>
                                {network.layers[hoveredNode.layerIndex - 1]?.biases?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>Bias: <span className="font-mono">{network.layers[hoveredNode.layerIndex - 1].biases[hoveredNode.nodeIndex].toFixed(4)}</span></p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.Z?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>Pre-activation (Z): <span className="font-mono">{network.layers[hoveredNode.layerIndex - 1].Z[sampleIndex][hoveredNode.nodeIndex].toFixed(4)}</span></p>
                                )}
                                {network.layers[hoveredNode.layerIndex - 1]?.A?.[sampleIndex]?.[hoveredNode.nodeIndex] !== undefined && (
                                    <p>Post-activation (A): <span className="font-mono">{network.layers[hoveredNode.layerIndex - 1].A[sampleIndex][hoveredNode.nodeIndex].toFixed(4)}</span></p>
                                )}
                            </>
                        ) : <p className="text-gray-500">Input node — values come from the dataset.</p>}
                    </div>
                ) : (
                    <p className="text-sm text-gray-500">Click a node or connection to see details</p>
                )}
            </div>
        </div>
    );
};

export default Graph;
