"use client";

import useStore from "@/hooks/store";
import { NeuronLayer } from "@/static/types";
import { useState } from "react";
import { Input } from "@/components/ui/input";

const Explain = () => {
    const { network, getExplanation, dataset } = useStore();
    const [sampleIndex, setSampleIndex] = useState(0);

    const renderMatrix = (matrix: number[][] | undefined, label: string, subLabel?: string) => (
        matrix && matrix.length > 0 ? (
            <div className="inline-block p-1 mx-1">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="grid border border-gray-400 px-1" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, auto)` }}>
                    {matrix.map((row, rowIndex) => (
                        row.map((val, colIndex) => (
                            <span key={`${rowIndex}-${colIndex}`} className="px-1 py-0.5 bg-white shadow-sm">
                                {val.toFixed(2)}
                            </span>
                        ))
                    ))}
                </div>
            </div>
        ) : null
    );

    const renderVector = (vector: number[] | undefined | null, label: string, subLabel?: string) => (
        vector && vector.length > 0 ? (
            <div className="inline-block px-1 mx-1">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="flex flex-row justify-center border border-gray-400 px-1 py-0.5 rounded bg-white shadow-sm">
                    {vector.map((val, index) => (
                        <span key={index} className="px-1 py-0.5">{val.toFixed(2)}</span>
                    ))}
                </div>
            </div>
        ) : null
    );

    return (
        <div className="mt-8 p-6 bg-gray-100 rounded-lg mx-8 shadow-md">
            <h2 className="text-xl font-semibold mb-4">Explanation</h2>
            {network?.layers && network?.layers[0].A?.length > 0 && (
            <div className="text-center">
                    <p className="mt-4 mb-2 font-medium">Forward Propagation:</p>
                    {dataset != "mnist" && (
                    <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                        {network?.layers.map((layer: NeuronLayer, layerIndex: number) => (
                            <div key={layerIndex} className="flex flex-row justify-center items-center">
                                {layerIndex < network.layers.length - 1 ? (
                                    <>  
                                        {renderVector((layerIndex == 0 ? network?.input[sampleIndex] : network?.layers[layerIndex- 1].A[sampleIndex]), (layerIndex == 0 ? "Input Vector": `Layer ${layerIndex} Output`))}
                                        <span className="text-lg mt-5">×</span>
                                        {renderMatrix(layer.weights, `Layer ${layerIndex + 1} Weights`)}
                                        <span className="text-lg mt-5">+</span>
                                        {renderVector(layer.biases, `Layer ${layerIndex + 1} Biases`)}
                                        <span className="text-lg mt-5">=</span>
                                        <div className="flex flex-col">
                                        {renderVector(layer.Z[sampleIndex], (layerIndex === network.layers.length - 2) ? `Output Preactivations` : `Layer ${layerIndex + 1} Preactivations`)}
                                        <span className="text-lg">↓</span>
                                        {renderVector(layer.A[sampleIndex], (layerIndex === network.layers.length - 2) ? `Output` : `Layer ${layerIndex + 1} Activations`)}
                                        </div>
                                    </>
                                ) : null}
                            </div>
                        ))}
                    </div>)}

                    <p className="mt-6 mb-2 font-medium">Backward Propagation:</p>
                    <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                        {network?.layers.slice().reverse().map((layer: NeuronLayer, reversedIndex: number) => {
                            const layerIndex = network.layers.length +1 - reversedIndex;
                            return (
                                <div key={layerIndex + "backprop"} className="flex flex-row justify-center items-center">
                                    {renderVector(layer.dZ[sampleIndex], `Layer ${layerIndex - 1} dZ`, "Gradient with respect to activations")}
                                    {renderMatrix(layer.dW, `Layer ${layerIndex - 1} dW`, "Gradient with respect to weights")}
                                    {renderVector(layer.db, `Layer ${layerIndex - 1} dB`, "Gradient with respect to biases")}
                                </div>
                            );
                        })}
                    </div>
            </div>
        )}
            <p className="whitespace-pre-line mt-4 text-gray-700 bg-white p-4 rounded shadow-sm border border-gray-300">
                {getExplanation()}
            </p>
        </div>
    );
};

export default Explain;


    {/* <p className="mb-2">Sample Index:</p>
    <Input
        type="number"
        value={sampleIndex}
        onChange={(e) => setSampleIndex(Math.max(1, Number(e.target.value)))}
        min={0}
        max={network?.input.length - 1}
        className="w-20 text-center border border-gray-400 rounded-md shadow-sm"
    /> */}