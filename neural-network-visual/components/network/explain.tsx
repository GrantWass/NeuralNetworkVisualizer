"use client";

import useStore from "@/hooks/store";
import { NeuronLayer } from "@/static/types";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

type PropagationView = 'forward' | 'backward';

const Explain = () => {
    const { network, getExplanation, dataset, losses, accuracies } = useStore();
    const [sampleIndex, setSampleIndex] = useState(0);
    const [expanded, setExpanded] = useState(false);
    const [view, setView] = useState<PropagationView>('forward');

    // Prepare chart data for losses
    const lossData = {
        labels: losses.map((_, index) => `Epoch ${index + 1}`),
        datasets: [
            {
                label: 'Training Loss',
                data: losses,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                tension: 0.1
            }
        ]
    };

    // Prepare chart data for accuracies
    const accuracyData = {
        labels: accuracies.map((_, index) => `Epoch ${index + 1}`),
        datasets: [
            {
                label: 'Accuracy',
                data: accuracies,
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                tension: 0.1
            }
        ]
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top' as const,
            },
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    const renderMatrix = (matrix: number[][] | undefined, label: string, subLabel?: string) => (
        matrix && matrix.length > 0 ? (
            <div className="inline-block p-1 mx-1">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="grid border border-gray-400 rounded bg-white shadow-sm" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, auto)` }}>
                    {matrix.map((row, rowIndex) => (
                        row.map((val, colIndex) => (
                            <span key={`${rowIndex}-${colIndex}`} className="px-1.5 py-0.5">
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

    // Split explanation into preview and full content
    const fullExplanation = getExplanation() || '';
    const explanationPreview = fullExplanation.split('\n').slice(0, 5).join('\n');
    const hasMoreContent = fullExplanation.split('\n').length > 5;

    return (
        <div className="mt-8 p-6 bg-gray-100 rounded-lg mx-8 shadow-md">
            {/* View Toggle */}
            <div className="flex justify-center mb-6">
                <div className="inline-flex rounded-md shadow-sm" role="group">
                    <button
                        type="button"
                        onClick={() => setView('forward')}
                        className={`px-4 py-2 text-base font-medium rounded-l-lg ${
                            view === 'forward' 
                                ? 'bg-black text-white' 
                                : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        Forward Propagation
                    </button>
                    <button
                        type="button"
                        onClick={() => setView('backward')}
                        className={`px-4 py-2 text-base font-medium rounded-r-lg ${
                            view === 'backward' 
                                ? 'bg-black text-white' 
                                : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        Backward Propagation
                    </button>
                </div>
            </div>

            {dataset != "mnist" && network?.layers && network?.layers[0].A?.length > 0 && (
                <div className="text-center">
                    {view === 'forward' ? (
                        <>
                            <p className="mt-4 mb-2 text-lg font-bold">Layer-by-layer computation</p>
                            {/* <div className="flex flex-col justify-center items-center flex-wrap gap-4">
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
                            </div> */}
                            <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                                {network?.layers.map((layer: NeuronLayer, layerIndex: number) => (
                                    <div key={layerIndex} className="flex flex-col items-center">
                                        {/* Layer Header */}
                                        {layerIndex < network.layers.length - 1 && (
                                            <h4 className="text-md font-semibold mb-2 text-gray-700">
                                                Layer {layerIndex + 1} Computation
                                            </h4>
                                        )}
                                        
                                        <div className="flex flex-row justify-center items-center">
                                            {layerIndex < network.layers.length - 1 ? (
                                                <>  
                                                    {/* Input/Previous Layer */}
                                                    <div className="flex flex-col items-center">
                                                        {renderVector(
                                                            (layerIndex == 0 ? network?.input[sampleIndex] : network?.layers[layerIndex- 1].A[sampleIndex]), 
                                                            (layerIndex == 0 ? "Input Features" : `Layer ${layerIndex} Activations`),
                                                            layerIndex == 0 ? "Original input data" : "Output from previous layer"
                                                        )}
                                                    </div>

                                                    <span className="text-lg mt-5 mx-2">×</span>

                                                    {/* Weights */}
                                                    <div className="flex flex-col items-center">
                                                        {renderMatrix(
                                                            layer.weights, 
                                                            `Weights`,
                                                            "Connection strengths between neurons"
                                                        )}
                                                    </div>

                                                    <span className="text-lg mt-5 mx-2">+</span>

                                                    {/* Biases */}
                                                    <div className="flex flex-col items-center">
                                                        {renderVector(
                                                            layer.biases, 
                                                            `Biases`
                                                            )}
                                                    </div>

                                                    <span className="text-lg mt-5 mx-2">=</span>

                                                    {/* Output */}
                                                    <div className="flex flex-col items-center">
                                                        <div className="flex flex-col items-center">
                                                            {layer.activation != "linear" && (<>
                                                            {renderVector(
                                                                layer.Z[sampleIndex], 
                                                                (layerIndex === network.layers.length - 2) ? "Raw Output" : `Pre-activations`
                                                            )}
                                                            <div className="relative flex flex-row justify-center items-center py-4">
                                                            <span className="absolute left-1/2 transform -translate-x-1/2 text-lg">↓</span>
                                                            <p className="absolute left-1/2 transform translate-x-2 text-sm text-gray-500 whitespace-nowrap">
                                                                {layer.activation}
                                                            </p>                                                            
                                                            </div>
                                                            </>)}
                                                            {renderVector(
                                                                layer.A[sampleIndex], 
                                                                (layerIndex === network.layers.length - 2) ? "Final Output" : `Activations`
                                                            )}
                                                        </div>
                                                    </div>
                                                </>
                                            ) : null}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <>
                            <p className="mt-6 mb-2 text-lg font-bold">Gradient calculations</p>
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
                        </>
                    )}
                </div>
            )}

            {/* Loss and Accuracy Charts */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white p-4 rounded-lg shadow-md border border-gray-200">
                    <h3 className="text-lg font-medium mb-2">Training Loss</h3>
                    <Line data={lossData} options={chartOptions} />
                </div>
                <div className="bg-white p-4 rounded-lg shadow-md border border-gray-200">
                    <h3 className="text-lg font-medium mb-2">Accuracy</h3>
                    <Line data={accuracyData} options={chartOptions} />
                </div>
            </div>

            {/* Expandable Explanation */}
            <div className="mt-6">
                <div className={`whitespace-pre-line text-gray-700 bg-white p-4 rounded shadow-sm border border-gray-300 ${!expanded ? 'max-h-34 overflow-hidden' : ''}`}>
                    <h3 className="text-xl font-semibold mb-4">Explanation</h3>
                    {expanded ? fullExplanation : explanationPreview}
                    {hasMoreContent && (
                    <div>
                        <button 
                            onClick={() => setExpanded(!expanded)}
                            className="font-bold cursor-pointer text-sm"
                        >
                            {expanded ? 'Show less' : 'Read more...'}
                        </button>
                    </div>
                )}
                </div>
            </div>
        </div>
    );
};

export default Explain;


//     <p className="mb-2">Sample Index:</p>
//     <Input
//         type="number"
//         value={sampleIndex}
//         onChange={(e) => setSampleIndex(Math.max(1, Number(e.target.value)))}
//         min={0}
//         max={network?.input.length - 1}
//         className="w-20 text-center border border-gray-400 rounded-md shadow-sm"
//     />