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
    const { network, getExplanation, dataset, losses, accuracies, learningRate, sampleIndex, setSampleIndex, name } = useStore();
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
                label: name === "accuracy" ? 'Accuracy' : 'Mean Absolute Error',
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

    const renderMatrix = (matrix: number[][] | undefined, label: string, subLabel?: string, extendDecimal?: boolean) => (
        matrix && matrix.length > 0 ? (
            <div className="inline-block p-1 mx-1">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="grid border border-gray-400 rounded bg-white shadow-sm" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, auto)` }}>
                    {matrix.map((row, rowIndex) => (
                        row.map((val, colIndex) => (
                            <span key={`${rowIndex}-${colIndex}`} className="px-1.5 py-0.5">
                                {val.toFixed(extendDecimal ? 3 : 2)}
                            </span>
                        ))
                    ))}
                </div>
            </div>
        ) : null
    );

    const renderVector = (vector: number[] | undefined | null, label: string, subLabel?: string, extendDecimal?: boolean) => (
        vector && vector.length > 0 ? (
            <div className="inline-block px-1 mx-1">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {subLabel && <p className="text-xs text-gray-500 text-center mb-1">{subLabel}</p>}
                <div className="flex flex-row justify-center border border-gray-400 px-1 py-0.5 rounded bg-white shadow-sm">
                    {vector.map((val, index) => (
                        <span key={index} className="px-1 py-0.5">{val.toFixed(extendDecimal ? 3 : 2)}</span>
                    ))}
                </div>
            </div>
        ) : null
    );

    // Split explanation into preview and full content
    const fullExplanation = getExplanation() || '';
    const explanationPreview = fullExplanation.split('\n').slice(0, 5).join('\n');
    const hasMoreContent = fullExplanation.split('\n').length > 5;

    const inputComponent = <div className="flex items-center gap-1">
        Input Features
        <div className="relative group">
            <span className="cursor-pointer text-sm text-gray-500">ⓘ</span>
            <div className="absolute z-10 hidden group-hover:block w-64 p-2 text-xs text-white bg-gray-700 rounded shadow-md top-6 left-0">
            {dataset === "california_housing" && (
                <>
                <strong>California Housing Features:</strong>
                <ul className="list-disc ml-4 mt-1">
                    <li>MedInc</li>
                    <li>HouseAge</li>
                    <li>AveRooms</li>
                    <li>AveBedrms</li>
                    <li>Population</li>
                    <li>AveOccup</li>
                    <li>Latitude</li>
                    <li>Longitude</li>
                </ul>
                </>
            )}
            {dataset === "iris" && (
                <>
                <strong>Iris Features:</strong>
                <ul className="list-disc ml-4 mt-1">
                    <li>Sepal length</li>
                    <li>Sepal width</li>
                    <li>Petal length</li>
                    <li>Petal width</li>
                </ul>
                </>
            )}
            </div>
        </div>
        </div>

    return (
        <div className="mt-8 p-6 bg-gray-100 rounded-lg mx-8 shadow-md">
            {dataset != "mnist" && network?.layers && network?.layers[0].A?.length > 0 && (
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
            </div>)}

            {dataset != "mnist" && network?.layers && network?.layers[0].A?.length > 0 && (
                <div className="text-center">
                    {view === 'forward' ? (
                        <>
                            <p className="mt-4 mb-2 text-lg font-bold">Layer-by-layer computation</p>
                            <p className="text-sm text-gray-600 mb-2">Forward Propagation: Input to Output</p>
                            <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                                {network?.layers.map((layer: NeuronLayer, layerIndex: number) => (
                                    <div key={layerIndex} className="flex flex-col items-center border-t border-gray-300 pt-4">
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
                                                    {/* <div className="flex flex-col items-center">
                                                        {renderVector(
                                                            (layerIndex == 0 ? network?.input[sampleIndex] : network?.layers[layerIndex- 1].A[sampleIndex]), 
                                                            (layerIndex == 0 ? inputComponent : `Layer ${layerIndex} Activations`),
                                                            layerIndex == 0 ? "Original input data" : "Output from previous layer"
                                                        )}
                                                    </div> */}
                                                    <div className="flex flex-col items-center">
                                                    {layerIndex === 0 ? (
                                                        <>
                                                        <div className="flex items-center gap-1 text-sm text-gray-600 mb-1 relative group">
                                                            <span>Input Features</span>
                                                            <span className="cursor-pointer text-gray-500 hover:text-gray-700 transition-colors duration-150">
                                                            <svg
                                                                xmlns="http://www.w3.org/2000/svg"
                                                                className="h-4 w-4"
                                                                fill="none"
                                                                viewBox="0 0 24 24"
                                                                stroke="currentColor"
                                                                strokeWidth={2}
                                                            >
                                                                <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z"
                                                                />
                                                            </svg>
                                                            </span>

                                                            {/* Tooltip */}
                                                            <div className="absolute z-10 hidden group-hover:block w-32 p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg bottom-6 left-1/2 transform -translate-x-1/2 transition-all duration-200">
                                                            {dataset === "california_housing" && (
                                                                <>
                                                                <p className="font-semibold mb-1">Dataset Features:</p>
                                                                <p className=" mb-1">{`(Normalized)`}</p>
                                                                <ul className="list-disc pl-5 space-y-1 text-sm text-gray-100 text-left">
                                                                    <li>MedInc</li>
                                                                    <li>HouseAge</li>
                                                                    <li>AveRooms</li>
                                                                    <li>AveBedrms</li>
                                                                    <li>Population</li>
                                                                    <li>AveOccup</li>
                                                                    <li>Latitude</li>
                                                                    <li>Longitude</li>
                                                                </ul>
                                                                </>
                                                            )}
                                                            {dataset === "iris" && (
                                                                <>
                                                                <p className="font-semibold mb-1">Iris Features:</p>
                                                                <p className=" mb-1">{`(Normalized)`}</p>
                                                                <ul className="list-disc pl-5 space-y-1 text-sm text-gray-100 text-left">
                                                                    <li>Sepal length</li>
                                                                    <li>Sepal width</li>
                                                                    <li>Petal length</li>
                                                                    <li>Petal width</li>
                                                                </ul>
                                                                </>
                                                            )}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-gray-500 text-center mb-1">Original input data</p>
                                                        </>
                                                    ) : (
                                                        <>
                                                        <p className="text-sm text-gray-600 text-center mb-1">{`Layer ${layerIndex} Activations`}</p>
                                                        <p className="text-xs text-gray-500 text-center mb-1">Output from previous layer</p>
                                                        </>
                                                    )}
                                                    {renderVector(
                                                        layerIndex === 0
                                                        ? network?.input[sampleIndex]
                                                        : network?.layers[layerIndex - 1].A[sampleIndex],
                                                        "",
                                                        ""
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
                            <p className="text-sm text-gray-600 mb-2">Backpropagation: Adjusting weights and biases</p>
                            <p className="text-sm text-gray-600 mb-2">
                                <strong className="text-xl text-gray-600">η</strong>
                                {` represents leanring rate (`}
                                <strong className="text-l text-gray-600">{learningRate}</strong>
                                {`)`}
                            </p>
                            <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                            {network?.layers.slice().reverse().map((layer: NeuronLayer, reversedIndex: number) => {

                                const layerIndex = network.layers.length - reversedIndex;

                                if (reversedIndex === 0) {
                                    return null; 
                                }

                                return (
                                <div key={`layer-${reversedIndex}-backprop`} className="flex flex-col gap-4 items-center border-t border-gray-300 pt-4">
                                    {layerIndex === network.layers.length - 1 ?
                                    <h2 className="text-lg font-semibold text-center">Output Layer</h2>
                                    : <h2 className="text-lg font-semibold text-center">Layer {layerIndex}</h2>}

                                    {/* Weights Equation */}
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderMatrix(layer.prevWeights, `Layer ${layerIndex} prevWeight`, "Previous Weights", true)}
                                    <span className="text-xxl mt-8">-</span>
                                    <span className="text-sm mt-8 text-gray-600"><strong className="text-xl text-gray-600">η</strong> ×</span>
                                    {renderMatrix(layer.dW, `Layer ${layerIndex} dW`, "dW (∇Weights)", true)}
                                    <span className="text-xl mt-8">=</span>
                                    {renderMatrix(layer.weights, `Layer ${layerIndex} Weight`, "Current Weights", true)}
                                    </div>

                                    {/* Biases Equation */}
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderVector(layer.prevBias, `Layer ${layerIndex} prevBias`, "Previous Biases", true)}
                                    <span className="text-xxl mt-8">-</span>
                                    <span className="text-sm mt-8 text-gray-600"><strong className="text-xl text-gray-600">η</strong> ×</span>
                                    {renderVector(layer.db, `Layer ${layerIndex} dB`, "dB (∇Biases)", true)}
                                    <span className="text-xl mt-8">=</span>
                                    {renderVector(layer.biases, `Layer ${layerIndex} Bias`, "Current Biases", true)}
                                     </div>
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
                    <h3 className="text-lg font-medium mb-2">{name === "accuracy" ? 'Accuracy' : 'Mean Absolute Error'}</h3>
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

