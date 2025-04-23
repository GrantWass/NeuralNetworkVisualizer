"use client";

import useStore from "@/hooks/store";
import { NeuronLayer } from "@/static/types";
import { useState } from "react";
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
import { InputInfo, OutputInfo } from "@/components/tooltips";
import { reshapeTo2D, transpose, multiplyMatrices } from "@/lib/utils"
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

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

type PropagationView = 'forward' | 'backward' | 'calculation';

const Explain = () => {
    const { network, getExplanation, dataset, losses, accuracies, learningRate, sampleIndex, originalData, name } = useStore();
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

    const renderMatrix = (matrix: number[][] | undefined | null, label: string, subLabel?: string, extendDecimal?: boolean) => (
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

    const renderVector = (vector: number[] | undefined | null, label: string, subLabel?: string, extendDecimal?: boolean, tooltip?: React.ReactNode) => (
        vector && vector.length > 0 ? (
            <div className="inline-block px-1 mx-1">
                <div className="flex flex-row items-center justify-center gap-1 relative group">
                <p className="text-sm text-gray-600 text-center mb-1">{label}</p>
                {tooltip as React.ReactNode}
                </div>
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
                        className={`px-4 py-2 text-base font-medium ${
                            view === 'backward' 
                                ? 'bg-black text-white' 
                                : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        Parameter Updates
                    </button>
                    <button
                        type="button"
                        onClick={() => setView('calculation')}
                        className={`px-4 py-2 text-base font-medium rounded-r-lg ${
                            view === 'calculation' 
                                ? 'bg-black text-white' 
                                : 'bg-white text-black hover:bg-gray-100 border border-gray-300'
                        }`}
                    >
                        Gradient Calculation
                    </button>
                </div>
            </div>)}

            {dataset != "mnist" && network?.layers && network?.layers[0].A?.length > 0 && (
                <div className="text-center">
                    {view === 'forward' ? (
                        <>
                            <p className="mt-4 mb-2 text-lg font-bold">Layer-by-layer computation</p>
                            <p className="text-sm text-gray-600 mb-2">Forward propagation from input to prediction</p>
                            <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                                {network?.layers.map((layer: NeuronLayer, layerIndex: number) => (
                                    <div key={layerIndex} className="flex flex-col items-center border-t border-gray-300 pt-4">
                                        {/* Layer Header */}
                                        {layerIndex < network.layers.length - 1 && (
                                            <h4 className="text-md font-semibold mb-2 text-gray-700">
                                                {network.layers.length - 2 === layerIndex ? "Output Layer Computation":`Layer ${layerIndex + 1} Computation`}
                                            </h4>
                                        )}
                                        
                                        <div className="flex flex-row justify-center items-center">
                                            {layerIndex < network.layers.length - 1 ? (
                                                <>  
                                                    <div className="flex flex-col items-center">
                                                    {renderVector(
                                                        layerIndex === 0
                                                        ? network?.input[sampleIndex]
                                                        : network?.layers[layerIndex - 1].A[sampleIndex],
                                                        (layerIndex == 0 ? "Input Features" : `Layer ${layerIndex} Activations`),
                                                        layerIndex == 0 ? "Original input data" : "Output from previous layer",
                                                        false,
                                                        layerIndex == 0 ? <InputInfo dataset={dataset} input={network?.input[sampleIndex]} originalInput={originalData[sampleIndex]} /> : null
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
                                                                (layerIndex === network.layers.length - 2) ? "Final Output" : `Activations`,
                                                                "",
                                                                (layerIndex === network.layers.length - 2) ? true : false,
                                                                (layerIndex === network.layers.length - 2) ? <OutputInfo dataset={dataset} output={layer.A[sampleIndex]} actual={originalData[sampleIndex]} />: null
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
                    ) : view === 'backward' ?
                        (<>
                            <p className="mt-6 mb-2 text-lg font-bold">Updating weight and bias values</p>
                            <p className="text-sm text-gray-600 mb-2">Adjusting parameters to minimize loss</p>
                            <p className="text-sm text-gray-600 mb-2">
                                <strong className="text-xl text-gray-600">η</strong>
                                {` represents learning rate (`}
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
                                    <h2 className="text-md font-semibold mb-2 text-gray-700">Output Layer</h2>
                                    : <h2 className="text-md font-semibold mb-2 text-gray-700">Layer {layerIndex}</h2>}

                                    {/* Weights Equation */}
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderMatrix(layer.prevWeights, `Previous Weights`, "", true)}
                                    <span className="text-xxl mt-8">-</span>
                                    <span className="text-sm mt-8 text-gray-600"><strong className="text-xl text-gray-600">η</strong> ×</span>
                                    {renderMatrix(layer.dW, `Change in Weights`, "dW (∇Weights)", true)}
                                    <span className="text-xl mt-8">=</span>
                                    {renderMatrix(layer.weights, `Current Weights`, "", true)}
                                    </div>

                                    {/* Biases Equation */}
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderVector(layer.prevBias, `Previous Biases`, "", true)}
                                    <span className="text-xxl mt-8">-</span>
                                    <span className="text-sm mt-8 text-gray-600"><strong className="text-xl text-gray-600">η</strong> ×</span>
                                    {renderVector(layer.db, `Change in Biases`, "dB (∇Biases)", true)}
                                    <span className="text-xl mt-8">=</span>
                                    {renderVector(layer.biases, `Current Biases`, "", true)}
                                     </div>
                                </div>
                                );
                            })}
                            </div>
                        </>
                    ) :
                    <div className="flex flex-col justify-center items-center flex-wrap">
                    <p className="mt-6 mb-2 text-lg font-bold">Propogating error backwards through model</p>
                    <p className="text-sm text-gray-600 mb-2">Calculating how each neuron contributed to the error</p>
                    <div className="flex flex-col justify-center items-center flex-wrap gap-4">
                    {network?.layers.slice().reverse().map((layer: NeuronLayer, index: number) => {

                        const initialContent = index == 0
                        const outputLayerIndex = network.layers.length - 2
                        const actual = originalData[sampleIndex].slice(dataset === "iris" ? -3 : -1)
                        const dA = multiplyMatrices([layer.dZ[sampleIndex]], transpose(network.layers[outputLayerIndex - index + 1]?.weights))

                        return (
                            <div key={`layer-${index}-update`} className="flex flex-col gap-2 items-center border-t border-gray-300 pt-6">
                            {/* Heading */}
                            <h2 className="text-md font-semibold text-gray-700">
                                {initialContent
                                ? "Error from Result"
                                : `Backpropagation — ${
                                    1 === index
                                        ? `Output Layer → Layer ${outputLayerIndex - index + 1}`
                                        : `Layer ${outputLayerIndex - index + 2} → ${(outputLayerIndex - index + 1) === 0 ? "Input" : ""} Layer ${(outputLayerIndex - index + 1) === 0 ? "" : outputLayerIndex - index + 1}`
                                    }`}
                            </h2>

                            {/* Error Calculation */}
                            {initialContent ? (
                                <div className="flex flex-col items-center gap-3">
                                <p className="text-sm text-gray-600">Calculate initial error (dZ) at the output:</p>
                                <p className="text-xs text-gray-600 italic">*Note that is is done with all samples at once, opposed to one sample as seen here*</p>
                                <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderVector(network.layers[outputLayerIndex].A[sampleIndex], `Prediction`, "", true)}
                                    <span className="text-2xl mt-6">-</span>
                                    {renderVector(actual, `Actual`, "", false)}
                                    <span className="text-2xl mt-6">=</span>
                                    {renderVector(network.layers[outputLayerIndex].dZ[sampleIndex], `Error`, "", true)}
                                </div>
                                </div>
                            ) : (
                                <div className="flex flex-col gap-4 items-center">

                                {/* dZ Calculation */}
                                <div className="flex flex-col items-center gap-2">
                                    {(outputLayerIndex - index + 1) !== 0 ?
                                    <>
                                    <p className="text-sm text-gray-600">
                                        To propagate the error backward, compute <code>dZ</code> by taking the dot product of the forward layer’s <code>dZ</code> and transposed weights, then apply the activation derivative.
                                    </p>
                                    <p className="text-xs text-gray-600 italic">*Note that is is done with all samples at once, opposed to one sample as seen here*</p>
                                    </>
                                    : 
                                    <p className="text-sm text-gray-600 italic"> No backward calculation needed here — this is the input layer</p>}                                
                                    <div className="flex flex-col items-center gap-2 flex-wrap justify-center">
                                        <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                        {renderVector(layer.dZ[sampleIndex], `dZ`, "Calculated above", true)}
                                        {(outputLayerIndex - index + 1) !== 0 ? <>
                                            <span className="text-2xl mt-6">×</span>
                                            {renderMatrix(transpose(network.layers[outputLayerIndex - index + 1]?.weights), `Wᵀ`, "Transposed weights from backwards layer", true)}
                                            <span className="text-2xl mt-6">=</span>
                                            {renderMatrix(dA, `dA`, "∇ Activations", true)}
                                        </> : null}
                                        </div>
                                        {(outputLayerIndex - index + 1) !== 0 ? <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                            {renderMatrix(dA, `dA`, "∇ Activations", true)}
                                            <span className="text-2xl mt-6">×</span>
                                            <div className="inline-block px-1 mx-1 text-center">
                                                <p className="text-sm text-gray-600 mb-1">{"Activation Derivative (σ′(Z))"}</p>
                                                <p className="text-xs text-gray-500 mb-1">Derivative of the activation function at this layer</p>
                                                <div className="border border-gray-400 rounded px-2 py-1 bg-white shadow-sm">
                                                    <span className="text-2xl">{`σ′(Z) `} </span>
                                                    <span className="text-xl text-gray-700 italic" >{`(${network.layers[outputLayerIndex - index ].activation})`} </span>
                                                </div>
                                            </div>
                                            <span className="text-2xl mt-6">=</span>
                                            {renderVector(network.layers[outputLayerIndex - index]?.dZ[sampleIndex], `dZ`, "Error to pass to backwards layer", true)}
                                        </div> : null}
                                    </div>
                                </div>
                                <h2 className="text-md font-semibold mt-6 text-gray-700">Calculating change in weight and biases</h2>

                                <div className="flex flex-row items-center justify-center gap-12 flex-wrap">

                                {/* dW Calculation */}
                                <div className="flex flex-col items-center gap-2">
                                    <p className="text-sm text-gray-600">Gradient of weights (dW) is based on current layer&apos;s dZ and previous activations:</p>
                                    <p className="text-xs text-gray-600 italic">*Note that is is done with all samples at once, opposed to one sample as seen here*</p>
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    {renderMatrix(reshapeTo2D(outputLayerIndex - index === -1 ? network.input[sampleIndex] :network.layers[outputLayerIndex - index].A[sampleIndex]), `Previous Activations`, "Aᵀ from layer below", false)}
                                    <span className="text-2xl mt-6">×</span>
                                    {renderVector(layer.dZ[sampleIndex], `dZ`, "", false)}
                                    <span className="text-2xl mt-6">=</span>
                                    {renderMatrix(layer.dW, `dW`, "∇ Weights", false)}
                                    </div>
                                </div>

                                {/* db Calculation */}
                                <div className="flex flex-col items-center gap-2">
                                    <p className="text-sm text-gray-600">Gradient of biases (dB) is the sum of dZ across samples:</p>
                                    <div className="flex flex-row items-center gap-2 flex-wrap justify-center">
                                    <span className="text-2xl mt-6">Σ</span>
                                    <div className="flex flex-col gap-1">
                                        {renderVector(layer.dZ[sampleIndex], `dZ`, "Different sample's dZ shown", false)}
                                        {renderVector(layer.dZ[sampleIndex + 1], ``, "", false)}
                                        ⋮
                                        {renderVector(layer.dZ[sampleIndex + 2], ``, "", false)}
                                    </div>
                                    <span className="text-2xl mt-6">=</span>
                                    {renderVector(layer.db, `dB`, "∇ Biases", true)}
                                    </div>
                                </div>
                                </div>
                                </div>
                            )}
                            </div>
                            )})}
                        </div>
                    </div>
                }
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
                    <ReactMarkdown
                        rehypePlugins={[rehypeRaw]}
                        components={{
                            a: ({ node, ...props }) => (
                            <a
                                {...props}
                                className="text-blue-500 underline font-medium hover:text-blue-600"
                                target="_blank"
                                rel="noopener noreferrer"
                            />
                            ),
                            ul: ({ node, ...props }) => (
                            <ul {...props} className="list-disc pl-5 mt-[-30] mb-[-10]" />
                            ),
                            li: ({ node, ...props }) => (
                            <li {...props} className="mb-[-15]" />
                            ),
                        }}
                    >
                        {expanded ? fullExplanation : explanationPreview}
                    </ReactMarkdown>
                    
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

