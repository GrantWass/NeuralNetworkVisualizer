"use client"

import { useState, useCallback, useEffect } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "@/hooks/use-toast"

interface HoveredConnection {
  layerIndex: number
  fromIndex: number
  toIndex: number
  weight: number
}

interface HoveredNode {
  layerIndex: number
  nodeIndex: number
}

class NeuronLayer {
  input_size: number
  output_size: number
  activation: string
  weights: number[][]
  activations: number[]
  biases: number[]
  dW: number[][]
  db: number[]

  constructor(input_size: number, output_size: number, activation: string) {
    this.input_size = input_size
    this.output_size = output_size
    this.activation = activation
    this.weights = []
    this.biases = []
    this.activations = []
    this.dW = []
    this.db = []
  }

  // Method to initialize weights and biases
  initWeightsAndBiases() {
    this.weights = Array(this.output_size)
      .fill(null)
      .map(() => Array(this.input_size).fill(0))
    this.biases = Array(this.output_size).fill(0)
  }
}


interface NetworkState {
  layers: NeuronLayer[]
}


const DATASETS = ["california_housing", "mnist", "iris"]
const ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "tanh", "linear"]
const DATASET_INFO: { [key: string]: string } = {
  california_housing: "The California Housing dataset contains information from the 1990 California census. It includes features like median income, housing median age, average rooms, etc.",
  mnist: "The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems.",
  iris: "The Iris dataset is a multivariate dataset introduced by Ronald Fisher. It consists of 50 samples from each of three species of Iris flowers."
};


const NeuralNetworkViz = () => {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState<boolean>(true)
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([4, 4])
  const [network, setNetwork] = useState<NetworkState | null>(null)
  const [hoveredConnection, setHoveredConnection] = useState<HoveredConnection | null>(null)
  const [hoveredNode, setHoveredNode] = useState<HoveredNode | null>(null)
  const [epoch, setEpoch] = useState<number>(0)
  const [learningRate, setLearningRate] = useState<number>(0.1)
  const [dataset, setDataset] = useState<string>(DATASETS[0])
  const [activations, setActivations] = useState<string[]>(["relu", "relu"])
  const [datasetInfo, setDatasetInfo] = useState<string>(DATASET_INFO[DATASETS[0]])

  const svgWidth = 1000
  const svgHeight = 300
  const nodeRadius = 20

  useEffect(() => {
    if (sessionId) {
      // Clear the session when component unmounts
      return () => {
        fetch(`http://localhost:8000/clear_session?session_id=${sessionId}`, { method: "POST" })
          .then((response) => response.json())
          .then((data) => console.log(data.message))
          .catch((error) => console.error("Error clearing session:", error))
      }
    }
  }, [sessionId])

  const initModel = async () => {
    try {
      const response = await fetch("http://localhost:8000/init_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          layer_sizes: hiddenLayers,
          activations: activations,
          dataset: dataset,
        }),
      })
      const data = await response.json()
      console.log("data", data)
  
      if (response.ok) {
        setSessionId(data.session_id)
        
        const layers = data.layer_sizes.map((size: number, index: number) => {
          const layer = new NeuronLayer(
            index === 0 ? size : data.layer_sizes[index - 1], // input_size
            size, // output_size
            activations[index] || "relu"
          )
          layer.initWeightsAndBiases()
  
          // Initialize layer with returned weights and biases from backend
          console.log("data.network.layers", data.network.layers)
          if (data.network.layers[index]) {
            layer.weights = data.network.layers[index].weights || layer.weights
            layer.biases = data.network.layers[index].biases[0] || layer.biases
          }
  
          return layer
        })
  
        setNetwork({
          layers,
        })
  
        toast({
          title: "Model Initialized",
          description: `Session ID: ${data.session_id}`,
        })
        setConfigOpen(false)
      } else {
        throw new Error(data.error || "Failed to initialize model")
      }
    } catch (error) {
      console.error("Error initializing model:", error)
      toast({
        title: "Error",
        description: "Failed to initialize model. Please try again.",
        variant: "destructive",
      })
    }
  }

  const clearSessionAndReset = async () => {
    if (!sessionId) return;
  
    try {
      // Clear session
      await fetch(`http://localhost:8000/clear_session?session_id=${sessionId}`, { method: "POST" });
      console.log("Session cleared");
  
      // Reset network state
      setNetwork(null);
  
      // Open the config modal
      setConfigOpen(true);
      setHoveredConnection(null);
      setHoveredNode(null);
      setEpoch(0);
      setLearningRate(0.1);
      setDataset(DATASETS[0]);
      setActivations(["relu", "relu"]);
      setHiddenLayers([4, 4]);
  
      toast({
        title: "Configuration Reset",
        description: "Session cleared, network reset, and configuration open.",
      });
    } catch (error) {
      console.error("Error clearing session:", error);
      toast({
        title: "Error",
        description: "Failed to reset configuration. Please try again.",
        variant: "destructive",
      });
    }
  };
  

  const runTrainingCycle = useCallback(async () => {
    if (!sessionId) {
      toast({
        title: "Error",
        description: "Please initialize the model first.",
        variant: "destructive",
      })
      return
    }

    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          learning_rate: learningRate,
          epochs: 1,
        }),
      })
      const data = await response.json()
      if (response.ok) {
        const result = data.training_results[0]
        setNetwork((prevNetwork) => {
          if (!prevNetwork) return null
          return {
            ...prevNetwork,
            weights: result.layers.map((layer: any) => layer.weights),
            biases: result.layers.map((layer: any) => layer.biases),
            activations: result.layers.map((layer: any) => layer.A),
          }
        })
        setEpoch((prevEpoch) => prevEpoch + 1)
        toast({
          title: "Training Cycle Complete",
          description: `Loss: ${result.loss.toFixed(4)}, Metric: ${result.metric.toFixed(4)}`,
        })
      } else {
        throw new Error(data.error || "Failed to run training cycle")
      }
    } catch (error) {
      console.error("Error running training cycle:", error)
      toast({
        title: "Error",
        description: "Failed to run training cycle. Please try again.",
        variant: "destructive",
      })
    }
  }, [sessionId, learningRate])

  const renderConnections = (): React.ReactNode => {
    if (!network) return null;
  
    const layerSpacing = svgWidth / (network.layers.length + 1);
    
    const connections = network.layers.flatMap((layer, layerIndex) => {
      return layer.weights.flatMap((nodeWeights: number[], fromIndex: number) =>
        nodeWeights.map((weight: number, toIndex: number) => {
          let fromX = (layerIndex + 1) * layerSpacing;
          let fromY = ((fromIndex + 1) * svgHeight) / (network.layers[layerIndex].output_size + 1);
          let toX = (layerIndex + 2) * layerSpacing;
          let toY = ((toIndex + 1) * svgHeight) / (network.layers[layerIndex + 1]?.output_size + 1);
  
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
              strokeWidth={Math.abs(weight) * 3}
              onMouseEnter={() => setHoveredConnection({ layerIndex, fromIndex, toIndex, weight })}
              onMouseLeave={() => setHoveredConnection(null)}
            />
          );
        })
      );
    });
  
    return connections.length > 0 ? connections : null; // Ensure you return a valid ReactNode
  };
  
  
  const renderNodes = () => {
    if (!network) return null
    const layerSpacing = svgWidth / (network.layers.length + 1)
    return network.layers.flatMap((layer, layerIndex) =>
      Array.from({ length: layer.output_size }, (_, nodeIndex) => {
        const cx = (layerIndex + 1) * layerSpacing
        const cy = ((nodeIndex + 1) * svgHeight) / (layer.output_size + 1)
  
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
            <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" fontSize="12">
              {/* {layer.activations[nodeIndex]?.toFixed(2) || "0.00"} */} 
            </text>
          </g>
        )
      })
    )
  }
  

  const renderLayerLabels = () => {
    if (!network) return null
    const layerSpacing = svgWidth / network.layers.length
    return network.layers.map((layer, index) => (
      <text
        key={index}
        x={(index + 1) * layerSpacing}
        y={svgHeight - 10}
        textAnchor="middle"
        fontSize="14"
        fontWeight="bold"
      >
        {/* {layer.name} */}
      </text>
    ))
  }

  const addHiddenLayer = () => {
    setHiddenLayers([...hiddenLayers, 4])
    setActivations([...activations, "relu"])
  }

  const removeHiddenLayer = () => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.slice(0, -1))
      setActivations([...activations.slice(0, -1)])
    }
  }

  const updateHiddenLayer = (index: number, value: number) => {
    const newHiddenLayers = [...hiddenLayers]
    newHiddenLayers[index] = value
    setHiddenLayers(newHiddenLayers)
  }

  const updateActivation = (index: number, value: string) => {
    const newActivations = [...activations]
    newActivations[index] = value
    setActivations(newActivations)
  }

  const handleDatasetChange = (newDataset: string) => {
    setDataset(newDataset)
    setDatasetInfo(DATASET_INFO[newDataset])
  }

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mt-4 mb-4">Interactive Neural Network Visualization</h1>
      <div className="mb-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex flex-row justify-between m-6">
        {configOpen ? (
        <>
        <div>
          <Label className="text-lg font-semibold">Hidden Layers</Label>
          {hiddenLayers.map((nodes, index) => (
            <div key={index} className="flex items-center gap-2 mt-2">
              <Input
                type="number"
                value={nodes}
                onChange={(e) => updateHiddenLayer(index, Math.max(1, Number(e.target.value)))}
                min={1}
                className="w-20"
              />
              <Label>nodes in Hidden Layer {index + 1}</Label>
              <Select value={activations[index]} onValueChange={(value) => updateActivation(index, value)}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select activation" />
                </SelectTrigger>
                <SelectContent>
                  {ACTIVATION_FUNCTIONS.map((af) => (
                    <SelectItem key={af} value={af}>
                      {af}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
          <div className="flex gap-2 mt-5">
            <Button onClick={addHiddenLayer}>Add Hidden Layer</Button>
            <Button onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>
              Remove Hidden Layer
            </Button>
          </div>
        </div>
        <div className="flex flex-col items-center gap-4 my-auto">
          <div className="flex items-center gap-4">
          <Label>Dataset:</Label>
          <Select value={dataset} onValueChange={handleDatasetChange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              {DATASETS.map((ds) => (
                <SelectItem key={ds} value={ds}>
                  {ds}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          </div>
          <p className="text-sm text-gray-600 mt-2 mx-8">{datasetInfo}</p>
        </div>
        </>) : null}
        </div>
        {!configOpen ? (<Button onClick={clearSessionAndReset} className="w-full">
          {"Change Configuration"}
        </Button>):
        <Button onClick={initModel} className="w-full">
          Initialize Model
        </Button>
        }
        <div className="flex items-center justify-between">
          <p className="font-semibold">Epoch: {epoch}</p>
        </div>
        <div className="flex items-center gap-4">
          <Label>Learning Rate:</Label>
          <Slider
            value={[learningRate]}
            onValueChange={(value) => setLearningRate(value[0])}
            max={1}
            step={0.01}
            className="w-64"
          />
          <span>{learningRate.toFixed(2)}</span>
        </div>
        <Button onClick={runTrainingCycle} className="w-full" disabled={!sessionId}>
          Run Training Cycle
        </Button>
      </div>
      <div className="grid place-items-center w-full mt-8 mb-6">
        <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
          {renderConnections()}
          {renderNodes()}
          {renderLayerLabels()}
        </svg>
      </div>



      <div className="mt-4 h-20">
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
              Node: Layer {hoveredNode.layerIndex + 1}, Node {hoveredNode.nodeIndex + 1}
            </p>
            <p>
              Activation: {network.layers[hoveredNode.layerIndex - 1]?.activations[hoveredNode.layerIndex]?.toFixed(4) || "N/A"}
            </p>
            {hoveredNode.layerIndex > 0 && (
              <p>Bias: {network.layers[hoveredNode.layerIndex - 1]?.biases[hoveredNode.nodeIndex]?.toFixed(4)|| "N/A"}</p>
            )}
          </div>
        ) : (
          <p>Hover over a node or connection to see details</p>
        )}
      </div>
    </div>
  )
}

export default NeuralNetworkViz

