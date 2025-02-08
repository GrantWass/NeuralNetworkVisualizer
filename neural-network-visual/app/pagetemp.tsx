"use client"

import { useState, useCallback } from "react"
import { createNetwork } from "@/lib/network"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

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

const NeuralNetworkViz = () => {
  const [inputNodes] = useState<number>(3)
  const [outputNodes] = useState<number>(2)
  const [hiddenLayers, setHiddenLayers] = useState<number[]>([4, 4])
  const [network, setNetwork] = useState(() => createNetwork(inputNodes, outputNodes, hiddenLayers))
  const [hoveredConnection, setHoveredConnection] = useState<HoveredConnection | null>(null)
  const [hoveredNode, setHoveredNode] = useState<HoveredNode | null>(null)
  const [epoch, setEpoch] = useState<number>(0)
  const [learningRate, setLearningRate] = useState<number>(0.1)

  const svgWidth = 800
  const svgHeight = 400
  const nodeRadius = 20
  const layerSpacing = svgWidth / (network.layers.length + 1)

  const runTrainingCycle = useCallback(() => {
    setNetwork((prevNetwork) => {
      const newNetwork = JSON.parse(JSON.stringify(prevNetwork))
      // Update weights and biases
      newNetwork.weights = newNetwork.weights.map((layerWeights: any[]) =>
        layerWeights.map((nodeWeights) => nodeWeights.map((weight: number) => weight + (Math.random() - 0.5) * learningRate)),
      )
      newNetwork.biases = newNetwork.biases.map((layerBiases: any[]) =>
        layerBiases.map((bias) => bias + (Math.random() - 0.5) * learningRate),
      )
      // Update activations
      newNetwork.activations = newNetwork.activations.map((layerActivations: string[]) =>
        layerActivations.map(() => Math.random()),
      )
      return newNetwork
    })
    setEpoch((prevEpoch) => prevEpoch + 1)
  }, [learningRate])

  const renderConnections = () => {
    return network.weights.flatMap((layerWeights: number[][], layerIndex: number) =>
      layerWeights.flatMap((nodeWeights: number[], fromIndex: number) =>
        nodeWeights.map((weight: number, toIndex: number) => {
          const fromX = (layerIndex + 1) * layerSpacing
          const fromY = ((fromIndex + 1) * svgHeight) / (network.layers[layerIndex].nodes + 1)
          const toX = (layerIndex + 2) * layerSpacing
          const toY = ((toIndex + 1) * svgHeight) / (network.layers[layerIndex + 1].nodes + 1)

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
          )
        }),
      ),
    )
  }

  const renderNodes = () => {
    return network.layers.flatMap((layer, layerIndex) =>
      Array.from({ length: layer.nodes }, (_, nodeIndex) => {
        const cx = (layerIndex + 1) * layerSpacing
        const cy = ((nodeIndex + 1) * svgHeight) / (layer.nodes + 1)

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
              {network.activations[layerIndex][nodeIndex].toFixed(2)}
            </text>
          </g>
        )
      }),
    )
  }

  const renderLayerLabels = () => {
    return network.layers.map((layer, index) => (
      <text
        key={index}
        x={(index + 1) * layerSpacing}
        y={svgHeight - 10}
        textAnchor="middle"
        fontSize="14"
        fontWeight="bold"
      >
        {layer.name}
      </text>
    ))
  }

  const updateNetwork = () => {
    setNetwork(createNetwork(inputNodes, outputNodes, hiddenLayers))
    setEpoch(0)
  }

  const addHiddenLayer = () => {
    setHiddenLayers([...hiddenLayers, 4])
  }

  const removeHiddenLayer = () => {
    if (hiddenLayers.length > 1) {
      setHiddenLayers(hiddenLayers.slice(0, -1))
    }
  }

  const updateHiddenLayer = (index: number, value: number) => {
    const newHiddenLayers = [...hiddenLayers]
    newHiddenLayers[index] = value
    setHiddenLayers(newHiddenLayers)
  }

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Interactive Neural Network Visualization</h1>
      <div className="mb-4 space-y-4">
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
            </div>
          ))}
          <div className="flex gap-2 mt-2">
            <Button onClick={addHiddenLayer}>Add Hidden Layer</Button>
            <Button onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>
              Remove Hidden Layer
            </Button>
          </div>
        </div>
        <Button onClick={updateNetwork} className="w-full">
          Update Network Structure
        </Button>
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
        <Button onClick={runTrainingCycle} className="w-full">
          Run Training Cycle
        </Button>
      </div>
      <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
        {renderConnections()}
        {renderNodes()}
        {renderLayerLabels()}
      </svg>
      <div className="mt-4 h-20">
        {" "}
        {hoveredConnection ? (
          <div>
            <p>
              Connection: Layer {hoveredConnection.layerIndex + 1}, Node {hoveredConnection.fromIndex + 1} to Node{" "}
              {hoveredConnection.toIndex + 1}
            </p>
            <p>Weight: {hoveredConnection.weight.toFixed(4)}</p>
          </div>
        ) : hoveredNode ? (
          <div>
            <p>
              Node: Layer {hoveredNode.layerIndex + 1}, Node {hoveredNode.nodeIndex + 1}
            </p>
            <p>Activation: {network.activations[hoveredNode.layerIndex][hoveredNode.nodeIndex].toFixed(4)}</p>
            {hoveredNode.layerIndex > 0 && (
              <p>Bias: {network.biases[hoveredNode.layerIndex - 1][hoveredNode.nodeIndex].toFixed(4)}</p>
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

