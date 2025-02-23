"use client"
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import useStore from "@/hooks/store";
import { DATASETS, DATASET_INFO, ACTIVATION_FUNCTIONS } from "@/static/constants";
import { renderConnections, renderNodes, renderLayerLabels } from "@/lib/network";


const NeuralNetworkViz = () => {
  const {
    sessionId,
    configOpen,
    hiddenLayers,
    network,
    hoveredConnection,
    hoveredNode,
    epoch,
    learningRate,
    dataset,
    activations,
    datasetInfo,
    setHoveredConnection,
    setHoveredNode,
    setLearningRate,
    initModel,
    clearSessionAndReset,
    runTrainingCycle,
    addHiddenLayer,
    removeHiddenLayer,
    updateHiddenLayer,
    updateActivation,
    handleDatasetChange
  } = useStore();

  const svgWidth = 1000;
  const svgHeight = 400;
  const nodeRadius = 20;

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
            </>
          ) : null}
        </div>
        {!configOpen ? (
          <Button onClick={clearSessionAndReset} className="w-full">
            {"Change Configuration"}
          </Button>
        ) : (
          <Button onClick={initModel} className="w-full">
            Initialize Model
          </Button>
        )}
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
        {renderConnections({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderNodes({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderLayerLabels({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
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
              <p>Bias: {network.layers[hoveredNode.layerIndex - 1]?.biases[hoveredNode.nodeIndex]?.toFixed(4) || "N/A"}</p>
            )}
          </div>
        ) : (
          <p>Hover over a node or connection to see details</p>
        )}
      </div>
    </div>
  );
};

export default NeuralNetworkViz;