"use client"
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import useStore from "@/hooks/store";
import { DATASETS, ACTIVATION_FUNCTIONS } from "@/static/constants";

const Config = () => {
  const {
    sessionId,
    configOpen,
    hiddenLayers,
    epoch,
    learningRate,
    dataset,
    activations,
    datasetInfo,
    setLearningRate,
    initModel,
    clearSessionAndReset,
    runTrainingCycle,
    addHiddenLayer,
    removeHiddenLayer,
    updateHiddenLayer,
    updateActivation,
    handleDatasetChange,
  } = useStore();

  return (
    <div className="mb-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex flex-row justify-between items-center p-4">
        <div>
          <Label className="text-lg font-semibold">Dataset</Label>
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
        <div className="flex gap-4">
          {!configOpen ?
          <Button onClick={clearSessionAndReset}>Change Model</Button>
          :
          <Button onClick={initModel}>Initialize Model</Button>}
          <Button onClick={runTrainingCycle} disabled={configOpen}>Run Model</Button>
        </div>
      </div>
      {configOpen ? (
        <div className="flex flex-row justify-between items-start p-4 border-t">
          <Button onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>Remove Hidden Layer</Button>
          <div className="flex flex-col gap-2">
            <Label className="text-lg font-semibold">Hidden Layers</Label>
            {hiddenLayers.map((nodes, index) => (
              <div key={index} className="flex items-center gap-2">
                <Input
                  type="number"
                  value={nodes}
                  onChange={(e) => updateHiddenLayer(index, Math.max(1, Number(e.target.value)))}
                  min={1}
                  className="w-20"
                />
                <Label>Layer {index + 1}</Label>
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
          </div>
          <Button onClick={addHiddenLayer}>Add Hidden Layer</Button>
        </div>
      ) : (
        <div className="flex flex-col gap-4 p-4 border-t">
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
      )}
    </div>
  );
};

export default Config;
