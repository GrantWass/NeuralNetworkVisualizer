"use client"
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import InfoPopup from "@/components/popup"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import useStore from "@/hooks/store";
import { DATASETS, ACTIVATION_FUNCTIONS } from "@/static/constants";
import { HIDDEN_LAYER_INFO, HIDDEN_LAYER_LEARN_MORE } from "@/static/explanation";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

const Config = () => {
  const {
    sessionId,
    configOpen,
    hiddenLayers,
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
    initModelFrontend,
    sampleIndex,
    setSampleIndex
  } = useStore();

  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const openPopup = () => setIsPopupOpen(true);
  const closePopup = () => setIsPopupOpen(false);

  useEffect(() => {
    initModelFrontend()
  }, [initModelFrontend]); 


  const onChangeModel = () => {
    handleDatasetChange("iris")
    clearSessionAndReset();
  }

  return (
    <div className="mb-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex flex-col justify-between justify-center p-4">
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
          <Button onClick={onChangeModel}>Change Model</Button>
          :
          <Button onClick={initModel}>INITIALIZE MODEL</Button>}
        </div>
      </div>
      <div className="p-4 bg-gray-100 rounded-lg">
      <div className="whitespace-pre-wrap prose">
      <ReactMarkdown
          rehypePlugins={[rehypeRaw]}
          components={{
            a: ({...props }) => (
              <a
                {...props}
                className="text-black-500 underline font-medium hover:text-blue-600"
                target="_blank"
                rel="noopener noreferrer"
              />
            ),
            ul: ({...props }) => (
              <ul {...props} className="list-disc pl-5 mt-[-30px] mb-[-10px]" />
            ),
            li: ({...props }) => (
              <li {...props} className="mb-[-15px]" />
            ),
          }}
        >
          {datasetInfo}
        </ReactMarkdown>
      </div>
      </div>
      </div>
      
      {configOpen ? (
        <div className="flex flex-row justify-between items-start p-4 border-t gap-6">
          <div className="flex flex-col justify-center gap-4">
            <Button onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>Remove Hidden Layer</Button>
            <Button onClick={addHiddenLayer}>Add Hidden Layer</Button>
          </div>
          <div className="flex flex-col gap-2">
            <Label className="text-lg font-semibold">Hidden Layers</Label>
            <div className="flex items-center gap-32">
              <p className="font-semibold">Size:</p>
              <p className="font-semibold">Activation:</p>
            </div>
            {hiddenLayers.map((nodes, index) => (
              <div key={index} className="flex items-center gap-2">
                <Input
                  type="number"
                  value={nodes}
                  onChange={(e) => updateHiddenLayer(index, Math.max(1, Number(e.target.value)))}
                  min={1}
                  max={10}
                  className="w-20"
                />
                <Label className="w-[75px]">Layer {index + 1}</Label>
                <Select value={activations[index]} onValueChange={(value) => updateActivation(index, value)}>
                  <SelectTrigger className="w-[150px]">
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
          <div className="p-4 bg-gray-100 rounded-lg">
          <div className="whitespace-pre-wrap prose">
            <ReactMarkdown
            rehypePlugins={[rehypeRaw]}
            components={{
              a: ({...props }) => (
                <a
                  {...props}
                  className="text-black-500 underline font-medium hover:text-blue-600"
                  target="_blank"
                  rel="noopener noreferrer"
                />
              ),
              ul: ({...props }) => (
                <ul {...props} className="list-disc pl-5 mt-[-30px] mb-[-10px]" />
              ),
              li: ({...props }) => (
                <li {...props} className="mb-[-15px]" />
              ),
            }}
          >
            {HIDDEN_LAYER_INFO}
          </ReactMarkdown>
          </div>
            <p className="font-bold cursor-pointer" onClick={openPopup} >Learn More</p> 
          </div>
          {isPopupOpen && (
            <InfoPopup
              title="Hidden Layer Information"
              message={HIDDEN_LAYER_LEARN_MORE}
              onClose={closePopup}
            />
          )}
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
            <Label className="ml-10">Input Sample Number:</Label>
            <Input
                type="number"
                value={sampleIndex}
                onChange={(e) => setSampleIndex(Math.max(0, Math.min(Number(e.target.value),25)))}
                min={0}
                max={25}
                className="w-15 text-center border border-gray-400 rounded-md shadow-sm"
            />
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
