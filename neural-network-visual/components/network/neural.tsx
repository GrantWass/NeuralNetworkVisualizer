"use client"

import useStore from "@/components/network/lib/store";
import { Network, GradientFlowOverlay } from "@/components/network/network";
import { useEffect, useRef, useState } from "react";

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
      <div className="w-5 h-1 bg-indigo-400 rounded" />
      <span>Positive weight</span>
    </div>
    <div className="flex items-center gap-1.5">
      <div className="w-5 h-1 bg-orange-400 rounded" />
      <span>Negative weight</span>
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
        sampleIndex,
        dataset,
        originalData,
        changedConnections,
        stepLayerHighlight,
        yMean,
        yStd,
        epoch,
    } = useStore();


    // Flash changed connections for 2s after each training cycle
    const [flashConnections, setFlashConnections] = useState<typeof changedConnections>([]);
    const [flashKey, setFlashKey] = useState(0);

    useEffect(() => {
        if (changedConnections.length > 0) {
            setFlashConnections(changedConnections);
            setFlashKey(k => k + 1);
            const timer = setTimeout(() => setFlashConnections([]), 2000);
            return () => clearTimeout(timer);
        }
    }, [changedConnections]);

    // Gradient flow animation: trigger on each completed training epoch
    const [gradAnimKey, setGradAnimKey] = useState(0);
    const [showGradFlow, setShowGradFlow] = useState(false);
    const prevEpochRef = useRef(0);

    useEffect(() => {
        if (epoch > 0 && epoch !== prevEpochRef.current) {
            prevEpochRef.current = epoch;
            setGradAnimKey(k => k + 1);
            setShowGradFlow(true);
            const timer = setTimeout(() => setShowGradFlow(false), 1800);
            return () => clearTimeout(timer);
        }
    }, [epoch]);

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
                    yMean={yMean}
                    yStd={yStd}
                />
                {/* Gradient flow animation — disabled for now
                {showGradFlow && network && (
                    <GradientFlowOverlay
                        network={network}
                        SVGWIDTH={1000}
                        SVGHEIGHT={500}
                        animKey={gradAnimKey}
                        dataset={dataset}
                    />
                )}
                */}
            </svg>

            {/* Legend */}
            <Legend />


        </div>
    );
};

export default Graph;
