"use client"

import useStore from "@/hooks/store";
import { useEffect, useState } from "react";
import { renderConnections, renderNodes, renderLayerLabels, maxNodes } from "@/lib/network";


const Graph = () => {
    const svgWidth = 1000;
    const svgHeight = 400;
    const nodeRadius = 20;
    const {
        network,
        runModel,
        setHoveredConnection,
        setHoveredNode,
        setRunModel
    } = useStore();

      const [dots, setDots] = useState<{ x: number; y: number; layerIndex: number; nodeIndex: number }[]>([]);
      const [backpropagation, setBackpropagation] = useState(false);

      useEffect(() => {
        if (!network || !runModel) return;
    
        const layerSpacing = svgWidth / (network.layers.length + 1);
        const totalNodes = network.layers[0].size;
    
        // Initialize dots at the first layer
        const initialDots = Array.from({ length: Math.min(totalNodes, maxNodes) }, (_, nodeIndex) => {
          const cx = layerSpacing;
          const cy = ((nodeIndex + 1) * svgHeight) / (Math.min(totalNodes, maxNodes) + 1);
          return { x: cx, y: cy, layerIndex: 0, nodeIndex };
        });
    
        setDots(initialDots);
        setBackpropagation(false);
    
        // Forward propagation animation
        const forwardAnimation = () => {
          let currentLayer = 0;
    
          const moveDots = () => {
            if (currentLayer >= network.layers.length - 1) {
              // Start backpropagation after forward propagation is complete
              setBackpropagation(true);
              return;
            }
    
            const nextLayer = currentLayer + 1;
            const nextLayerSpacing = (nextLayer + 1) * layerSpacing;
    
            setDots((prevDots) =>
              prevDots.map((dot) => {
                if (dot.layerIndex === currentLayer) {
                  const nextNodeIndex = Math.floor(Math.random() * network.layers[nextLayer].size);
                  const nextY = ((nextNodeIndex + 1) * svgHeight) / (Math.min(network.layers[nextLayer].size, maxNodes) + 1);
                  return {
                    x: nextLayerSpacing,
                    y: nextY,
                    layerIndex: nextLayer,
                    nodeIndex: nextNodeIndex,
                  };
                }
                return dot;
              })
            );
    
            currentLayer++;
            setTimeout(moveDots, 500); // Adjust the delay as needed
          };
    
          moveDots();
        };
    
        forwardAnimation();
      }, [network, svgWidth, svgHeight, runModel]); 
    
      useEffect(() => {
        if (!backpropagation || !network || !runModel) return;
    
        // Backpropagation animation
        const backpropagationAnimation = () => {
          let currentLayer = network.layers.length - 1;
    
          const moveDotsBack = () => {
            if (currentLayer <= 0) return;
            const layerSpacing = svgWidth / (network.layers.length + 1);

            const prevLayer = currentLayer - 1;
            const prevLayerSpacing = (prevLayer + 1) * layerSpacing;
    
            setDots((prevDots) =>
              prevDots.map((dot) => {
                if (dot.layerIndex === currentLayer) {
                  const prevNodeIndex = Math.floor(Math.random() * network.layers[prevLayer].size);
                  const prevY = ((prevNodeIndex + 1) * svgHeight) / (Math.min(network.layers[prevLayer].size, maxNodes) + 1);
                  return {
                    x: prevLayerSpacing,
                    y: prevY,
                    layerIndex: prevLayer,
                    nodeIndex: prevNodeIndex,
                  };
                }
                return dot;
              })
            );
    
            currentLayer--;
            setTimeout(moveDotsBack, 500); // Adjust the delay as needed
          };
    
          moveDotsBack();
          // setRunModel(false);
        };
    
        backpropagationAnimation();
      }, [backpropagation, network, svgWidth, svgHeight, runModel]);

  return (
    <div className="grid place-items-center w-full mt-8 mb-6">
        <svg width={svgWidth} height={svgHeight} className="border border-gray-300 rounded">
        {renderConnections({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderNodes({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {renderLayerLabels({ network, svgWidth, svgHeight, nodeRadius, setHoveredConnection, setHoveredNode })}
        {/* {dots.map((dot, index) => (
          <circle
            key={index}
            cx={dot.x}
            cy={dot.y}
            r={nodeRadius / 2}
            fill="#FF0000"
          />
        ))} */}
        </svg>
      </div>
  );
};

export default Graph;