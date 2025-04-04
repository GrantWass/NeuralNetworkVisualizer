import Config from "@/components/network/config";
import Graph from "@/components/network/graph";
import Explain from "@/components/network/explain";


const NeuralNetworkViz = () => {

  return (
    <div className="p-4 max-w-9xl mx-auto">
      <h1 className="text-3xl font-bold mt-4 mb-4">Interactive Neural Network Visualization</h1>
      <Config/>
      <Graph />
      <Explain />
    </div>
  );
};

export default NeuralNetworkViz;