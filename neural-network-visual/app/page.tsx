import Config from "@/components/network/config";
import Graph from "@/components/network/graph";
import Explain from "@/components/network/explain";
import ContactInfo from "./contact";


export const metadata = {
  title: "Neural Network Visualizer | Interactive Deep Learning Tool",
  description: "Visualize forward and backward propagation in a neural network with an interactive graph, detailed configuration options, and explanations.",
  keywords: [
    "Neural Network Visualizer",
    "Deep Learning",
    "Backpropagation",
    "Forward Propagation",
    "Machine Learning",
    "AI Visualization",
    "Neural Network Demo",
    "Interactive Neural Net"
  ],
  openGraph: {
    title: "Neural Network Visualizer",
    description: "Explore and understand neural networks through an interactive visualization. Configure layers, track metrics, and learn backpropagation.",
    url: "https://nn-visual.com/",
    siteName: "Neural Network Visualizer",
    locale: "en_US",
    type: "website",
  },
};

const NeuralNetworkViz = () => {
  return (
    <div className="p-4 max-w-9xl mx-auto relative">
      <h1 className="text-3xl font-bold mt-4 mb-4">Interactive Neural Network Visualization</h1>
      
      {/* Visualization Components */}
      <Config />
      <Graph />
      <Explain />
      <ContactInfo />

     </div>
  );
};


export default NeuralNetworkViz;