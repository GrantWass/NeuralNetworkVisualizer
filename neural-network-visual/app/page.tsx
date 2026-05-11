import Config from "@/components/network/config";
import Graph from "@/components/network/neural";
import Explain from "@/components/network/explain";
import ContactInfo from "./contact";


export const metadata = {
  title: "Neural Network Visualizer | Interactive Deep Learning Tool",
  description:
    "Step through forward propagation, backpropagation, and gradient descent interactively — with real matrix math, live training charts, prediction confidence bars, a step-by-step mode, and a built-in ML glossary. Free and open to everyone learning AI.",
  keywords: [
    "Neural Network Visualizer",
    "Interactive Neural Network",
    "Backpropagation Visualization",
    "Forward Propagation",
    "Deep Learning Tutorial",
    "Machine Learning Education",
    "Gradient Descent Interactive",
    "Neural Network Playground",
    "AI Visualization Tool",
    "Learn Neural Networks",
    "Step by Step Backpropagation",
    "Machine Learning Glossary",
    "Neural Network Demo",
    "Iris Dataset Neural Network",
  ],
  openGraph: {
    title: "Neural Network Visualizer — See the Math, Step by Step",
    description:
      "Watch a neural network learn in real time. Step through each layer's computation, see weights update after every training cycle, explore a built-in ML glossary, and understand backpropagation through actual matrix equations.",
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