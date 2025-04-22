import Config from "@/components/network/config";
import Graph from "@/components/network/graph";
import Explain from "@/components/network/explain";


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
    // images: [
    //   {
    //     url: "https://your-domain.com/preview.png", // Replace with your actual image path
    //     width: 1200,
    //     height: 630,
    //     alt: "Neural Network Visualizer Screenshot",
    //   },
    // ],
    locale: "en_US",
    type: "website",
  },
  // twitter: {
  //   card: "summary_large_image",
  //   title: "Neural Network Visualizer",
  //   description: "An interactive and educational way to understand how neural networks work. Perfect for students and AI enthusiasts.",
  //   images: ["https://your-domain.com/preview.png"],
  //   creator: "@yourTwitterHandle", // optional
  // },
};

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