interface NetworkState {
  layers: NeuronLayer[];
}

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

interface RenderNetworkProps {
  network: NetworkState | null;
  svgWidth: number;
  svgHeight: number;
  nodeRadius: number;
  setHoveredConnection: (hovered: HoveredConnection | null) => void;
  setHoveredNode: (hovered: HoveredNode | null) => void;
}

class NeuronLayer {
    input_size: number
    output_size: number
    activation: string
    weights: number[][]
    activations: number[]
    biases: number[]
    dW: number[][]
    db: number[]
    name: string
  
    constructor(input_size: number, output_size: number, activation: string, layerIndex: number, totalLayers: number) {
      this.input_size = input_size
      this.output_size = output_size
      this.activation = activation
      this.weights = []
      this.biases = []
      this.activations = []
      this.dW = []
      this.db = []

      if (layerIndex === 0) {
        this.name = "Input Layer";
      } else if (layerIndex === totalLayers - 1) {
        this.name = "Output Layer";
      } else {
        this.name = `Hidden Layer ${layerIndex}`;
      }
    }

    initWeightsAndBiases() {
        this.weights = Array(this.output_size)
          .fill(null)
          .map(() => Array(this.input_size).fill(0))
        this.biases = Array(this.output_size).fill(0)
    }
}

export { NeuronLayer };
export type { NetworkState, HoveredConnection, HoveredNode, RenderNetworkProps };
