interface NetworkState {
  input: number[][]
  layers: NeuronLayer[];
  initialized: boolean
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
    size: number
    activation: string
    weights: number[][]
    biases: number[]
    A: number[][]
    Z: number[][]
    dW: number[][]
    db: number[]
    dZ: number[][]
    prevBias: number[]
    prevWeights: number[][]
    name: string
  
    constructor(size: number, activation: string, layerIndex: number, totalLayers: number) {
      this.size = size
      this.activation = activation
      this.weights = []
      this.biases = []
      this.A = []
      this.Z = []
      this.dW = []
      this.db = []
      this.dZ = []
      this.prevBias = []
      this.prevWeights = []

      if (layerIndex === 0) {
        this.name = "Input Layer";
      } else if (layerIndex === totalLayers - 1) {
        this.name = "Output Layer";
      } else {
        this.name = `Hidden Layer ${layerIndex}`;
      }
    }

    initWeightsAndBiases(size: number, nextSize: number) {
        this.weights = Array(size)
          .fill(null)
          .map(() => Array(nextSize).fill(0))
        this.biases = Array(size).fill(0)
    }
}

export { NeuronLayer };
export type { NetworkState, HoveredConnection, HoveredNode, RenderNetworkProps };
