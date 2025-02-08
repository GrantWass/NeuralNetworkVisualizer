// store.ts
import { create } from 'zustand';
import { toast } from './use-toast'

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

class NeuronLayer {
  input_size: number
  output_size: number
  activation: string
  weights: number[][]
  activations: number[]
  biases: number[]
  dW: number[][]
  db: number[]

  constructor(input_size: number, output_size: number, activation: string) {
    this.input_size = input_size
    this.output_size = output_size
    this.activation = activation
    this.weights = []
    this.biases = []
    this.activations = []
    this.dW = []
    this.db = []
  }

  // Method to initialize weights and biases
  initWeightsAndBiases() {
    this.weights = Array(this.output_size)
      .fill(null)
      .map(() => Array(this.input_size).fill(0))
    this.biases = Array(this.output_size).fill(0)
  }
}

const DATASETS = ["california_housing", "mnist", "iris"]
const DATASET_INFO: { [key: string]: string } = {
  california_housing: "The California Housing dataset contains information from the 1990 California census. It includes features like median income, housing median age, average rooms, etc.",
  mnist: "The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems.",
  iris: "The Iris dataset is a multivariate dataset introduced by Ronald Fisher. It consists of 50 samples from each of three species of Iris flowers."
};

interface TrainingState {
  sessionId: string | null;
  epoch: number;
  learningRate: number;
  dataset: string;
  activations: string[];
  hiddenLayers: number[];
  network: NetworkState | null;
  hoveredConnection: HoveredConnection | null;
  hoveredNode: HoveredNode | null;
  configOpen: boolean;
  datasetInfo: string;
}

interface TrainingActions {
  setSessionId: (sessionId: string | null) => void;
  setEpoch: (epoch: number) => void;
  setLearningRate: (learningRate: number) => void;
  setDataset: (dataset: string) => void;
  setActivations: (activations: string[]) => void;
  setHiddenLayers: (hiddenLayers: number[]) => void;
  setNetwork: (network: NetworkState | null) => void;
  setHoveredConnection: (hoveredConnection: HoveredConnection | null) => void;
  setHoveredNode: (hoveredNode: HoveredNode | null) => void;
  setConfigOpen: (configOpen: boolean) => void;
  setDatasetInfo: (datasetInfo: string) => void;
  initModel: () => Promise<void>;
  clearSessionAndReset: () => Promise<void>;
  runTrainingCycle: () => Promise<void>;
}

const useStore = create<TrainingState & TrainingActions>((set, get) => ({
  sessionId: null,
  epoch: 0,
  learningRate: 0.1,
  dataset: DATASETS[0],
  activations: ["relu", "relu"],
  hiddenLayers: [4, 4],
  network: null,
  hoveredConnection: null,
  hoveredNode: null,
  configOpen: true,
  datasetInfo: DATASET_INFO[DATASETS[0]],

  setSessionId: (sessionId) => set({ sessionId }),
  setEpoch: (epoch) => set({ epoch }),
  setLearningRate: (learningRate) => set({ learningRate }),
  setDataset: (dataset) => set({ dataset, datasetInfo: DATASET_INFO[dataset] }),
  setActivations: (activations) => set({ activations }),
  setHiddenLayers: (hiddenLayers) => set({ hiddenLayers }),
  setNetwork: (network) => set({ network }),
  setHoveredConnection: (hoveredConnection) => set({ hoveredConnection }),
  setHoveredNode: (hoveredNode) => set({ hoveredNode }),
  setConfigOpen: (configOpen) => set({ configOpen }),
  setDatasetInfo: (datasetInfo) => set({ datasetInfo }),

  initModel: async () => {
    const { hiddenLayers, activations, dataset } = get();
    try {
      const response = await fetch("http://localhost:8000/init_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          layer_sizes: hiddenLayers,
          activations: activations,
          dataset: dataset,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        set({ sessionId: data.session_id });
        const layers = data.layer_sizes.map((size: number, index: number) => {
          const layer = new NeuronLayer(
            index === 0 ? size : data.layer_sizes[index - 1],
            size,
            activations[index] || "relu"
          );
          layer.initWeightsAndBiases();
          if (data.network.layers[index]) {
            layer.weights = data.network.layers[index].weights || layer.weights;
            layer.biases = data.network.layers[index].biases[0] || layer.biases;
          }
          return layer;
        });
        set({ network: { layers }, configOpen: false });
        toast({
          title: "Model Initialized",
          description: `Session ID: ${data.session_id}`,
        });
      } else {
        throw new Error(data.error || "Failed to initialize model");
      }
    } catch (error) {
      console.error("Error initializing model:", error);
      toast({
        title: "Error",
        description: "Failed to initialize model. Please try again.",
        variant: "destructive",
      });
    }
  },

  clearSessionAndReset: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`http://localhost:8000/clear_session?session_id=${sessionId}`, { method: "POST" });
      set({
        sessionId: null,
        network: null,
        configOpen: true,
        hoveredConnection: null,
        hoveredNode: null,
        epoch: 0,
        learningRate: 0.1,
        dataset: DATASETS[0],
        activations: ["relu", "relu"],
        hiddenLayers: [4, 4],
      });
      toast({
        title: "Configuration Reset",
        description: "Session cleared, network reset, and configuration open.",
      });
    } catch (error) {
      console.error("Error clearing session:", error);
      toast({
        title: "Error",
        description: "Failed to reset configuration. Please try again.",
        variant: "destructive",
      });
    }
  },

  runTrainingCycle: async () => {
    const { sessionId, learningRate } = get();
    if (!sessionId) {
      toast({
        title: "Error",
        description: "Please initialize the model first.",
        variant: "destructive",
      });
      return;
    }
    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          learning_rate: learningRate,
          epochs: 1,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        const result = data.training_results[0];
        set((state) => ({
          network: {
            ...state.network,
            layers: result.layers || []
          },
          epoch: state.epoch + 1,
        }));
        toast({
          title: "Training Cycle Complete",
          description: `Loss: ${result.loss.toFixed(4)}, Metric: ${result.metric.toFixed(4)}`,
        });
      } else {
        throw new Error(data.error || "Failed to run training cycle");
      }
    } catch (error) {
      console.error("Error running training cycle:", error);
      toast({
        title: "Error",
        description: "Failed to run training cycle. Please try again.",
        variant: "destructive",
      });
    }
  },
}));

export default useStore;