import { create } from 'zustand';
import { toast } from "sonner"
import { DATASETS, DATASET_INFO } from '@/static/constants';
import { NetworkState, HoveredConnection, HoveredNode, NeuronLayer} from "@/static/types";

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
  addHiddenLayer: () => void;
  removeHiddenLayer: () => void;
  updateHiddenLayer: (index: number, value: number) => void;
  updateActivation: (index: number, value: string) => void;
  handleDatasetChange: (newDataset: string) => void;
  getExplanation: () => string;
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

  setSessionId: (sessionId) => {
    const prevSessionId = get().sessionId;

    if (prevSessionId) {
      fetch(`http://localhost:8000/clear_session?session_id=${prevSessionId}`, { method: "POST" })
        .then((response) => response.json())
        .then((data) => console.log("Session Cleared:", data.message))
        .catch((error) => console.error("Error clearing session:", error));
    }

    set({ sessionId });
  },

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
        const totalLayers = hiddenLayers.length + 2;
        const layers = data.layer_sizes.map((size: number, index: number) => {
          const input_size = index === 0 ? 0 : data.layer_sizes[index - 1];
          const output_size = size;
          const layer = new NeuronLayer(
            input_size,
            output_size,
            activations[index] || "relu",
            index,
            totalLayers
          );
          layer.initWeightsAndBiases();
          if (data.network.layers[index]) {
            layer.weights = data.network.layers[index].weights || layer.weights;
            layer.biases = data.network.layers[index].biases[0] || layer.biases;
          }
          return layer;
        });
        set({ network: { layers }, configOpen: false });
        toast("Model Initialized", {
          description: `Session ID: ${data.session_id}`,
        });
      } else {
        throw new Error(data.error || "Failed to initialize model");
      }
    } catch (error) {
      console.error("Error initializing model:", error);
      toast("Error", {
        description: "Failed to initialize model. Please try again.",
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
      toast("Configuration Reset", {
        description: "Session cleared, network reset, and configuration open.",
      });
    } catch (error) {
      console.error("Error clearing session:", error);
      toast("Error", {
        description: "Failed to reset configuration. Please try again.",
      });
    }
  },

  runTrainingCycle: async () => {
    const { sessionId, learningRate } = get();
    if (!sessionId) {
      toast("Error", {
        description: "Please initialize the model first.",
      });
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/train", {
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

        // Update network layers individually
        set((state) => {
          const updatedLayers = state.network?.layers.map((layer, index) => {
            const resultLayer = result.layers[index];
            if (resultLayer) {
              // Update individual layer properties independently
              layer.weights = resultLayer.weights ? resultLayer.weights : layer.weights;
              layer.biases = resultLayer.biases ? resultLayer.biases[0] : layer.biases;
              layer.Z = resultLayer.Z ? resultLayer.Z[0] : layer.Z; // TODO Keep all samples of Z and handle in UI
              layer.A = resultLayer.A ? resultLayer.A[0] : layer.A; // TODO Keep all samples of Z and handle in UI
              layer.dW = resultLayer.dW ? resultLayer.dW : layer.dW;
              layer.db = resultLayer.db ? resultLayer.db : layer.db;
              layer.dZ = resultLayer.dZ ? resultLayer.dZ : layer.dZ; // TODO Keep all samples of Z and handle in UI
              layer.activation = resultLayer.activation ? resultLayer.activation : layer.activation;
            }
            return layer;
          }) || [];

          return {
            ...state,
            network: {
              ...state.network,
              layers: updatedLayers,  // Update layers with new data
            },
            epoch: state.epoch + 1
          };
        });

        toast("Training Cycle Complete", {
          description: `Loss: ${result.loss.toFixed(4)}, Metric (${result.name}): ${result.metric.toFixed(4)}`,
        });
      } else {
        throw new Error(data.error || "Failed to run training cycle");
      }
    } catch (error) {
      console.error("Error running training cycle:", error);
      toast("Error", {
        description: "Failed to run training cycle. Please try again.",
      });
    }
  },

  getExplanation: () => {
    if (!get().network) {
      return "The neural network has not been initialized yet. Configure the network and click 'Initialize Model' to start."
    }

    if (get().epoch === 0) {
      return "The model has been initialized with random weights and biases. Each node represents a neuron, and each line represents a connection between neurons. The thickness of the line represents the strength of the connection (weight)."
    }

    return `
      Training Cycle ${get().epoch} completed:
      1. Forward Propagation: Input data is fed through the network, layer by layer. Each neuron computes a weighted sum of its inputs and applies an activation function.
      2. Loss Calculation: The network's output is compared to the actual target values, and a loss is calculated.
      3. Backward Propagation: The error is propagated backwards through the network, calculating gradients for each weight and bias.
      4. Parameter Update: Weights and biases are updated using the calculated gradients and the current learning rate (${get().learningRate}).
      
      Hover over nodes and connections to see their current values. As training progresses, you should see the network adjust its weights to better fit the data.
    `
  },
  addHiddenLayer: () => {
    const { hiddenLayers, activations } = get();
    set({
      hiddenLayers: [...hiddenLayers, 4],
      activations: [...activations, "relu"],
    });
  },

  removeHiddenLayer: () => {
    const { hiddenLayers, activations } = get();
    if (hiddenLayers.length > 1) {
      set({
        hiddenLayers: hiddenLayers.slice(0, -1),
        activations: activations.slice(0, -1),
      });
    }
  },

  updateHiddenLayer: (index: number, value: number) => {
    const { hiddenLayers } = get();
    const newHiddenLayers = [...hiddenLayers];
    newHiddenLayers[index] = value;
    set({ hiddenLayers: newHiddenLayers });
  },

  updateActivation: (index: number, value: string) => {
    const { activations } = get();
    const newActivations = [...activations];
    newActivations[index] = value;
    set({ activations: newActivations });
  },

  handleDatasetChange: (newDataset: string) => {
    set({
      dataset: newDataset,
      datasetInfo: DATASET_INFO[newDataset],
    });
  },
}));

export default useStore;
