import { create } from 'zustand';
import { toast } from "sonner"
import { DATASETS, DATASET_INFO } from '@/static/constants';
import { NetworkState, HoveredConnection, HoveredNode, NeuronLayer} from "@/static/types";
import { getExplanationText } from '@/static/explanation';

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
  runModel: boolean;
  loss: number;
  metric: number;
  name: string;
  losses: number[];
  accuracies: number[];
  sampleIndex: number;
}

interface TrainingActions {
  setSessionId: (sessionId: string | null) => void;
  setEpoch: (epoch: number) => void;
  setLearningRate: (learningRate: number) => void;
  setNetwork: (network: NetworkState | null) => void;
  setHoveredConnection: (hoveredConnection: HoveredConnection | null) => void;
  setHoveredNode: (hoveredNode: HoveredNode | null) => void;
  setConfigOpen: (configOpen: boolean) => void;
  initModel: () => Promise<void>;
  initModelFrontend: () => void;  
  clearSessionAndReset: () => Promise<void>;
  runTrainingCycle: () => Promise<void>;
  addHiddenLayer: () => void;
  removeHiddenLayer: () => void;
  updateHiddenLayer: (index: number, value: number) => void;
  updateActivation: (index: number, value: string) => void;
  handleDatasetChange: (newDataset: string) => void;
  getExplanation: () => string;
  setRunModel: (runModel: boolean) => void;
  setSampleIndex: (sampleIndex: number) => void;
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
  runModel: false,
  loss: 0,
  metric: 0,
  name: "",
  losses: [],
  accuracies: [],
  sampleIndex: 0,

  setEpoch: (epoch) => set({ epoch }),
  setSampleIndex: (sampleIndex) => set({ sampleIndex }),
  setLearningRate: (learningRate) => set({ learningRate }),
  setNetwork: (network) => set({ network }),
  setHoveredConnection: (hoveredConnection) => {
    if (hoveredConnection){
      get().setHoveredNode(null)
    }
    set({ hoveredConnection })
  },
  setHoveredNode: (hoveredNode) => {
    if (hoveredNode){
      get().setHoveredConnection(null)
    }
    set({ hoveredNode })
  },
  setConfigOpen: (configOpen) => set({ configOpen }),
  setRunModel: (runModel) => set({ runModel }),

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
          const layer = new NeuronLayer(
            size,
            data.network.layers[index] ? data.network.layers[index].activation : "",
            index,
            totalLayers
          );
          if (data.network.layers[index]) {
            layer.weights = data.network.layers[index].weights || layer.weights;
            layer.biases = data.network.layers[index].biases[0] || layer.biases;
          }
          return layer;
        });
        set({ network: { input: [[]], layers, initialized: true }, configOpen: false });
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

  initModelFrontend: () => {
    const { hiddenLayers, activations, dataset } = get();
    var inputSize = 0;
    var outputSize = 0;

    if (dataset === "mnist") {
      inputSize = 784;
      outputSize = 10;
    } else if (dataset === "iris") {
      inputSize = 4;
      outputSize = 3;
    } else if (dataset === "california_housing") {
      inputSize = 8;
      outputSize = 1;
    }

    const layerSizes = [inputSize, ...hiddenLayers, outputSize];
    const totalLayers = layerSizes.length;
    const layers = layerSizes.map((size, index) => {
      const activation = index === totalLayers - 1 ? "softmax" : activations[index] || "relu";
      
      const layer = new NeuronLayer(size, activation, index, totalLayers);
      layer.initWeightsAndBiases(size, index + 1 == layerSizes.length ? 0 : layerSizes[index + 1]);
      return layer;
    });
    set({ network: { input: [[]], layers, initialized: false }, configOpen: true, });
    get().clearSessionAndReset();

  },

  clearSessionAndReset: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`http://localhost:8000/clear_session?session_id=${sessionId}`, { method: "POST" });
      set({
        sessionId: null,
        configOpen: true,
        hoveredConnection: null,
        hoveredNode: null,
        epoch: 0,
        learningRate: 0.1,
        dataset: DATASETS[0],
        activations: ["relu", "relu"],
        hiddenLayers: [4, 4],
        losses: [],
        accuracies: []
      });
      toast("Configuration Reset", {
        description: "Session cleared, network reset, and configuration open.",
      });
      get().setSessionId(null);
    } catch (error) {
      console.error("Error clearing session:", error);
      toast("Error", {
        description: "Failed to reset configuration. Please try again.",
      });
    }
  },

  runTrainingCycle: async () => {
    const { sessionId, learningRate, dataset } = get();
    if (!sessionId) {
      toast("Error", {
        description: "Please initialize the model first.",
      });
      return;
    }
    
    get().setRunModel(true);

    try {
      const epochs = dataset === "mnist" ? 1 : 2; // Set epochs based on dataset

      const response = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          learning_rate: learningRate,
          epochs: epochs,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        const result = data.training_results[epochs - 1]; // Always get the last one
        const resultPrev = epochs > 1 ? data.training_results[epochs - 2] : null;

        // Update network layers individually
        set((state) => {
          const updatedLayers = state.network?.layers.map((layer, index) => {
            const resultLayer = result.layers[index];
            if (resultLayer) {
              // Update individual layer properties independently
              layer.weights = resultLayer.weights ? resultLayer.weights : layer.weights;
              layer.biases = resultLayer.biases ? resultLayer.biases[0] : layer.biases;
              layer.Z = resultLayer.Z ? resultLayer.Z : layer.Z; 
              layer.A = resultLayer.A ? resultLayer.A : layer.A;
              layer.dW = resultLayer.dW ? resultLayer.dW : layer.dW;
              layer.db = resultLayer.db ? resultLayer.db[0] : layer.db;
              layer.dZ = resultLayer.dZ ? resultLayer.dZ : layer.dZ;
              layer.activation = resultLayer.activation ? resultLayer.activation : layer.activation;
              layer.prevBias = resultPrev?.layers[index]?.biases ? resultPrev.layers[index].biases[0] : layer.prevBias;
              layer.prevWeights = resultPrev?.layers[index]?.weights ? resultPrev.layers[index].weights : layer.prevWeights;
            }
            return layer;
          }) || [];
          return {
            ...state,
            network: {
              ...state.network,
              input: result.input,
              initialized: true,
              layers: updatedLayers,  // Update layers with new data
            },
            epoch: state.epoch + 1,
            loss: result.loss,
            metric: result.metric,
            name: result.name,
            losses: [...state.losses, result.loss],
            accuracies: [...state.accuracies, result.metric],

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

  addHiddenLayer: () => {
    const { hiddenLayers, activations } = get();
    set({
      hiddenLayers: [...hiddenLayers, 4],
      activations: [...activations, "relu"],
    });
    get().initModelFrontend();
  },

  removeHiddenLayer: () => {
    const { hiddenLayers, activations } = get();
    if (hiddenLayers.length > 1) {
      set({
        hiddenLayers: hiddenLayers.slice(0, -1),
        activations: activations.slice(0, -1),
      });
    }
    get().initModelFrontend();
  },

  updateHiddenLayer: (index: number, value: number) => {
    const { hiddenLayers } = get();
    const newHiddenLayers = [...hiddenLayers];
    newHiddenLayers[index] = value;
    set({ hiddenLayers: newHiddenLayers });
    get().initModelFrontend();
  },

  updateActivation: (index: number, value: string) => {
    const { activations } = get();
    const newActivations = [...activations];
    newActivations[index] = value;
    set({ activations: newActivations });
    get().initModelFrontend();
  },

  handleDatasetChange: (newDataset: string) => {
    set({
      dataset: newDataset,
      datasetInfo: DATASET_INFO[newDataset],
    });
    get().initModelFrontend();
  },

  getExplanation: () => {
    const text : string = getExplanationText()
    return text
  },
  
}));

export default useStore;
