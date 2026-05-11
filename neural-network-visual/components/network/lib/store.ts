import { create } from 'zustand';
import { toast } from "sonner"
import { DATASETS } from '@/components/network/static/constants';
import { NetworkState, HoveredConnection, HoveredNode, NeuronLayer} from "@/components/network/static/types";
import { getExplanationText, DATASET_INFO } from '@/components/network/static/explanation';

interface ChangedConnection {
  li: number;
  fi: number;
  ti: number;
  delta: number;
  positive: boolean;
}

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
  prevLoss: number;
  metric: number;
  name: string;
  losses: number[];
  accuracies: number[];
  sampleIndex: number;
  originalData: number[][];
  changedConnections: ChangedConnection[];
  stepLayerHighlight: number | null;
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
  setStepLayerHighlight: (index: number | null) => void;
  setWeight: (layerIndex: number, fromIndex: number, toIndex: number, newValue: number) => Promise<void>;
}

const URL = process.env.NEXT_PUBLIC_API_URL;

const useStore = create<TrainingState & TrainingActions>((set, get) => ({
  sessionId: null,
  epoch: 0,
  learningRate: 0.1,
  dataset: DATASETS[1],
  activations: ["relu", "relu"],
  hiddenLayers: [4, 4],
  network: null,
  hoveredConnection: null,
  hoveredNode: null,
  configOpen: true,
  datasetInfo: DATASET_INFO[DATASETS[1]],
  runModel: false,
  loss: 0,
  prevLoss: 0,
  metric: 0,
  name: "",
  losses: [],
  accuracies: [],
  sampleIndex: 0,
  originalData: [],
  changedConnections: [],
  stepLayerHighlight: null,

  setEpoch: (epoch) => set({ epoch }),
  setSampleIndex: (sampleIndex) => set({ sampleIndex }),
  setLearningRate: (learningRate) => set({ learningRate }),
  setStepLayerHighlight: (stepLayerHighlight) => set({ stepLayerHighlight }),
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
      fetch(`${URL}/clear_session?session_id=${prevSessionId}`, { method: "POST" })
        .then((response) => response.json())
        .then((data) => console.log("Session Cleared:", data.message))
        .catch((error) => console.error("Error clearing session:", error));
    }

    set({ sessionId });
  },

  initModel: async () => {
    const { hiddenLayers, activations, dataset } = get();
    try {
      const response = await fetch(`${URL}/init_model`, {
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
        set({ network: { input: [[]], layers, initialized: true }, originalData: data.original_train_data, configOpen: false });
        const arch = data.layer_sizes.join(" → ");
        toast.success("Model ready", {
          description: `Architecture: ${arch} · ${data.layer_sizes.reduce((a: number, b: number, i: number, arr: number[]) => i < arr.length - 1 ? a + arr[i] * arr[i+1] + arr[i+1] : a, 0)} parameters`,
        });
      } else {
        throw new Error(data.error || "Failed to initialize model");
      }
    } catch (error) {
      console.error("Error initializing model:", error);
      toast.error("Initialization failed", {
        description: "Could not connect to the backend. Make sure the server is running.",
      });
    }
  },

  initModelFrontend: () => {
    const { hiddenLayers, activations, dataset } = get();
    let inputSize = 0;
    let outputSize = 0;
    let accuracyMetric = "accuracy";

    if (dataset === "iris") {
      inputSize = 4;
      outputSize = 3;
    } else if (dataset === "auto_mpg") {
      inputSize = 4;
      outputSize = 1;
      accuracyMetric = "mae";
    } else if (dataset === "xor") {
      inputSize = 2;
      outputSize = 1;
    }

    const layerSizes = [inputSize, ...hiddenLayers, outputSize];
    const totalLayers = layerSizes.length;
    const outputActivation = dataset === "iris" ? "softmax" : dataset === "xor" ? "sigmoid" : "linear";
    const layers = layerSizes.map((size, index) => {
      const activation = index === totalLayers - 1 ? outputActivation : activations[index] || "relu";
      
      const layer = new NeuronLayer(size, activation, index, totalLayers);
      layer.initWeightsAndBiases(size, index + 1 == layerSizes.length ? 0 : layerSizes[index + 1]);
      return layer;
    });
    set({ network: { input: [[]], layers, initialized: false }, name: accuracyMetric, configOpen: true, });
    get().clearSessionAndReset();

  },

  clearSessionAndReset: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    try {
      await fetch(`${URL}/clear_session?session_id=${sessionId}`, { method: "POST" });
      set({
        sessionId: null,
        configOpen: true,
        hoveredConnection: null,
        hoveredNode: null,
        epoch: 0,
        learningRate: 0.1,
        activations: ["relu", "relu"],
        hiddenLayers: [4, 4],
        losses: [],
        accuracies: [],
        originalData: [],
      });
      toast("Model reset");
      get().setSessionId(null);
    } catch (error) {
      console.error("Error clearing session:", error);
      toast.error("Reset failed", {
        description: "Could not clear the session. Please try again.",
      });
    }
  },

  runTrainingCycle: async () => {
    const { sessionId, learningRate } = get();
    if (!sessionId) {
      toast.error("No model initialized", {
        description: "Complete step 3 to initialize the model before training.",
      });
      return;
    }

    get().setRunModel(true);
    const prevLossValue = get().loss;


    try {
      const epochs = 1

      const response = await fetch(`${URL}/train`, {
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
            const currBiases = layer.biases;
            const currWeights = layer.weights;
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
              layer.prevBias = resultPrev?.layers[index]?.biases ? resultPrev.layers[index].biases[0] : currBiases;
              layer.prevWeights = resultPrev?.layers[index]?.weights ? resultPrev.layers[index].weights : currWeights;
            }
            return layer;
          }) || [];

          // Compute top 5 most-changed connections for highlighting
          const changedConns: ChangedConnection[] = [];
          updatedLayers.forEach((layer, li) => {
            if (layer.prevWeights?.length > 0 && layer.weights?.length > 0) {
              layer.weights.forEach((row, fi) => {
                row.forEach((currW, ti) => {
                  const prevW = layer.prevWeights[fi]?.[ti] ?? currW;
                  const diff = currW - prevW;
                  const delta = Math.abs(diff);
                  if (delta > 0.001) {
                    changedConns.push({ li, fi, ti, delta, positive: diff > 0 });
                  }
                });
              });
            }
          });
          changedConns.sort((a, b) => b.delta - a.delta);

          return {
            ...state,
            network: {
              ...state.network,
              input: result.input,
              initialized: true,
              layers: updatedLayers,
            },
            epoch: state.epoch + epochs,
            loss: result.loss,
            prevLoss: prevLossValue,
            metric: result.metric,
            name: result.name,
            losses: [...state.losses, result.loss],
            accuracies: [...state.accuracies, result.metric],
            changedConnections: changedConns.slice(0, 5),
          };
        });

        const lossArrow = prevLossValue > 0
          ? result.loss < prevLossValue ? "↓" : result.loss > prevLossValue ? "↑" : "→"
          : "";
        const metricLabel = result.name === "accuracy"
          ? `${result.metric.toFixed(1)}% accuracy`
          : `MAE ${result.metric.toFixed(3)}`;
        const { epoch: currentEpoch } = get();
        toast.success(`Epoch ${currentEpoch}`, {
          description: `Loss ${result.loss.toFixed(4)} ${lossArrow} · ${metricLabel}`,
        });

      } else {
        throw new Error(data.error || "Failed to run training cycle");
      }
    } catch (error) {
      console.error("Error running training cycle:", error);
      toast.error("Training failed", {
        description: "Could not complete the training cycle. Please try again.",
      });
    }
  },

  addHiddenLayer: () => {
    const { hiddenLayers, activations } = get();
    if (hiddenLayers.length < 3) {
      set({
        hiddenLayers: [...hiddenLayers, 4],
        activations: [...activations, "relu"],
      });
      get().initModelFrontend();
    }
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
    const { network, epoch, learningRate, dataset, loss, prevLoss, metric, name, sessionId } = get();
    const text : string = getExplanationText(network, epoch, learningRate, dataset, loss, prevLoss, metric, name, sessionId)
    return text
  },

  setWeight: async (layerIndex, fromIndex, toIndex, newValue) => {
    const { sessionId, network } = get();
    if (!sessionId || !network) return;
    try {
      const response = await fetch(`${URL}/set_weights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, layer_index: layerIndex, from_index: fromIndex, to_index: toIndex, new_value: newValue }),
      });
      const data = await response.json();
      if (response.ok) {
        set((state) => {
          const updatedLayers = state.network?.layers.map((layer, index) => {
            const resultLayer = data.layers[index];
            if (resultLayer) {
              if (index === layerIndex && layer.weights[fromIndex]) {
                layer.weights[fromIndex][toIndex] = newValue;
              }
              layer.A = resultLayer.A ?? layer.A;
              layer.Z = resultLayer.Z ?? layer.Z;
            }
            return layer;
          }) ?? [];
          return { ...state, network: { ...state.network!, layers: updatedLayers } };
        });
        toast.success("Weight updated", {
          description: `Layer ${layerIndex}, neuron ${fromIndex} → ${toIndex} set to ${newValue.toFixed(4)}`,
        });
      } else {
        throw new Error(data.error);
      }
    } catch {
      toast.error("Weight update failed", { description: "Could not apply the new weight value." });
    }
  },

}));

export default useStore;
