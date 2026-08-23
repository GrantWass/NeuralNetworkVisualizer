import type { StateCreator } from "zustand";
import { toast } from "sonner";
import { DATASETS } from "@/components/network/static/constants";
import { NetworkState, NeuronLayer } from "@/components/network/static/types";
import { getExplanationText, DATASET_INFO } from "@/components/network/static/explanation";
import { forwardPassSingle, computeStratifiedIndices } from "../networkUtils";
import { fetchWithTimeout } from "../apiUtils";

interface ChangedConnection {
  li: number;
  fi: number;
  ti: number;
  delta: number;
  positive: boolean;
}

interface DigitPrediction {
  predictedClass: number;
  confidences: number[];
}

const EPOCH_CAPS: Record<string, number | null> = {
  xor: null, iris: 100, auto_mpg: 200, mnist: 300,
};

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function serverErrorMessage(data: any): string {
  // FastAPI errors use "detail", older paths used "error"
  return data?.detail || data?.error || "";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleSessionExpired(set: any): void {
  set({ sessionId: null });
  toast.error("Session expired", {
    description: "Training sessions last 3 hours. Please re-initialize the model (step 3) to continue.",
  });
}

export interface TrainingSlice {
  sessionId: string | null;
  epoch: number;
  learningRate: number;
  trainingEpochs: number;
  dataset: string;
  activations: string[];
  hiddenLayers: number[];
  network: NetworkState | null;
  datasetInfo: string;
  runModel: boolean;
  loss: number;
  prevLoss: number;
  metric: number;
  name: string;
  losses: number[];
  accuracies: number[];
  testLosses: number[];
  testAccuracies: number[];
  requestToken: number;
  sampleIndex: number;
  sampleIndexMap: number[];
  originalData: number[][];
  changedConnections: ChangedConnection[];
  yMean: number | null;
  yStd: number | null;
  drawnDigitPrediction: DigitPrediction | null;
  drawnDigitPixels: number[] | null;
  isInitializing: boolean;
  submittableScore: number | null;
  xorEpochsTo100: number | null;
  setSessionId: (sessionId: string | null) => void;
  setEpoch: (epoch: number) => void;
  setLearningRate: (learningRate: number) => void;
  setTrainingEpochs: (n: number) => void;
  getDisplayIndex: (uiIndex: number) => number;
  setNetwork: (network: NetworkState | null) => void;
  setRunModel: (runModel: boolean) => void;
  setSampleIndex: (sampleIndex: number) => void;
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
  setWeight: (layerIndex: number, fromIndex: number, toIndex: number, newValue: number) => Promise<void>;
  predictDigit: (pixels: number[]) => Promise<void>;
  clearDigitPrediction: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const createTrainingSlice: StateCreator<any, [], [], TrainingSlice> = (set, get) => ({
  sessionId: null,
  epoch: 0,
  learningRate: 0.1,
  trainingEpochs: 1,
  dataset: DATASETS[1],
  activations: ["relu", "relu"],
  hiddenLayers: [4, 4],
  network: null,
  datasetInfo: DATASET_INFO[DATASETS[1]],
  runModel: false,
  loss: 0,
  prevLoss: 0,
  metric: 0,
  name: "",
  losses: [],
  accuracies: [],
  testLosses: [],
  testAccuracies: [],
  requestToken: 0,
  sampleIndex: 0,
  sampleIndexMap: [],
  originalData: [],
  changedConnections: [],
  yMean: null,
  yStd: null,
  drawnDigitPrediction: null,
  drawnDigitPixels: null,
  isInitializing: false,
  submittableScore: null,
  xorEpochsTo100: null,

  setEpoch: (epoch) => set({ epoch }),
  setSampleIndex: (sampleIndex) => set({ sampleIndex }),
  setTrainingEpochs: (n) => set({ trainingEpochs: n }),
  getDisplayIndex: (uiIndex) => {
    const { sampleIndexMap } = get();
    return sampleIndexMap[uiIndex] ?? uiIndex;
  },
  setLearningRate: (learningRate) => set({ learningRate }),
  setNetwork: (network) => set({ network }),
  setRunModel: (runModel) => set({ runModel }),

  setSessionId: (sessionId) => {
    const prevSessionId = get().sessionId;
    if (prevSessionId) {
      fetch(`${API_URL}/clear_session?session_id=${prevSessionId}`, { method: "POST" })
        .then((response) => response.json())
        .then((data) => console.log("Session Cleared:", data.message))
        .catch((error) => console.error("Error clearing session:", error));
    }
    set({ sessionId });
  },

  initModel: async () => {
    const { hiddenLayers, activations, dataset, sessionId: prevSessionId } = get();
    // Don't orphan the previous Lambda session (3h TTL) when re-initializing
    if (prevSessionId) {
      fetch(`${API_URL}/clear_session?session_id=${prevSessionId}`, { method: "POST" }).catch(() => {});
    }
    set({ isInitializing: true, requestToken: get().requestToken + 1 });
    try {
      const response = await fetchWithTimeout(`${API_URL}/init_model`, {
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
        const sampleIndexMap = computeStratifiedIndices(data.original_train_data, dataset);
        set({
          network: { input: [[]], layers, initialized: true },
          originalData: data.original_train_data,
          sampleIndexMap,
          configOpen: false,
          yMean: data.y_mean ?? null,
          yStd: data.y_std ?? null,
          // A fresh model means a fresh training history
          epoch: 0,
          loss: 0,
          prevLoss: 0,
          metric: 0,
          losses: [],
          accuracies: [],
          testLosses: [],
          testAccuracies: [],
          submittableScore: null,
          xorEpochsTo100: null,
        });
        const arch = data.layer_sizes.join(" → ");
        if (!get().tourActive) toast.success("Model ready", {
          description: `Architecture: ${arch} · ${data.layer_sizes.reduce((a: number, b: number, i: number, arr: number[]) => i < arr.length - 1 ? a + arr[i] * arr[i + 1] + arr[i + 1] : a, 0)} parameters`,
        });
      } else {
        throw new Error(serverErrorMessage(data) || "Failed to initialize model");
      }
    } catch (error) {
      console.error("Error initializing model:", error);
      const msg = error instanceof Error ? error.message : "";
      toast.error("Initialization failed", {
        description: msg || "Could not connect to the backend. Make sure the server is running.",
      });
    } finally {
      set({ isInitializing: false });
    }
  },

  initModelFrontend: () => {
    const { hiddenLayers, activations, dataset } = get();
    let inputSize = 0;
    let outputSize = 0;
    let accuracyMetric = "accuracy";

    if (dataset === "iris") {
      inputSize = 4; outputSize = 3;
    } else if (dataset === "auto_mpg") {
      inputSize = 4; outputSize = 1; accuracyMetric = "mae";
    } else if (dataset === "xor") {
      inputSize = 2; outputSize = 1;
    } else if (dataset === "mnist") {
      inputSize = 784; outputSize = 10;
    }

    const layerSizes = [inputSize, ...hiddenLayers, outputSize];
    const totalLayers = layerSizes.length;
    const outputActivation = dataset === "iris" || dataset === "mnist"
      ? "softmax"
      : dataset === "xor" ? "sigmoid" : "linear";
    const layers = layerSizes.map((size, index) => {
      const activation = index === totalLayers - 1 ? outputActivation : activations[index] || "relu";
      const layer = new NeuronLayer(size, activation, index, totalLayers);
      layer.initWeightsAndBiases(size, index + 1 === layerSizes.length ? 0 : layerSizes[index + 1]);
      return layer;
    });
    set({ network: { input: [[]], layers, initialized: false }, name: accuracyMetric, configOpen: true });
    get().clearSessionAndReset();
  },

  clearSessionAndReset: async () => {
    const { sessionId, dataset } = get();
    const resetDefaults = dataset === "xor"
      ? { learningRate: 0.3, activations: ["tanh", "tanh"], hiddenLayers: [4, 4] }
      : { learningRate: 0.1, activations: ["relu", "relu"], hiddenLayers: [4, 4] };
    // Always reset local state — even without a live session — so switching
    // datasets or architecture can't leave stale losses/metrics behind
    try {
      if (sessionId) {
        await fetchWithTimeout(`${API_URL}/clear_session?session_id=${sessionId}`, { method: "POST" });
      }
    } catch (error) {
      // Orphaned server sessions expire via TTL; the local reset still proceeds
      console.error("Error clearing session:", error);
    }
    set({
      sessionId: null,
      configOpen: true,
      hoveredConnection: null,
      hoveredNode: null,
      epoch: 0,
      loss: 0,
      prevLoss: 0,
      metric: 0,
      losses: [],
      accuracies: [],
      testLosses: [],
      testAccuracies: [],
      originalData: [],
      sampleIndexMap: [],
      yMean: null,
      yStd: null,
      submittableScore: null,
      xorEpochsTo100: null,
      trainingEpochs: 1,
      requestToken: get().requestToken + 1,
      ...resetDefaults,
    });
    if (sessionId) toast("Model reset");
  },

  runTrainingCycle: async () => {
    const { sessionId, learningRate, yStd, trainingEpochs } = get();
    if (get().runModel) return; // guard against double-fire while a cycle is in flight
    if (!sessionId) {
      toast.error("No model initialized", {
        description: "Complete step 3 to initialize the model before training.",
      });
      return;
    }

    get().setRunModel(true);
    const prevLossValue = get().loss;
    const requestToken = get().requestToken;

    try {
      const response = await fetchWithTimeout(`${API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          learning_rate: learningRate,
          epochs: trainingEpochs,
        }),
      }, 120_000); // large epoch counts on MNIST can exceed the default 15s

      const data = await response.json();
      // Config changed while this request was in flight — discard the stale result
      if (get().requestToken !== requestToken) return;

      if (response.ok) {
        // Trust what the server returned rather than assuming `epochs` results
        const results = data.training_results;
        const result = results[results.length - 1];
        const resultPrev = results.length > 1 ? results[results.length - 2] : null;
        const appliedEpochs = results.length;

        const scaleMetric = (value: number, metricName: string) =>
          metricName === "mae" && yStd !== null ? value * yStd : value;
        const batchMetrics: number[] = results.map((r: { metric: number; name: string }) => scaleMetric(r.metric, r.name));
        const batchLosses: number[] = results.map((r: { loss: number }) => r.loss);
        const batchTestLosses: number[] = results.map((r: { test_loss: number }) => r.test_loss);
        const batchTestMetrics: number[] = results.map((r: { test_metric: number; test_name: string }) => scaleMetric(r.test_metric, r.test_name));

        set((state: TrainingSlice) => {
          const updatedLayers = state.network?.layers.map((layer, index) => {
            const resultLayer = result.layers[index];
            const currBiases = layer.biases;
            const currWeights = layer.weights;
            if (resultLayer) {
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

          const changedConns: ChangedConnection[] = [];
          updatedLayers.forEach((layer, li) => {
            if (layer.prevWeights?.length > 0 && layer.weights?.length > 0) {
              layer.weights.forEach((row: number[], fi: number) => {
                row.forEach((currW: number, ti: number) => {
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

          // Server-side bookkeeping is authoritative (assignment epoch caps are
          // enforced against it) — fall back to local accumulation if absent
          const newEpoch = typeof data.epochs_trained === "number"
            ? data.epochs_trained
            : state.epoch + appliedEpochs;
          const cap = EPOCH_CAPS[state.dataset];
          const newAccuracies = [...state.accuracies, ...batchMetrics];
          const newTestAccuracies = [...state.testAccuracies, ...batchTestMetrics];

          let newXorEpochsTo100 = state.xorEpochsTo100;
          if (state.dataset === "xor" && newXorEpochsTo100 === null) {
            // xor's test set equals its train set, so either series works
            const hitIdx = batchTestMetrics.findIndex((m) => m >= 100);
            if (hitIdx !== -1) newXorEpochsTo100 = state.epoch + hitIdx + 1;
          }

          let newSubmittableScore: number | null = null;
          if (state.dataset === "xor") {
            newSubmittableScore = newXorEpochsTo100;
          } else if (newTestAccuracies.length === newEpoch) {
            // Score on the held-out test set — the same basis the backend uses
            // to verify submissions and grade assignments
            const scoreEpoch = cap !== null ? Math.min(newEpoch, cap) : newEpoch;
            newSubmittableScore = newTestAccuracies[scoreEpoch - 1] ?? null;
          }

          return {
            ...state,
            network: {
              ...state.network,
              input: result.input,
              initialized: true,
              layers: updatedLayers,
            },
            epoch: newEpoch,
            loss: result.loss,
            prevLoss: prevLossValue,
            metric: result.name === "mae" && yStd !== null ? result.metric * yStd : result.metric,
            name: result.name,
            losses: [...state.losses, ...batchLosses],
            accuracies: newAccuracies,
            testLosses: [...state.testLosses, ...batchTestLosses],
            testAccuracies: newTestAccuracies,
            changedConnections: changedConns.slice(0, 5),
            xorEpochsTo100: newXorEpochsTo100,
            submittableScore: newSubmittableScore,
          };
        });

        const lossArrow = prevLossValue > 0
          ? result.loss < prevLossValue ? "↓" : result.loss > prevLossValue ? "↑" : "→"
          : "";
        // Same scaling as the store/chart: MAE must be shown in original units
        const displayMetric = scaleMetric(result.metric, result.name);
        const metricLabel = result.name === "accuracy"
          ? `${displayMetric.toFixed(1)}% accuracy`
          : `MAE ${displayMetric.toFixed(3)}`;
        const { epoch: currentEpoch } = get();
        if (!get().tourActive) toast.success(`Epoch ${currentEpoch}`, {
          description: `Loss ${result.loss.toFixed(4)} ${lossArrow} · ${metricLabel}`,
        });
      } else {
        if (response.status === 404) {
          handleSessionExpired(set);
          return;
        }
        throw new Error(serverErrorMessage(data) || "Failed to run training cycle");
      }
    } catch (error) {
      console.error("Error running training cycle:", error);
      const msg = error instanceof Error ? error.message : "";
      toast.error("Training failed", {
        description: msg.startsWith("Failed to") || !msg
          ? "Could not complete the training cycle. Please try again."
          : msg,
      });
    } finally {
      get().setRunModel(false);
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
    // XOR: tanh avoids sigmoid's vanishing gradients (max derivative 0.25) without dying ReLU risk on 4 samples
    // MNIST: larger hidden layers to handle 784 inputs; smaller LR for stability
    const datasetDefaults =
      newDataset === "xor"
        ? { activations: ["tanh", "tanh"], hiddenLayers: [4, 4], learningRate: 0.3 }
        : newDataset === "mnist"
        ? { activations: ["relu", "relu"], hiddenLayers: [12, 10], learningRate: 0.05 }
        : { activations: ["relu", "relu"], hiddenLayers: [4, 4], learningRate: 0.1 };
    set({
      dataset: newDataset,
      datasetInfo: DATASET_INFO[newDataset],
      drawnDigitPrediction: null,
      ...datasetDefaults,
    });
    get().initModelFrontend();
  },

  getExplanation: () => {
    const { network, epoch, learningRate, dataset, loss, prevLoss, metric, name, sessionId } = get();
    return getExplanationText(network, epoch, learningRate, dataset, loss, prevLoss, metric, name, sessionId);
  },

  setWeight: async (layerIndex: number, fromIndex: number, toIndex: number, newValue: number) => {
    const { sessionId, network } = get();
    if (!sessionId || !network) return;
    try {
      const response = await fetchWithTimeout(`${API_URL}/set_weights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, layer_index: layerIndex, from_index: fromIndex, to_index: toIndex, new_value: newValue }),
      });
      const data = await response.json();
      if (response.ok) {
        set((state: TrainingSlice) => {
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
        if (response.status === 404) {
          handleSessionExpired(set);
          return;
        }
        throw new Error(serverErrorMessage(data) || "Failed to update weight");
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : "";
      toast.error("Weight update failed", { description: msg.startsWith("Failed to") || !msg ? "Could not apply the new weight value." : msg });
    }
  },

  predictDigit: async (pixels: number[]) => {
    const { sessionId } = get();
    if (!sessionId) {
      toast.error("No model initialized", { description: "Initialize the model before predicting." });
      return;
    }
    try {
      const response = await fetchWithTimeout(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, pixels }),
      });
      const data = await response.json();
      if (response.ok) {
        const prediction = { predictedClass: data.predicted_class, confidences: data.confidences };
        set((state: TrainingSlice) => {
          if (!state.network) return { drawnDigitPrediction: prediction, drawnDigitPixels: pixels };
          const si = state.sampleIndexMap[state.sampleIndex] ?? state.sampleIndex;
          const { Z: zAll, A: aAll } = forwardPassSingle(state.network.layers, pixels);

          const newInput = [...state.network.input];
          newInput[si] = pixels;

          const updatedLayers = state.network.layers.map((layer, li) => {
            if (li >= aAll.length) return layer;
            const newA = [...(layer.A ?? [])];
            const newZ = [...(layer.Z ?? [])];
            newA[si] = aAll[li];
            newZ[si] = zAll[li];
            layer.A = newA;
            layer.Z = newZ;
            return layer;
          });

          return {
            drawnDigitPrediction: prediction,
            drawnDigitPixels: pixels,
            network: { ...state.network, input: newInput, layers: updatedLayers },
          };
        });
      } else {
        if (response.status === 404) {
          handleSessionExpired(set);
          return;
        }
        throw new Error(serverErrorMessage(data) || "Prediction failed");
      }
    } catch (error) {
      console.error("Prediction error:", error);
      const msg = error instanceof Error ? error.message : "";
      toast.error("Prediction failed", { description: msg.startsWith("Prediction") || !msg ? "Could not run prediction." : msg });
    }
  },

  clearDigitPrediction: () => set({ drawnDigitPrediction: null, drawnDigitPixels: null }),
});
