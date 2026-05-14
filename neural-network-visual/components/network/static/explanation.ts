import { NetworkState } from "./types";

const getTrendComment = (loss: number, prevLoss: number, epoch: number): string => {
  if (epoch <= 1 || prevLoss === 0) return "";
  const pct = ((prevLoss - loss) / prevLoss) * 100;
  if (pct > 10) return " Loss dropped sharply this cycle — the network is learning quickly and the current settings are working well.";
  if (pct > 2) return " Loss is steadily decreasing — training is still making healthy progress, so keep going.";
  if (pct >= -1) return " Loss is still improving, but more slowly — the model may be nearing a flatter part of the curve, so keep training or try a small learning-rate tweak.";
  return " Loss increased this cycle — the learning rate may be a little too aggressive, causing the weights to overshoot before settling.";
};

export const getExplanationText = (
  network: NetworkState | null,
  epoch: number,
  learningRate: number,
  dataset: string,
  loss: number,
  prevLoss: number,
  metric: number,
  name: string,
  sessionId: string | null
) => {
  // ── not initialized ────────────────────────────────────────────────────────
  if (!network || !sessionId) {
    if (dataset === "iris") {
      return `Configure your network using the steps above, then click **Initialize Model** to begin.

**What you'll be training:** A network that reads four flower measurements — sepal length, sepal width, petal length, and petal width — and predicts which of three Iris species the flower belongs to: Setosa, Versicolor, or Virginica.

The network learns by seeing labeled examples and adjusting its weights until it can reliably distinguish the three species from raw measurements alone.`;
    }
    if (dataset === "auto_mpg") {
      return `Configure your network using the steps above, then click **Initialize Model** to begin.

**What you'll be training:** A network that takes four characteristics of a car — engine displacement, horsepower, weight, and acceleration — and predicts its fuel efficiency in miles per gallon.

Unlike classification, this is a regression task: the output is a continuous number rather than a category. The network learns to approximate the relationship between a car's specs and how efficiently it burns fuel.`;
    }
    if (dataset === "xor") {
      return `Configure your network using the steps above, then click **Initialize Model** to begin.

**What you'll be training:** A network that learns the XOR rule — given two binary inputs A and B, output 1 when exactly one of them is 1, and 0 otherwise.

XOR is the classic proof that hidden layers matter. A single layer of neurons can only draw straight-line boundaries, and XOR cannot be separated by a straight line. You need at least one hidden layer to solve it. Watch whether your network architecture is deep enough.`;
    }
  }

  // ── initialized, epoch 0 ───────────────────────────────────────────────────
  if (epoch === 0) {
    const inputLayer = network?.layers[0];
    const outputLayer = network?.layers[network.layers.length - 1];
    const hiddenCount = (network?.layers.length ?? 2) - 2;
    const totalParams = network?.layers.reduce((sum, layer) => {
      return sum + (layer.weights?.flat().length ?? 0) + (layer.biases?.length ?? 0);
    }, 0) ?? 0;

    const archDesc = hiddenCount === 0
      ? `no hidden layers (direct input → output)`
      : hiddenCount === 1
      ? `1 hidden layer with ${network?.layers[1]?.weights?.[0]?.length ?? "?"} neurons`
      : `${hiddenCount} hidden layers`;

    if (dataset === "iris") {
      return `Your network is ready. Weights have been randomly initialized — at this point, predictions are essentially random guesses.

**Architecture:** ${inputLayer?.weights?.length ?? 4} inputs → ${archDesc} → ${outputLayer?.weights?.[0]?.length ?? 3} outputs (${totalParams} learnable parameters total).

**What to expect:** When you click Run Training Cycle, the network will see all 150 flower samples, compute predictions, measure how wrong it was (cross-entropy loss), then adjust every weight by a small amount in the direction that reduces the error. After enough cycles, it should learn to separate the three species reliably. Random guessing would give ~33% accuracy — anything above that means the network is finding real patterns.`;
    }
    if (dataset === "auto_mpg") {
      return `Your network is ready. Weights have been randomly initialized — at this point, MPG predictions are essentially random.

**Architecture:** ${inputLayer?.weights?.length ?? 4} inputs → ${archDesc} → 1 output (${totalParams} learnable parameters total).

**What to expect:** The network will learn to map (displacement, horsepower, weight, acceleration) → MPG by minimizing mean squared error across the training set. A naive model that always predicts the mean MPG would have a MAE of roughly 6–7. A well-trained network should get below 3 MPG error. Watch the MAE curve to see how quickly it drops.`;
    }
    if (dataset === "xor") {
      return `Your network is ready. Weights have been randomly initialized — at this point, outputs are near 0.5 (maximum uncertainty).

**Architecture:** 2 inputs → ${archDesc} → 1 output (${totalParams} learnable parameters total).

**What to expect:** XOR has only 4 training patterns, so each cycle passes through all of them. The network needs to learn a non-linear decision boundary — with a learning rate around 0.3 and sigmoid activations, it should converge to 100% accuracy within 50–200 cycles. If it gets stuck at 75% (3 of 4 correct), re-initialize — random weight initialization can land near a saddle point that's hard to escape.`;
    }
  }

  // ── training in progress ───────────────────────────────────────────────────
  const trend = getTrendComment(loss, prevLoss, epoch);

  if (dataset === "iris") {
    const pct = metric.toFixed(1);
    const correctPer10 = Math.round(metric / 10);
    let progressComment = "";
    if (metric >= 95) progressComment = `At ${pct}%, your model is classifying almost every flower correctly. This is near the theoretical limit for this dataset — the three species do overlap slightly in measurement space.`;
    else if (metric >= 80) progressComment = `At ${pct}%, your model correctly identifies roughly ${correctPer10} out of 10 flowers. The species it's most likely confusing are Versicolor and Virginica, which have overlapping petal measurements.`;
    else if (metric >= 50) progressComment = `At ${pct}%, your model is doing better than random (which would give ~33%), but there's still significant room for improvement. It's learned some structure but hasn't fully separated the classes.`;
    else progressComment = `At ${pct}%, your model is barely above random guessing (~33%). The weights haven't found a useful representation yet — keep training or try adjusting the learning rate.`;

    return `**After ${epoch} training cycle${epoch === 1 ? "" : "s"}**

${progressComment}${trend}

**Loss (cross-entropy): ${loss.toFixed(4)}**
Cross-entropy measures how confident and correct the predictions are. A loss of 0 would mean perfect certainty on every sample. Values above 1 suggest the network is frequently wrong or unsure; below 0.3 is generally good for this dataset.

**What to try:** ${metric >= 90 ? "Your model is performing well. Try comparing learning rates using the Compare LR button, or explore how the weights change across epochs in the Forward Pass tab." : metric >= 60 ? "Keep running training cycles and watch the loss curve. If it plateaus, try a learning rate of 0.05–0.15." : "Run more cycles — the model is still in the early stages of learning. If loss isn't decreasing after 20+ cycles, try adjusting the learning rate."}`;
  }

  if (dataset === "auto_mpg") {
    let progressComment = "";
    if (metric < 2) progressComment = `Your mean absolute error is ${metric.toFixed(2)} MPG — predictions are within 2 MPG on average. That's strong performance for this dataset, where cars range from about 9 to 46 MPG.`;
    else if (metric < 4) progressComment = `Your mean absolute error is ${metric.toFixed(2)} MPG — predictions are in the right ballpark, typically off by 2–4 MPG. There's still room to improve, especially on edge cases like very high or low efficiency vehicles.`;
    else if (metric < 8) progressComment = `Your mean absolute error is ${metric.toFixed(2)} MPG — the model has a sense of the relationship between car specs and efficiency, but predictions can be significantly off. Keep training.`;
    else progressComment = `Your mean absolute error is ${metric.toFixed(2)} MPG — still quite high. A naive model that always predicts the mean MPG would have MAE around 6–7, so the network is only slightly better than that baseline. More training or a different learning rate may help.`;

    return `**After ${epoch} training cycle${epoch === 1 ? "" : "s"}**

${progressComment}${trend}

**Loss (MSE): ${loss.toFixed(4)}**
Mean squared error penalizes large errors more than small ones (because errors are squared before averaging). It's useful for regression but harder to interpret directly — MAE is usually the more intuitive metric here.

**What to try:** ${metric < 3 ? "The model is performing well. Try the Compare LR feature to see if a different learning rate would converge faster." : "If the loss curve has flattened, try a slightly different learning rate. For regression, 0.01–0.1 often works well."}`;
  }

  if (dataset === "xor") {
    const correct = Math.round((metric / 100) * 4);
    let progressComment = "";
    if (metric >= 100) progressComment = `Your network has learned XOR perfectly — it correctly classifies all 4 patterns. The hidden layer has found a non-linear boundary that a single-layer network couldn't learn.`;
    else if (metric >= 75) progressComment = `Your model gets ${correct} out of 4 XOR patterns right. It's close — one pattern is still being misclassified. The decision boundary is forming but hasn't fully converged.`;
    else if (metric >= 50) progressComment = `Your model gets ${correct} out of 4 XOR patterns right — same as random guessing. The weights haven't found the right non-linear structure yet.`;
    else progressComment = `Your model is getting fewer than half the patterns right. This can happen early in training or if the learning rate is causing weights to oscillate.`;

    return `**After ${epoch} training cycle${epoch === 1 ? "" : "s"}**

${progressComment}${trend}

**Loss (binary cross-entropy): ${loss.toFixed(4)}**
Binary cross-entropy measures how far each sigmoid output is from the true 0 or 1 label. A loss of 0 means perfect confidence on all 4 patterns. Values above 0.5 mean the network is still uncertain or wrong on multiple examples.


**What to try:** ${metric >= 100 ? "XOR is solved. Try re-initializing with fewer hidden neurons to see the minimum architecture required, or switch to a harder dataset." : "XOR converges most reliably around LR 0.2–0.4. If stuck at 75%, re-initialize — the sigmoid activations mean no dead neurons, but some random seeds land near saddle points that are hard to escape."}

**Tip:** If the loss curve has flattened, switch to the **Compute Gradients** tab and inspect the dW values — when they're near zero, weight updates have effectively stalled. The dZ column at the output layer tells you why: large dZ means the network is confidently wrong, while dZ near zero means predictions are close to the decision boundary. For XOR, a plateau at 75% usually means the weights settled near a saddle point where gradients from the four patterns nearly cancel each other out — a fresh re-initialization is often all it takes to escape.`;
  }

  return "";
};

// Hidden layer info
export const HIDDEN_LAYER_INFO = `**Hidden layers** let the model learn **abstract features** by stacking transformations through weights and activation functions.

Common hidden activations include **ReLU** for fast feature extraction, **tanh** for smoother centered outputs, and **sigmoid** for older or simpler setups. The right choice depends on how sharply you want the layer to respond to inputs.`;

// Hidden layer detailed explanation
export const HIDDEN_LAYER_LEARN_MORE = `More hidden layers support **deeper learning** but usually need more data and computation.
Each layer has **neurons** that mix the previous layer's outputs, add a bias, and then apply an activation function before passing information onward.

Hidden layers often behave like **feature detectors**:
- **Early layers** capture simple patterns
- **Deeper layers** recognize more abstract relationships

Too many layers can lead to **overfitting** (memorizing data).
Regularization techniques like **dropout** and **batch normalization** help mitigate this.`;

export const DATASET_INFO: { [key: string]: string } = {
  auto_mpg: `The [**Auto MPG dataset**](https://archive.ics.uci.edu/ml/datasets/auto+mpg) contains **automobile fuel efficiency data**.
- **Inputs:** Displacement, Horsepower, Weight, Acceleration
- **Output:** Miles per gallon (numeric)`,

  iris: `The [**Iris dataset**](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic dataset with **flower measurements**.
- **Inputs:** Sepal length, Sepal width, Petal length, Petal width
- **Output:** Class label (0 = Setosa, 1 = Versicolor, 2 = Virginica)`,

  xor: `The **XOR dataset** is the simplest non-linear problem in ML.
- **Inputs:** Two binary values (A and B)
- **Output:** 1 if exactly one input is 1, else 0
- **Why it matters:** A single-layer network *cannot* solve it — it requires at least one hidden layer.`,

  mnist: `The [**MNIST dataset**](http://yann.lecun.com/exdb/mnist/) is the classic handwritten digit benchmark.
- **Inputs:** 784 pixel values (28×28 grayscale image, normalized 0–1)
- **Output:** Digit class (0–9) via softmax
- **Train on the data, then draw your own digit to see what the network predicts.**`
};
