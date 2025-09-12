import useStore from "@/hooks/store";

export const getExplanationText = () => {
  const { network, epoch, learningRate, dataset, loss, metric, name, sessionId } = useStore();

  let datasetExplanation = "";
  let lossExplanation = "";
  let accuracyExplanation = "";

  if (dataset === "mnist") {
    datasetExplanation = `Dataset: [**MNIST (Handwritten Digits)**](https://en.wikipedia.org/wiki/MNIST_database)  
Task: **Multi-class classification** (10 classes: digits 0-9)  
Output: A **probability distribution** over 10 digits. The highest probability is the predicted digit.  
Interpretation: The network learns to recognize **handwritten digits** by adjusting weights to reduce **classification errors**.`;

    lossExplanation = `Loss Function: [**Cross-Entropy Loss**](https://en.wikipedia.org/wiki/Cross_entropy)  
**Cross-entropy** measures the difference between predicted probabilities and true labels. **Lower values** indicate better predictions.`;

    accuracyExplanation = `Accuracy:  
Calculated as the percentage of **correct predictions**. A higher value indicates the model is correctly identifying more digits.`;
  } else if (dataset === "iris") {
    datasetExplanation = `Dataset: [**Iris Flower Dataset**](https://en.wikipedia.org/wiki/Iris_flower_data_set)  
Task: **Multi-class classification** (3 species)  
Output: **Probability distribution** over 3 classes (Setosa, Versicolor, Virginica)  
Interpretation: The model learns patterns in **sepal/petal measurements** to classify flower species.`;

    lossExplanation = `Loss Function: [**Cross-Entropy Loss**](https://en.wikipedia.org/wiki/Cross_entropy)  
Measures the difference between predicted and actual species labels. **Lower values** indicate more accurate predictions.`;

    accuracyExplanation = `Accuracy:  
Indicates how often the model correctly classifies **flower species** based on input features.`;
  } else if (dataset === "auto_mpg") {
    datasetExplanation = `Dataset: [**Auto MPG Dataset**](https://archive.ics.uci.edu/ml/datasets/auto+mpg)  
Task: **Regression** (Predicting fuel efficiency)  
Output: A single **numeric value** representing predicted MPG  
Interpretation: The model finds patterns in features like **displacement**, **horsepower**, **weight**, and **acceleration** to predict fuel efficiency.`;

    lossExplanation = `Loss Function: [**Mean Squared Error (MSE)**](https://en.wikipedia.org/wiki/Mean_squared_error)  
**MSE** calculates the average squared difference between predicted and actual MPG values. The model tries to **minimize** this.`;

    accuracyExplanation = `Metric: **Mean Absolute Error (MAE)**  
**MAE** is the average of absolute differences between predicted and true MPG values. **Lower MAE** means better predictions.`;
  }

  if (!network || !sessionId) {
    return `The neural network has not been initialized yet.  
Configure the network and click "Initialize Model" to begin.

${datasetExplanation}`;
  }

  if (epoch === 0) {
    return `The model has been initialized with **random weights** and biases.  
- **Nodes**: Neurons in the network  
- **Connections**: Weights between neurons  
- **Thickness**: Strength of each connection  

The model is not trained yet. Once training begins, the weights will adjust to fit the data.

${datasetExplanation}

${lossExplanation}

${accuracyExplanation}`;
  }

  return `Training Cycle **${epoch}** Completed  
  
1. [**Forward Propagation**](https://en.wikipedia.org/wiki/Feedforward_neural_network): Inputs pass through the network, each neuron computes a **weighted sum** and applies an **activation function**.  
2. [**Loss Calculation**](https://en.wikipedia.org/wiki/Loss_function): Measures how far predictions are from actual values.  
3. [**Backward Propagation**](https://en.wikipedia.org/wiki/Backpropagation): The error is **back-propagated** to update weights.  
4. **Parameter Update**: Weights and biases are adjusted using the **learning rate** (${learningRate}).

${datasetExplanation}

${lossExplanation}

${accuracyExplanation}

Current Metrics:  
- **Loss**: ${loss ? loss.toFixed(4) : "N/A"}  
- **Metric** (${name}): ${metric ? metric.toFixed(2) : "N/A"}${name === "accuracy" ? "%" : ""}

How to Interpret Results:  
- A **lower loss** means better performance.  
- In **classification tasks**, the class with the highest probability is the prediction.  
- In **regression tasks**, the output value should approach **real-world targets**.`;
};

// Hidden layer info
export const HIDDEN_LAYER_INFO = `**Hidden layers** allow the model to learn **abstract features** by transforming inputs through weights and activation functions.`;

// Hidden layer detailed explanation
export const HIDDEN_LAYER_LEARN_MORE = `More hidden layers support **deeper learning** but may require more data and computation.  
Each layer has **neurons** that process information before passing it to the next layer.  
**Activation functions** transform outputs between layers.  

Hidden layers act as **feature detectors**:  
- **Early layers** capture simple patterns  
- **Deeper layers** recognize complex features  

Too many layers can lead to **overfitting** (memorizing data).  
Regularization techniques like **dropout** and **batch normalization** help mitigate this.`;

export const DATASET_INFO: { [key: string]: string } = {
  auto_mpg: `The [**Auto MPG dataset**](https://archive.ics.uci.edu/ml/datasets/auto+mpg) contains **automobile fuel efficiency data**.  
- **Inputs:** Displacement, Horsepower, Weight, Acceleration  
- **Output:** Miles per gallon (numeric)`,

  mnist: `The [**MNIST dataset**](https://en.wikipedia.org/wiki/MNIST_database) is a collection of **handwritten digit images**.  
- ** Inputs:** 28x28 grayscale images (784 pixels)  
- ** Output:** A digit label (0–9)`,

  iris: `The [**Iris dataset**](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic dataset with **flower measurements**.  
- **Inputs:** Sepal length, Sepal width, Petal length, Petal width  
- **Output:** Class label (0 = Setosa, 1 = Versicolor, 2 = Virginica)`
};
