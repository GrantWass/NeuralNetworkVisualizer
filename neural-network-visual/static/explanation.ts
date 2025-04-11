import useStore from "@/hooks/store";

export const getExplanationText = () => {
  const { network, epoch, learningRate, dataset, loss, metric, name } = useStore();
  
  let datasetExplanation = "";
  let lossExplanation = "";
  let accuracyExplanation = "";

  if (dataset === "mnist") {
    datasetExplanation = `Dataset: MNIST (Handwritten Digits)  
      Task: Multi-class classification (10 classes, digits 0-9)  
      Output: A probability distribution over the 10 digits. The highest probability corresponds to the predicted digit.  
      Interpretation: The network is learning to recognize handwritten numbers by adjusting weights to minimize classification errors.
    `;
    lossExplanation = `Loss Function: Cross-entropy loss  
      The loss measures the difference between the predicted probabilities and the actual class labels. A lower loss means the model's predictions are closer to the true labels. The model aims to minimize this value during training.
    `;
    accuracyExplanation = `Accuracy:  
      Accuracy is calculated as the percentage of correct predictions. In classification tasks like MNIST, a higher accuracy indicates that the model is correctly identifying more digits.
    `;
  } else if (dataset === "iris") {
    datasetExplanation = `Dataset: Iris Flower Dataset  
      Task: Multi-class classification (3 species)  
      Output: A probability distribution over the 3 classes (Setosa, Versicolor, Virginica).  
      Interpretation: The network is learning patterns in petal and sepal measurements to correctly classify flower species.
    `;
    lossExplanation = `Loss Function: Cross-entropy loss  
      The loss here measures the difference between the predicted probabilities and the true class labels for the flowers.
    `;
    accuracyExplanation = `Accuracy:  
      The model's accuracy is determined by how often it correctly classifies the flowers into the correct species based on the input features (sepal and petal dimensions). The accuracy metric will increase as the model correctly classifies more samples.
    `;
  } else if (dataset === "california_housing") {
    datasetExplanation = `Dataset: California Housing Prices  
      Task: Regression (Predicting continuous values)  
      Output: A single numerical value representing the predicted house price.  
      Interpretation: The network is learning relationships between features like median income, house age, and population density to estimate home prices.
    `;
    lossExplanation = `Loss Function: Mean Squared Error (MSE)  
      MSE calculates the average squared difference between the predicted and actual house prices. The model tries to minimize this value during training to improve predictions.
    `;
    accuracyExplanation=`Our accuracy metric for this dataset is Mean Absolute Error (MAE). This is the average difference between predicted and actual house prices. For regression tasks like this, MAE will decrease as the predicted house prices become closer to the actual prices.
    `
  }

  if (!network || !network.initialized) {
    return `The neural network has not been initialized yet. Configure the network and click 'Initialize Model' to start.

    ${datasetExplanation}`;
  }

  if (epoch === 0) {
    return `The model has been initialized with random weights and biases.  
      - Each node represents a neuron, and each line represents a connection (weight).  
      - The thickness of each connection represents the strength of the weight.  
      The network is not yet trained. Once training starts, the model will adjust its parameters to fit the data.

      ${datasetExplanation}
      ${lossExplanation}
      ${accuracyExplanation}`;
  }

  return `Training Cycle ${epoch} Completed:  
    1. Forward Propagation: The input is processed through the network, with each neuron applying a weighted sum and an activation function.  
    2. Loss Calculation: The difference between the predicted and actual values is computed using the loss function. For classification tasks, the loss is typically cross-entropy loss, and for regression tasks, it is MSE.  
    3. Backward Propagation: The error is propagated backward to adjust weights using gradient descent.  
    4. Parameter Update: Weights and biases are updated using the learning rate (${learningRate}).

    ${datasetExplanation}
    ${lossExplanation}
    ${accuracyExplanation}
    
    Current Metrics:  
    - Loss: ${loss ? loss.toFixed(4) : "N/A"}  
    - Metric (${name}): ${metric ? metric.toFixed(2) : "N/A"}${name == "accuracy" ? "%": ""}

    How to Interpret Results:  
    - If training is successful, the network will gradually reduce its loss and improve predictions.  
    - For classification tasks, the highest output probability is the predicted class.  
    - For regression tasks, the output is a continuous value that should approach real-world values.`;
}