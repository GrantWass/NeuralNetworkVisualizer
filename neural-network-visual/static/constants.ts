const DATASETS = ["california_housing", "mnist", "iris"];

const DATASET_INFO: { [key: string]: string } = {
  california_housing: `The California Housing dataset contains information from the 1990 California census.\n\n- Inputs (Features): Median income, Housing median age, Average rooms per household, Average bedrooms per household, Population, Households, Latitude, Longitude.\n- Output (Target): Median house value (continuous numeric value).`,

  mnist: `The MNIST dataset is a large database of handwritten digits, commonly used for training image processing and deep learning models.\n\n- Inputs (Features): 28x28 grayscale images of handwritten digits (784 pixels in total, each represented as a value between 0 and 255).\n- Output (Target): A digit label (0-9), representing the digit in the image.`,

  iris: `The Iris dataset is a well-known multivariate dataset introduced by Ronald Fisher. It consists of flower measurements for three species of Iris.\n\n- Inputs (Features): Sepal length, Sepal width, Petal length, Petal width.\n- Output (Target): A class label (0, 1, or 2), corresponding to the three species: Iris Setosa, Iris Versicolor, and Iris Virginica.`
};

const ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "tanh", "linear"];

const HIDDEN_LAYER_INFO = `Hidden layers in a neural network enable the model to learn complex patterns in data by transforming inputs through weighted connections and activation functions.`;

const HIDDEN_LAYER_LEARN_MORE = `- More hidden layers allow for deeper learning but may require more training data and computational power.\n- Each hidden layer has a set number of neurons, which determine how much information is processed.\n- Activation functions decide how the weighted sum of inputs is transformed before passing to the next layer. \n \n Hidden layers act as feature extractors, learning increasingly abstract representations of input data as information passes through each layer.\n\n- Lower layers capture simple patterns, like edges in image recognition, while deeper layers learn complex features.\n- Too many layers can lead to overfitting, meaning the model memorizes training data rather than generalizing well.\n- Techniques like dropout and batch normalization help improve performance and stability.`;


export { DATASETS, DATASET_INFO, ACTIVATION_FUNCTIONS, HIDDEN_LAYER_INFO, HIDDEN_LAYER_LEARN_MORE };
