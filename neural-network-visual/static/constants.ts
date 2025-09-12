const DATASETS = ["auto_mpg", "iris"];

const ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "tanh", "linear"];

const DATASET_INPUT_FEATURES: { [key: string]: string[] } = {
  auto_mpg: [
    "Displacement",
    "Horsepower", 
    "Weight",
    "Acceleration"
  ],
  iris: [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
  ]
};

const DATASET_INPUT_FEATURES_SHORT: { [key: string]: string[] } = {
  auto_mpg: [
    "Displacement",
    "Horsepower",
    "Weight", 
    "Acceleration"
  ],
  iris: [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
  ]
};


export { DATASETS, ACTIVATION_FUNCTIONS, DATASET_INPUT_FEATURES, DATASET_INPUT_FEATURES_SHORT };
