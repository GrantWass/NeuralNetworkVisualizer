const DATASETS = ["california_housing", "iris"];
// const DATASETS = ["california_housing", "mnist", "iris"];

const ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "tanh", "linear"];

const DATASET_INPUT_FEATURES: { [key: string]: string[] } = {
  california_housing: [
    "Median income",
    "Median house age",
    "Average rooms (qty)",
    "Average bedrooms (qty)",
    "Population",
    "Average occupancy",
    "Latitude",
    "Longitude"
  ],
  mnist: [
    "784 grayscale pixel values (28x28 image)"
  ],
  iris: [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
  ]
};

const DATASET_INPUT_FEATURES_SHORT: { [key: string]: string[] } = {
  california_housing: [
    "Avg income",
    "House age",
    "Rooms (qty)",
    "Bedrooms (qty)",
    "Population",
    "Avg occupancy",
    "Latitude",
    "Longitude"
  ],
  mnist: [
    "784 grayscale pixel values (28x28 image)"
  ],
  iris: [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
  ]
};


export { DATASETS, ACTIVATION_FUNCTIONS, DATASET_INPUT_FEATURES, DATASET_INPUT_FEATURES_SHORT };
