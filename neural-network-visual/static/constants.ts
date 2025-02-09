const DATASETS = ["california_housing", "mnist", "iris"]
const DATASET_INFO: { [key: string]: string } = {
  california_housing: "The California Housing dataset contains information from the 1990 California census. It includes features like median income, housing median age, average rooms, etc.",
  mnist: "The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems.",
  iris: "The Iris dataset is a multivariate dataset introduced by Ronald Fisher. It consists of 50 samples from each of three species of Iris flowers."
};
const ACTIVATION_FUNCTIONS = ["relu", "sigmoid", "tanh", "linear"];

export { DATASETS, DATASET_INFO, ACTIVATION_FUNCTIONS }