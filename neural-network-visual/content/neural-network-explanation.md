## The problem neural networks solve

Traditional computer programs follow explicit instructions. A programmer writes rules, conditions, and logic that tell the computer exactly what to do in every situation. This works well for structured problems with clear steps, like sorting numbers, calculating taxes, or validating passwords. But many real-world problems are not easily reducible to rules.

Consider image recognition. How would you write a traditional program that recognizes a cat in a photograph? You might try rules like:

- Cats have ears
- Cats have whiskers
- Cats have fur
- Cats have certain proportions

The problem is that real images are messy. Lighting changes. Angles change. Backgrounds vary. Cats can be partially hidden, blurry, upside down, or stylized. The number of possible cases becomes effectively infinite.

This is where neural networks become powerful. Instead of manually writing rules, we allow the system to *learn patterns directly from data*. A neural network is shown many examples of inputs and correct outputs, and over time it adjusts itself to produce increasingly accurate predictions.

For example:

- Show a network millions of labeled images and it learns image recognition
- Show it text from the internet and it learns language patterns
- Show it customer behavior and it learns recommendations
- Show it historical data and it learns prediction patterns

Neural networks excel at problems involving pattern recognition, approximation, high-dimensional relationships, complex nonlinear behavior, perception, and language. At their core, neural networks are systems that learn mathematical representations of patterns hidden inside data. Rather than being programmed with explicit knowledge, they discover statistical structure through experience.

## How a neural network represents information

A neural network is built from layers of interconnected computational units called *neurons*. Each neuron performs a simple mathematical operation, but when many neurons are connected together, complex behavior emerges.

The basic structure of a neural network contains three kinds of layers:

1. Input layer
2. Hidden layers
3. Output layer

The input layer receives raw information from the outside world. Depending on the task, these inputs could represent pixel values in an image, audio waveforms, sensor measurements, numerical features, or words and tokens. Each neuron then passes information forward through weighted connections.

### Weights, biases, and activations

Weights determine how strongly one neuron influences another. Large positive weights amplify signals, negative weights suppress them, and small weights have little influence. During learning, the network continuously adjusts these weights to improve its predictions.

Biases are additional learnable values added before activation. They allow neurons to shift their behavior independently of incoming signals, acting somewhat like adjustable thresholds.

After computing weighted inputs, neurons apply an *activation function*. Activation functions introduce nonlinearity into the network. Without them, even very deep neural networks would collapse into simple linear systems.

Common activation functions include:

- ReLU
- Sigmoid
- Tanh
- Softmax

These functions determine how strongly a neuron responds to incoming information. In the visualization, neuron brightness or color intensity often represents activation strength. Highly active neurons indicate features the network currently considers important.

### Layers and feature hierarchies

Neural networks learn hierarchical representations. Early layers typically detect simple patterns like edges, colors, shapes, and local structures. Middle layers combine these into more complex concepts, while deeper layers represent increasingly abstract features like faces, objects, semantic meaning, or language structure.

The network gradually transforms raw input data into increasingly meaningful internal representations until a final prediction can be made.

## How a neural network learns

A neural network begins with random weights, so its initial predictions are effectively meaningless. Learning is the process of gradually adjusting those weights so the network produces more accurate outputs over time.

This happens through four major steps:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Gradient descent

Together, these steps form the foundation of modern deep learning.

### Forward propagation and loss

During forward propagation, information flows through the network from input to output. Suppose we are training a network to recognize handwritten digits. The input image is converted into numerical values and fed into the input layer. Each neuron processes information from the previous layer using weighted sums, biases, and activation functions.

As signals move through the network, neurons respond to patterns they detect in the data. Eventually, the output layer produces a prediction. For example, 10 output neurons might represent digits 0–9, with the highest activation becoming the network’s prediction.

The prediction is then compared to the correct answer using a *loss function*, which quantifies error numerically. Common examples include mean squared error and cross entropy loss. If the network predicts correctly, the loss is small. If the prediction is very wrong, the loss becomes large.

The entire goal of training is to minimize loss over time.

### Backpropagation and gradient descent

Once the error is known, the network must determine which weights contributed to that error. This is the role of backpropagation.

Backpropagation propagates error information backward through the network using calculus and the chain rule. Conceptually, it answers questions like:

- Which connections were most responsible for the mistake?
- Which neurons activated too strongly?
- Which weights should increase or decrease?

Each parameter receives a *gradient*, which measures how changing that parameter would affect the loss. Large gradients indicate parameters with strong influence on the error, while small gradients indicate parameters with little impact.

Once gradients are computed, the network updates its parameters using gradient descent. The idea is simple:

> Slightly adjust weights in the direction that reduces error.

This process repeats across thousands or millions of training examples. Useful connections strengthen, harmful connections weaken, and internal representations gradually become organized.

Modern neural networks may contain millions or billions of learnable parameters, but the core learning mechanism remains surprisingly elegant:

1. Make a prediction
2. Measure error
3. Compute responsibility
4. Update parameters
5. Repeat

## Reading the visualization

The visualization on this page is designed to make these abstract concepts more concrete. As data flows through the network, you can observe how information transforms layer by layer.

### Nodes and connections

Each circle represents a neuron. Neuron brightness, color, or size often corresponds to activation strength. Highly active neurons indicate features currently being emphasized by the network, while inactive neurons contribute little to the current prediction.

Lines between neurons represent weights. Thicker or brighter connections usually indicate stronger influence, while different colors may represent positive and negative weights. As training progresses, these connections evolve continuously as the network learns.

### Layers and training dynamics

Information moves from left to right through the input, hidden, and output layers. Early layers often respond to simple patterns, while deeper layers encode more abstract information. Watching activations propagate through the network helps reveal how complex predictions emerge from many small computations.

During training, you may observe:

- Activations stabilizing
- Weights reorganizing
- Predictions becoming more confident
- Loss decreasing over time

The visualization transforms neural network learning from abstract mathematics into an observable process. Rather than viewing training as a black box, you can watch the system gradually structure itself through experience.

## What's next

Neural networks are the foundation of modern AI, but they are only the beginning.

Once you understand the concepts on this page, you can explore deeper topics such as:

- Convolutional neural networks (CNNs)
- Transformers and attention
- Recurrent neural networks (RNNs)
- Reinforcement learning
- Embeddings and representation learning
- Optimization theory
- Generative models

Some excellent resources for continuing your learning journey include:

- [3Blue1Brown — Neural Networks Series](https://www.3blue1brown.com/topics/neural-networks)
- [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Playground](https://playground.tensorflow.org/)

The best way to truly understand neural networks is to experiment with them directly. Watch the visualization, change parameters, train small models, and observe what changes.

Neural networks become far less mysterious once you can see learning happening in real time.