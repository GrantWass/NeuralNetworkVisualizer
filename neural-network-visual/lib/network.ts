const createNetwork = (inputNodes: number, outputNodes: number, hiddenLayers: number[]) => {
    const layers = [
      { nodes: inputNodes, name: "Input" },
      ...hiddenLayers.map((nodes, i) => ({ nodes, name: `Hidden ${i + 1}` })),
      { nodes: outputNodes, name: "Output" },
    ]
  
    const weights = layers.slice(0, -1).map((layer, i) =>
      Array(layer.nodes)
        .fill(0)
        .map(() =>
          Array(layers[i + 1].nodes)
            .fill(0)
            .map(() => Math.random() * 2 - 1),
        ),
    )
  
    const biases = layers.slice(1).map((layer) =>
      Array(layer.nodes)
        .fill(0)
        .map(() => Math.random() * 2 - 1),
    )
  
    const activations = layers.map((layer) =>
      Array(layer.nodes)
        .fill(0)
        .map(() => Math.random()),
    )
  
    return { layers, weights, biases, activations }
  }

  export { createNetwork }