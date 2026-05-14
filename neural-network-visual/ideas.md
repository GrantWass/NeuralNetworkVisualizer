# Feature Ideas

Remaining ideas from the audit. The decision boundary canvas and gradient flow animation are already implemented.

---

## Home Page (/)

### Loss Landscape Slice
After training, let users click any connection to "lock" it for perturbation. A weight slider appears; dragging it shows the resulting loss change in real time as a mini sparkline next to that connection. Grounds the abstract loss landscape concept by tying it to a specific weight in the live network.

### Activation Function Side-by-Side Comparator
Replace the current modal popup charts with a live scrubbing widget in the explanation article: pick two activation functions, drag a slider for the pre-activation input value, see both outputs and derivatives update simultaneously. Makes saturation and dead zones tactile.

### Dead Neuron Heatmap
After training with ReLU, render a per-neuron firing rate grid below the network. Each cell's brightness = fraction of training samples that activated the neuron (> 0). Dead neurons glow red. Toggle between activation functions (ReLU vs Leaky ReLU vs ELU) and retrain to see the difference. The visualization *is* the explanation — no annotation needed.

### Weight Distribution Histogram (live)
A small histogram of all network weights, updated each epoch. Shows random initialization → trained distribution. Click a layer label to filter to just that layer's weights. Makes weight explosion and collapse visible before the loss curve shows symptoms.

### MNIST Saliency Map ("What the Network Sees")
After training on MNIST, compute input gradients: how much does each pixel affect the predicted class? Render as a color overlay on the drawn digit. Users draw a digit, see the prediction, then toggle a "What mattered?" overlay to see which pixels drove the decision. A simplified but accurate saliency visualization.

---

## Transformer Page (/transformer)

### Custom Sentence Input (re-enable BERT)
The `@xenova/transformers` inference code is already written but commented out. Re-enable it using DistilBERT (~260 MB, half the size of BERT-base) so users can type any short sentence and see real attention weights — not just the 7 pre-curated examples. The live inference code path already exists in the codebase.

### Head Specialization Radar Chart
For each attention head, compute 5 behavioral metrics from the attention matrix: diagonal dominance (self-attention), forward bias, backward bias, sparsity (how peaked the distribution is), and CLS token focus. Render as a radar/spider chart per head. Instead of just naming heads, users get a visual fingerprint they can compare across examples.

### Positional Encoding Visualizer
Render the sinusoidal position encoding matrix as a heatmap (position × dimension). Two draggable sliders select positions A and B; compute their cosine similarity and highlight both rows. Users discover the periodicity and distance properties by touching the matrix rather than reading the formula.

### QKV "Matching Game" — Intuition Builder
Simplified widget: 5 abstract "query" tokens and 5 "key" tokens shown as colored bar vectors. User drags to adjust their dot-product similarity and watches softmax weights update live. A temperature slider shows how sharpness affects the distribution. The math becomes a consequence of what the user feels, not a prerequisite.

### Attention Rollout / Flow Visualization
Compute attention rollout (Abnar & Zuidema 2020): multiply attention matrices through all 12 layers to show where information actually flows to a selected token. Visualize as a Sankey-like flow diagram. Reveals the actual information path, which is often completely different from any single layer's heatmap — a non-traditional view that goes beyond what the standard visualization shows.

### Encoder vs. Decoder — Side-by-side Inference Trace
Show the same sentence running through an encoder (bidirectional) and a decoder (causal mask) side by side. When the user hovers a token in one panel, the corresponding row highlights in both. The current enc/decoder section is only text + a static mask diagram; this makes the difference immediate.

### Token Probability Distribution at Inference Time
For decoder-style generation, show the full softmax vocabulary distribution as a bar chart for each generation step. Let users "sample" manually — click a bar to select a token, watch the sequence extend, and see the next distribution shift. Makes "the model predicts the next token" concrete and shows stochasticity firsthand.
