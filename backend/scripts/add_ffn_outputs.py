"""
add_ffn_outputs.py

Adds FFN intermediate activations as named outputs to the existing
gpt2-viz-q8.onnx model. No re-training or re-export needed — we just
expose internal graph tensors via Identity nodes.

New outputs per layer (l0..l11):
  l{N}.ffn_in     [1, seq, 768]  — LayerNorm'd residual entering the FFN
  l{N}.ffn_hidden [1, seq, 3072] — post-GELU hidden activations
  l{N}.ffn_out    [1, seq, 768]  — FFN output (before residual add-back)

Usage:
    python3.13 add_ffn_outputs.py
"""

import onnx
from onnx import helper, TensorProto

MODEL_IN  = "/Users/grantwasserman/Documents/GitHub/NeuralNetworkVisualizer/neural-network-visual/public/models/gpt2-viz-q8.onnx"
MODEL_OUT = MODEL_IN  # overwrite in-place (model is in git so it's recoverable)

# Tensor names identified by tracing the graph (see inspect_onnx.py for derivation)
LAYER_TENSORS = [
    (0,  "layer_norm_1",  "mul_222",  "view_11"),
    (1,  "layer_norm_3",  "mul_373",  "view_22"),
    (2,  "layer_norm_5",  "mul_524",  "view_33"),
    (3,  "layer_norm_7",  "mul_675",  "view_44"),
    (4,  "layer_norm_9",  "mul_826",  "view_55"),
    (5,  "layer_norm_11", "mul_977",  "view_66"),
    (6,  "layer_norm_13", "mul_1128", "view_77"),
    (7,  "layer_norm_15", "mul_1279", "view_88"),
    (8,  "layer_norm_17", "mul_1430", "view_99"),
    (9,  "layer_norm_19", "mul_1581", "view_110"),
    (10, "layer_norm_21", "mul_1732", "view_121"),
    (11, "layer_norm_23", "mul_1883", "view_132"),
]

def main():
    print(f"Loading {MODEL_IN} ...")
    model = onnx.load(MODEL_IN)
    graph = model.graph

    existing_outputs = {o.name for o in graph.output}
    print(f"Existing outputs: {len(existing_outputs)}")

    new_nodes   = []
    new_outputs = []

    for layer_idx, ffn_in_t, ffn_hidden_t, ffn_out_t in LAYER_TENSORS:
        for suffix, internal_name in [
            ("ffn_in",     ffn_in_t),
            ("ffn_hidden", ffn_hidden_t),
            ("ffn_out",    ffn_out_t),
        ]:
            out_name = f"l{layer_idx}.{suffix}"
            if out_name in existing_outputs:
                print(f"  skipping {out_name} (already exists)")
                continue

            # Identity node exposes the internal tensor as a named output
            identity = helper.make_node(
                "Identity",
                inputs=[internal_name],
                outputs=[out_name],
                name=f"expose_{out_name}",
            )
            new_nodes.append(identity)

            # Output declaration (float32, dynamic shape)
            vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            new_outputs.append(vi)

    graph.node.extend(new_nodes)
    graph.output.extend(new_outputs)

    print(f"Added {len(new_nodes)} Identity nodes and {len(new_outputs)} new outputs")

    # Shape inference populates type/shape on all intermediate tensors,
    # which the checker requires for outputs we added.
    print("Running shape inference ...")
    model = onnx.shape_inference.infer_shapes(model)
    print("Shape inference done.")

    print(f"Saving to {MODEL_OUT} ...")
    onnx.save(model, MODEL_OUT)
    print("Done.")

if __name__ == "__main__":
    main()
