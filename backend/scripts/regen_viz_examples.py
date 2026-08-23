"""
regen_viz_examples.py

Re-runs the GPT-2 viz examples through the updated ONNX model (which now
exports ffn_in / ffn_hidden / ffn_out per layer) and writes one JSON file
per example plus a manifest, into neural-network-visual/public/data/gpt2-examples/.

Usage:
    python3.13 regen_viz_examples.py
"""

import json
import os

import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer

MODEL_PATH  = "/Users/grantwasserman/Documents/GitHub/NeuralNetworkVisualizer/neural-network-visual/public/models/gpt2-viz-q8.onnx"
OUT_DIR     = "/Users/grantwasserman/Documents/GitHub/NeuralNetworkVisualizer/neural-network-visual/public/data/gpt2-examples"

N_LAYERS  = 12
N_HEADS   = 12
HEAD_DIM  = 64
EMBED_DIM = 768
FFN_DIM   = 3072
MAX_TOKENS = 8

EXAMPLES = [
    {"id": "shakespeare", "label": "to be or not to",          "sentence": "to be or not to"},
    {"id": "starwars",    "label": '"May the force be with"',   "sentence": "May the force be with"},
    {"id": "nursery",     "label": '"Jack and Jill went up the"',"sentence": "Jack and Jill went up the"},
    {"id": "fairytale",   "label": '"once upon a"',             "sentence": "once upon a"},
]

def r4(x): return round(float(x), 4)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

print("Loading tokenizer…")
tok = GPT2Tokenizer.from_pretrained("gpt2")

print("Loading ONNX model…")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
output_names = [o.name for o in sess.get_outputs()]
print(f"  {len(output_names)} outputs available")
has_ffn = any("ffn" in n for n in output_names)
print(f"  FFN outputs present: {has_ffn}")

results = []

for ex in EXAMPLES:
    print(f"\nProcessing: {ex['sentence']}")
    ids   = tok(ex["sentence"])["input_ids"][:MAX_TOKENS]
    n     = len(ids)
    inp   = np.array([ids], dtype=np.int64)

    out   = dict(zip(output_names, sess.run(output_names, {"input_ids": inp})))

    tokens = [tok.decode([i]).strip() or f"[{i}]" for i in ids]
    print(f"  tokens ({n}): {tokens}")

    # ── next-word probs ──────────────────────────────────────────────────────
    logits    = out["logits"][0, n - 1]          # (vocab,)
    probs     = softmax(logits)
    top_ids   = np.argsort(probs)[::-1][:20]
    nwp       = [{"token": tok.decode([int(i)]).strip() or f"[{i}]",
                  "prob":  r4(probs[i])} for i in top_ids]

    # ── per-layer ────────────────────────────────────────────────────────────
    layers = []
    for li in range(N_LAYERS):
        q    = out[f"l{li}.q"][0]      # (12, n, 64)
        k    = out[f"l{li}.k"][0]
        v    = out[f"l{li}.v"][0]
        attn = out[f"l{li}.attn"][0]   # (12, n, n)

        mha  = [[[r4(attn[h, r, c]) for c in range(n)] for r in range(n)] for h in range(N_HEADS)]
        avg  = [[r4(sum(attn[h,r,c] for h in range(N_HEADS))/N_HEADS) for c in range(n)] for r in range(n)]
        qv   = [[[r4(q[h,s,d]) for d in range(HEAD_DIM)] for s in range(n)] for h in range(N_HEADS)]
        kv   = [[[r4(k[h,s,d]) for d in range(HEAD_DIM)] for s in range(n)] for h in range(N_HEADS)]
        vv   = [[[r4(v[h,s,d]) for d in range(HEAD_DIM)] for s in range(n)] for h in range(N_HEADS)]

        layer = {
            "layerIdx":          li,
            "multiHeadAttention": mha,
            "attentionMatrix":    avg,
            "queryVectors":       qv,
            "keyVectors":         kv,
            "valueVectors":       vv,
        }

        if has_ffn:
            fi  = out[f"l{li}.ffn_in"]      # shape varies — flatten to (n, 768)
            fh  = out[f"l{li}.ffn_hidden"]
            fo  = out[f"l{li}.ffn_out"]
            # squeeze leading batch dim if present
            fi  = fi.reshape(n, EMBED_DIM)
            fh  = fh.reshape(n, FFN_DIM)
            fo  = fo.reshape(n, EMBED_DIM)
            layer["ffnIn"]     = [[r4(fi[s,d])  for d in range(EMBED_DIM)] for s in range(n)]
            layer["ffnHidden"] = [[r4(fh[s,d])  for d in range(FFN_DIM)]   for s in range(n)]
            layer["ffnOut"]    = [[r4(fo[s,d])  for d in range(EMBED_DIM)] for s in range(n)]

        layers.append(layer)

    results.append({
        "id":           ex["id"],
        "label":        ex["label"],
        "tokens":       tokens,
        "layers":       layers,
        "nextWordProbs": nwp,
    })
    print(f"  done. top token: {nwp[0]['token']!r} ({nwp[0]['prob']:.3f})")

os.makedirs(OUT_DIR, exist_ok=True)
manifest = []
for ex in results:
    fname = f"{ex['id']}.json"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        json.dump(ex, f, separators=(",", ":"))
    manifest.append({"id": ex["id"], "label": ex["label"], "file": f"/data/gpt2-examples/{fname}"})
    print(f"  wrote {fname} ({os.path.getsize(path) / 1e6:.1f} MB)")

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=1)
print(f"Wrote manifest ({len(manifest)} examples) → {manifest_path}")
