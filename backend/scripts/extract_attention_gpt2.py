"""
extract_attention_gpt2.py

GPT-2 variant of extract_attention.py. Key differences vs BERT:

  - No [CLS] / [SEP] tokens — the "attention sink" problem goes away entirely
  - Causal (lower-triangular) attention — each token can only attend to tokens
    that came before it, mirroring how modern autoregressive LLMs (GPT, Claude)
    actually work
  - BPE tokenizer — words may be split into sub-word pieces; displayed with the
    leading-space marker (Ġ) stripped so they read naturally

Setup:
    pip install torch transformers
    python extract_attention_gpt2.py

Output:
    examples_gpt2.json — inspect with analyze_gpt2.py, then drop into
    app/transformer/data.json if the patterns look better than BERT

Layer choice:
    GPT-2 small has 12 layers (0–11). Layers 4–7 tend to show the most
    interpretable content patterns; early layers are mostly positional and
    late layers are task-specific. Default is layer 5.
"""

import json
import torch
from transformers import GPT2Tokenizer, GPT2Model

MODEL_NAME = "gpt2"
LAYER = 5

# Same sentences as the BERT script so we can compare directly.
# The full set is listed here; run analyze_gpt2.py after to decide which to keep.
EXAMPLES = [
    {
        "id": "anaphora",
        "label": "Anaphora resolution",
        "sentence": "The animal didn't cross the street because it was too tired",
    },
    {
        "id": "modifier",
        "label": "PP attachment",
        "sentence": "She saw the man with the telescope",
    },
    {
        "id": "ditransitive",
        "label": "Semantic roles",
        "sentence": "Mary gave John a book",
    },
    {
        "id": "agreement",
        "label": "Subject-verb agreement",
        "sentence": "The cat on the mats is sleeping",
    },
    {
        "id": "reflexive",
        "label": "Reflexive pronoun",
        "sentence": "John accidentally hurt himself while cooking",
    },
    {
        "id": "coordination",
        "label": "Coordination",
        "sentence": "Cats and dogs make wonderful pets",
    },
]


def clean_token(t: str) -> str:
    """Strip GPT-2 BPE artifacts for display (Ġ = leading space, Ċ = newline)."""
    return t.replace("Ġ", " ").replace("Ċ", "\n").strip()


def extract_for_sentence(model, tokenizer, sentence: str, layer: int) -> dict:
    """Return tokens and attention matrices for one sentence."""
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = [clean_token(t) for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])]

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions: tuple of 12 tensors, each (batch=1, num_heads=12, seq, seq)
    # GPT-2 uses causal masking so the upper triangle is always zero.
    attentions = outputs.attentions[layer][0]  # (12, seq, seq)

    main_attention = attentions.mean(dim=0).tolist()

    # Heads 0, 3, 7, 11 — same indices as BERT script for easy comparison
    head_indices = [0, 3, 7, 11]
    multi_head = [attentions[h].tolist() for h in head_indices]

    return {
        "tokens": tokens,
        "attentionMatrix": main_attention,
        "multiHeadAttention": multi_head,
        "headLayer": layer,
        "headIndices": head_indices,
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2Model.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    print(f"Loaded. Extracting attention from layer {LAYER}.\n")

    results = []
    for ex in EXAMPLES:
        print(f"  Processing: {ex['sentence']}")
        weights = extract_for_sentence(model, tokenizer, ex["sentence"], LAYER)
        results.append({
            "id": ex["id"],
            "label": ex["label"],
            "sentence": ex["sentence"],
            **weights,
        })
        print(f"    -> {len(weights['tokens'])} tokens")

    output_path = "examples_gpt2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {output_path}")
    print("Run analyze_gpt2.py to see the top attention pairs, then decide what to keep.")


if __name__ == "__main__":
    main()
