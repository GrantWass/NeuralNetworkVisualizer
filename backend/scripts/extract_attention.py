"""
extract_attention.py

Run this locally (NOT on Lambda) to extract real BERT attention weights
for the pre-computed example sentences. Outputs a JSON file you'll drop
into the frontend as the source of truth.

Setup:
    pip install torch transformers
    python extract_attention.py

Output:
    examples.json — drop this into your Next.js project at
    app/transformer/data.json (or wherever the frontend expects it)

Why this is separate:
    - Loading BERT takes 10+ seconds, way too slow for the actual page
    - We pre-compute weights for the curated examples once
    - The Lambda endpoint handles user-provided sentences on demand
"""

import json
import math
import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"
LAYER = 6
NUM_HEADS = 12
HEAD_DIM = 64  # hidden_dim (768) / num_heads (12)

EXAMPLES = [
    # ── Known-good (kept from v1) ─────────────────────────────────────────
    {
        "id": "anaphora",
        "label": "Anaphora resolution",
        "sentence": "The animal didn't cross the street because it was too tired",
        # "it" → "animal": pronoun resolves to correct antecedent over distractor "street"
    },
    {
        "id": "modifier",
        "label": "PP attachment",
        "sentence": "She saw the man with the telescope",
        # "with" → "man": PP attaches to the noun rather than the verb
    },
    {
        "id": "ditransitive",
        "label": "Semantic roles",
        "sentence": "Mary gave John a book",
        # "gave" → "Mary": verb identifies its agent; "a" splits between "book"/"gave"
    },

    # ── New candidates — inspect heatmaps before keeping ─────────────────
    {
        "id": "winograd",
        "label": "Pronoun resolution",
        "sentence": "The trophy doesn't fit in the suitcase because it is too large",
        # Classic Winograd: "it" should resolve to "trophy" (the large object)
    },
    {
        "id": "agreement",
        "label": "Subject-verb agreement",
        "sentence": "The cat on the mats is sleeping",
        # "is" should attend to "cat" over the distractor plural "mats"
    },
    {
        "id": "negation",
        "label": "Negation",
        "sentence": "The dog did not chase the cat",
        # "not" should pull toward "chase" — the verb it scopes over
    },
    {
        "id": "reflexive",
        "label": "Reflexive pronoun",
        "sentence": "John accidentally hurt himself while cooking",
        # "himself" should collapse onto "John" — short, unambiguous coreference chain
    },
    {
        "id": "coordination",
        "label": "Coordination",
        "sentence": "Cats and dogs make wonderful pets",
        # "and" should bridge both conjuncts; conjuncts should attend to each other
    },
]


def extract_for_sentence(model, tokenizer, sentence: str, layer: int) -> dict:
    """Return tokens, attention weights, and raw Q/K scores for one sentence."""
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Hook the query and key projections inside BertSelfAttention to capture
    # the intermediate tensors before the dot-product and softmax.
    store: dict = {}

    def q_hook(_, __, output):  # noqa: ANN001
        # output: (batch=1, seq_len, hidden=768) — reshape to (heads, seq, head_dim)
        b, s, _ = output.shape
        store["q"] = output.detach().view(b, s, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)[0]

    def k_hook(_, __, output):  # noqa: ANN001
        b, s, _ = output.shape
        store["k"] = output.detach().view(b, s, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)[0]

    attn_self = model.encoder.layer[layer].attention.self
    h1 = attn_self.query.register_forward_hook(q_hook)
    h2 = attn_self.key.register_forward_hook(k_hook)

    with torch.no_grad():
        outputs = model(**inputs)

    h1.remove()
    h2.remove()

    # Softmax attention weights — shape (num_heads, seq_len, seq_len)
    attentions = outputs.attentions[layer][0]
    main_attention = attentions.mean(dim=0).tolist()

    head_indices = [0, 3, 7, 11]
    multi_head = [attentions[h].tolist() for h in head_indices]

    # Raw pre-softmax scores: Q @ K^T / sqrt(head_dim)
    q = store["q"]  # (num_heads, seq_len, head_dim)
    k = store["k"]  # (num_heads, seq_len, head_dim)
    raw = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)  # (num_heads, seq, seq)

    # Average across all 12 heads — parallel to attentionMatrix
    raw_scores_matrix = raw.mean(dim=0).tolist()

    # Per-head raw scores for the 4 displayed heads
    multi_head_raw_scores = [raw[h].tolist() for h in head_indices]

    # Q and K vectors for each of the 4 displayed heads
    # Shape: 4 × seq_len × HEAD_DIM — lets the frontend show real vector patterns
    query_vectors = [q[h].tolist() for h in head_indices]
    key_vectors   = [k[h].tolist() for h in head_indices]

    return {
        "tokens": tokens,
        "attentionMatrix": main_attention,
        "multiHeadAttention": multi_head,
        "headLayer": layer,
        "headIndices": head_indices,
        "rawScoresMatrix": raw_scores_matrix,
        "multiHeadRawScores": multi_head_raw_scores,
        "queryVectors": query_vectors,
        "keyVectors": key_vectors,
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
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
        print(f"    -> {len(weights['tokens'])} tokens, "
              f"{len(weights['multiHeadAttention'])} heads")

    output_path = "examples.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {output_path}")
    print(f"Drop this into your Next.js project (e.g. app/transformer/data.json)")


if __name__ == "__main__":
    main()