"""
analyze_gpt2.py

Prints the top content-word attention pairs for each example in
examples_gpt2.json so you can decide which ones to keep.

Usage:
    python analyze_gpt2.py
"""

import json

data = json.load(open("examples_gpt2.json"))

for ex in data:
    tokens = ex["tokens"]
    matrix = ex["attentionMatrix"]
    n = len(tokens)

    # Collect all off-diagonal pairs
    pairs = [
        (matrix[i][j], tokens[i], tokens[j])
        for i in range(n)
        for j in range(n)
        if i != j and matrix[i][j] > 0  # upper triangle is zero (causal mask)
    ]
    pairs.sort(reverse=True)

    print(f"{'=' * 60}")
    print(f"{ex['id']}  |  {ex['label']}")
    print(f"  sentence : {ex['sentence']}")
    print(f"  tokens   : {tokens}")
    print(f"  top pairs:")
    for w, src, tgt in pairs[:6]:
        print(f"    {src:14s} -> {tgt:14s}  {w:.3f}")
    print()
