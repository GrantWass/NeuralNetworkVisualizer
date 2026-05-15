"""
extract_enc_dec_trace.py

Extracts real attention weights and next-token distributions for two new
interactive sections on the transformer page:

  Section 6 — Encoder vs. Decoder side-by-side:
    BERT (encoder, bidirectional) and GPT-2 (decoder, causal) are both run
    on the same sentence so the two attention matrices can be compared directly.

  Section 7 — Token-by-token generation:
    GPT-2 is greedy-decoded for GEN_STEPS steps from GENERATION_PROMPT.
    At each step the full softmax distribution is captured; only the top
    TOP_K tokens are kept for the frontend bar chart.

Output:
    enc_dec_trace.json — drop into neural-network-visual/app/transformer/

Setup:
    pip install torch transformers
    python extract_enc_dec_trace.py
"""

import json
import torch
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,
)

BERT_LAYER = 6
# Layer 11 head 8 shows clear causal dependencies (sat→cat, on→sat, mat→sat)
# with no attention-sink dominance. The 12-head average is swamped by sinks.
GPT2_LAYER = 11
GPT2_HEAD = 8
TRACE_SENTENCE = "The cat sat on the mat"
GENERATION_PROMPT = "The cat"
GEN_STEPS = 5
TOP_K = 12


def clean_gpt2_token(t: str) -> str:
    return t.replace("Ġ", " ").replace("Ċ", "\n").strip()


# ── Encoder (BERT) ────────────────────────────────────────────────────────────

def extract_bert_attention(sentence: str, layer: int):
    print(f"  Loading bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # Average over all 12 heads at the chosen layer
    attentions = outputs.attentions[layer][0]   # (12, seq, seq)
    avg = attentions.mean(dim=0).tolist()

    print(f"  -> {len(tokens)} tokens: {tokens}")
    return {"tokens": tokens, "attentionMatrix": avg}


# ── Decoder (GPT-2) ───────────────────────────────────────────────────────────

def extract_gpt2_attention(sentence: str, layer: int, head: int):
    print(f"  Loading gpt2 (attention)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [clean_gpt2_token(t) for t in raw_tokens]

    with torch.no_grad():
        outputs = model(**inputs)

    # Use a single head rather than the average — the 12-head average is
    # dominated by attention-sink heads that route ~80% of every row to
    # position 0, obscuring the causal patterns we want to show.
    attn = outputs.attentions[layer][0][head].tolist()   # (seq, seq)

    print(f"  -> {len(tokens)} tokens: {tokens}  (layer={layer} head={head})")
    return {
        "tokens": tokens,
        "attentionMatrix": attn,
        "headLayer": layer,
        "headIndex": head,
    }


# ── Generation steps (GPT-2 LM) ──────────────────────────────────────────────

def extract_generation_steps(prompt: str, steps: int, top_k: int):
    print(f"  Loading gpt2 (LM head)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    results = []

    for step in range(steps):
        context_str = tokenizer.decode(input_ids[0])

        with torch.no_grad():
            out = model(input_ids)

        logits = out.logits[0, -1, :]           # (vocab_size,)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k)

        dist = [
            {
                "token": clean_gpt2_token(tokenizer.decode([idx.item()])),
                "prob": round(prob.item(), 6),
            }
            for idx, prob in zip(top_ids, top_probs)
        ]

        greedy_id = top_ids[0].unsqueeze(0).unsqueeze(0)
        greedy_token = dist[0]["token"]
        input_ids = torch.cat([input_ids, greedy_id], dim=1)

        results.append({
            "step": step,
            "context": context_str,
            "distribution": dist,
            "greedyToken": greedy_token,
        })

        top3 = [(d["token"], f"{d['prob']*100:.1f}%") for d in dist[:3]]
        print(f"  step {step}: context={context_str!r}  top-3={top3}  greedy={greedy_token!r}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Extracting BERT encoder attention: {TRACE_SENTENCE!r}")
    encoder = extract_bert_attention(TRACE_SENTENCE, BERT_LAYER)

    print(f"\nExtracting GPT-2 decoder attention: {TRACE_SENTENCE!r}")
    decoder = extract_gpt2_attention(TRACE_SENTENCE, GPT2_LAYER, GPT2_HEAD)

    print(f"\nExtracting GPT-2 generation steps from: {GENERATION_PROMPT!r}")
    gen_steps = extract_generation_steps(GENERATION_PROMPT, GEN_STEPS, TOP_K)

    output = {
        "traceSentence": TRACE_SENTENCE,
        "encoder": encoder,
        "decoder": decoder,
        "generationPrompt": GENERATION_PROMPT,
        "generationSteps": gen_steps,
    }

    out_path = "enc_dec_trace.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {out_path}")
    print("Copy to: neural-network-visual/app/transformer/enc_dec_trace.json")


if __name__ == "__main__":
    main()
