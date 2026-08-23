import { NextResponse } from "next/server";
import OpenAI from "openai";

const MAX_TOKENS = 64;
const MAX_TOKEN_LENGTH = 128;

// Simple in-memory cache
const embeddingCache = new Map<string, number[]>();

export async function POST(req: Request) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "OPENAI_API_KEY not configured" }, { status: 503 });
  }

  let tokens: unknown;
  try {
    const body = await req.json();
    tokens = body?.tokens;
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON with a `tokens` array." }, { status: 400 });
  }

  if (
    !Array.isArray(tokens) ||
    tokens.length === 0 ||
    tokens.length > MAX_TOKENS ||
    !tokens.every((t) => typeof t === "string" && t.length > 0 && t.length <= MAX_TOKEN_LENGTH)
  ) {
    return NextResponse.json(
      { error: `tokens must be a non-empty array of at most ${MAX_TOKENS} strings (each ≤ ${MAX_TOKEN_LENGTH} chars).` },
      { status: 400 },
    );
  }

  const openai = new OpenAI({ apiKey });
  const vectors: number[][] = [];

  try {
    for (const token of tokens as string[]) {
      if (embeddingCache.has(token)) {
        vectors.push(embeddingCache.get(token)!);
      } else {
        const res = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: token,
          dimensions: 3,
        });
        const embedding = res.data[0].embedding;
        embeddingCache.set(token, embedding);
        vectors.push(embedding);
      }
    }
  } catch (e) {
    console.error("Embedding request failed:", e);
    return NextResponse.json({ error: "Failed to compute embeddings. Please try again." }, { status: 502 });
  }

  return NextResponse.json({ vectors });
}
