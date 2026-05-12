import { NextResponse } from "next/server";
import OpenAI from "openai";

// Simple in-memory cache
const embeddingCache = new Map<string, number[]>();

export async function POST(req: Request) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "OPENAI_API_KEY not configured" }, { status: 503 });
  }

  const openai = new OpenAI({ apiKey });
  const { tokens }: { tokens: string[] } = await req.json();

  const vectors: number[][] = [];

  for (const token of tokens) {
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

  return NextResponse.json({ vectors });
}
