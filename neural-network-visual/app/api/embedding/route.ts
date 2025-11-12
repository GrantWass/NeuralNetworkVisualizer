import { NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// Simple in-memory cache
const embeddingCache = new Map<string, number[]>();

export async function POST(req: Request) {
  const { tokens }: { tokens: string[] } = await req.json();

  const vectors: number[][] = [];

  for (const token of tokens) {
    if (embeddingCache.has(token)) {
      // Reuse cached embedding
      vectors.push(embeddingCache.get(token)!);
    } else {
      // Fetch new embedding
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
