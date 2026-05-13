"use client";
import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const COLORS = [
  "#6366f1", "#ec4899", "#10b981", "#f59e0b",
  "#3b82f6", "#ef4444", "#8b5cf6", "#14b8a6",
  "#f97316", "#84cc16", "#06b6d4", "#a855f7",
];

const DEFAULT_WORDS = "king, queen, man, woman, dog, cat";

type Point3D = { word: string; x: number; y: number; z: number; color: string };
type Status = "idle" | "loading-model" | "computing" | "done" | "error";

function parseWords(input: string): string[] {
  return input
    .split(/[\s,]+/)
    .map((w) => w.trim().toLowerCase())
    .filter(Boolean)
    .filter((w, i, arr) => arr.indexOf(w) === i)
    .slice(0, 12);
}

function normalizePoints(raw: number[][]): [number, number, number][] {
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const p of raw) {
    for (let d = 0; d < 3; d++) {
      if (p[d] < mins[d]) mins[d] = p[d];
      if (p[d] > maxs[d]) maxs[d] = p[d];
    }
  }
  return raw.map(
    (p) =>
      p.map((v, d) => {
        const range = maxs[d] - mins[d] || 1;
        return ((v - mins[d]) / range) * 2 - 1;
      }) as [number, number, number]
  );
}

function WordDot({ word, x, y, z, color }: Point3D) {
  const [hovered, setHovered] = useState(false);
  return (
    <group position={[x, y, z]}>
      <mesh
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[hovered ? 0.075 : 0.055, 24, 24]} />
        <meshStandardMaterial color={color} roughness={0.3} metalness={0.1} />
      </mesh>
      <Html distanceFactor={8} center>
        <div
          style={{
            background: "rgba(0,0,0,0.72)",
            color: "#fff",
            padding: "2px 7px",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: "var(--font-geist-mono, monospace)",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            marginTop: 22,
            border: `1px solid ${color}`,
            letterSpacing: "0.02em",
          }}
        >
          {word}
        </div>
      </Html>
    </group>
  );
}

function Scene({ points }: { points: Point3D[] }) {
  const spread = useMemo(() => {
    if (points.length === 0) return 2;
    let max = 0;
    for (const p of points) {
      const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (d > max) max = d;
    }
    return Math.max(max, 0.5);
  }, [points]);

  return (
    <>
      <ambientLight intensity={0.7} />
      <pointLight position={[4, 4, 4]} intensity={1.2} />
      <pointLight position={[-4, -4, -4]} intensity={0.4} />
      <Line points={[[-spread * 1.4, 0, 0], [spread * 1.4, 0, 0]]} color="black" lineWidth={1} />
      <Line points={[[0, -spread * 1.4, 0], [0, spread * 1.4, 0]]} color="black" lineWidth={1} />
      <Line points={[[0, 0, -spread * 1.4], [0, 0, spread * 1.4]]} color="black" lineWidth={1} />
      {points.map((p) => (
        <WordDot key={p.word} {...p} />
      ))}
      <OrbitControls
        autoRotate
        autoRotateSpeed={0.6}
        enablePan={false}
        minDistance={spread * 0.3}
        maxDistance={spread * 4}
      />
    </>
  );
}

export function EmbeddingExplorer() {
  const [inputValue, setInputValue] = useState(DEFAULT_WORDS);
  const [points, setPoints] = useState<Point3D[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [visualizedWords, setVisualizedWords] = useState<string[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const extractorRef = useRef<any>(null);

  const runCompute = useCallback(async (words: string[]) => {
    if (words.length < 2) {
      setError("Enter at least 2 words.");
      return;
    }

    setError(null);

    try {
      if (!extractorRef.current) {
        setStatus("loading-model");
        setModelProgress(0);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const xf = (await import("@xenova/transformers")) as any;
        const { pipeline, env } = xf;
        env.allowLocalModels = false;
        extractorRef.current = await pipeline(
          "feature-extraction",
          "Xenova/all-MiniLM-L6-v2",
          {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            progress_callback: (p: any) => {
              if (p.status === "progress" && p.total) {
                setModelProgress(Math.round((p.loaded / p.total) * 100));
              }
            },
          }
        );
      }

      setStatus("computing");

      // Process each word individually — batching changes output tensor shape
      const embeddings: number[][] = [];
      for (const word of words) {
        const out = await extractorRef.current(word, {
          pooling: "mean",
          normalize: true,
        });
        embeddings.push(Array.from(out.data) as number[]);
      }

      const { UMAP } = await import("umap-js");
      const nNeighbors = Math.max(2, Math.min(words.length - 1, 5));
      const umap = new UMAP({ nComponents: 3, nNeighbors, minDist: 0.2, spread: 1.5 });
      const raw: number[][] = umap.fit(embeddings);
      const normalized = normalizePoints(raw);

      setPoints(
        words.map((word, i) => ({
          word,
          x: normalized[i][0],
          y: normalized[i][1],
          z: normalized[i][2],
          color: COLORS[i % COLORS.length],
        }))
      );
      setVisualizedWords(words);
      setStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setStatus("error");
    }
  }, []);

  const compute = useCallback(() => {
    runCompute(parseWords(inputValue));
  }, [inputValue, runCompute]);

  useEffect(() => {
    runCompute(parseWords(DEFAULT_WORDS));
  }, [runCompute]);

  const buttonLabel =
    status === "loading-model"
      ? `Loading ${modelProgress}%`
      : status === "computing"
      ? "Computing…"
      : "Visualize";

  const busy = status === "loading-model" || status === "computing";

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Word Embeddings in 3D</h2>
      <p className="text-muted-foreground leading-relaxed max-w-2xl">
        Before attention runs, each token is converted into a dense vector
        called an <strong>embedding</strong> — a point in high-dimensional
        space where meaning is encoded as geometry. Words with similar meanings
        land close together; unrelated words are far apart. Enter words below
        to see their embeddings projected into 3D using UMAP.
      </p>

      <div className="flex gap-2 max-w-lg">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => !busy && e.key === "Enter" && compute()}
          placeholder="king, queen, man, woman, dog, cat"
          className="flex-1 font-mono text-sm"
        />
        <Button onClick={compute} disabled={busy}>
          {buttonLabel}
        </Button>
      </div>

      {status === "loading-model" && (
        <div className="space-y-1 max-w-lg">
          <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-300"
              style={{ width: `${modelProgress}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Loading model… {modelProgress}%
          </p>
        </div>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}

      {points.length > 0 && (
        <div className="space-y-2">
          <div
            className="rounded-lg border border-border bg-card overflow-hidden"
            style={{ height: 420 }}
          >
            <Canvas camera={{ position: [0, 0, 2.8], fov: 55 }}>
              <Scene points={points} />
            </Canvas>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-y-1 px-1">
            <p className="text-xs text-muted-foreground">
              Showing:{" "}
              {visualizedWords.map((w, i) => (
                <span key={w}>
                  <span style={{ color: COLORS[i % COLORS.length] }}>{w}</span>
                  {i < visualizedWords.length - 1 && (
                    <span className="text-muted-foreground/50"> · </span>
                  )}
                </span>
              ))}
            </p>
            <p className="text-xs text-muted-foreground">
              Drag to rotate · scroll to zoom
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
