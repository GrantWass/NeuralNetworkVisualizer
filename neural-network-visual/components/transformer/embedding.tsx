"use client";
import React, { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

export const DatasetInfo: React.FC = () => {
  const markdownContent = `
# Dataset Overview

This visualization uses a set of **word tokens** embedded into a 3D space using the OpenAI [**text-embedding-3-small**](https://platform.openai.com/docs/guides/embeddings) model. Each token is converted into a vector representing its semantic meaning.

- Embeddings capture **semantic similarity**: tokens with similar meanings appear closer together.
- Traditionally, embeddings are high-dimensional (e.g., 1536 dimensions for this model).
- For visualization, we project these embeddings into 3D space.
- Typically, techniques like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) or [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) are used for dimensionality reduction.
- However, OpenAI's embedding API allows us to directly request 3D embeddings for simplicity.

These embeddings are a key component in **Transformers** and large language models (LLMs), where they serve as the initial vector representation of tokens that the model uses to understand and generate language.

---

Feel free to hover, pan, and click on the tokens to explore relationships interactively.

`;

  return (
    <ReactMarkdown
      rehypePlugins={[rehypeRaw]}
      components={{
        a: ({ ...props }) => (
          <a
            {...props}
            className="text-black-500 underline font-medium hover:text-blue-600"
            target="_blank"
            rel="noopener noreferrer"
          />
          
        ),
        ul: ({ ...props }) => <ul {...props} className="list-disc pl-5 mt-2 mb-2" />,
        li: ({ ...props }) => <li {...props} className="mb-1 text-sm" />,
        h1: ({ ...props }) => <h1 {...props} className="text-xl font-bold mt-4 mb-2" />,
        h2: ({ ...props }) => <h2 {...props} className="text-lg font-semibold mt-3 mb-2" />,
        h3: ({ ...props }) => <h3 {...props} className="text-md font-medium mt-2 mb-1" />,
        p: ({ ...props }) => <p {...props} className="text-gray-700 mb-2 text-sm" />,
        strong: ({ ...props }) => <strong {...props} className="font-semibold" />,
      }}
    >
      {markdownContent}
    </ReactMarkdown>
  );
};


export const EmbeddingsSection: React.FC<{ tokens: string[] }> = ({ tokens }) => {
  const [coords, setCoords] = useState<number[][]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const fetchEmbeddings = async () => {
      const res = await fetch("/api/embedding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokens }),
      });
      const data = await res.json();
      setCoords(data.vectors);
    };
    fetchEmbeddings();
  }, [tokens]);

  if (!coords.length)
    return <div className="p-4 border rounded-md">Calculating embeddings...</div>;

  const scale = 10;

  return (
    <section className="p-4 border rounded-md">
      <div className="flex flex-col lg:flex-row gap-0">
        {/* Left side: plot + title */}
        <div className="flex-1 flex flex-col md:flex-none lg:w-[500px]">
          <h2 className="text-xl font-semibold mb-2">Word Embeddings (3D)</h2>
          <Canvas
            style={{ width: "100%", aspectRatio: "1/1" }}
            camera={{ position: [12, 8, 25], fov: 60 }}
            className="hover:cursor-pointer"
          >
            <ambientLight intensity={0.5} />
            <OrbitControls enablePan enableZoom enableRotate />

            {/* X axis - red */}
            <lineSegments>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([-1 * scale, 0, 0, 1 * scale, 0, 0]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="red" />
            </lineSegments>

            {/* Y axis - green */}
            <lineSegments>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([0, -1 * scale, 0, 0, 1 * scale, 0]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="green" />
            </lineSegments>

            {/* Z axis - blue */}
            <lineSegments>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([0, 0, -1 * scale, 0, 0, 1 * scale]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="blue" />
            </lineSegments>

            {/* Points */}
            {coords.map(([x, y, z], i) => {
              const isSel = selected === tokens[i];
              return (
                <group key={i} position={[x * scale, y * scale, z * scale]}>
                  <mesh onClick={() => setSelected(tokens[i])}>
                    <sphereGeometry args={[isSel ? 0.8 : 0.5, 16, 16]} />
                    <meshStandardMaterial color={isSel ? "#f43f5e" : "#3b82f6"} />
                  </mesh>
                  <Text
                    position={[0.5, 0.5, 0.5]}
                    fontSize={0.5}
                    color="black"
                    anchorX="left"
                    anchorY="bottom"
                  >
                    {tokens[i]}
                  </Text>
                </group>
              );
            })}
          </Canvas>
        </div>

        {/* Right side: description + selected token info */}
        <div className="lg:flex-1 lg:w-[300px] bg-gray-50 p-4 rounded-md flex flex-col gap-4">
          <DatasetInfo/>

          <div>
            <h3 className="font-medium mb-1">Selected Token:</h3>
            <p>{selected ?? "None"}</p>
            {selected && (
              <p className="text-sm text-slate-500">
                3D coordinates:{" "}
                {coords[tokens.indexOf(selected)]
                  ?.map((v) => v.toFixed(3))
                  .join(", ") ?? "N/A"}
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};
