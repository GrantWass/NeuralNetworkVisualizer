import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Neural Network Visual — Interactive ML Visualizations";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "72px 80px",
          background: "#0f0f13",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Accent bar */}
        <div
          style={{
            width: 64,
            height: 4,
            background: "#6366f1",
            borderRadius: 2,
            marginBottom: 32,
          }}
        />
        <div
          style={{
            fontSize: 56,
            fontWeight: 700,
            color: "#ffffff",
            lineHeight: 1.15,
            marginBottom: 24,
            letterSpacing: "-0.5px",
          }}
        >
          Neural Network Visual
        </div>
        <div
          style={{
            fontSize: 28,
            color: "#a1a1aa",
            lineHeight: 1.5,
            maxWidth: 800,
          }}
        >
          Interactive visualizations for neural networks and transformer
          attention — free, hands-on, and built for students.
        </div>
        <div
          style={{
            marginTop: "auto",
            fontSize: 20,
            color: "#6366f1",
            fontWeight: 500,
          }}
        >
          nn-visual.com
        </div>
      </div>
    ),
    { ...size }
  );
}
