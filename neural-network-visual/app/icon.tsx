import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 6,
          background: "#6366f1",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="20" height="20" viewBox="0 0 14 14" fill="none">
          <circle cx="3" cy="7" r="2" fill="white" />
          <circle cx="11" cy="4" r="1.5" fill="white" fill-opacity="0.7" />
          <circle cx="11" cy="10" r="1.5" fill="white" fill-opacity="0.7" />
          <line x1="5" y1="6.5" x2="9.5" y2="4.5" stroke="white" stroke-width="1" stroke-opacity="0.6" />
          <line x1="5" y1="7.5" x2="9.5" y2="9.5" stroke="white" stroke-width="1" stroke-opacity="0.6" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
