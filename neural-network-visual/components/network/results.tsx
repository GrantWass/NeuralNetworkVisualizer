import React from "react";
import { formatActual, formatValue, outputMap } from "@/components/network/lib/utils";

type RenderResultsParams = {
    SVGWIDTH: number;
    SVGHEIGHT: number;
    cx: number;
    cy: number;
    dataset?: string;
    ni: number; // node index
    activationValue: number;
    original?: number[];
    fontSize: number;
};

export const renderResults = ({
    SVGWIDTH,
    SVGHEIGHT,
    cx,
    cy,
    dataset,
    ni,
    activationValue,
    original,
    fontSize,
}: RenderResultsParams) => {
    if (!dataset) return null;

    const x = cx + SVGWIDTH * 0.11;

    if (dataset === "auto_mpg") {
        const actual = formatActual(original || [], dataset);
        return (
            <>
                <text x={x} y={cy - SVGHEIGHT * 0.07} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                    {outputMap[dataset]?.[ni]}
                </text>
                <text x={x} y={cy - SVGHEIGHT * 0.01} fontSize={fontSize + 1} fontWeight="bold" textAnchor="middle" fill="#111827">
                    {formatValue(activationValue, dataset)}
                </text>
                <text x={x} y={cy + SVGHEIGHT * 0.07} fontSize={fontSize - 1} textAnchor="middle" fill="#9ca3af">
                    actual {actual}
                </text>
            </>
        );
    }

    if (dataset === "iris") {
        const label = outputMap[dataset]?.[ni];
        const isActual = formatActual(original || [], dataset) === label;
        return (
            <>
                <text x={x} y={cy - SVGHEIGHT * 0.04} fontSize={fontSize} fontWeight={isActual ? "bold" : "normal"} textAnchor="middle" fill={isActual ? "#111827" : "#6b7280"}>
                    {label}
                </text>
                <text x={x} y={cy + SVGHEIGHT * 0.03} fontSize={fontSize} textAnchor="middle" fill="#374151">
                    {formatValue(activationValue, dataset)}
                </text>
                {isActual && (
                    <text x={x} y={cy + SVGHEIGHT * 0.09} fontSize={fontSize - 1} textAnchor="middle" fill="#16a34a" fontWeight="bold">
                        ✓ actual
                    </text>
                )}
            </>
        );
    }

    if (dataset === "xor") {
        const predicted = activationValue >= 0.5 ? 1 : 0;
        const confidence = predicted === 1 ? activationValue : 1 - activationValue;
        const actual = original && original.length > 0 ? original[original.length - 1] : null;
        const isCorrect = actual !== null && predicted === Math.round(actual);
        return (
            <>
                <text x={x} y={cy - SVGHEIGHT * 0.05} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                    XOR output
                </text>
                <text x={x} y={cy + SVGHEIGHT * 0.02} fontSize={fontSize + 1} fontWeight="bold" textAnchor="middle" fill="#111827">
                    {predicted} ({(confidence * 100).toFixed(0)}%)
                </text>
                {actual !== null && (
                    <text x={x} y={cy + SVGHEIGHT * 0.09} fontSize={fontSize - 1} textAnchor="middle" fill={isCorrect ? "#16a34a" : "#dc2626"} fontWeight="bold">
                        {isCorrect ? `✓ correct (${Math.round(actual)})` : `✗ actual: ${Math.round(actual)}`}
                    </text>
                )}
            </>
        );
    }

    return (
        <>
            <text x={x} y={cy - SVGHEIGHT * 0.04} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                {outputMap[dataset]?.[ni]}
            </text>
            <text x={x} y={cy + SVGHEIGHT * 0.03} fontSize={fontSize} textAnchor="middle" fill="#111827">
                {formatValue(activationValue, dataset)}
            </text>
        </>
    );
};

export const renderInputValues = (cx: number, cy: number, original: number[], nodeIndex: number, activationValue: number, fontSize: number, INPUTLABELOFFSET: number, SVGWIDTH: number, SVGHEIGHT: number) => (
  <text
    x={cx - INPUTLABELOFFSET}
    y={cy + SVGHEIGHT * 0.04}
    textAnchor="middle"
    fontSize={fontSize}
  >
    <tspan fill="#374151">{original[nodeIndex].toFixed(1)}</tspan>
    <tspan fill="#9ca3af"> ({activationValue.toFixed(2)})</tspan>
  </text>
);