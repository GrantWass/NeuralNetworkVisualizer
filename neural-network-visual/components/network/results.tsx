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
    yMean?: number | null;
    yStd?: number | null;
    isPredicted?: boolean;
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
    yMean,
    yStd,
    isPredicted,
}: RenderResultsParams) => {
    if (!dataset) return null;

    const x = cx + SVGWIDTH * 0.11;

    // Tightened vertical offsets — label and value sit close together as a pair
    const labelY  = cy - SVGHEIGHT * 0.02;   // just above node center
    const valueY  = cy + SVGHEIGHT * 0.018;  // just below node center
    const thirdY  = cy + SVGHEIGHT * 0.052;  // auxiliary line (actual / correct)

    if (dataset === "auto_mpg") {
        const displayValue = yMean != null && yStd != null ? activationValue * yStd + yMean : activationValue;
        const actual = formatActual(original || [], dataset);
        return (
            <>
                <text x={x} y={cy - SVGHEIGHT * 0.045} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                    {outputMap[dataset]?.[ni]}
                </text>
                <text x={x} y={cy - SVGHEIGHT * 0.005} fontSize={fontSize + 1} fontWeight="bold" textAnchor="middle" fill="#111827">
                    {formatValue(displayValue, dataset)}
                </text>
                <text x={x} y={cy + SVGHEIGHT * 0.045} fontSize={fontSize - 1} textAnchor="middle" fill="#9ca3af">
                    actual {actual}
                </text>
            </>
        );
    }

    if (dataset === "iris" || dataset === "mnist") {
        const label = outputMap[dataset]?.[ni];
        const orig = original || [];
        let actualIdx = -1;
        if (dataset === "iris") {
            const tail = orig.slice(-3);
            actualIdx = tail.findIndex(v => v === 1);
        } else {
            const labels = orig.slice(784);
            actualIdx = labels.length >= 10 ? labels.indexOf(Math.max(...labels)) : -1;
        }
        const isActual = actualIdx === ni;
        const isCorrect = isPredicted && isActual;
        const isWrongPred = isPredicted && !isActual;

        let labelFill = "#9ca3af";
        let labelWeight: "bold" | "normal" = "normal";
        if (isPredicted) { labelFill = isCorrect ? "#16a34a" : "#dc2626"; labelWeight = "bold"; }
        else if (isActual) { labelFill = "#16a34a"; labelWeight = "bold"; }

        let valueFill = "#6b7280";
        if (isPredicted) valueFill = isCorrect ? "#16a34a" : "#dc2626";

        return (
            <>
                <text x={x} y={labelY} fontSize={fontSize} textAnchor="middle">
                    <tspan fontWeight={labelWeight} fill={labelFill}>{label}</tspan>
                    {isPredicted && <tspan fill={isCorrect ? "#16a34a" : "#dc2626"} fontWeight="bold">{isCorrect ? " ▶ ✓" : " ▶"}</tspan>}
                    {!isPredicted && isActual && <tspan fill="#16a34a" fontWeight="bold"> ✓</tspan>}
                    {isWrongPred && <tspan fill="#9ca3af" fontSize={fontSize - 1}> </tspan>}
                </text>
                <text x={x} y={valueY} fontSize={fontSize} fontWeight={isPredicted ? "bold" : "normal"} textAnchor="middle" fill={valueFill}>
                    {formatValue(activationValue, dataset)}
                </text>
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
                <text x={x} y={cy - SVGHEIGHT * 0.04} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                    XOR output
                </text>
                <text x={x} y={valueY} fontSize={fontSize + 1} fontWeight="bold" textAnchor="middle" fill="#111827">
                    {predicted} ({(confidence * 100).toFixed(0)}%)
                </text>
                {actual !== null && (
                    <text x={x} y={thirdY} fontSize={fontSize - 1} textAnchor="middle" fill={isCorrect ? "#16a34a" : "#dc2626"} fontWeight="bold">
                        {isCorrect ? `✓ correct (${Math.round(actual)})` : `✗ actual: ${Math.round(actual)}`}
                    </text>
                )}
            </>
        );
    }

    return (
        <>
            <text x={x} y={labelY} fontSize={fontSize} fontWeight="bold" textAnchor="middle" fill="#374151">
                {outputMap[dataset]?.[ni]}
            </text>
            <text x={x} y={valueY} fontSize={fontSize} textAnchor="middle" fill="#111827">
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
