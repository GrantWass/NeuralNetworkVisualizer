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

    return (
        <>
            <text
                x={cx + SVGWIDTH * 0.11}
                y={cy + (dataset === "iris" ? 0 : -SVGHEIGHT * 0.1)}
                fontSize={fontSize}
                textAnchor="middle"
                fontWeight="bold"
            >
                {outputMap[dataset]?.[ni]}
            </text>

            {dataset === "auto_mpg" && (
                <>
                    <text
                        x={cx + SVGWIDTH * 0.11}
                        y={cy - SVGHEIGHT * 0.06}
                        fontSize={fontSize}
                        textAnchor="middle"
                    >
                        Predicted Value:
                    </text>
                    <text
                        x={cx + SVGWIDTH * 0.11}
                        y={cy + SVGHEIGHT * 0.04}
                        fontSize={fontSize}
                        textAnchor="middle"
                    >
                        Actual Value:
                    </text>
                    <text
                        x={cx + SVGWIDTH * 0.11}
                        y={cy + SVGHEIGHT * 0.08}
                        fontSize={fontSize}
                        textAnchor="middle"
                    >
                        {formatActual(original || [], dataset)}
                    </text>
                </>
            )}

            {dataset === "iris" && (
                <text
                    x={cx + SVGWIDTH * 0.11}
                    y={cy + SVGHEIGHT * 0.08}
                    fontSize={fontSize}
                    textAnchor="middle"
                >
                    {formatActual(original || [], dataset) === outputMap[dataset]?.[ni]
                        ? "Actual Answer"
                        : ""}
                </text>
            )}

            <text
                x={cx + SVGWIDTH * 0.11}
                y={(dataset === "iris" ? cy : cy - SVGHEIGHT * 0.06) + SVGHEIGHT * 0.04}
                fontSize={fontSize}
                textAnchor="middle"
            >
                {formatValue(activationValue, dataset)}
            </text>
        </>
    );
};

export const renderInputValues = (cx: number, cy: number, original: number[], nodeIndex: number, activationValue: number, fontSize: number, INPUTLABELOFFSET: number, SVGWIDTH: number, SVGHEIGHT: number) => (
  <>
    <text
      x={cx - INPUTLABELOFFSET - (10 + SVGWIDTH * .012)}
      y={cy + (SVGHEIGHT * 0.04)}
      className=""
      textAnchor="middle"
      fontSize={fontSize}
    >
      {original[nodeIndex].toFixed(2)}
    </text>
    <text
      x={cx - INPUTLABELOFFSET + (SVGWIDTH * .01) + (Math.abs(original[nodeIndex]) > 100 ? (SVGWIDTH * .02) : (SVGWIDTH * .01))}
      y={cy + (SVGHEIGHT * 0.04)}
      className=""
      textAnchor="middle"
      fontSize={fontSize}
    >
      {`(${activationValue.toFixed(2)})`}
    </text>
  </>
);