import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function transpose(matrix: number[][]) {
  if (!matrix || !(matrix.length > 0)){
    return null
  }
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

export function reshapeTo2D(array: number[]) {
  const result = [];
  for (let i = 0; i < array.length; i += 1) {
    result.push(array.slice(i, i + 1));
  }
  return result;
}

export function multiplyMatrices(a: number[][] | null, b: number[][] | null) {
  if (!a || !b || !a[0] || !b[0] || !(a.length > 0)|| !(b.length > 0) || !(a[0].length > 0)|| !(b[0].length > 0)){
    return null
  }

  const aRows = a.length;
  const aCols = a[0].length;
  const bRows = b.length;
  const bCols = b[0].length;

  if (aCols !== bRows) {
    console.log(aRows, aCols, bRows, bCols)
    throw new Error("Columns of A must match rows of B");
  }

  const result = Array.from({ length: aRows }, () =>
    Array.from({ length: bCols }, () => 0)
  );

  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      for (let k = 0; k < aCols; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

  export const formatValue = (value: number, dataset: string) => {
    if (dataset === "iris") {
      return `${(value * 100).toFixed(1)}%`;
    }
    if (dataset === "auto_mpg") {
      return `${value.toFixed(1)} MPG`;
    }
    return value.toFixed(2);
  };

  export const outputMap: { [key: string]: string[] } = {
    auto_mpg: ["MPG"],
    iris: ["Setosa", "Versicolor", "Virginica"],
  };
  
  export const formatActual = (original: number[], dataset: string) => {
    if (!original || original.length < 3){
      return;
    }
    if (dataset === "iris") {
      const actualResults = original.slice(-3)
      const index = actualResults.findIndex((v) => v === 1);
      return outputMap[dataset]?.[index] ?? "Unknown";
    } else if (dataset === "auto_mpg") {
      const actualResults = original.slice(-1)
      return formatValue(actualResults[0], dataset);
    }
    return "N/A";
  };