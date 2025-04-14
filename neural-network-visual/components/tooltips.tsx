import { DATASET_INPUT_FEATURES } from "@/static/constants";

const InputInfo = ({ dataset, input, originalInput }: { dataset: string; input: number[], originalInput: number[] }) => {        
  
    const features = DATASET_INPUT_FEATURES[dataset] || [];
  
    return (
      <>
        <div className="flex items-center gap-1 text-sm text-gray-600 mb-1 relative group">
          <span className="cursor-pointer text-gray-500 hover:text-gray-700 transition-colors duration-150">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z"
              />
            </svg>
          </span>
  
          {/* Tooltip */}
          <div className="absolute z-10 hidden w-max group-hover:block p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg bottom-6 left-1/2 transform -translate-x-1/2 transition-all duration-200">
            {features.length > 0 && (
              <>
                <p className="font-semibold mb-1">Features & Input:</p>
                <p className="mb-1">(Normalized)</p>
                <ul className="space-y-1 text-sm text-gray-100 text-left">
                  {features.map((feature, idx) => (
                    <li key={feature} className="flex justify-between">
                      <span>{feature}</span>
                      <div className="ml-5">
                      <span className="font-mono">{originalInput[idx]?.toFixed(3) ?? "N/A"}</span>
                      <span className="ml-2 font-mono">{`(${input[idx]?.toFixed(3)})` ?? "N/A"}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>
      </>
    );
  };

  const OutputInfo = ({ dataset, output, actual }: { dataset: string; output: number[]; actual: number[] }) => {
    const outputMap: { [key: string]: string[] } = {
      california_housing: ["Median House Value"],
      iris: ["Setosa", "Versicolor", "Virginica"],
    };
  
    const formatValue = (value: number, dataset: string) => {
      if (dataset === "iris") {
        return `${(value * 100).toFixed(1)}%`;
      }
      if (dataset === "california_housing") {
        const dollars = value * 100000; // adjust scale
        return dollars.toLocaleString("en-US", {
          style: "currency",
          currency: "USD",
          maximumFractionDigits: 0,
        });
      }
      return value.toFixed(3);
    };

    const formatActual = () => {
      if (dataset === "iris") {
        const actualResults = actual.slice(-3)
        const index = actualResults.findIndex((v) => v === 1);
        return outputMap[dataset]?.[index] ?? "Unknown";
      } else if (dataset === "california_housing") {
        const actualResults = actual.slice(-1)
        return formatValue(actualResults[0], dataset);
      }
      return "N/A";
    };
  
    const outputs = outputMap[dataset] || [];
  
    return (
      <>
        <div className="flex items-center gap-1 text-sm text-gray-600 mb-1 relative group">
          <span className="cursor-pointer text-gray-500 hover:text-gray-700 transition-colors duration-150">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z"
              />
            </svg>
          </span>
  
          {/* Tooltip */}
          <div className="absolute z-10 hidden group-hover:block w-56 p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg bottom-6 left-1/2 transform -translate-x-1/2 transition-all duration-200">
            {outputs.length > 0 && (
              <>
                <p className="font-semibold mb-1">Model Output:</p>
                <ul className="space-y-1 text-sm text-gray-100 text-left">
                  {outputs.map((label, idx) => (
                    <li key={label} className="flex justify-between">
                      <span>{label}</span>
                      <span className="font-mono">
                        {formatValue(output[idx], dataset)}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          <p className="font-semibold">Actual Result: <span className="font-mono text-white">{formatActual()}</span></p>
          </div>
        </div>
      </>
    );
  };  
  
  export { InputInfo, OutputInfo };
  