import { DATASET_INPUT_FEATURES } from "@/components/network/static/constants";

const InputInfo = ({ dataset, input, originalInput }: { dataset: string; input: number[], originalInput: number[] }) => {
  const features = DATASET_INPUT_FEATURES[dataset] || [];

  return (
    <div className="flex items-center gap-1 text-sm text-gray-600 mb-1 relative group">
      <span className="cursor-pointer text-indigo-400 hover:text-indigo-600 transition-colors duration-150">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
        </svg>
      </span>

      <div className="absolute z-50 hidden group-hover:block p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg bottom-full mb-1 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
        {features.length > 0 && originalInput?.length > 0 && (
          <>
            <p className="font-semibold mb-2">Input Features</p>
            <div className="flex gap-4 text-xs text-gray-400 mb-1 pb-1 border-b border-gray-600">
              <span className="w-28">Feature</span>
              <span className="w-12 text-right">Raw</span>
              <span className="w-16 text-right">Normalized</span>
            </div>
            <ul className="space-y-1">
              {features.map((feature, idx) => (
                <li key={feature} className="flex gap-4 text-xs">
                  <span className="w-28 text-gray-300">{feature}</span>
                  <span className="w-12 text-right font-mono text-gray-400">{originalInput[idx]?.toFixed(2) ?? "—"}</span>
                  <span className="w-16 text-right font-mono text-white">{input[idx]?.toFixed(4) ?? "—"}</span>
                </li>
              ))}
            </ul>
            <p className="text-xs text-gray-500 mt-2 pt-1 border-t border-gray-600">The network receives the normalized values</p>
          </>
        )}
      </div>
    </div>
  );
};

const OutputInfo = ({ dataset, output, actual }: { dataset: string; output: number[]; actual: number[] }) => {
  const outputMap: { [key: string]: string[] } = {
    auto_mpg: ["MPG"],
    iris: ["Setosa", "Versicolor", "Virginica"],
    xor: ["Output"],
  };

  const actualIdx = dataset === "iris"
    ? actual.slice(-3).findIndex((v) => v === 1)
    : -1;

  const outputs = outputMap[dataset] || [];

  return (
    <div className="flex items-center gap-1 text-sm text-gray-600 mb-1 relative group">
      <span className="cursor-pointer text-indigo-400 hover:text-indigo-600 transition-colors duration-150">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
        </svg>
      </span>

      <div className="absolute z-50 hidden group-hover:block w-52 p-3 text-sm text-white bg-gray-800 rounded-lg shadow-lg bottom-full mb-1 left-1/2 transform -translate-x-1/2">
        {outputs.length > 0 && (
          <>
            <p className="font-semibold mb-2">Raw Output Values</p>
            <div className="flex justify-between text-xs text-gray-400 mb-1 pb-1 border-b border-gray-600">
              <span>Neuron</span>
              <span>Activation</span>
            </div>
            <ul className="space-y-1">
              {outputs.map((label, idx) => (
                <li key={label} className="flex justify-between items-center text-xs">
                  <span className={actualIdx === idx ? "text-yellow-300 font-medium" : "text-gray-300"}>
                    {label}{actualIdx === idx ? " ← actual" : ""}
                  </span>
                  <span className="font-mono text-white">{output[idx]?.toFixed(4) ?? "—"}</span>
                </li>
              ))}
            </ul>
            <p className="text-xs text-gray-500 mt-2 pt-1 border-t border-gray-600">
              {dataset === "iris" ? "Softmax applied — values sum to 1" :
               dataset === "xor" ? "Sigmoid output — closer to 1 = predicts XOR=1" :
               "Linear output — raw predicted value"}
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export { InputInfo, OutputInfo };
