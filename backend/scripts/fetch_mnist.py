"""
Download MNIST from OpenML and save the first 10,000 samples as a CSV.
Columns: pixel_0 ... pixel_783, label
Pixel values are stored as integers 0-255.
Run from the backend/ directory: python scripts/fetch_mnist.py
"""
import os
import numpy as np
from sklearn.datasets import fetch_openml

N = 10000
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist.csv")
out_path = os.path.normpath(out_path)

print("Fetching MNIST from OpenML (may take a moment)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:N].astype(np.uint8)   # (10000, 784)
y = mnist.target[:N].astype(np.uint8) # (10000,)

header = ",".join([f"pixel_{i}" for i in range(784)] + ["label"])
data = np.hstack([X, y.reshape(-1, 1)])

np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%d")
print(f"Saved {N} samples to {out_path}")
