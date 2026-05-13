import os
import numpy as np
from utils import one_hot_encode, train_test_split_np, StandardScalerNP

def load_dataset(dataset_name):
    if dataset_name == "iris":
        X_train, X_test, Y_train, Y_test, inp, out, act, data = load_iris_dataset()
        return X_train, X_test, Y_train, Y_test, inp, out, act, data, None, None
    elif dataset_name == "auto_mpg":
        return load_auto_mpg_dataset()
    elif dataset_name == "xor":
        X_train, X_test, Y_train, Y_test, inp, out, act, data = load_xor_dataset()
        return X_train, X_test, Y_train, Y_test, inp, out, act, data, None, None
    else:
        raise ValueError("Unsupported dataset. Choose 'iris', 'auto_mpg', or 'xor'.")

def load_xor_dataset():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    original_train_data = np.hstack((X, Y)).tolist()
    return X, X, Y, Y, 2, 1, "sigmoid", original_train_data

def load_auto_mpg_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "auto_mpg.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("auto_mpg.csv not found. Run scripts/fetch_datasets.py to download.")

    raw = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    # Columns: displacement, horsepower, weight, acceleration, mpg
    X = raw[:, :4]
    y = raw[:, 4].reshape(-1, 1)

    # Remove rows with NaNs if any
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, random_state=42) #type: ignore
    original_train_data = np.hstack((X_train, y_train)).tolist()

    scaler = StandardScalerNP()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_scaler = StandardScalerNP()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test, X_train.shape[1], y_train.shape[1], "linear", original_train_data, float(y_scaler.mean_[0]), float(y_scaler.scale_[0])

def load_iris_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("iris.csv not found. Run scripts/fetch_datasets.py to download.")

    raw = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    # Columns: sepal_length,sepal_width,petal_length,petal_width,class
    X = raw[:, :4]
    y = raw[:, 4].astype(int)
    Y = one_hot_encode(y, num_classes=3)

    X_train, X_test, Y_train, Y_test = train_test_split_np(X, Y, test_size=0.2, random_state=42)  # type: ignore
    original_train_data = np.hstack((X_train, Y_train)).tolist()

    scaler = StandardScalerNP()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test, X_train.shape[1], Y_train.shape[1], "softmax", original_train_data