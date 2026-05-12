import os
import csv
from typing import Tuple

# Requires: scikit-learn (and pandas used internally by OpenML when as_frame=True)
from sklearn.datasets import load_iris, fetch_openml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, header: Tuple[str, ...], rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def fetch_iris() -> str:
    iris = load_iris()  # features in iris.data, targets in iris.target (0,1,2)
    X = iris.data # type: ignore
    y = iris.target # type: ignore
    out_path = os.path.join(DATA_DIR, "iris.csv")
    rows = [[X[i, 0], X[i, 1], X[i, 2], X[i, 3], int(y[i])] for i in range(X.shape[0])]
    save_csv(out_path, ("sepal_length","sepal_width","petal_length","petal_width","class"), rows)
    print(f"Saved iris CSV to {out_path}")
    return out_path


def fetch_auto_mpg() -> str:
    # Use OpenML autoMpg dataset
    ds = fetch_openml("autoMpg", version=1, as_frame=True)
    df = ds.data.copy()
    y = ds.target

    # Clean horsepower which may contain '?' -> drop rows where horsepower isn't numeric
    df = df[["displacement","horsepower","weight","acceleration"]].copy()
    # Coerce horsepower to numeric, drop NaNs
    df["horsepower"] = df["horsepower"].apply(lambda v: None if (isinstance(v, str) and v.strip() == "?") else v)
    df = df.dropna()
    # Align target with cleaned df index
    y = y.loc[df.index]

    rows = [[float(df.iloc[i]["displacement"]), float(df.iloc[i]["horsepower"]), float(df.iloc[i]["weight"]), float(df.iloc[i]["acceleration"]), float(y.iloc[i])] for i in range(len(df))]
    out_path = os.path.join(DATA_DIR, "auto_mpg.csv")
    save_csv(out_path, ("displacement","horsepower","weight","acceleration","mpg"), rows)
    print(f"Saved auto_mpg CSV to {out_path}")
    return out_path


def main() -> None:
    ensure_dir(DATA_DIR)
    fetch_iris()
    fetch_auto_mpg()
    print("Datasets downloaded via scikit-learn and saved as CSVs.")


if __name__ == "__main__":
    main()
