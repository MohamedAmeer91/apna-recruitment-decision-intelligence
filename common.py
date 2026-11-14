"""Shared helpers for part1.py .. part4.py. Importing this module runs nothing."""
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUT_DIR = ROOT / "outputs"
X_NAMES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


def load_data():
    """Read data/records.csv. X is 800x7 (X1..X7), y the outcome, value the days column."""
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    mat = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64)
    return {"header": header, "X": mat[:, 0:7], "y": mat[:, 7].astype(int), "value": mat[:, 8]}


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = f"{float(v):.4f}"
        return "0.0000" if s == "-0.0000" else s
    return str(v)


def write_results(name, pairs):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / name, "w", newline="\n") as f:
        for k, v in pairs:
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(name, header, rows):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / name, "w", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sample_sd(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.sum(a) / len(a)
    return math.sqrt(np.sum((a - m) ** 2) / (len(a) - 1))


def sample_sd_cols(A):
    m = np.sum(A, axis=0) / A.shape[0]
    return np.sqrt(np.sum((A - m) ** 2, axis=0) / (A.shape[0] - 1))


def ols_closed_form(x, y):
    """Simple regression slope = cov/var (both divisor n-1), intercept, fitted values."""
    n = len(x)
    mx = np.sum(x) / n
    my = np.sum(y) / n
    cov = np.sum((x - mx) * (y - my)) / (n - 1)
    var = np.sum((x - mx) ** 2) / (n - 1)
    slope = cov / var
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept, "fitted": intercept + slope * x}


def stratified_split(y):
    tr = []
    te = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = (7 * len(idx)) // 10
        tr.extend(idx[:k].tolist())
        te.extend(idx[k:].tolist())
    return np.array(sorted(tr)), np.array(sorted(te))


def standardise(Xtr, Xte):
    mean = np.sum(Xtr, axis=0) / Xtr.shape[0]
    sd = sample_sd_cols(Xtr)
    return (Xtr - mean) / sd, (Xte - mean) / sd, mean, sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y))


def train_logistic(Z, y, eta, steps, record=()):
    n, d = Z.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Z @ w + b))
    for t in range(1, steps + 1):
        p = sigmoid(Z @ w + b)
        err = p - y
        gw = Z.T @ err / n
        gb = np.sum(err) / n
        w = w - eta * gw
        b = b - eta * gb
        if t in record:
            losses[t] = bce(y, sigmoid(Z @ w + b))
    return w, b, losses


def predict_proba(Z, w, b):
    return sigmoid(Z @ w + b)


def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "specificity": _div(tn, tn + fp),
        "fpr": _div(fp, fp + tn),
    }


def knn_pos_counts(Zref, yref, Zq, k):
    """Number of positive neighbours among the k nearest (ties by file position)."""
    d2 = np.sum((Zq[:, None, :] - Zref[None, :, :]) ** 2, axis=2)
    order = np.argsort(d2, axis=1, kind="stable")[:, :k]
    return np.sum(yref[order], axis=1)


def knn_proba(Zref, yref, Zq, k):
    return knn_pos_counts(Zref, yref, Zq, k) / k


def knn_predict(Zref, yref, Zq, k):
    return knn_pos_counts(Zref, yref, Zq, k) >= (k + 1) // 2


def cost(fp, fn):
    return 50 * fp + 400 * fn


def fit_split_artifacts():
    """Task 2.1 split + standardisation and the Task 2.2 logistic model (deterministic)."""
    d = load_data()
    tr, te = stratified_split(d["y"])
    Ztr, Zte, mean, sd = standardise(d["X"][tr], d["X"][te])
    w, b, losses = train_logistic(Ztr, d["y"][tr], 0.1, 1000, record=(0, 1, 10, 100, 1000))
    return {"data": d, "tr": tr, "te": te, "Ztr": Ztr, "Zte": Zte, "mean": mean, "sd": sd,
            "ytr": d["y"][tr], "yte": d["y"][te], "w": w, "b": b, "losses": losses}
