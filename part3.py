"""Part 3: K-means, hierarchical clustering on a supplied table, PCA by seven steps."""
import sys

import numpy as np

import common as c

DIST = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def kmeans(Z, K):
    n = Z.shape[0]
    step = n // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    passes = 0
    while True:
        passes += 1
        d2 = np.sum((Z[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        new = np.argmin(d2, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            mem = labels == k
            if np.any(mem):
                cent[k] = np.sum(Z[mem], axis=0) / np.sum(mem)
    wcss = float(np.sum(d2[np.arange(n), labels]))
    return labels, wcss, passes


def link_dist(D, A, B, rule):
    vals = [D[i, j] for i in A for j in B]
    if rule == "single":
        return min(vals)
    if rule == "complete":
        return max(vals)
    return sum(vals) / len(vals)


def agglomerate(D, rule):
    clusters = [(i,) for i in range(D.shape[0])]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = link_dist(D, clusters[i], clusters[j], rule)
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = clusters[i], clusters[j]
        merges.append((a, b, h))
        clusters = [cl for t, cl in enumerate(clusters) if t not in (i, j)] + [tuple(sorted(a + b))]
        clusters.sort(key=lambda cl: cl[0])
    return merges


def name(cl):
    return "+".join(f"E{i + 1}" for i in cl)


def main():
    d = c.load_data()
    X, y, value = d["X"][:, :5], d["y"], d["value"]
    n = X.shape[0]
    mean = np.sum(X, axis=0) / n
    sd = c.sample_sd_cols(X)
    Z = (X - mean) / sd
    out = []
    # 3.1
    wcss = {}
    runs = {}
    for K in range(2, 7):
        labels, w, passes = kmeans(Z, K)
        wcss[K] = w
        runs[K] = labels
        print(f"K={K} passes={passes}", file=sys.stderr)
    out += [(f"wcss_k{K}", wcss[K]) for K in range(2, 7)]
    # 3.2
    elbow = None
    for K in (3, 4, 5):
        if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
            elbow = K
            break
    if elbow is None:
        raise SystemExit("elbow_k: no K in {3,4,5} satisfies the rule; stop and report")
    out.append(("elbow_k", elbow))
    # 3.3
    lab = runs[3]
    mean_x3 = [np.sum(X[lab == k, 2]) / np.sum(lab == k) for k in range(3)]
    order = sorted(range(3), key=lambda k: mean_x3[k])
    seg_rows = []
    seg_of = np.zeros(n, dtype=int)
    for s, k in enumerate(order):
        mem = lab == k
        seg_of[mem] = s
        cnt = int(np.sum(mem))
        seg_rows.append(["ABC"[s], cnt] + [np.sum(X[mem, j]) / cnt for j in range(5)]
                        + [np.sum(y[mem]) / cnt, np.sum(value[mem]) / cnt])
    out += [(f"size_segment_{r[0]}", r[1]) for r in seg_rows]
    c.write_csv("part3_segments.csv",
                ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
                seg_rows)
    # 3.4
    h_rows = []
    cuts = {}
    for rule in ("single", "complete", "average"):
        merges = agglomerate(DIST, rule)
        heights = [m[2] for m in merges]
        for no, (a, b, h) in enumerate(merges, 1):
            if b[0] < a[0]:
                a, b = b, a
            h_rows.append([rule, no, name(a), name(b), h])
        gaps = [heights[i + 1] - heights[i] for i in range(4)]
        j = int(np.argmax(gaps))
        cuts[rule] = 6 - (j + 1)
    c.write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], h_rows)
    out += [(f"{r}_clusters_at_cut", cuts[r]) for r in ("single", "complete", "average")]
    # 3.5
    mu = np.sum(Z, axis=0) / n
    assert np.max(np.abs(mu)) < 1e-9
    Xc = Z - mu
    S = Xc.T @ Xc / (n - 1)
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(-vals, kind="stable")
    vals, vecs = vals[idx], vecs[:, idx].copy()
    for j in range(5):
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] = -vecs[:, j]
    W = vecs[:, :2]
    P = Xc @ W
    total = float(np.sum(vals))
    ratio = vals / total
    cum = np.cumsum(ratio)
    out.append(("eigenvalue_sum", total))
    out += [(f"eigenvalue_{i + 1}", vals[i]) for i in range(5)]
    out += [(f"explained_ratio_{i + 1}", ratio[i]) for i in range(5)]
    out += [(f"cumulative_ratio_{i + 1}", cum[i]) for i in range(5)]
    out.append(("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1))
    out += [(f"pc1_loading_X{j + 1}", vecs[j, 0]) for j in range(5)]
    out += [(f"pc2_loading_X{j + 1}", vecs[j, 1]) for j in range(5)]
    for r in range(3):
        out += [(f"row{r + 1}_pc1", P[r, 0]), (f"row{r + 1}_pc2", P[r, 1])]
    out += [(f"mean_pc1_segment_{'ABC'[s]}", np.sum(P[seg_of == s, 0]) / np.sum(seg_of == s)) for s in range(3)]
    c.write_results("part3_results.txt", out)


if __name__ == "__main__":
    main()
