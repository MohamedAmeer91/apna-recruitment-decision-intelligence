"""Part 4: split purity, voting, AdaBoost by hand, out-of-bag share, network sizing."""
import math

import numpy as np

import common as c


def entropy(pos, n):
    if n == 0 or pos == 0 or pos == n:
        return 0.0
    p = pos / n
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1.0 - (p * p + (1 - p) * (1 - p))


def conv_out(size, k, pad, stride):
    return (size - k + 2 * pad) // stride + 1


def dense_params(i, o):
    return i * o + o


def conv_params(kh, kw, cin, cout):
    return (kh * kw * cin + 1) * cout


def prf(tp, fp, fn):
    m = c.metrics(tp, fp, 0, fn)
    return m["precision"], m["recall"], m["f1"]


def main():
    a = c.fit_split_artifacts()
    d = a["data"]
    tr, te = a["tr"], a["te"]
    Xtr, Xte = d["X"][tr], d["X"][te]
    ytr, yte = a["ytr"], a["yte"]
    out = []
    # 4.1
    n = len(ytr)
    P = int(np.sum(ytr))
    h_root, g_root = entropy(P, n), gini(P, n)
    out += [("root_entropy", h_root), ("root_gini", g_root)]
    conds = {"S1": lambda X: X[:, 5] == 1, "S2": lambda X: X[:, 6] == 1,
             "S3": lambda X: X[:, 2] > 0.5, "S4": lambda X: X[:, 0] <= 24}
    split_rows = []
    gains = {}
    for nm, f in conds.items():
        yes = f(Xtr)
        ny, py = int(np.sum(yes)), int(np.sum(ytr[yes]))
        nn, pn = n - ny, P - py
        ig = h_root - (ny / n * entropy(py, ny) + nn / n * entropy(pn, nn))
        gg = g_root - (ny / n * gini(py, ny) + nn / n * gini(pn, nn))
        gains[nm] = (ig, gg)
        split_rows.append([nm, ny, py, nn, pn, ig, gg])
    c.write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
                split_rows)
    best_ig = max(conds, key=lambda s: (gains[s][0], -list(conds).index(s)))
    best_gg = max(conds, key=lambda s: (gains[s][1], -list(conds).index(s)))
    out += [("best_split_by_info_gain", best_ig), ("best_split_by_gini_gain", best_gg)]
    # 4.2
    yes_tr = conds[best_ig](Xtr)
    rate_yes = float(np.sum(ytr[yes_tr])) / int(np.sum(yes_tr))
    rate_no = float(np.sum(ytr[~yes_tr])) / int(np.sum(~yes_tr))
    out += [("stump_rate_yes", rate_yes), ("stump_rate_no", rate_no)]
    _, best_k = c.knn_cv(a["Ztr"], ytr)
    p_log = c.predict_proba(a["Zte"], a["w"], a["b"])
    p_knn = c.knn_proba(a["Ztr"], ytr, a["Zte"], best_k)
    pred_knn = c.knn_predict(a["Ztr"], ytr, a["Zte"], best_k)
    p_stump = np.where(conds[best_ig](Xte), rate_yes, rate_no)
    pred_log = p_log >= 0.3
    pred_stump = p_stump >= 0.3
    votes = pred_log.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)
    pred_hard = votes >= 2
    pred_soft = (p_log + p_knn + p_stump) / 3 >= 0.3 - 1e-12
    vrows = []
    for nm, pr in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
        tp, fp, tn, fn = c.confusion(yte, pr)
        precision, recall, f1 = prf(tp, fp, fn)
        vrows.append([nm, tp, fp, tn, fn, precision, recall, f1])
    c.write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)
    # 4.3
    ys = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
    w = np.full(8, 1 / 8)
    ada_rows = []
    alphas = []
    errs = []
    for rnd, h in ((1, h1), (2, h2)):
        eps = float(np.sum(w[h != ys]))
        alpha = 0.5 * math.log((1 - eps) / eps)
        u = w * np.exp(-alpha * ys * h)
        wn = u / np.sum(u)
        for i in range(8):
            ada_rows.append([rnd, i + 1, w[i], u[i], wn[i]])
        errs.append(eps)
        alphas.append(alpha)
        w = wn
    c.write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
    score = alphas[0] * h1 + alphas[1] * h2
    acc = float(np.sum(np.sign(score) == ys)) / 8
    out += [("round1_error", errs[0]), ("round1_alpha", alphas[0]), ("round2_error", errs[1]),
            ("round2_alpha", alphas[1]), ("ensemble_accuracy", acc)]
    # 4.4
    out.append(("oob_fraction", (1 - 1 / 560) ** 560))
    # 4.5
    mlp = [("mlp_hidden1", dense_params(784, 16)), ("mlp_hidden2", dense_params(16, 16)),
           ("mlp_output", dense_params(16, 10))]
    s1 = conv_out(28, 5, 2, 1)
    p1 = conv_out(s1, 2, 0, 2)
    s2 = conv_out(p1, 5, 0, 1)
    p2 = conv_out(s2, 2, 0, 2)
    flat = p2 * p2 * 16
    cnn = [("cnn_conv1", conv_params(5, 5, 1, 6)), ("cnn_conv2", conv_params(5, 5, 6, 16)),
           ("cnn_dense1", dense_params(flat, 32)), ("cnn_dense2", dense_params(32, 10))]
    c.write_csv("part4_architecture.csv", ["layer", "parameters"], [[k, v] for k, v in mlp + cnn])
    out += [("mlp_total", sum(v for _, v in mlp)), ("cnn_total", sum(v for _, v in cnn)),
            ("fc_layer_params", dense_params(128 * 128 * 3, 1024)), ("conv_layer_params", conv_params(3, 3, 3, 64)),
            ("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)]
    c.write_results("part4_results.txt", out)


if __name__ == "__main__":
    main()
