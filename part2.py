"""Part 2: logistic regression from scratch, KNN with CV, cost-based threshold."""
import math

import numpy as np

import common as c


def main():
    a = c.fit_split_artifacts()
    Ztr, Zte, ytr, yte, w, b = a["Ztr"], a["Zte"], a["ytr"], a["yte"], a["w"], a["b"]
    out = []
    # 2.1
    out += [("train_rows", len(ytr)), ("train_positives", int(np.sum(ytr))),
            ("test_rows", len(yte)), ("test_positives", int(np.sum(yte)))]
    # 2.2
    L = a["losses"]
    out += [("loss_epoch_0", L[0]), ("loss_epoch_1", L[1]), ("loss_epoch_10", L[10]),
            ("loss_epoch_100", L[100]), ("loss_epoch_1000", L[1000])]
    out += [(f"weight_{nm}", w[j]) for j, nm in enumerate(c.X_NAMES)]
    p_te = c.predict_proba(Zte, w, b)
    out += [("bias", b), ("test_bce", c.bce(yte, p_te))]
    # 2.3
    for eta in (0.01, 0.1, 1.0):
        _, _, ls = c.train_logistic(Ztr, ytr, eta, 200, record=(200,))
        out.append((f"loss_epoch_200_eta_{eta}", ls[200]))
    # 2.4
    out += [(f"odds_ratio_{nm}", math.exp(w[j])) for j, nm in enumerate(c.X_NAMES)]
    # 2.5
    for label, tau in (("0.5", 0.5), ("0.3", 0.3)):
        tp, fp, tn, fn = c.confusion(yte, p_te >= tau)
        m = c.metrics(tp, fp, tn, fn)
        out += [(f"tau_{label}_tp", tp), (f"tau_{label}_fp", fp), (f"tau_{label}_tn", tn), (f"tau_{label}_fn", fn)]
        out += [(f"tau_{label}_{k}", m[k]) for k in ("accuracy", "precision", "recall", "f1", "specificity")]
    out.append(("blind_accuracy", int(np.sum(yte == 0)) / len(yte)))
    # 2.6
    rows = []
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = c.confusion(yte, p_te >= tau)
        m = c.metrics(tp, fp, tn, fn)
        rows.append([tau, tp, fp, tn, fn, m["recall"], m["fpr"]])
    c.write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)
    # 2.7
    cv_rows, best_k = c.knn_cv(Ztr, ytr)
    c.write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)
    out.append(("knn_best_k", best_k))
    pred = c.knn_predict(Ztr, ytr, Zte, best_k)
    tp, fp, tn, fn = c.confusion(yte, pred)
    m = c.metrics(tp, fp, tn, fn)
    out += [("knn_tp", tp), ("knn_fp", fp), ("knn_tn", tn), ("knn_fn", fn)]
    out += [(f"knn_{k}", m[k]) for k in ("accuracy", "precision", "recall", "f1")]
    # 2.8
    p_tr = c.predict_proba(Ztr, w, b)
    best_idx, best_cost = None, None
    for k in range(1, 20):
        tp, fp, tn, fn = c.confusion(ytr, p_tr >= k / 20)
        cst = c.cost(fp, fn)
        if best_cost is None or cst < best_cost:
            best_idx, best_cost = k, cst
    tau_star = best_idx / 20
    _, fp, _, fn = c.confusion(yte, p_te >= tau_star)
    _, fp5, _, fn5 = c.confusion(yte, p_te >= 0.5)
    out += [("cost_tau", tau_star), ("cost_train_at_tau", best_cost), ("cost_test_at_tau", c.cost(fp, fn)),
            ("cost_test_at_0.5", c.cost(fp5, fn5)), ("cost_test_flag_nobody", c.cost(0, int(np.sum(yte))))]
    c.write_results("part2_results.txt", out)


if __name__ == "__main__":
    main()
