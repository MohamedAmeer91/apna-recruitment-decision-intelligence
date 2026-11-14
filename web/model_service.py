"""Cached model state built from common.py on the project's 560/240 split."""
import math
from functools import lru_cache

import numpy as np

import common as c
from web.outputs_reader import read_results, read_table

FEATURES = ["experience_months", "interview_reschedules_90d", "salary_gap_ratio",
            "expected_salary_inr_thousands", "profile_completeness_score",
            "competing_offer_flag", "referred_flag"]
STUMP_CUT = 24.0


def _row_metrics(tp, fp, tn, fn):
    m = c.metrics(tp, fp, tn, fn)
    m.update(tp=tp, fp=fp, tn=tn, fn=fn, cost_inr=c.cost(fp, fn))
    return m


class Model:
    def __init__(self):
        a = c.fit_split_artifacts()
        self.a = a
        self.yte = a["yte"]
        self.ytr = a["ytr"]
        self.p_te = c.predict_proba(a["Zte"], a["w"], a["b"])
        self.positives = int(np.sum(self.yte))
        self.negatives = len(self.yte) - self.positives
        self.nobody_cost = c.cost(0, self.positives)
        self.cv_rows, self.best_k = c.knn_cv(a["Ztr"], self.ytr)
        d = a["data"]
        Xtr, Xte = d["X"][a["tr"]], d["X"][a["te"]]
        yes = Xtr[:, 0] <= STUMP_CUT
        self.rate_yes = float(np.sum(self.ytr[yes])) / int(np.sum(yes))
        self.rate_no = float(np.sum(self.ytr[~yes])) / int(np.sum(~yes))
        self.p_stump = np.where(Xte[:, 0] <= STUMP_CUT, self.rate_yes, self.rate_no)
        self.p_knn = c.knn_proba(a["Ztr"], self.ytr, a["Zte"], self.best_k)
        self.pred_knn = c.knn_predict(a["Ztr"], self.ytr, a["Zte"], self.best_k)
        # segments: centroids come from the committed part3_segments.csv means
        self.seg_rows = read_table("part3_segments.csv")
        X5 = d["X"][:, :5]
        self.all_mean = np.sum(X5, axis=0) / len(X5)
        self.all_sd = c.sample_sd_cols(X5)
        self.centroids = np.array([[r[f"mean_X{j + 1}"] for j in range(5)] for r in self.seg_rows])
        self.cent_z = (self.centroids - self.all_mean) / self.all_sd
        self.X5 = X5
        self.seg_names = [r["segment"] for r in self.seg_rows]
        self.cost_tau = read_results("part2_results.txt")["cost_tau"]
        self._knn_cache = {}

    # ---- thresholds
    def threshold(self, tau):
        tp, fp, tn, fn = c.confusion(self.yte, self.p_te >= tau)
        m = _row_metrics(tp, fp, tn, fn)
        m.update(tau=tau, positives=self.positives, negatives=self.negatives,
                 cost_flag_nobody_inr=self.nobody_cost,
                 saving_vs_nobody_inr=self.nobody_cost - m["cost_inr"])
        return m

    def curve(self):
        pts = []
        for i in range(101):
            tau = i / 100
            t = self.threshold(tau)
            pts.append({"tau": tau, "fpr": t["fpr"], "tpr": t["recall"], "precision": t["precision"],
                        "f1": t["f1"], "cost_inr": t["cost_inr"]})
        # AUC by trapezoid over the ROC points (derived, not a graded number)
        xs = [p["fpr"] for p in pts][::-1]
        ys = [p["tpr"] for p in pts][::-1]
        auc = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2 for i in range(len(xs) - 1))
        return {"auc": auc, "points": pts, "cost_tau_train": self.cost_tau, "default_tau": 0.5}

    # ---- knn
    def knn(self, k):
        if k not in self._knn_cache:
            pred = c.knn_predict(self.a["Ztr"], self.ytr, self.a["Zte"], k)
            self._knn_cache[k] = _row_metrics(*c.confusion(self.yte, pred))
        m = dict(self._knn_cache[k])
        m.update(k=k, is_best=(k == self.best_k), best_k=self.best_k,
                 cv=[{"K": r[0], "mean_cv_f1": r[1]} for r in self.cv_rows])
        return m

    # ---- ensemble (mirrors part4.py: knn hard majority, logistic and stump at tau)
    def ensemble(self, tau):
        yte = self.yte
        pred_log = self.p_te >= tau
        pred_stump = self.p_stump >= tau
        votes = pred_log.astype(int) + self.pred_knn.astype(int) + pred_stump.astype(int)
        soft = (self.p_te + self.p_knn + self.p_stump) / 3 >= tau - 1e-12
        rows = []
        for nm, pr in (("logistic", pred_log), ("knn", self.pred_knn), ("stump", pred_stump),
                       ("hard_vote", votes >= 2), ("soft_vote", soft)):
            m = _row_metrics(*c.confusion(yte, pr))
            rows.append({"model": nm, **{k: m[k] for k in
                         ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "cost_inr")}})
        return {"tau": tau, "models": rows}

    # ---- segments
    def assign(self, X5rows):
        Z = (np.asarray(X5rows, dtype=float) - self.all_mean) / self.all_sd
        d2 = np.sum((Z[:, None, :] - self.cent_z[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1)

    def segment_points(self):
        lab = self.assign(self.X5)
        return [{"x1": float(r[0]), "x2": float(r[1]), "x3": float(r[2]), "x4": float(r[3]),
                 "x5": float(r[4]), "segment": self.seg_names[i]} for r, i in zip(self.X5, lab)]

    # ---- scoring
    def score(self, req):
        raw = np.array([getattr(req, f) for f in FEATURES], dtype=float)
        z = (raw - self.a["mean"]) / self.a["sd"]
        w, b = self.a["w"], float(self.a["b"])
        contrib = w * z
        logit = b + float(np.sum(contrib))
        prob = 1.0 / (1.0 + math.exp(-logit))
        tau = self.cost_tau if req.tau is None else req.tau
        seg = self.seg_names[int(self.assign(raw[:5][None, :])[0])]
        return {
            "probability": prob, "logit": logit, "bias": b, "tau": tau, "flagged": prob >= tau,
            "contributions": [{"feature": FEATURES[j], "x_name": c.X_NAMES[j], "value": float(raw[j]),
                               "z": float(z[j]), "weight": float(w[j]), "contribution": float(contrib[j])}
                              for j in range(7)],
            "segment": seg,
            "note": "Synthetic data. Probability from the Part 2 logistic model; not a ranking of people.",
        }


@lru_cache(maxsize=1)
def get_model():
    return Model()
