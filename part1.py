"""Part 1: estimation, inference and regression baseline (all 800 rows)."""
import math

import numpy as np

import common as c


def r2_of(y, fitted):
    my = np.sum(y) / len(y)
    return 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - my) ** 2)


def half_sds(fitted, resid):
    order = np.argsort(fitted, kind="stable")
    lo = c.sample_sd(resid[order[:400]])
    hi = c.sample_sd(resid[order[400:]])
    return lo, hi, hi / lo


def main():
    d = c.load_data()
    X, y = d["X"], d["value"]
    n = len(y)
    x4 = X[:, 3]
    out = []
    # 1.1
    mean = np.sum(y) / n
    sd = c.sample_sd(y)
    se = sd / math.sqrt(n)
    out += [("n", n), ("mean_value", mean), ("sd_value", sd), ("se_value", se),
            ("ci95_low", mean - 1.96 * se), ("ci95_high", mean + 1.96 * se)]
    # 1.2
    z = (mean - 8.0) / se
    out += [("z_stat", z), ("reject_h0", abs(z) > 1.96)]
    # 1.3
    out += [("n_required", int(math.ceil((1.96 * sd / 0.5) ** 2)))]
    # 1.4
    fit = c.ols_closed_form(x4, y)
    f = fit["fitted"]
    tss = np.sum((y - mean) ** 2)
    ess = np.sum((f - mean) ** 2)
    rss = np.sum((y - f) ** 2)
    out += [("reg_slope", fit["slope"]), ("reg_intercept", fit["intercept"]), ("tss", tss),
            ("ess", ess), ("rss", rss), ("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss),
            ("r2", 1 - rss / tss), ("rmse", math.sqrt(rss / n)), ("mae", np.sum(np.abs(y - f)) / n)]
    # 1.5
    mx = np.sum(x4) / n
    sx = c.sample_sd(x4)
    zz = (x4 - mx) / sx
    m = 0.0
    b = 0.0
    J = {}
    for t in range(0, 501):
        e = y - (m * zz + b)
        if t in (0, 1, 10, 100, 500):
            J[t] = np.sum(e ** 2) / n
        if t == 500:
            break
        gm = -(2 / n) * np.sum(zz * e)
        gb = -(2 / n) * np.sum(e)
        m -= 0.1 * gm
        b -= 0.1 * gb
    gd_slope = m / sx
    gd_int = b - m * mx / sx
    out += [("gd_J0", J[0]), ("gd_J1", J[1]), ("gd_J10", J[10]), ("gd_J100", J[100]), ("gd_J500", J[500]),
            ("gd_slope", gd_slope), ("gd_intercept", gd_int), ("gd_slope_gap", abs(gd_slope - fit["slope"]))]
    # 1.6
    lo, hi, ratio = half_sds(f, y - f)
    out += [("resid_sd_lower_raw", lo), ("resid_sd_upper_raw", hi), ("resid_sd_ratio_raw", ratio)]
    ly = np.log(y)
    lfit = c.ols_closed_form(np.log(x4), ly)
    lf = lfit["fitted"]
    lrss = np.sum((ly - lf) ** 2)
    ll_r2 = r2_of(ly, lf)
    out += [("loglog_intercept", lfit["intercept"]), ("loglog_slope", lfit["slope"]),
            ("loglog_r2", ll_r2), ("loglog_rmse", math.sqrt(lrss / n))]
    lo, hi, ratio = half_sds(lf, ly - lf)
    out += [("resid_sd_lower_log", lo), ("resid_sd_upper_log", hi), ("resid_sd_ratio_log", ratio)]
    # 1.7
    A = np.column_stack([np.ones(n), np.log(x4), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(A.T @ A, A.T @ ly)
    r2 = r2_of(ly, A @ beta)
    out += [(f"mr_beta_{i}", beta[i]) for i in range(8)]
    out += [("mr_r2", r2), ("mr_adj_r2", 1 - (1 - r2) * (n - 1) / (n - 7 - 1)),
            ("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))]
    c.write_results("part1_results.txt", out)


if __name__ == "__main__":
    main()
