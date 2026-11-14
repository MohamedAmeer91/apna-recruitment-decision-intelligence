import {api} from "../api.js";
import {el, card, kvTable, pick} from "../ui.js";
import {barChart} from "../charts.js";

const GROUPS = [
  ["Mean and sample size", ["n", "mean_value", "sd_value", "se_value", "ci95_low", "ci95_high", "z_stat", "reject_h0", "n_required"]],
  ["Simple regression baseline", ["reg_slope", "reg_intercept", "tss", "ess", "rss", "tss_equals_ess_plus_rss", "r2", "rmse", "mae"]],
  ["Gradient descent check", ["gd_J0", "gd_J1", "gd_J10", "gd_J100", "gd_J500", "gd_slope", "gd_intercept", "gd_slope_gap"]],
  ["Residual spread (funnel test)", ["resid_sd_lower_raw", "resid_sd_upper_raw", "resid_sd_ratio_raw", "resid_sd_lower_log", "resid_sd_upper_log", "resid_sd_ratio_log"]],
  ["Log-log and multiple regression", ["loglog_intercept", "loglog_slope", "loglog_r2", "loglog_rmse", "loglog_adj_r2", "mr_r2", "mr_adj_r2"]],
];

export async function render(root) {
  const d = await api("/api/part1");
  const r = d.results;
  root.append(
    el("h1", {}, "Part 1: days to an offer decision"),
    el("p", {class: "lead"}, "Is the planning mean of 8.0 days still right? The sample mean is " + r.mean_value.toFixed(4) +
      " with a 95% interval of " + r.ci95_low.toFixed(4) + " to " + r.ci95_high.toFixed(4) + ", so the hypothesis is rejected (z = " + r.z_stat.toFixed(4) +
      "), though the gap is under one day."),
    el("div", {class: "grid2"},
      card("Residual spread ratio, upper vs lower half",
        barChart({label: "Residual spread ratio, raw model versus log model", xLabel: "model", yLabel: "ratio",
          items: [{name: "raw", value: r.resid_sd_ratio_raw}, {name: "log-log", value: r.resid_sd_ratio_log}], highlight: "log-log"}),
        el("p", {class: "note"}, "A ratio near 1 means constant spread. The raw model is funnel-shaped; the log-log model fixes it.")),
      card("Gradient descent cost",
        barChart({label: "Gradient descent cost at selected updates", xLabel: "update", yLabel: "cost J",
          items: [["0", "gd_J0"], ["1", "gd_J1"], ["10", "gd_J10"], ["100", "gd_J100"], ["500", "gd_J500"]].map(([n, k]) => ({name: n, value: r[k]})), highlight: "500"}),
        el("p", {class: "note"}, "Cost flattens at " + r.gd_J500.toFixed(4) + ": the same minimum the closed form solves for."))),
    el("h2", {}, "Results from outputs"),
    el("div", {class: "grid2"}, GROUPS.map(([t, keys]) => card(t, kvTable(pick(r, keys))))),
    card("Multiple regression coefficients", kvTable(pick(r, Object.keys(r).filter(k => k.startsWith("mr_beta"))))),
    el("p", {class: "note"}, d.notes.join(" ")));
}
