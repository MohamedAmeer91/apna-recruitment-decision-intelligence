import {api} from "../api.js";
import {el, card, table, kvTable, pick, inr, errorBox} from "../ui.js";
import {barChart} from "../charts.js";

export async function render(root) {
  const p4 = await api("/api/part4");
  const r = p4.results;
  root.append(el("h1", {}, "Part 4: comparing models"),
    el("p", {class: "lead"}, "Which split is purest, does voting help, how does boosting reweight points, and how big would later neural networks be?"));

  // split comparison
  const metric = el("select", {id: "metric", "aria-label": "Split metric"},
    el("option", {value: "info_gain"}, "Information gain"), el("option", {value: "gini_gain"}, "Gini gain"));
  const splitChart = el("div");
  const drawSplit = () => splitChart.replaceChildren(barChart({label: "Gain by candidate split", xLabel: "split", yLabel: metric.value.replace("_", " "),
    items: p4.tables.splits.map(s => ({name: s.split, value: s[metric.value]})),
    highlight: r[metric.value === "info_gain" ? "best_split_by_info_gain" : "best_split_by_gini_gain"]}));
  metric.addEventListener("change", drawSplit);
  root.append(el("h2", {}, "Split comparison"),
    el("div", {class: "grid2"},
      card("Gain by split", el("label", {for: "metric"}, "Metric"), metric, splitChart,
        el("p", {class: "note"}, "Best split by both measures: " + r.best_split_by_info_gain + " (experience at most 24 months). Root entropy " + r.root_entropy.toFixed(4) + ", root Gini " + r.root_gini.toFixed(4) + ".")),
      card("Candidate splits (training rows)", table(p4.tables.splits))));
  drawSplit();

  // ensemble voting lab
  const out = el("div", {"aria-live": "polite"});
  const slider = el("input", {type: "range", id: "etau", min: "0.05", max: "0.95", step: "0.01", value: "0.3", "aria-label": "Ensemble threshold"});
  const lbl = el("b", {}, "0.30");
  async function update() {
    const tau = parseFloat(slider.value);
    lbl.textContent = tau.toFixed(2);
    try {
      const d = await api("/api/ensemble?tau=" + tau);
      const best = d.models.reduce((a, b) => (b.cost_inr < a.cost_inr ? b : a));
      out.replaceChildren(table(d.models, ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "cost_inr"]),
        barChart({label: "Cost in INR by model", xLabel: "model", yLabel: "cost (INR)",
          items: d.models.map(m => ({name: m.model.replace("_vote", ""), value: m.cost_inr})), highlight: best.model.replace("_vote", "")}),
        el("p", {class: "note"}, "Lowest cost at this threshold: " + best.model + " (" + inr(best.cost_inr) + "). KNN uses its own majority vote; logistic and stump use the threshold."));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  }
  slider.addEventListener("input", () => update());
  root.append(el("h2", {}, "Ensemble voting lab"),
    card("Logistic, KNN, stump, hard and soft vote on test rows",
      el("label", {for: "etau"}, "Threshold ", lbl), slider, out),
    el("p", {class: "note"}, "At 0.30 the table equals part4_voting.csv."));
  await update();

  // adaboost
  const rounds = [1, 2].map(n => p4.tables.adaboost.filter(x => x.round === n));
  root.append(el("h2", {}, "AdaBoost by hand (8 points, 2 rounds)"),
    el("div", {class: "grid2"}, rounds.map((rows, i) => card("Round " + (i + 1), table(rows, ["point", "weight_before", "weight_unnormalised", "weight_after"])))),
    card("Boosting numbers", kvTable(pick(r, ["round1_error", "round1_alpha", "round2_error", "round2_alpha", "ensemble_accuracy", "oob_fraction"]))),
    el("h2", {}, "Sizing later neural networks (parameter counts)"),
    el("div", {class: "grid2"},
      card("Layer parameters", table(p4.tables.architecture)),
      card("Totals", kvTable(pick(r, ["mlp_total", "cnn_total", "fc_layer_params", "conv_layer_params", "rnn_total"])))));
}
