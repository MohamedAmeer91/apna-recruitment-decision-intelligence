import {api} from "../api.js";
import {el, card, kvTable, table, inr, fmt, debounce, errorBox, pick} from "../ui.js";
import {lineChart, barChart} from "../charts.js";

export async function render(root) {
  const [p2, curve] = await Promise.all([api("/api/part2"), api("/api/curve")]);
  const r = p2.results;
  const names = ["X1 experience", "X2 reschedules", "X3 salary gap", "X4 salary", "X5 profile", "X6 competing", "X7 referred"];
  root.append(
    el("h1", {}, "Part 2: who will decline an offer?"),
    el("p", {class: "lead"}, "A logistic regression built from scratch, trained on 560 rows and tested on 240. Cost rule from the brief: INR 50 per unneeded follow-up (false positive), INR 400 per missed decline (false negative)."),
    el("h2", {}, "Threshold lab"));

  // ---- threshold lab
  const out = el("div", {"aria-live": "polite"});
  const err = el("div");
  const slider = el("input", {type: "range", id: "tau", min: "0", max: "1", step: "0.01", value: "0.5", "aria-label": "Alert threshold"});
  const tauLabel = el("b", {}, "0.50");
  const rocBox = el("div"), costBox = el("div");
  const setTau = v => { slider.value = v; update(); };
  const controls = card("Alert threshold",
    el("label", {for: "tau"}, "Flag a candidate when predicted probability is at least ", tauLabel),
    slider,
    el("div", {style: "display:flex;gap:8px;flex-wrap:wrap;margin-top:8px"},
      el("button", {class: "sec", onclick: () => setTau(0.5)}, "0.50"),
      el("button", {class: "sec", onclick: () => setTau(0.3)}, "0.30"),
      el("button", {class: "sec", onclick: () => setTau(curve.cost_tau_train)}, "Cost-chosen " + curve.cost_tau_train.toFixed(2))),
    el("p", {class: "note"}, "AUC of the ROC curve on test rows: " + curve.auc.toFixed(4) + " (derived view, not a graded number)."),
    err, out);
  const pts = curve.points;
  async function update() {
    const tau = parseFloat(slider.value);
    tauLabel.textContent = tau.toFixed(2);
    try {
      const t = await api("/api/threshold?tau=" + tau);
      err.replaceChildren();
      out.replaceChildren(
        el("div", {class: "cm"},
          el("div", {class: "h"}), el("div", {class: "h"}, "Predicted decline"), el("div", {class: "h"}, "Predicted accept"),
          el("div", {class: "h"}, "Declined"), el("div", {}, el("span", {class: "big"}, t.tp), " TP"), el("div", {}, el("span", {class: "big"}, t.fn), " FN"),
          el("div", {class: "h"}, "Accepted"), el("div", {}, el("span", {class: "big"}, t.fp), " FP"), el("div", {}, el("span", {class: "big"}, t.tn), " TN")),
        el("div", {class: "metrics"}, [["Precision", t.precision], ["Recall", t.recall], ["F1", t.f1], ["Accuracy", t.accuracy], ["Specificity", t.specificity]]
          .map(([l, v]) => el("div", {}, el("b", {}, v.toFixed(4)), el("span", {}, l)))),
        el("div", {class: "metrics"},
          el("div", {}, el("b", {class: "bad"}, inr(t.cost_inr)), el("span", {}, "cost at this threshold")),
          el("div", {}, el("b", {}, inr(t.cost_flag_nobody_inr)), el("span", {}, "cost flagging nobody")),
          el("div", {}, el("b", {class: t.saving_vs_nobody_inr >= 0 ? "good" : "bad"}, inr(t.saving_vs_nobody_inr)), el("span", {}, "saving vs nobody"))));
      rocBox.replaceChildren(lineChart({
        label: "ROC curve, true positive rate against false positive rate", xLabel: "false positive rate", yLabel: "true positive rate",
        xRange: [0, 1], yRange: [0, 1],
        series: [{pts: [[0, 0], [1, 1]], color: "var(--neutral)", dash: "4 4"}, {pts: pts.map(p => [p.fpr, p.tpr])}],
        markers: [{x: t.fpr, y: t.recall}]}));
      costBox.replaceChildren(lineChart({
        label: "Cost in INR against threshold", xLabel: "threshold", yLabel: "cost (INR)", xRange: [0, 1],
        series: [{pts: pts.map(p => [p.tau, p.cost_inr])}],
        markers: [{x: tau, y: t.cost_inr, color: "var(--bad)"}]}));
    } catch (e) { err.replaceChildren(errorBox(e)); }
  }
  slider.addEventListener("input", debounce(update, 60));
  root.append(el("div", {class: "grid2"}, controls,
    el("div", {}, card("ROC curve", rocBox), el("div", {style: "height:12px"}), card("Cost curve (test rows)", costBox))));
  await update();

  // ---- KNN selector
  const knnOut = el("div", {"aria-live": "polite"});
  const sel = el("select", {id: "k", "aria-label": "K nearest neighbours"},
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21].map(k => el("option", {value: k, selected: k === 5}, "K = " + k)));
  async function updateKnn() {
    try {
      const k = await api("/api/knn?k=" + sel.value);
      knnOut.replaceChildren(
        el("div", {class: "metrics"}, [["TP", k.tp], ["FP", k.fp], ["TN", k.tn], ["FN", k.fn]].map(([l, v]) => el("div", {}, el("b", {}, v), el("span", {}, l)))),
        el("div", {class: "metrics"}, [["Precision", k.precision], ["Recall", k.recall], ["F1", k.f1]].map(([l, v]) => el("div", {}, el("b", {}, v.toFixed(4)), el("span", {}, l))),
          el("div", {}, el("b", {}, inr(k.cost_inr)), el("span", {}, "cost on test"))),
        el("p", {class: "note"}, k.is_best ? "K = " + k.k + " is the cross-validated best." : "Cross-validated best is K = " + k.best_k + "."),
        barChart({label: "Cross-validated F1 by K", xLabel: "K", yLabel: "mean CV F1",
          items: k.cv.map(c => ({name: String(c.K), value: c.mean_cv_f1})), highlight: String(k.k)}));
    } catch (e) { knnOut.replaceChildren(errorBox(e)); }
  }
  sel.addEventListener("change", updateKnn);
  root.append(el("h2", {}, "KNN selector"),
    card("K nearest neighbours (standardised features, test rows)", el("label", {for: "k"}, "Choose K"), sel, knnOut));
  await updateKnn();

  // ---- static results
  const weights = names.map((n, i) => ({feature: n, weight: r["weight_X" + (i + 1)], odds_ratio: r["odds_ratio_X" + (i + 1)]}));
  root.append(el("h2", {}, "Trained model"),
    el("div", {class: "grid2"},
      card("Weights and odds ratios (per training SD)", table(weights, ["feature", "weight", "odds_ratio"]), el("p", {class: "note"}, "Bias " + fmt(r.bias) + ". Test log loss " + fmt(r.test_bce) + ".")),
      card("Learning rate comparison (loss after 200 updates)", kvTable(pick(r, ["loss_epoch_200_eta_0.01", "loss_epoch_200_eta_0.1", "loss_epoch_200_eta_1.0"])))),
    el("h2", {}, "Committed results"),
    el("div", {class: "grid2"},
      card("Threshold grid (part2_thresholds.csv)", table(p2.tables.thresholds)),
      card("Cost-chosen threshold", kvTable(pick(r, ["cost_tau", "cost_train_at_tau", "cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody", "blind_accuracy"])))));
}
