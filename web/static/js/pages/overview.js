import {api} from "../api.js";
import {el} from "../ui.js";

export async function render(root) {
  const d = await api("/api/overview");
  root.append(
    el("h1", {}, d.title),
    el("p", {class: "lead"}, "A recruitment analytics capstone in four parts: estimation and regression, a classifier chosen with money, segmentation, and model comparison. This site shows the committed results and lets you probe the model."),
    el("p", {class: "note"}, d.synthetic_notice),
    el("h2", {}, "Three questions"),
    el("ol", {}, d.questions.map(q => el("li", {}, q))),
    el("h2", {}, "Headline numbers"),
    el("div", {class: "grid"}, d.kpis.map(k =>
      el("div", {class: "card kpi"}, el("div", {class: "v"}, k.value), el("div", {class: "l"}, k.label), el("div", {class: "n"}, k.note)))),
    el("h2", {}, "Key findings"),
    el("ul", {class: "find"}, d.findings.map(f => el("li", {}, el("b", {}, "Part " + f.part + ": "), f.text))),
    el("h2", {}, "Explore"),
    el("p", {}, "Open Part 2 Model Lab to drag the alert threshold and watch the confusion matrix, ROC curve and INR cost update, then try Score a record."),
  );
}
