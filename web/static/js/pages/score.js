import {post} from "../api.js";
import {el, card, table, errorBox} from "../ui.js";

const FIELDS = [
  ["experience_months", "Experience (months)", 0, 600, 1, 24],
  ["interview_reschedules_90d", "Interview reschedules, last 90 days", 0, 30, 1, 4],
  ["salary_gap_ratio", "Salary gap ratio (0 to 1)", 0, 1, 0.01, 0.5],
  ["expected_salary_inr_thousands", "Expected salary (INR thousands)", 1, 1000, 0.1, 18],
  ["profile_completeness_score", "Profile completeness (0 to 100)", 0, 100, 1, 50],
];
const FLAGS = [["competing_offer_flag", "Has a competing offer"], ["referred_flag", "Was referred"]];

export async function render(root) {
  const out = el("div", {"aria-live": "polite"});
  const inputs = {};
  const form = el("form", {novalidate: "novalidate"});
  const grid = el("div", {class: "fields"});
  for (const [id, label, min, max, step, def] of FIELDS) {
    inputs[id] = el("input", {type: "number", id, min, max, step, value: def, required: "required"});
    grid.append(el("div", {}, el("label", {for: id}, label), inputs[id]));
  }
  for (const [id, label] of FLAGS) {
    inputs[id] = el("select", {id}, el("option", {value: "0"}, "No"), el("option", {value: "1"}, "Yes"));
    grid.append(el("div", {}, el("label", {for: id}, label), inputs[id]));
  }
  inputs.tau = el("input", {type: "number", id: "tau", min: 0, max: 1, step: 0.01, placeholder: "0.15 (cost-chosen)"});
  grid.append(el("div", {}, el("label", {for: "tau"}, "Alert threshold (optional)"), inputs.tau));
  form.append(grid, el("div", {style: "margin-top:12px"}, el("button", {type: "submit"}, "Score record")));
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const body = {};
    for (const [id] of FIELDS) body[id] = parseFloat(inputs[id].value);
    for (const [id] of FLAGS) body[id] = parseInt(inputs[id].value, 10);
    if (inputs.tau.value !== "") body.tau = parseFloat(inputs.tau.value);
    if (Object.entries(body).some(([, v]) => Number.isNaN(v))) { out.replaceChildren(errorBox(new Error("Please fill every field with a number."))); return; }
    try {
      const d = await post("/api/score", body);
      out.replaceChildren(
        el("div", {class: "metrics"},
          el("div", {}, el("b", {}, (d.probability * 100).toFixed(1) + "%"), el("span", {}, "probability of declining")),
          el("div", {}, el("b", {class: d.flagged ? "bad" : "good"}, d.flagged ? "Follow up" : "No alert"), el("span", {}, "at threshold " + d.tau.toFixed(2))),
          el("div", {}, el("b", {}, "Segment " + d.segment), el("span", {}, "nearest candidate group"))),
        el("h3", {style: "margin-top:16px"}, "What moved the score (log-odds)"),
        table(d.contributions.map(c => ({feature: c.x_name + " " + c.feature, value: c.value, z: c.z, weight: c.weight, contribution: c.contribution})),
          ["feature", "value", "z", "weight", "contribution"]),
        el("p", {class: "note"}, "Bias " + d.bias.toFixed(4) + ", total logit " + d.logit.toFixed(4) + ". " + d.note));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  });
  root.append(el("h1", {}, "Score a record"),
    el("p", {class: "lead"}, "Enter a candidate record and get the Part 2 logistic model's probability of declining an offer. Inputs are standardised with the training mean and SD, as in the project."),
    card("", form), el("div", {style: "height:12px"}), card("Result", out));
}
