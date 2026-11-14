import {api} from "../api.js";
import {el, card, table, kvTable, pick, legend} from "../ui.js";
import {scatter, barChart} from "../charts.js";

const COLORS = {A: "var(--seg-a)", B: "var(--seg-b)", C: "var(--seg-c)"};
const AXES = {x1: "experience (months)", x2: "interview reschedules", x3: "salary gap ratio", x4: "expected salary (INR thousands)", x5: "profile completeness"};

export async function render(root) {
  const [seg, pts, p3] = await Promise.all([api("/api/segments"), api("/api/segments/points"), api("/api/part3")]);
  root.append(el("h1", {}, "Part 3: candidate segments"),
    el("p", {class: "lead"}, "K-means on five standardised features, K = " + seg.elbow_k + " chosen by the elbow rule. Segment C declines most often; segment A almost never."));

  // segment explorer
  const sel = el("div", {class: "grid2"});
  const mk = id => el("select", {id, "aria-label": id === "ax" ? "Horizontal axis" : "Vertical axis"},
    Object.entries(AXES).map(([k, v]) => el("option", {value: k, selected: (id === "ax" ? "x1" : "x3") === k}, v)));
  const ax = mk("ax"), ay = mk("ay");
  sel.append(el("div", {}, el("label", {for: "ax"}, "Horizontal axis"), ax), el("div", {}, el("label", {for: "ay"}, "Vertical axis"), ay));
  const chart = el("div");
  const filter = el("select", {id: "seg", "aria-label": "Show segment"}, el("option", {value: "all"}, "All segments"),
    seg.segments.map(s => el("option", {value: s.segment}, "Segment " + s.segment)));
  const detail = el("div");
  function draw() {
    const show = filter.value;
    const data = pts.points.filter(p => show === "all" || p.segment === show).map(p => ({x: p[ax.value], y: p[ay.value], group: p.segment}));
    chart.replaceChildren(scatter({label: "Records plotted by " + AXES[ax.value] + " and " + AXES[ay.value] + ", coloured by segment",
      xLabel: AXES[ax.value], yLabel: AXES[ay.value], points: data, colors: COLORS}),
      legend(seg.segments.map(s => ({name: "Segment " + s.segment, color: COLORS[s.segment]}))));
    const s = seg.segments.find(x => x.segment === show);
    detail.replaceChildren(s ? el("div", {}, el("h3", {}, "Segment " + s.segment + ": " + s.label),
      kvTable(pick(s, ["size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"]))) :
      el("p", {class: "note"}, "Pick one segment to see its profile. Points are assigned to the nearest segment centroid in standardised space; the counts match part3_results (322, 255, 223)."));
  }
  [ax, ay, filter].forEach(s => s.addEventListener("change", draw));
  root.append(el("h2", {}, "Segment explorer"),
    el("div", {class: "grid2"}, card("Records", sel, el("label", {for: "seg"}, "Segment"), filter, chart), card("Profile", detail)));
  draw();

  root.append(el("h2", {}, "Segment table"), card("", table(seg.segments, ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])),
    el("h2", {}, "Choosing K and compressing features"),
    el("div", {class: "grid2"},
      card("Within-cluster sum of squares (elbow at K = " + seg.elbow_k + ")",
        barChart({label: "WCSS by K", xLabel: "K", yLabel: "WCSS", items: seg.elbow.map(e => ({name: String(e.K), value: e.wcss})), highlight: String(seg.elbow_k)})),
      card("PCA explained variance ratio",
        barChart({label: "Explained variance ratio by principal component", xLabel: "component", yLabel: "ratio",
          items: seg.pca.explained.map((v, i) => ({name: "PC" + (i + 1), value: v})), highlight: "PC1"}),
        el("p", {class: "note"}, p3.results.components_for_80pct + " components keep at least 80% of the variance."))),
    card("PCA loadings", table(["X1", "X2", "X3", "X4", "X5"].map((n, j) => ({feature: n, pc1: seg.pca.loadings.pc1[j], pc2: seg.pca.loadings.pc2[j]})))),
    el("h2", {}, "Hierarchical clustering (supplied 6-cohort table)"),
    el("div", {class: "grid2"},
      card("Clusters at the largest-gap cut", kvTable(pick(p3.results, ["single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut"]))),
      card("Merge heights", table(p3.tables.hierarchical))));
}
