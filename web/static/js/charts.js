const NS = "http://www.w3.org/2000/svg";
const W = 520, H = 300, M = {l: 52, r: 14, t: 14, b: 38};

function s(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}

function ext(a, pad = 0) {
  let lo = Math.min(...a), hi = Math.max(...a);
  if (lo === hi) hi = lo + 1;
  return [lo - pad * (hi - lo), hi + pad * (hi - lo)];
}

const nice = v => (Math.abs(v) >= 100 ? String(Math.round(v)) : String(+v.toPrecision(2)));

function frame(label, xl, yl, xr, yr, xt = 5, yt = 5) {
  const svg = s("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label});
  const X = v => M.l + (v - xr[0]) / (xr[1] - xr[0]) * (W - M.l - M.r);
  const Y = v => H - M.b - (v - yr[0]) / (yr[1] - yr[0]) * (H - M.t - M.b);
  for (let i = 0; i <= yt; i++) {
    const v = yr[0] + (yr[1] - yr[0]) * i / yt;
    svg.append(s("line", {x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)"}),
      s("text", {x: M.l - 6, y: Y(v) + 4, "text-anchor": "end"}, nice(v)));
  }
  for (let i = 0; i <= xt && xt > 0; i++) {
    const v = xr[0] + (xr[1] - xr[0]) * i / xt;
    svg.append(s("text", {x: X(v), y: H - M.b + 16, "text-anchor": "middle"}, nice(v)));
  }
  svg.append(s("text", {x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle"}, xl),
    s("text", {x: 12, y: H / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${H / 2})`}, yl));
  return {svg, X, Y};
}

/** series: [{pts:[[x,y]], color, dash}]; markers: [{x,y,color}] */
export function lineChart({label, xLabel, yLabel, series, xRange, yRange, markers = []}) {
  const all = series.flatMap(q => q.pts);
  const xr = xRange || ext(all.map(p => p[0]));
  const yr = yRange || ext(all.map(p => p[1]), 0.05);
  const {svg, X, Y} = frame(label, xLabel, yLabel, xr, yr);
  for (const q of series) {
    svg.append(s("polyline", {
      points: q.pts.map(p => X(p[0]) + "," + Y(p[1])).join(" "), fill: "none",
      stroke: q.color || "var(--accent)", "stroke-width": 2, "stroke-dasharray": q.dash || "none"}));
  }
  for (const m of markers) {
    svg.append(s("circle", {cx: X(m.x), cy: Y(m.y), r: 6, fill: m.color || "var(--accent)",
      stroke: "var(--card)", "stroke-width": 2}));
  }
  return svg;
}

export function barChart({label, xLabel, yLabel, items, highlight}) {
  const ymax = Math.max(...items.map(i => i.value), 0) * 1.1 || 1;
  const {svg, X, Y} = frame(label, xLabel, yLabel, [0, items.length], [0, ymax], 0);
  const bw = (W - M.l - M.r) / items.length;
  items.forEach((it, i) => {
    svg.append(
      s("rect", {x: X(i) + bw * 0.15, y: Y(it.value), width: bw * 0.7, height: Y(0) - Y(it.value), rx: 3,
        fill: it.name === highlight ? "var(--accent)" : "var(--neutral)"}),
      s("text", {x: X(i) + bw / 2, y: H - M.b + 16, "text-anchor": "middle"}, it.name));
  });
  return svg;
}

export function scatter({label, xLabel, yLabel, points, colors}) {
  const xr = ext(points.map(p => p.x), 0.03), yr = ext(points.map(p => p.y), 0.03);
  const {svg, X, Y} = frame(label, xLabel, yLabel, xr, yr);
  for (const p of points) {
    svg.append(s("circle", {cx: X(p.x), cy: Y(p.y), r: 3, fill: colors[p.group], "fill-opacity": 0.6}));
  }
  return svg;
}
