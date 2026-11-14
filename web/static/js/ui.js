export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

export const fmt = (v, d = 4) =>
  typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v);

export const inr = v => "INR " + Math.round(v).toLocaleString("en-IN");

export function table(rows, cols, labels) {
  cols = cols || Object.keys(rows[0] || {});
  const head = el("thead", {}, el("tr", {}, cols.map(c => el("th", {scope: "col"}, labels ? (labels[c] || c) : c))));
  const body = el("tbody", {}, rows.map(r => el("tr", {}, cols.map(c => el("td", {}, fmt(r[c]))))));
  return el("div", {class: "scroll"}, el("table", {}, head, body));
}

export const kvTable = (obj, keys) =>
  table((keys || Object.keys(obj)).map(k => ({key: k, value: obj[k]})), ["key", "value"]);

export const card = (title, ...kids) =>
  el("section", {class: "card"}, title ? el("h3", {}, title) : null, kids);

export const errorBox = e => el("div", {class: "alert", role: "alert"}, e.message || String(e));

export function debounce(fn, ms = 120) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export const pick = (res, keys) => Object.fromEntries(keys.filter(k => k in res).map(k => [k, res[k]]));

export function legend(items) {
  return el("div", {class: "legend"}, items.map(i =>
    el("span", {}, el("span", {class: "dot", style: "background:" + i.color}), i.name)));
}
