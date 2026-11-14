import {el, errorBox} from "./ui.js";

const ROUTES = [
  ["overview", "Overview", () => import("./pages/overview.js")],
  ["part1", "Part 1", () => import("./pages/part1.js")],
  ["part2", "Part 2 Model Lab", () => import("./pages/part2.js")],
  ["part3", "Part 3 Segments", () => import("./pages/part3.js")],
  ["part4", "Part 4 Ensembles", () => import("./pages/part4.js")],
  ["score", "Score a record", () => import("./pages/score.js")],
  ["memo", "Memo", () => import("./pages/text.js").then(m => ({render: r => m.render(r, "/api/memo")}))],
  ["limits", "Limitations", () => import("./pages/text.js").then(m => ({render: r => m.render(r, "/api/limitations")}))],
];

const nav = document.getElementById("nav");
const app = document.getElementById("app");

nav.append(...ROUTES.map(([id, label]) => el("a", {href: "#/" + id, "data-id": id}, label)));

async function route() {
  const id = (location.hash.replace("#/", "") || "overview");
  const found = ROUTES.find(r => r[0] === id) || ROUTES[0];
  nav.querySelectorAll("a").forEach(a => {
    if (a.dataset.id === found[0]) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  app.replaceChildren(el("p", {}, "Loading..."));
  try {
    const mod = await found[2]();
    const root = el("div");
    app.replaceChildren(root);
    await mod.render(root);
    document.title = found[1] + " | Apna Recruitment Decision Intelligence";
  } catch (e) {
    app.replaceChildren(errorBox(e));
  }
}

window.addEventListener("hashchange", route);
route();
