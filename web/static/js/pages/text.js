import {api} from "../api.js";
import {el, card} from "../ui.js";

export async function render(root, url) {
  const d = await api(url);
  root.append(el("h1", {}, d.title), card("", d.paragraphs.map(p => el("p", {}, p))));
}
