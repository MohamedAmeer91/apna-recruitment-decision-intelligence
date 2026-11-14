export async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error("Network error: the server is not reachable.");
  }
  let body = null;
  try { body = await res.json(); } catch (e) { /* not json */ }
  if (!res.ok) {
    const er = body && body.error;
    const detail = er && er.details ? " (" + er.details.map(d => d.field + ": " + d.message).join("; ") + ")" : "";
    throw new Error((er ? er.message : "Request failed " + res.status) + detail);
  }
  return body;
}

export const post = (path, data) =>
  api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
