let passcode = "";
try { passcode = localStorage.getItem("vq_pass") || ""; } catch {}
export const setPasscode = (p: string) => { passcode = p; try { localStorage.setItem("vq_pass", p); } catch {} };

export class ApiError extends Error { status: number; constructor(status: number, msg: string) { super(msg); this.status = status; } }

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as any) };
  if (passcode) headers["x-passcode"] = passcode;
  if (opts.body && !(opts.body instanceof FormData)) headers["content-type"] = "application/json";
  const r = await fetch(path, { ...opts, headers });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new ApiError(r.status, msg);
  }
  return r.json();
}

/** Server-sent events. Returns a function that closes the stream. */
export function stream(path: string, onEvent: (e: any) => void, onEnd: () => void): () => void {
  const url = path + (path.includes("?") ? "&" : "?") + (passcode ? `passcode=${encodeURIComponent(passcode)}` : "");
  const es = new EventSource(url);
  let ended = false;
  const end = () => { if (!ended) { ended = true; es.close(); onEnd(); } };
  es.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "done") { onEvent(e); end(); } else onEvent(e);
  };
  es.onerror = () => end();
  return () => { ended = true; es.close(); };
}
