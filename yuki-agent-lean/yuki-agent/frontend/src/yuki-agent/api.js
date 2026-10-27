// Client for the YUKI agent API. Never put an xAI/Grok key in the browser: it lives on the server only.
// Public deployments use signed anonymous tokens (POST /api/auth/anonymous) kept in localStorage.
const KEY = "yuki_token";
let pending = null;

async function ensureToken(baseUrl, force = false) {
  if (!force) { const t = localStorage.getItem(KEY); if (t) return t; }
  pending ||= fetch(baseUrl + "/api/auth/anonymous", { method: "POST" })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d?.token) { localStorage.setItem(KEY, d.token); return d.token; } return null; })
    .catch(() => null)
    .finally(() => { pending = null; });
  return pending;
}

export function createAgentApi({ baseUrl = "", token } = {}) {
  const doFetch = async (path, init = {}, retry = true) => {
    const tok = token || (await ensureToken(baseUrl));
    const r = await fetch(baseUrl + path, { ...init, headers: { ...(init.headers || {}), ...(tok ? { Authorization: `Bearer ${tok}` } : {}) } });
    if (r.status === 401 && retry && !token) { localStorage.removeItem(KEY); await ensureToken(baseUrl, true); return doFetch(path, init, false); }
    return r;
  };
  const fail = async (r, fallback) => { throw new Error((await r.json().catch(() => ({}))).detail || `${fallback} (${r.status})`); };
  const json = async (path, opts = {}) => {
    const r = await doFetch(path, { ...opts, headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
    if (!r.ok) await fail(r, "Request failed");
    return r.json();
  };
  return {
    health: () => json("/api/health"),
    documents: () => json("/api/documents"),
    projects: () => json("/api/projects"),
    createProject: (name) => json("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
    sessions: () => json("/api/agent/sessions"),
    session: (id) => json(`/api/agent/sessions/${id}`),
    deleteSession: (id) => json(`/api/agent/sessions/${id}`, { method: "DELETE" }),
    clearHistory: () => json("/api/agent/sessions", { method: "DELETE" }),
    confirm: (pending_id, approve) => json("/api/agent/confirm", { method: "POST", body: JSON.stringify({ pending_id, approve }) }),
    async upload(file) {
      const fd = new FormData(); fd.append("file", file);
      const r = await doFetch("/api/documents", { method: "POST", body: fd });
      if (!r.ok) await fail(r, "Upload failed");
      return r.json();
    },
    // ---- voice (server-side Grok STT/TTS; the hook falls back to the browser's Web Speech API)
    voiceConfig: () => json("/api/voice/config"),
    async transcribe(wavBlob, language) {
      const fd = new FormData(); fd.append("language", language || ""); fd.append("file", wavBlob, "speech.wav");
      const r = await doFetch("/api/voice/transcribe", { method: "POST", body: fd });
      if (!r.ok) await fail(r, "Transcription failed");
      return r.json();
    },
    async speak(text, language) {
      const r = await doFetch("/api/voice/speak", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, language }) });
      if (!r.ok) await fail(r, "Speech failed");
      return r.blob();
    },
    // Streams `activity` events as they really happen, then one `final` event. Abort with the signal to cancel.
    async chatStream(body, { onActivity, signal }) {
      const r = await doFetch("/api/agent/chat/stream", { method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) await fail(r, "Request failed");
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "", final = null;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          const ev = /^event: (.+)$/m.exec(block)?.[1], data = /^data: (.+)$/m.exec(block)?.[1];
          if (!ev || !data) continue;
          const payload = JSON.parse(data);
          if (ev === "activity") onActivity?.(payload);
          else if (ev === "final") final = payload;
          else if (ev === "error") throw new Error(payload.error);
        }
      }
      if (!final) throw new Error("The connection closed before an answer arrived.");
      return final;
    },
  };
}
