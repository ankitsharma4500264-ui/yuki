import { useCallback, useRef, useState } from "react";

export function useAgentChat(api, { selectedDocIds, projectId, preferences }) {
  const [messages, setMessages] = useState([]);
  const [activity, setActivity] = useState([]);
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const abortRef = useRef(null);
  const lastRef = useRef("");

  const run = useCallback(async (text, regenerate = false) => {
    setBusy(true); setActivity([]);
    if (!regenerate) setMessages((m) => [...m, { role: "user", content: text }]);
    else setMessages((m) => { const c = [...m]; if (c.at(-1)?.role === "assistant") c.pop(); return c; });
    const ctrl = new AbortController(); abortRef.current = ctrl; lastRef.current = text;
    try {
      const final = await api.chatStream(
        { message: text, session_id: sessionId, selected_document_ids: selectedDocIds, project_id: projectId || null, preferences, regenerate },
        { signal: ctrl.signal, onActivity: (s) => setActivity((a) => { const c = a.filter((x) => x.id !== s.id); return [...c, s].sort((x, y) => x.id - y.id); }) });
      setSessionId(final.session_id);
      setMessages((m) => [...m, { role: "assistant", ...final }]);
    } catch (e) {
      if (e.name !== "AbortError") setMessages((m) => [...m, { role: "assistant", status: "error", content: e.message, error: e.message }]);
      else setMessages((m) => [...m, { role: "assistant", status: "cancelled", content: "Generation cancelled." }]);
    } finally { setBusy(false); abortRef.current = null; }
  }, [api, sessionId, selectedDocIds, projectId, preferences]);

  return {
    messages, activity, busy, sessionId,
    send: (t) => run(t, false),
    regenerate: () => lastRef.current && run(lastRef.current, true),
    cancel: () => abortRef.current?.abort(),
    newChat: () => { setMessages([]); setSessionId(null); setActivity([]); },
    loadSession: async (id) => {
      const { messages: ms } = await api.session(id);
      setSessionId(id);
      setMessages(ms.map((m) => ({ role: m.role, content: m.content, ...(m.metadata || {}) })));
    },
    patchMessage: (index, patch) => setMessages((m) => m.map((x, i) => (i === index ? { ...x, ...patch } : x))),
  };
}
