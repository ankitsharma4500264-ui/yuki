"use client";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createAgentApi } from "./api";
import { useAgentChat } from "./useAgentChat";
import { useVoice } from "./useVoice";
import { Rich } from "./markdown";
import "./agent.css";

const TOOLS = [
  { icon: "💡", label: "Explain", color: "#ff9f1c", prompt: "Explain this chapter in simple English" },
  { icon: "📝", label: "Summarize", color: "#2ec4b6", prompt: "Summarize my notes" },
  { icon: "🎯", label: "Quiz me", color: "#ff5d8f", prompt: "Generate 10 MCQs on this topic" },
  { icon: "🃏", label: "Flashcards", color: "#8e6bff", prompt: "Make 10 flashcards" },
  { icon: "⚖️", label: "Compare", color: "#3a86ff", prompt: "Compare these two papers" },
  { icon: "🔍", label: "Find gaps", color: "#06d6a0", prompt: "What are the knowledge gaps here?" },
  { icon: "🚀", label: "Ideas", color: "#f15bb5", prompt: "Give me a research idea about RAG" },
];
const LANGS = [["auto", "Auto"], ["en", "EN"], ["hi", "हिं"]];

function Switch({ checked, onChange, icon, label }) {
  return (
    <label className="ya-switch"><span>{icon} {label}</span>
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} /><i className="ya-knob" /></label>
  );
}

// Markdown + clickable [S1] citation chips (see markdown.jsx; no raw HTML is ever injected).
function Answer({ text }) {
  return <div className="ya-answer"><Rich text={text} /></div>;
}

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <details className="ya-sources" open>
      <summary>Sources ({sources.filter((s) => s.cited).length} cited, {sources.length} retrieved)</summary>
      {[...sources].sort((a, b) => b.cited - a.cited).map((s) => (
        <div key={s.label} id={`ya-src-${s.label}`} className={"ya-card" + (s.cited ? "" : " dim")}>
          <b>[{s.label}] {s.filename}</b>{s.page ? ` · p.${s.page}` : ""}{s.score != null ? ` · score ${s.score}` : ""}
          <p>{s.snippet}{s.snippet?.length >= 500 ? "…" : ""}</p>
        </div>))}
    </details>
  );
}

function Activity({ steps, live }) {
  if (!steps?.length) return null;
  return (
    <details className="ya-activity" open={live}>
      <summary>{live ? "Working…" : `Activity (${steps.length} steps)`}</summary>
      <ul>{steps.map((s) => (
        <li key={s.id} className={s.status}>
          <span className="dot" /> {s.label}{s.detail ? ` — ${s.detail}` : ""}{s.duration_ms ? ` (${s.duration_ms} ms)` : ""}
        </li>))}</ul>
    </details>
  );
}

function Flashcards({ cards }) {
  const [open, setOpen] = useState({});
  return (
    <div className="ya-grid">{cards.map((c, i) => (
      <button key={i} style={{ "--i": i }} className={"ya-flash" + (open[i] ? " flip" : "")} onClick={() => setOpen((o) => ({ ...o, [i]: !o[i] }))}>
        <small>{open[i] ? "Answer" : "Question"}{c.source ? ` · ${c.source}` : ""}</small>
        <div>{open[i] ? c.answer : c.question}</div>
      </button>))}</div>
  );
}

function Quiz({ questions }) {
  const [picked, setPicked] = useState({}); const [shown, setShown] = useState({});
  return (
    <div className="ya-quiz">{questions.map((q, i) => (
      <div key={i} className="ya-card">
        <b>Q{i + 1}. {q.question}</b>
        {q.type === "mcq" && q.options.map((o, j) => (
          <label key={j} className={shown[i] ? (j === q.answer_index ? "ok" : picked[i] === j ? "bad" : "") : ""}>
            <input type="radio" name={`q${i}`} onChange={() => setPicked((p) => ({ ...p, [i]: j }))} /> {"ABCD"[j]}. {o}
          </label>))}
        <button className="ya-btn grape" onClick={() => setShown((s) => ({ ...s, [i]: true }))}>✅ Check answer</button>
        {shown[i] && q.type === "mcq" && <span className={"ya-pop " + (picked[i] === q.answer_index ? "ok" : "bad")}>{picked[i] === q.answer_index ? "🎉 Great job!" : "💪 Try again next time!"}</span>}
        {shown[i] && <p className="ya-expl">{q.type === "mcq" ? `Answer: ${"ABCD"[q.answer_index]}. ` : `Answer: ${q.answer}. `}{q.explanation}{q.source ? ` [${q.source}]` : ""}</p>}
      </div>))}</div>
  );
}

function Message({ m, index, api, onPatch, onRegenerate, isLast, onSpeak, canSpeak }) {
  if (m.role === "user") return <div className="ya-msg user">{m.content}</div>;
  const study = m.structured?.flashcards || m.structured?.questions;
  const resolveAction = async (approve) => {
    try { const r = await api.confirm(m.pending_action.id, approve); onPatch(index, { pending_action: null, answer: r.answer, content: r.answer, status: r.status, structured: r.structured, warnings: [] }); }
    catch (e) { onPatch(index, { pending_action: null, answer: e.message, content: e.message, status: "error" }); }
  };
  return (
    <div className={`ya-msg bot ${m.status === "error" ? "err" : ""}`}><span className="ya-avatar">🦊</span>
      {study
        ? <div className="ya-note">{m.structured.flashcards ? `${m.structured.flashcards.length} flashcards. Tap a card to flip it.`
            : `${m.structured.questions.length} questions. The answer key stays hidden until you check each question.`}</div>
        : <Answer text={m.answer || m.content} />}
      {m.structured?.flashcards && <Flashcards cards={m.structured.flashcards} />}
      {m.structured?.questions && <Quiz questions={m.structured.questions} />}
      {m.pending_action && (
        <div className="ya-confirm"><button className="ya-btn primary" onClick={() => resolveAction(true)}>💾 Save note</button>
          <button className="ya-btn" onClick={() => resolveAction(false)}>✖ Cancel</button></div>)}
      {m.warnings?.map((w, i) => <div key={i} className="ya-warn">⚠ {w}</div>)}
      <Sources sources={m.sources} />
      <Activity steps={m.tool_activity} />
      <div className="ya-actions">
        <button className="ya-btn" onClick={() => navigator.clipboard.writeText(m.answer || m.content)}>📋 Copy</button>
        {canSpeak && <button className="ya-btn" onClick={() => onSpeak(m.answer || m.content)}>🔊 Listen</button>}
        {isLast && <button className="ya-btn" onClick={onRegenerate}>🔄 Again</button>}
      </div>
    </div>
  );
}

export default function AgentPanel({ apiBase = "", token }) {
  const api = useMemo(() => createAgentApi({ baseUrl: apiBase, token }), [apiBase, token]);
  const [docs, setDocs] = useState([]); const [selected, setSelected] = useState([]);
  const [projects, setProjects] = useState([]); const [projectId, setProjectId] = useState("");
  const [sessions, setSessions] = useState([]); const [input, setInput] = useState(""); const [err, setErr] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(false);
  const saved = (k, d) => { try { return JSON.parse(localStorage.getItem("yuki_" + k)) ?? d; } catch { return d; } };
  const [dark, setDark] = useState(() => saved("dark", false));
  const [fx, setFx] = useState(() => saved("fx", true));
  useEffect(() => { try { localStorage.setItem("yuki_dark", dark); localStorage.setItem("yuki_fx", fx); } catch { /* storage unavailable */ } }, [dark, fx]);
  const [sideOpen, setSideOpen] = useState(false);   // sidebar is a drawer on small screens
  const submitRef = useRef(null), spokenRef = useRef(0);
  const voice = useVoice(api, { onTranscript: (t) => submitRef.current?.(t) });
  const preferences = useMemo(() => ({ ...(voice.lang !== "auto" ? { language: voice.lang } : {}), voice: autoSpeak }), [voice.lang, autoSpeak]);
  const chat = useAgentChat(api, { selectedDocIds: selected, projectId, preferences });
  useEffect(() => {   // read new completed answers aloud when "Read replies aloud" is on
    const m = chat.messages; const last = m[m.length - 1];
    if (m.length > spokenRef.current && last?.role === "assistant" && last.status === "completed" && autoSpeak) voice.speak(last.answer || last.content);
    spokenRef.current = m.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.messages]);

  const refresh = useCallback(() => Promise.all([api.documents(), api.projects(), api.sessions()])
    .then(([d, p, s]) => { setDocs(d.documents); setProjects(p.projects); setSessions(s.sessions); })
    .catch((e) => setErr(e.message)), [api]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (!chat.busy) api.sessions().then((s) => setSessions(s.sessions)).catch(() => {}); }, [chat.busy, api]);

  const submit = (t = input) => { if (t.trim() && !chat.busy) { setInput(""); chat.send(t.trim()); } };
  submitRef.current = submit;
  const upload = async (e) => { const f = e.target.files[0]; if (!f) return; setErr("");
    try { const d = await api.upload(f); setSelected((s) => [...s, d.id]); await refresh(); } catch (x) { setErr(x.message); } e.target.value = ""; };

  return (
    <div className={"ya-root" + (dark ? " dark" : "") + (fx ? "" : " calm")}>
      <div className="ya-blobs" aria-hidden="true"><i /><i /><i /></div>
      <aside className={"ya-side" + (sideOpen ? " open" : "")}>
        <div className="ya-brand"><span className="ya-logo">🦊</span><div><b>YUKI</b><small>Your study buddy ❄️</small></div></div>
        <button className="ya-btn primary big" onClick={chat.newChat}>✨ New chat</button>
        <h4>⚙️ Settings</h4>
        <Switch icon="🔊" label="Read aloud" checked={autoSpeak} onChange={(v) => { setAutoSpeak(v); if (!v) voice.stopSpeaking(); }} />
        <Switch icon="🌙" label="Dark mode" checked={dark} onChange={setDark} />
        <Switch icon="🎈" label="Animations" checked={fx} onChange={setFx} />
        <div className="ya-seg" role="group" aria-label="Language">{LANGS.map(([v, l]) => (
          <button key={v} type="button" className={voice.lang === v ? "on" : ""} onClick={() => voice.setLang(v)}>{l}</button>))}</div>
        <h4>📚 My documents</h4>
        {docs.filter((d) => d.status === "ready").map((d) => (
          <label key={d.id} className="ya-switch doc"><span>📄 {d.filename}</span><input type="checkbox" role="switch" checked={selected.includes(d.id)}
            onChange={() => setSelected((s) => (s.includes(d.id) ? s.filter((x) => x !== d.id) : [...s, d.id]))} /><i className="ya-knob" /></label>))}
        <label className="ya-btn sky">⬆️ Upload PDF/TXT<input type="file" hidden accept=".pdf,.txt,.md" onChange={upload} /></label>
        <h4>🗂️ Project</h4>
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
          <option value="">(none)</option>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        <button className="ya-btn mint" onClick={async () => { const n = window.prompt("Project name"); if (n) { await api.createProject(n); refresh(); } }}>➕ Project</button>
        <h4>🕘 History</h4>
        {sessions.map((s) => (<div key={s.id} className="ya-hist"><button onClick={() => chat.loadSession(s.id)}>💬 {s.title}</button>
          <button title="Delete" onClick={async () => { await api.deleteSession(s.id); refresh(); }}>✕</button></div>))}
        {sessions.length > 0 && <button className="ya-btn coral" onClick={async () => { if (window.confirm("Delete all chat history?")) { await api.clearHistory(); chat.newChat(); refresh(); } }}>🗑️ Clear history</button>}
      </aside>
      <main className="ya-main">
        <button className="ya-btn ya-menu" onClick={() => setSideOpen((o) => !o)}>{sideOpen ? "✕ Close" : "☰ Menu"}</button>
        {err && <div className="ya-warn">{err}</div>}
        <div className="ya-feed">
          {chat.messages.length === 0 && <div className="ya-empty"><div className="ya-mascot">🦊</div><h2>Hi, I'm <span>YUKI</span>!</h2><p>Pick your documents, then choose a magic power ✨</p>
            <div className="ya-tools">{TOOLS.map((t, i) => <button key={t.label} className="ya-tool" style={{ "--c": t.color, "--i": i }} onClick={() => submit(t.prompt)}>
              <span>{t.icon}</span>{t.label}</button>)}</div></div>}
          {chat.messages.map((m, i) => <Message key={i} m={m} index={i} api={api} onPatch={chat.patchMessage}
            onRegenerate={chat.regenerate} isLast={i === chat.messages.length - 1} onSpeak={voice.speak} canSpeak={voice.canSpeak} />)}
          {chat.busy && <div className="ya-msg bot"><span className="ya-avatar">🦊</span><div className="ya-typing"><i /><i /><i /></div><Activity steps={chat.activity} live />
            <button className="ya-btn coral" onClick={chat.cancel}>⏹ Stop</button></div>}
        </div>
        {voice.error && <div className="ya-warn">{voice.error}</div>}
        {chat.messages.length > 0 && <div className="ya-chips">{TOOLS.map((t) => <button key={t.label} className="ya-pill" style={{ "--c": t.color }} disabled={chat.busy} onClick={() => submit(t.prompt)}>{t.icon} {t.label}</button>)}
          {voice.speaking && <button className="ya-pill" style={{ "--c": "#ef476f" }} onClick={voice.stopSpeaking}>■ Stop audio</button>}</div>}
        <form className="ya-input" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <textarea value={input} rows={2} placeholder="Ask me anything about your documents… 🌈" onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
          {voice.canListen && <button type="button" className={"ya-btn ya-mic" + (voice.listening ? " on" : "")} disabled={voice.transcribing || chat.busy}
            title={voice.listening ? "Stop recording" : "Speak (Hindi or English)"}
            onClick={voice.listening ? voice.stopListening : voice.startListening}>{voice.transcribing ? "…" : voice.listening ? "■" : "🎤"}</button>}
          <button className="ya-btn primary send" disabled={chat.busy}>Send 🚀</button>
        </form>
      </main>
    </div>
  );
}
