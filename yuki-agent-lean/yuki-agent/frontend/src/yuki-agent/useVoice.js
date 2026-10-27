import { useCallback, useEffect, useRef, useState } from "react";
import { blobToWav16k } from "./wav";
import { detectLang, speakable } from "./speakable";

const SR = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null;
const BCP = { en: "en-IN", hi: "hi-IN" };
const MAX_RECORD_MS = 60000;

// lang: "auto" | "en" | "hi".  Server voice (Grok) is used when the backend has a key; otherwise the browser's
// Web Speech API (note: Chrome's recogniser sends audio to Google).
export function useVoice(api, { onTranscript }) {
  const [sttOn, setSttOn] = useState(false);
  const [ttsOn, setTtsOn] = useState(false);
  const [lang, setLang] = useState("auto");
  const [listening, setListening] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState("");
  const rec = useRef(null), recog = useRef(null), audio = useRef(null), timer = useRef(null);
  const server = sttOn;   // server speech-to-text
  const canListen = sttOn || !!SR;
  const canSpeak = ttsOn || (typeof window !== "undefined" && "speechSynthesis" in window);

  useEffect(() => { api.voiceConfig().then((c) => { setSttOn(!!c.server_stt); setTtsOn(!!c.server_tts); }).catch(() => { setSttOn(false); setTtsOn(false); }); }, [api]);

  const stopSpeaking = useCallback(() => {
    audio.current?.pause(); audio.current = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, []);

  const speakBrowser = useCallback((text, l) => {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text); u.lang = BCP[l];
    const v = window.speechSynthesis.getVoices();
    u.voice = v.find((x) => x.lang.replace("_", "-").toLowerCase() === BCP[l].toLowerCase()) || v.find((x) => x.lang.toLowerCase().startsWith(l)) || null;
    u.onend = u.onerror = () => setSpeaking(false);
    setSpeaking(true); window.speechSynthesis.speak(u);
  }, []);

  const speak = useCallback(async (text) => {
    stopSpeaking(); setError("");
    const clean = speakable(text); if (!clean) return;
    const l = lang === "auto" ? detectLang(clean) : lang;
    if (ttsOn) {
      try {
        const url = URL.createObjectURL(await api.speak(clean, l));
        const a = new Audio(url); audio.current = a;
        a.onended = a.onerror = () => { setSpeaking(false); URL.revokeObjectURL(url); };
        setSpeaking(true); await a.play(); return;
      } catch (e) { setSpeaking(false); setError(e.message); }
    }
    speakBrowser(clean, l);
  }, [api, lang, ttsOn, stopSpeaking, speakBrowser]);

  const stopListening = useCallback(() => {
    clearTimeout(timer.current);
    if (rec.current && rec.current.state !== "inactive") rec.current.stop();
    recog.current?.stop();
  }, []);

  const startListening = useCallback(async () => {
    setError(""); stopSpeaking();
    try {
      if (server) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const r = new MediaRecorder(stream), chunks = [];
        r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
        r.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop()); setListening(false); setTranscribing(true);
          try {
            const wav = await blobToWav16k(new Blob(chunks, { type: r.mimeType }));
            const out = await api.transcribe(wav, lang === "auto" ? "" : lang);
            if (out.text) onTranscript(out.text); else setError(lang === "hi" ? "कुछ सुनाई नहीं दिया।" : "I didn't catch that.");
          } catch (e) { setError(e.message); } finally { setTranscribing(false); }
        };
        rec.current = r; r.start(); setListening(true);
        timer.current = setTimeout(stopListening, MAX_RECORD_MS);
      } else if (SR) {
        const r = new SR(); r.lang = BCP[lang === "hi" ? "hi" : "en"]; r.interimResults = false;
        r.onresult = (e) => onTranscript(e.results[0][0].transcript);
        r.onerror = (e) => setError(e.error === "not-allowed" ? "Microphone permission denied." : `Voice error: ${e.error}`);
        r.onend = () => setListening(false);
        recog.current = r; r.start(); setListening(true);
      }
    } catch { setListening(false); setError("Microphone unavailable or permission denied."); }
  }, [api, lang, server, onTranscript, stopListening, stopSpeaking]);

  return { server, lang, setLang, listening, transcribing, speaking, error, canListen, canSpeak,
           startListening, stopListening, speak, stopSpeaking };
}
