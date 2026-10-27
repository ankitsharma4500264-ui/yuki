// Same idea as yuki_agent/lang.py:speakable, for the browser-TTS fallback.
export const detectLang = (t) => {
  const dev = (t.match(/[\u0900-\u097F]/g) || []).length, lat = (t.match(/[A-Za-z]/g) || []).length;
  return dev && dev / (dev + lat) >= 0.3 ? "hi" : "en";
};
export function speakable(text, max = 1500) {
  let s = (text || "").replace(/```[\s\S]*?```/g, " ").replace(/\[S\d+(?:\s*,\s*S\d+)*\]/g, "")
    .replace(/https?:\/\/\S+/g, "").replace(/[`*_>#|~]/g, "").replace(/^\s*(?:[-\u2022]|\d+[.)])\s+/gm, "")
    .replace(/\n+/g, ". ").replace(/([.!?\u0964])\s*\.\s*/g, "$1 ").replace(/\s{2,}/g, " ").replace(/\s+([.,!?\u0964])/g, "$1").trim();
  if (s.length > max) { const cut = s.slice(0, max); const e = Math.max(...[".", "!", "?", "\u0964"].map((c) => cut.lastIndexOf(c))); s = e > max / 2 ? cut.slice(0, e + 1) : cut; }
  return s;
}
