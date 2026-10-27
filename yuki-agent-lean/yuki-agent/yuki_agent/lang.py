"""Hindi/English helpers: language detection, spoken-text cleanup, and the few fixed UI strings."""
from __future__ import annotations

import re

_DEV = re.compile(r"[\u0900-\u097F]")
_LAT = re.compile(r"[A-Za-z]")
LANGS = ("en", "hi")

STRINGS = {
    "insufficient": {
        "en": ("I can't establish an answer to that from the selected sources. The retrieved material doesn't "
               "cover it. Try selecting other documents or rephrasing the question."),
        "hi": ("चुने गए स्रोतों में इसका उत्तर नहीं मिला। दूसरे दस्तावेज़ चुनकर या प्रश्न को दूसरे शब्दों में पूछकर देखें।"),
    },
    "confirm_save": {"en": "Please confirm before I save this:", "hi": "सेव करने से पहले कृपया पुष्टि करें:"},
    "nothing_saved": {"en": "Okay, nothing was saved.", "hi": "ठीक है, कुछ भी सेव नहीं किया गया।"},
    "model_down": {
        "en": ("The language model is unavailable ({detail}). Check your GROQ_API_KEY (Groq) or, for the local setup, "
               "start Ollama and pull the model, then try again."),
        "hi": "भाषा मॉडल अभी उपलब्ध नहीं है। अपनी GROQ_API_KEY जाँचें या Ollama चालू करके दोबारा कोशिश करें।",
    },
}


def t(key, lang="en", **kw):
    s = STRINGS[key].get(lang) or STRINGS[key]["en"]
    return s.format(**kw) if kw else s


def detect_language(text):
    """'hi' if the text is meaningfully Devanagari, else 'en' (romanised Hinglish is treated as 'en')."""
    dev, lat = len(_DEV.findall(text or "")), len(_LAT.findall(text or ""))
    if dev == 0:
        return "en"
    return "hi" if dev / (dev + lat) >= 0.3 else "en"


def pick_language(prefs, message):
    """Return (ui_language, forced_language_or_None). Forced = tell the model explicitly which language to write."""
    pref = (prefs or {}).get("language")
    if pref in LANGS:
        return pref, pref
    lang = detect_language(message)
    return lang, ("hi" if lang == "hi" else None)


LANG_INSTRUCTION = {
    "hi": ("Write all user-facing text in Hindi (Devanagari script). Keep citation labels like [S1], file names, "
           "numbers and JSON keys unchanged."),
    "en": "Write all user-facing text in English. Keep citation labels, file names and JSON keys unchanged.",
}

_CODE = re.compile(r"```.*?```", re.S)
_CITE = re.compile(r"\[S\d+(?:\s*,\s*S\d+)*\]")
_URL = re.compile(r"https?://\S+")


def speakable(text, max_chars=1500):
    """Strip markdown, citation markers, URLs and code so text-to-speech reads naturally; cap at a sentence end."""
    s = _CODE.sub(" ", text or "")
    s = _CITE.sub("", s)
    s = _URL.sub("", s)
    s = re.sub(r"[`*_>#|~]", "", s)
    s = re.sub(r"^\s*(?:[-\u2022]|\d+[.)])\s+", "", s, flags=re.M)
    s = re.sub(r"[\u26a0\u2705\u274c]\ufe0f?", "", s)
    s = re.sub(r"\n+", ". ", s)
    s = re.sub(r"([.!?\u0964])\s*\.\s*", r"\1 ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([.,!?\u0964])", r"\1", s)
    if len(s) > max_chars:
        cut = s[:max_chars]
        end = max(cut.rfind(c) for c in (".", "!", "?", "\u0964"))
        s = cut[:end + 1] if end > max_chars * 0.5 else cut
    return s
