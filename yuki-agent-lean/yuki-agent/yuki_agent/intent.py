"""Intent classifier: fast rule-based routing, optional model routing via a validated JSON tool-call protocol."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .context import build_messages, extract_json


@dataclass
class Intent:
    name: str
    tool: Optional[str]
    confidence: float
    params: dict = field(default_factory=dict)


RULES = [  # order matters. Hindi patterns avoid \\b because Devanagari vowel signs are not word characters.
    ("save_note", "save_research_note",
     r"^\s*(?:please |kindly |can you |could you )?(?:save|store|add|keep)\b.{0,60}\b(?:to|in|into)\s+(?:my |the |a )?(?:\w+\s+){0,2}(?:projects?|notes?)\b|"
     r"\bsave (?:this|that|it|the (?:summary|answer|response|explanation|quiz|flashcards))\b|सेव कर|सहेज|नोट\w* में|प्रोजेक्ट में"),
    ("flashcards", "generate_flashcards", r"\bflash ?cards?\b|फ़?्?लैश ?कार्ड"),
    ("quiz", "generate_quiz",
     r"\b(quiz|mcqs?|multiple[- ]choice|practice (questions|test)|test me)\b|क्विज़?|प्रश्नोत्तरी|एमसीक्यू|बहुविकल्पीय|मेरा टेस्ट"),
    ("study_plan", "create_study_plan",
     r"\b(study|revision|exam|learning) (plan|schedule|timetable)\b|\bplan my (study|revision)\b|"
     r"(पढ़ाई|अध्ययन|रिवीज़?न|परीक्षा).{0,12}(योजना|प्लान|शेड्यूल|समय[- ]?सारणी)"),
    ("compare", "compare_documents", r"\bcompar(e|ison|ing)\b|\bdifferences? between\b|\bsimilarities\b|तुलना|अंतर बता|फ़र्क|फर्क"),
    ("gaps", "analyze_knowledge_gaps",
     r"\b(knowledge|research|literature) gaps?\b|\bwhat('s| is) missing\b|\bunanswered (questions|issues)\b|\bgaps? in\b|"
     r"अंतराल|क्या कमी"),
    ("ideas", "generate_research_ideas",
     r"\bresearch (ideas?|topics?|questions?|directions?|proposals?)\b|\bthesis (topic|idea)s?\b|\bwhat (could|should) i research\b|"
     r"(शोध|रिसर्च).{0,8}(विचार|विषय|प्रश्न|आइडिया|टॉपिक)"),
    ("summarize", "summarize_document",
     r"\b(summari[sz]e|summary|tl;?dr|recap|overview)\b|सारांश|संक्षेप|समरी|\bsaransh\b"),
    ("explain", "explain_concept",
     r"\b(explain|simple (english|terms|words)|eli5|teach me|break (it )?down)\b|समझा|आसान भाषा|सरल भाषा|\bsamjha(o|iye)\b"),
    ("chat", None,
     r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|नमस्ते|नमस्कार|हैलो|धन्यवाद|शुक्रिया)[\s!.,।]*$"),
]
_COMPILED = [(n, t, re.compile(p, re.I)) for n, t, p in RULES]
_HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
GENERIC = {"document", "documents", "chapter", "paper", "papers", "notes", "pdf", "file", "files", "this", "these",
           "that", "those", "it", "them", "selected", "my", "the", "uploaded", "above"}


def extract_params(message):
    p = {}
    message = message.translate(_HI_DIGITS)
    # only a number tied to the thing being counted ("15 MCQs", "quiz of 5"), never "chapter 3"
    unit = r"(?:mcqs?|questions?|flash ?cards?|cards?|quiz(?:zes)?|प्रश्न|सवाल|सवालों|फ़?्?लैश ?कार्ड|sawal|saval)"
    m = re.search(r"\b(\d{1,3})\s*(?:[\w\u0900-\u097F]+\s+){0,2}" + unit, message, re.I) \
        or re.search(unit + r"\s*(?:of|:|-)?\s*(\d{1,3})\b", message, re.I)
    if m:
        p["count"] = max(1, min(int(m.group(1)), 50))
    m = re.search(r"\b(easy|medium|hard|difficult|advanced|beginner)\b|(आसान|मध्यम|कठिन)", message, re.I)
    if m:
        w = (m.group(1) or m.group(2)).lower()
        p["difficulty"] = {"difficult": "hard", "advanced": "hard", "beginner": "easy", "आसान": "easy",
                           "मध्यम": "medium", "कठिन": "hard"}.get(w, w)
    if re.search(r"\b(simple|simply|beginner|eli5|basic|easy|child)\b|आसान|सरल", message, re.I):
        p["level"] = "beginner"
    elif re.search(r"\b(advanced|expert|graduate|technical)\b", message, re.I):
        p["level"] = "advanced"
    if re.search(r"\b(short|brief|quick|tl;?dr)\b", message, re.I):
        p["length"] = "short"
    elif re.search(r"\b(detailed|long|thorough|comprehensive)\b", message, re.I):
        p["length"] = "long"
    m = re.search(r"focus(?:ing)? on\s+(.+?)[.?!]*$", message, re.I)
    if m:
        p["focus"] = m.group(1)[:300]
    m = re.search(r"\b(?:called|titled|named)\s+[\"\u201c']?(.+?)[\"\u201d']?\s*$", message, re.I)
    if m:
        p["title"] = m.group(1)[:120]
    return p


def extract_topic(message):
    m = re.search(r"\b(?:on|about|regarding|covering|from)\b\s+(.+?)[.?!]*$", message, re.I)
    topic = m.group(1) if m else message
    topic = re.sub(r"^(my|the|these|this|those|our)\s+", "", topic.strip(), flags=re.I)
    toks = re.findall(r"[\w\u0900-\u097F]+", topic.lower())
    if not toks or all(t in GENERIC for t in toks):
        return "key concepts and main ideas"
    return topic[:500]


def classify(message, has_documents=False):
    for name, tool, rx in _COMPILED:
        if rx.search(message):
            return Intent(name, tool, 0.9, extract_params(message))
    if has_documents:
        return Intent("qa", "search_documents", 0.5, extract_params(message))
    if "?" in message or re.match(r"\s*(what|why|how|who|when|define|explain)\b", message, re.I) \
            or re.search(r"(^|\s)(क्या|क्यों|कैसे|कौन|कब|कहाँ|कहां|कितने|कितना)(\s|$)", message):
        return Intent("explain", "explain_concept", 0.5, extract_params(message))
    return Intent("chat", None, 0.4, {})


# ----- model-driven routing (JSON tool-call protocol, schema-validated, never code execution)
ROUTABLE = ("search_documents", "explain_concept", "summarize_document", "generate_flashcards", "generate_quiz",
            "compare_documents", "generate_research_ideas", "analyze_knowledge_gaps", "create_study_plan")


def llm_route(ctx, message, registry, doc_ids):
    """Ask the model to pick a tool. Returns (tool, arguments) or None. Output is validated by the executor."""
    tools = [t for t in registry.describe() if t["name"] in ROUTABLE]
    prompt = ("Choose the single best tool for the request, or none for plain conversation.\nTools:\n"
              + "\n".join(f"- {t['name']}: {t['description']} params={list(t['parameters']['properties'])}" for t in tools)
              + f"\nSelected document ids: {doc_ids}\nRequest: {message}\n"
              'Reply ONLY with JSON: {"tool": "<name or null>", "arguments": {...}}')
    try:
        obj = extract_json(ctx.llm_chat(build_messages(prompt), json_mode=True, temperature=0.0))
    except ValueError:
        return None
    tool = obj.get("tool") if isinstance(obj, dict) else None
    args = obj.get("arguments") if isinstance(obj, dict) else None
    if tool not in ROUTABLE or not isinstance(args, dict):
        return None
    return tool, args
