"""Citation manager: every citation shown to the user maps to a chunk that was really retrieved."""
from __future__ import annotations

import re

from .safety import sanitize_evidence

LABEL_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
QUOTE_RE = re.compile(r'"([^"\n]{20,})"|\u201c([^\u201d\n]{20,})\u201d')


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


class CitationManager:
    def __init__(self):
        self._by_chunk = {}
        self._by_label = {}

    def register(self, hit):
        r = hit.record
        if r.chunk_id in self._by_chunk:
            return self._by_chunk[r.chunk_id]
        label = f"S{len(self._by_label) + 1}"
        self._by_chunk[r.chunk_id] = label
        self._by_label[label] = {
            "label": label, "chunk_id": r.chunk_id, "document_id": r.document_id, "filename": r.filename,
            "page": r.page, "score": None if hit.score is None else round(hit.score, 4), "text": r.text}
        return label

    def has(self, label):
        return label in self._by_label

    def get(self, label):
        return self._by_label.get(label)

    def _format(self, label):
        s = self._by_label[label]
        where = f"{s['filename']}, page {s['page']}" if s["page"] else s["filename"]
        return f"[{label}] ({where})\n\"\"\"\n{sanitize_evidence(s['text'])}\n\"\"\""

    def evidence_block(self, hits):
        labels = [self.register(h) for h in hits]
        return self.evidence_for_labels(labels)

    def evidence_for_labels(self, labels):
        return "<evidence>\n" + "\n\n".join(self._format(l) for l in labels) + "\n</evidence>"

    def finalize(self, text, extra_labels=(), only_cited=False):
        """Strip citation markers that do not exist; return (clean_text, sources, dropped_labels)."""
        cited, dropped = set(extra_labels), []

        def repl(m):
            labels = [l.strip() for l in m.group(1).split(",")]
            valid = [l for l in labels if l in self._by_label]
            dropped.extend(l for l in labels if l not in self._by_label)
            cited.update(valid)
            return "[" + ", ".join(valid) + "]" if valid else ""

        clean = LABEL_RE.sub(repl, text)
        if dropped:   # tidy only where we removed an invalid marker
            clean = re.sub(r"[ \t]+([.,;:])(?=\s|$)", r"\1", clean)
        order = sorted(self._by_label, key=lambda l: int(l[1:]))
        sources = []
        for l in order:
            s = self._by_label[l]
            if only_cited and l not in cited:
                continue
            sources.append({"label": l, "document_id": s["document_id"], "filename": s["filename"],
                            "page": s["page"], "chunk_id": s["chunk_id"], "score": s["score"],
                            "snippet": s["text"][:500], "cited": l in cited})
        return clean, sources, dropped

    def unverified_quotes(self, text):
        corpus = _norm(" ".join(s["text"] for s in self._by_label.values()))
        bad = []
        for m in QUOTE_RE.finditer(text):
            q = m.group(1) or m.group(2)
            if _norm(q) not in corpus:
                bad.append(q[:80])
        return bad
