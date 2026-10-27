"""Response formatter: the one structured shape returned to the frontend."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class AgentResponse:
    session_id: str = ""
    status: str = "completed"   # completed | needs_input | needs_confirmation | insufficient_evidence | error | cancelled
    answer: str = ""
    sources: List[dict] = field(default_factory=list)
    tool_activity: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structured: Optional[dict] = None       # e.g. {"type": "flashcards", "flashcards": [...]}
    pending_action: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class ActivityLog:
    """Records only steps that really happen; optionally streams them to the client as they occur."""

    def __init__(self, on_event=None):
        self.steps = []
        self._on_event = on_event

    def _emit(self, step):
        if self._on_event:
            try:
                self._on_event(dict(step))
            except Exception:
                pass

    def start(self, label, tool=None):
        step = {"id": len(self.steps), "label": label, "tool": tool, "status": "running", "detail": None,
                "duration_ms": None}
        self.steps.append(step)
        self._emit(step)
        return step

    def finish(self, step, status="done", detail=None, duration_ms=None):
        step.update(status=status, detail=detail, duration_ms=duration_ms)
        self._emit(step)

    def note(self, label, detail=None, tool=None):
        step = self.start(label, tool)
        self.finish(step, "done", detail, 0)
        return step
