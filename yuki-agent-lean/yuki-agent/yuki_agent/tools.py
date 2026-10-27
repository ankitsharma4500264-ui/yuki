"""Tool registry, strict argument validation, and the controlled tool executor."""
from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .citations import CitationManager
from .errors import (AgentTimeout, Cancelled, InsufficientEvidence, PermissionDenied, ProviderError,
                     ToolArgError, ToolFailure, VectorStoreError, YukiError)


# ------------------------------------------------------------------ specs
@dataclass
class ArgSpec:
    name: str
    type: str                       # str | int | list[str] | enum
    description: str = ""
    required: bool = False
    default: Any = None
    choices: tuple = ()
    min: Optional[int] = None
    max: Optional[int] = None
    max_len: int = 4000
    max_items: int = 20


@dataclass
class ToolSpec:
    name: str
    description: str
    args: List[ArgSpec]
    handler: Callable
    doc_args: tuple = ()            # argument names holding document ids (ownership is verified)
    requires_confirmation: bool = False
    destructive: bool = False
    preview: Optional[Callable] = None
    timeout_s: Optional[float] = None

    def json_schema(self):
        props, req = {}, []
        for a in self.args:
            if a.type == "int":
                p = {"type": "integer"}
                if a.min is not None:
                    p["minimum"] = a.min
                if a.max is not None:
                    p["maximum"] = a.max
            elif a.type == "list[str]":
                p = {"type": "array", "items": {"type": "string"}, "maxItems": a.max_items}
            elif a.type == "enum":
                p = {"type": "string", "enum": list(a.choices)}
            else:
                p = {"type": "string", "maxLength": a.max_len}
            if a.description:
                p["description"] = a.description
            if a.default is not None:
                p["default"] = a.default
            props[a.name] = p
            if a.required:
                req.append(a.name)
        return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def _coerce(a: ArgSpec, v):
    if a.type == "str":
        if not isinstance(v, str):
            raise ToolArgError(f"'{a.name}' must be text.")
        v = v.strip()
        if a.required and not v:
            raise ToolArgError(f"'{a.name}' must not be empty.")
        if len(v) > a.max_len:
            raise ToolArgError(f"'{a.name}' is too long (max {a.max_len} characters).")
        return v
    if a.type == "int":
        if isinstance(v, bool):
            raise ToolArgError(f"'{a.name}' must be an integer.")
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            v = int(v)
        if not isinstance(v, int):
            raise ToolArgError(f"'{a.name}' must be an integer.")
        if a.min is not None and v < a.min or a.max is not None and v > a.max:
            raise ToolArgError(f"'{a.name}' must be between {a.min} and {a.max}.")
        return v
    if a.type == "enum":
        if not isinstance(v, str) or v.strip().lower() not in a.choices:
            raise ToolArgError(f"'{a.name}' must be one of: {', '.join(a.choices)}.")
        return v.strip().lower()
    if a.type == "list[str]":
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ToolArgError(f"'{a.name}' must be a list of text values.")
        v = [x.strip() for x in v if x.strip()]
        if len(v) > a.max_items:
            raise ToolArgError(f"'{a.name}' has too many items (max {a.max_items}).")
        if any(len(x) > a.max_len for x in v):
            raise ToolArgError(f"An item in '{a.name}' is too long.")
        return list(dict.fromkeys(v))
    raise ToolArgError(f"Unsupported argument type for '{a.name}'.")


def validate_args(spec: ToolSpec, raw):
    if not isinstance(raw, dict):
        raise ToolArgError("Tool arguments must be an object.")
    unknown = set(raw) - {a.name for a in spec.args}
    if unknown:
        raise ToolArgError(f"Unknown argument(s): {', '.join(sorted(unknown))}.")
    out = {}
    for a in spec.args:
        if raw.get(a.name) is not None:
            out[a.name] = _coerce(a, raw[a.name])
        elif a.required:
            raise ToolArgError(f"Missing required argument '{a.name}'.")
        else:
            out[a.name] = list(a.default) if isinstance(a.default, list) else a.default
    return out


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def get(self, name) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self):
        return list(self._tools)

    def describe(self):
        return [{"name": t.name, "description": t.description, "parameters": t.json_schema(),
                 "requires_confirmation": t.requires_confirmation, "destructive": t.destructive}
                for t in self._tools.values()]


# ------------------------------------------------------------------ context / results
@dataclass
class ToolOutput:
    data: dict = field(default_factory=dict)
    text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    grounded: Optional[bool] = None
    cited_labels: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool: str
    status: str                      # ok | error | needs_confirmation | insufficient_evidence
    data: dict = field(default_factory=dict)
    text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    pending_action: Optional[dict] = None
    duplicate: bool = False
    duration_ms: int = 0
    grounded: Optional[bool] = None
    cited_labels: List[str] = field(default_factory=list)

    @property
    def ok(self):
        return self.status == "ok"


@dataclass
class ToolContext:
    user_id: str
    session_id: Optional[str]
    settings: Any
    db: Any
    retriever: Any
    llm: Any
    citations: CitationManager = field(default_factory=CitationManager)
    project_id: Optional[str] = None
    deadline: Optional[float] = None
    confirmed: bool = False
    cancel: Any = None
    language: str = "en"                 # UI language for fixed strings
    force_language: Optional[str] = None  # tell the model explicitly which language to write

    def remaining(self):
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("Generation was cancelled.")
        if self.remaining() <= 0:
            raise AgentTimeout("The request took too long and was stopped.")

    def llm_chat(self, messages, json_mode=False, temperature=0.2):
        self.check()
        timeout = max(1.0, min(self.settings.llm_timeout_s, self.remaining()))
        if self.force_language and messages and messages[0].get("role") == "system":
            from .lang import LANG_INSTRUCTION
            messages = [{**messages[0], "content": messages[0]["content"] + "\n\n" + LANG_INSTRUCTION[self.force_language]},
                        *messages[1:]]
        return self.llm.chat(messages, json_mode=json_mode, temperature=temperature, timeout=timeout)


# ------------------------------------------------------------------ executor
class ToolExecutor:
    """Validates and runs tool calls: schema check, ownership check, dedupe, call cap, timeout, confirmation."""

    def __init__(self, registry: ToolRegistry, settings, ctx: ToolContext):
        self.registry, self.settings, self.ctx = registry, settings, ctx
        self.calls = 0
        self._cache: Dict[str, ToolResult] = {}
        self.log: List[dict] = []

    def execute(self, name, raw_args) -> ToolResult:
        t0 = time.monotonic()
        res = self._execute(name, raw_args)
        res.duration_ms = int((time.monotonic() - t0) * 1000)
        self.log.append({"tool": name, "status": res.status, "duplicate": res.duplicate,
                         "duration_ms": res.duration_ms})
        return res

    def _fail(self, name, msg, status="error"):
        return ToolResult(tool=name, status=status, error=msg)

    def _execute(self, name, raw_args):
        spec = self.registry.get(name)
        if spec is None:
            return self._fail(name, f"Unknown tool '{name}'.")
        try:
            args = validate_args(spec, raw_args)
        except ToolArgError as e:
            return self._fail(name, f"Invalid arguments: {e}")

        key = name + json.dumps(args, sort_keys=True) + str(self.ctx.confirmed)
        if key in self._cache:
            prev = self._cache[key]
            return ToolResult(**{**prev.__dict__, "duplicate": True})
        if self.calls >= self.settings.max_tool_calls:
            return self._fail(name, "Tool-call limit reached for this request.")

        try:
            self._check_permissions(spec, args)
        except PermissionDenied as e:
            return self._fail(name, str(e))

        if spec.requires_confirmation and not self.ctx.confirmed:
            preview = spec.preview(args) if spec.preview else json.dumps(args)[:600]
            pid = self.ctx.db.create_pending(self.ctx.user_id, self.ctx.session_id, name, args, preview)
            res = ToolResult(tool=name, status="needs_confirmation",
                             pending_action={"id": pid, "tool": name, "preview": preview})
            self._cache[key] = res
            return res

        self.calls += 1
        timeout = min(spec.timeout_s or self.settings.tool_timeout_s, self.ctx.remaining())
        if timeout <= 0:
            return self._fail(name, "The request ran out of time.")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(spec.handler, args, self.ctx)
        try:
            out = fut.result(timeout=timeout)
            res = ToolResult(tool=name, status="ok", data=out.data, text=out.text, warnings=out.warnings,
                             grounded=out.grounded, cited_labels=out.cited_labels)
        except concurrent.futures.TimeoutError:
            res = self._fail(name, f"The '{name}' tool timed out after {int(timeout)}s.")
        except InsufficientEvidence as e:
            res = self._fail(name, str(e), status="insufficient_evidence")
        except (Cancelled, AgentTimeout, ProviderError, VectorStoreError):
            raise   # infrastructure problems are reported by the controller, not hidden as tool errors
        except (ToolFailure, YukiError) as e:
            res = self._fail(name, str(e))
        finally:
            pool.shutdown(wait=False)
        self._cache[key] = res
        return res

    def _check_permissions(self, spec, args):
        for an in spec.doc_args:
            v = args.get(an)
            ids = [v] if isinstance(v, str) else list(v or [])
            if ids and len(self.ctx.db.get_documents(self.ctx.user_id, ids)) != len(set(ids)):
                raise PermissionDenied("One or more documents were not found or are not accessible.")
