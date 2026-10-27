"""Wires settings -> concrete services."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .controller import AgentController
from .db import Database
from .providers import build_provider
from .voice import VoiceService
from .rag import RagService, Retriever, build_embedder, build_store


@dataclass
class Services:
    settings: Settings
    db: Database
    store: Any
    rag: RagService
    retriever: Retriever
    llm: Any
    controller: AgentController
    voice: Any = None


def build_services(settings=None, llm=None):
    s = settings or Settings.from_env()
    os.makedirs(s.data_dir, exist_ok=True)
    os.makedirs(s.upload_dir, exist_ok=True)
    db, embedder, store = Database(s.db_path), build_embedder(s), build_store(s)
    retriever = Retriever(store, embedder, s)
    llm = llm or build_provider(s)
    return Services(s, db, store, RagService(db, store, embedder, s), retriever, llm,
                    AgentController(s, db, retriever, llm), VoiceService(s))
