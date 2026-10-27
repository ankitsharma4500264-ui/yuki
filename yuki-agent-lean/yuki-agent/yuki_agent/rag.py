"""RAG pipeline: extraction, chunking, embeddings, vector stores, retrieval, ingestion."""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

from .errors import ValidationError, VectorStoreError, YukiError

WORDS_PER_TOKEN = 0.75   # tokenizer-free approximation: 700 tokens ~ 525 words
STOPWORDS = set("a an the and or of to in on for with is are was were be by as at it this that these those "
                "from what which who how why when does do did can could should would about into than then "
                "है हैं था थे थी का की के में और से को पर यह वह ये वे क्या कि एक भी तो ही था".split())


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    index: int


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    user_id: str
    filename: str
    page: Optional[int]
    index: int
    text: str


@dataclass
class Hit:
    record: ChunkRecord
    score: Optional[float]      # vector similarity (0..1)
    final: float = 0.0          # score after optional lexical rerank


# ---------------------------------------------------------------- extraction / chunking
def extract_pdf(path, max_pages=None):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise YukiError("PyMuPDF is not installed (pip install pymupdf).")
    pages = []
    with fitz.open(path) as doc:
        if max_pages and doc.page_count > max_pages:
            raise ValidationError(f"This PDF has {doc.page_count} pages; the limit is {max_pages}. Split it and upload the parts.")
        for i, page in enumerate(doc, 1):
            pages.append((i, page.get_text("text") or ""))
    return pages


def extract_text_file(path):
    with open(path, "rb") as f:
        return [(None, f.read().decode("utf-8", errors="replace"))]


def extract_pages(path, max_pages=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path, max_pages)
    if ext in (".txt", ".md"):
        return extract_text_file(path)
    raise ValidationError("Unsupported file type.")


def chunk_pages(pages, chunk_tokens=700, overlap_tokens=100):
    """Overlapping word-window chunks. Chunks never cross page boundaries so page numbers stay exact."""
    size = max(20, int(chunk_tokens * WORDS_PER_TOKEN))
    overlap = min(int(overlap_tokens * WORDS_PER_TOKEN), size // 2)
    step = max(1, size - overlap)
    out, idx = [], 0
    for page_no, text in pages:
        words = text.split()
        start = 0
        while words and start < len(words):
            out.append(Chunk(" ".join(words[start:start + size]), page_no, idx))
            idx += 1
            if start + size >= len(words):
                break
            start += step
    return out


# ---------------------------------------------------------------- embeddings
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9\u0900-\u097F]+", text.lower()) if t not in STOPWORDS]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests/offline smoke checks only."""

    def __init__(self, dim=512):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise YukiError("sentence-transformers is not installed (pip install sentence-transformers).")
        self._model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()


class FastEmbedEmbedder:
    """Light ONNX embeddings (no PyTorch): small enough for serverless bundles such as Vercel."""

    def __init__(self, model_name, cache_dir=None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise YukiError("fastembed is not installed (pip install fastembed).")
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, texts):
        out = []
        for v in self._model.embed(list(texts)):
            v = [float(x) for x in v]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def build_embedder(settings):
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    if settings.embedding_backend == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model, os.getenv("FASTEMBED_CACHE_PATH"))
    return SentenceTransformerEmbedder(settings.embedding_model)


# ---------------------------------------------------------------- vector stores
class MemoryStore:
    """In-process store (tests, tiny corpora). Same interface as ChromaStore."""

    def __init__(self):
        self._rows = []   # (ChunkRecord, embedding)

    def add(self, records, embeddings):
        self._rows.extend(zip(records, embeddings))

    def _match(self, rec, user_id, document_ids):
        return rec.user_id == user_id and (not document_ids or rec.document_id in document_ids)

    def query(self, embedding, n, user_id, document_ids=None):
        scored = []
        for rec, emb in self._rows:
            if self._match(rec, user_id, document_ids):
                scored.append(Hit(rec, sum(a * b for a, b in zip(embedding, emb))))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:n]

    def get_document_chunks(self, document_id, user_id):
        recs = [r for r, _ in self._rows if r.document_id == document_id and r.user_id == user_id]
        return sorted(recs, key=lambda r: r.index)

    def delete_document(self, document_id, user_id):
        self._rows = [(r, e) for r, e in self._rows if not (r.document_id == document_id and r.user_id == user_id)]

    def delete_user(self, user_id):
        self._rows = [(r, e) for r, e in self._rows if r.user_id != user_id]

    def count(self):
        return len(self._rows)


class ChromaStore:
    def __init__(self, persist_dir):
        try:
            import chromadb
        except ImportError:
            raise YukiError("chromadb is not installed (pip install chromadb).")
        os.makedirs(persist_dir, exist_ok=True)
        try:
            client = chromadb.PersistentClient(
                path=persist_dir, settings=chromadb.config.Settings(anonymized_telemetry=False))
            self._col = client.get_or_create_collection(
                "yuki_chunks", metadata={"hnsw:space": "cosine"})
        except Exception:
            raise VectorStoreError("Could not open the vector database.")

    @staticmethod
    def _where(user_id, document_ids):
        conds = [{"user_id": {"$eq": user_id}}]
        if document_ids:
            conds.append({"document_id": {"$in": list(document_ids)}})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    @staticmethod
    def _rec(cid, text, m):
        return ChunkRecord(cid, m["document_id"], m["user_id"], m["filename"],
                           m["page"] if m.get("page", 0) > 0 else None, m["index"], text)

    def add(self, records, embeddings):
        try:
            for i in range(0, len(records), 256):
                rs = records[i:i + 256]
                self._col.add(
                    ids=[r.chunk_id for r in rs], embeddings=embeddings[i:i + 256],
                    documents=[r.text for r in rs],
                    metadatas=[{"document_id": r.document_id, "user_id": r.user_id, "filename": r.filename,
                                "page": r.page or 0, "index": r.index} for r in rs])
        except Exception:
            raise VectorStoreError("Failed to write to the vector database.")

    def query(self, embedding, n, user_id, document_ids=None):
        try:
            if self._col.count() == 0:
                return []
            res = self._col.query(query_embeddings=[embedding], n_results=n,
                                  where=self._where(user_id, document_ids),
                                  include=["documents", "metadatas", "distances"])
        except Exception:
            raise VectorStoreError("The vector database query failed.")
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(self._rec(cid, doc, meta), 1.0 - float(dist)))
        return hits

    def get_document_chunks(self, document_id, user_id):
        try:
            res = self._col.get(where=self._where(user_id, [document_id]), include=["documents", "metadatas"])
        except Exception:
            raise VectorStoreError("The vector database query failed.")
        recs = [self._rec(c, d, m) for c, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(recs, key=lambda r: r.index)

    def delete_document(self, document_id, user_id):
        try:
            self._col.delete(where=self._where(user_id, [document_id]))
        except Exception:
            raise VectorStoreError("Failed to delete from the vector database.")

    def delete_user(self, user_id):
        try:
            self._col.delete(where=self._where(user_id, None))
        except Exception:
            raise VectorStoreError("Failed to delete from the vector database.")

    def count(self):
        try:
            return self._col.count()
        except Exception:
            raise VectorStoreError("The vector database is unavailable.")


def build_store(settings):
    return MemoryStore() if settings.vector_backend == "memory" else ChromaStore(settings.chroma_dir)


# ---------------------------------------------------------------- retrieval
class Retriever:
    def __init__(self, store, embedder, settings):
        self.store, self.embedder, self.settings = store, embedder, settings

    def search(self, query, user_id, document_ids=None, top_k=None, min_score=None):
        top_k = top_k or self.settings.top_k
        min_score = self.settings.min_score if min_score is None else min_score
        emb = self.embedder.embed([query])[0]
        n = top_k * 3 if self.settings.rerank else top_k
        hits = [h for h in self.store.query(emb, n, user_id, document_ids or None)
                if h.score is not None and h.score >= min_score]
        qtok = set(tokenize(query))
        for h in hits:
            if self.settings.rerank and qtok:
                overlap = len(qtok & set(tokenize(h.record.text))) / len(qtok)
                h.final = 0.8 * h.score + 0.2 * overlap
            else:
                h.final = h.score
        hits.sort(key=lambda h: h.final, reverse=True)
        return hits[:top_k]

    def document_chunks(self, document_id, user_id):
        return self.store.get_document_chunks(document_id, user_id)


# ---------------------------------------------------------------- ingestion
class RagService:
    def __init__(self, db, store, embedder, settings):
        self.db, self.store, self.embedder, self.settings = db, store, embedder, settings

    def ingest_pages(self, user_id, filename, pages, path=None, doc_id=None):
        doc_id = doc_id or self.db.add_document(user_id, filename, path)
        try:
            chunks = chunk_pages(pages, self.settings.chunk_tokens, self.settings.chunk_overlap)
            if not chunks:
                raise ValidationError("No extractable text found (scanned PDFs need OCR, which is not included).")
            if len(chunks) > self.settings.max_chunks_per_doc:
                raise ValidationError(f"This document is too large to index ({len(chunks)} sections; limit "
                                      f"{self.settings.max_chunks_per_doc}). Split it and upload the parts.")
            recs = [ChunkRecord(f"{doc_id}:{c.index}", doc_id, user_id, filename, c.page, c.index, c.text)
                    for c in chunks]
            embs = []
            for i in range(0, len(recs), 64):
                embs.extend(self.embedder.embed([r.text for r in recs[i:i + 64]]))
            self.store.add(recs, embs)
        except Exception as e:
            try:
                self.store.delete_document(doc_id, user_id)
            except Exception:
                pass
            self.db.set_document_status(doc_id, "failed", error=str(e) if isinstance(e, YukiError) else "Processing failed.")
            raise
        self.db.set_document_status(doc_id, "ready", pages=len([p for p in pages if p[1].strip()]), chunks=len(recs))
        return self.db.get_document(user_id, doc_id)

    def ingest_file(self, user_id, filename, path, doc_id=None):
        return self.ingest_pages(user_id, filename, extract_pages(path, self.settings.max_pages), path=path, doc_id=doc_id)

    def delete_document(self, user_id, doc_id):
        doc = self.db.get_document(user_id, doc_id)
        if not doc:
            raise YukiError("Document not found.")
        self.store.delete_document(doc_id, user_id)
        self.db.delete_document(user_id, doc_id)
        if doc.get("path") and os.path.exists(doc["path"]):
            os.remove(doc["path"])

    def delete_user_data(self, user_id):
        docs = self.db.delete_user_data(user_id)
        self.store.delete_user(user_id)
        for d in docs:
            if d.get("path") and os.path.exists(d["path"]):
                os.remove(d["path"])
        return len(docs)
