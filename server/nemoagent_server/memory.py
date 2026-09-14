"""Long-term memory: RAG over past dialogs.

Two collections, because the two embedding models live in different vector spaces:
  * "text"  — dialog turns with Nemotron, embedded by nvidia/nemotron-3-embed-1b;
  * "vl"    — DeepSeek analyses (images + documents + answers), embedded by
              nvidia/llama-nemotron-embed-vl-1b-v2 (text and data-URI images in one space).

Storage: SQLite (source of truth) + an in-memory normalized numpy matrix per collection for
cosine search. Thousands of turns search in well under a millisecond; embeddings are the only
network cost and they are computed server-side through NIM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import settings
from .nim import NIMClient

log = logging.getLogger("memory")

COLLECTION_MODEL = {"text": settings.EMBED_TEXT_MODEL, "vl": settings.EMBED_VL_MODEL}


class MemoryStore:
    def __init__(self, nim: NIMClient, db_path: Path = settings.MEMORY_DB):
        self.nim = nim
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                session_id TEXT,
                kind TEXT,
                text TEXT NOT NULL,
                meta TEXT,
                ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vectors(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                collection TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_vectors_collection ON vectors(collection);
            CREATE INDEX IF NOT EXISTS ix_memories_session ON memories(session_id);
            """
        )
        self._lock = threading.Lock()
        self._mat: dict[str, np.ndarray] = {}
        self._ids: dict[str, list[int]] = {}
        self._load()

    # ----------------------------------------------------------- persistence
    def _load(self) -> None:
        for coll in COLLECTION_MODEL:
            rows = self._db.execute("SELECT memory_id, dim, vec FROM vectors WHERE collection=?", (coll,)).fetchall()
            ids, vecs = [], []
            for mid, dim, blob in rows:
                v = np.frombuffer(blob, dtype=np.float32)
                if v.size != dim:
                    continue
                ids.append(mid)
                vecs.append(v)
            self._ids[coll] = ids
            self._mat[coll] = np.vstack(vecs) if vecs else np.zeros((0, 2048), dtype=np.float32)
            log.info("memory[%s]: %d vectors", coll, len(ids))

    def count(self) -> dict[str, int]:
        return {c: len(i) for c, i in self._ids.items()}

    @staticmethod
    def _norm(v: list[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a

    def _insert(self, coll: str, session_id: Optional[str], kind: str, text: str, meta: dict, vecs: list[np.ndarray]) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO memories(collection, session_id, kind, text, meta, ts) VALUES(?,?,?,?,?,?)",
                (coll, session_id, kind, text, json.dumps(meta, ensure_ascii=False), time.time()),
            )
            mid = int(cur.lastrowid)
            for v in vecs:
                self._db.execute("INSERT INTO vectors(memory_id, collection, dim, vec) VALUES(?,?,?,?)",
                                 (mid, coll, int(v.size), v.astype(np.float32).tobytes()))
                self._ids[coll].append(mid)
                self._mat[coll] = np.vstack([self._mat[coll], v[None, :]]) if self._mat[coll].size else v[None, :].copy()
            self._db.commit()
        return mid

    # ------------------------------------------------------------- writing
    async def remember_dialog(self, session_id: str, user_text: str, assistant_text: str, meta: Optional[dict] = None) -> Optional[int]:
        """Index one user/assistant exchange in the text collection."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text and not assistant_text:
            return None
        doc = f"User: {user_text[:4000]}\nAssistant: {assistant_text[:6000]}"
        try:
            vec = (await self.nim.embed(COLLECTION_MODEL["text"], [doc], "passage"))[0]
        except Exception as e:  # noqa: BLE001
            log.warning("embedding failed (text): %s", e)
            return None
        return self._insert("text", session_id, "dialog", doc, meta or {}, [self._norm(vec)])

    async def remember_analysis(self, session_id: str, question: str, answer: str, files: list[dict],
                                image_data_uris: Optional[list[str]] = None) -> Optional[int]:
        """Index a DeepSeek attachment/screen analysis in the VL collection.

        The text (question + file names + answer) and every image (as data-URI) each get their own
        vector, all pointing at the same memory row, so the memory is reachable by text or by
        visual similarity.
        """
        names = ", ".join(f.get("name", "?") for f in files) if files else ""
        doc = f"Attachments: {names}\nQuestion: {question[:2000]}\nAnalysis: {answer[:6000]}"
        inputs = [doc] + list(image_data_uris or [])[:8]
        try:
            vecs = await self.nim.embed(COLLECTION_MODEL["vl"], inputs, "passage")
        except Exception as e:  # noqa: BLE001
            log.warning("embedding failed (vl): %s", e)
            try:  # images may be rejected (too big etc.) — fall back to text only
                vecs = await self.nim.embed(COLLECTION_MODEL["vl"], [doc], "passage")
            except Exception as e2:  # noqa: BLE001
                log.warning("embedding failed (vl, text only): %s", e2)
                return None
        meta = {"files": [{"name": f.get("name"), "mime": f.get("mime")} for f in files]}
        return self._insert("vl", session_id, "analysis", doc, meta, [self._norm(v) for v in vecs])

    # ------------------------------------------------------------- search
    def _search_vec(self, coll: str, q: np.ndarray, top_k: int, exclude_session: Optional[str], min_score: float) -> list[dict]:
        with self._lock:
            mat = self._mat.get(coll)
            ids = list(self._ids.get(coll, []))
        if mat is None or mat.shape[0] == 0:
            return []
        scores = mat @ q
        order = np.argsort(-scores)
        best: dict[int, float] = {}
        for idx in order:
            s = float(scores[idx])
            if s < min_score:
                break
            mid = ids[idx]
            if mid in best:
                continue
            best[mid] = s
            if len(best) >= top_k * 3:
                break
        if not best:
            return []
        placeholders = ",".join("?" * len(best))
        rows = self._db.execute(
            f"SELECT id, session_id, kind, text, meta, ts FROM memories WHERE id IN ({placeholders})", list(best)
        ).fetchall()
        out = []
        for mid, sid, kind, text, meta, ts in rows:
            if exclude_session and sid == exclude_session:
                continue
            out.append({"id": mid, "collection": coll, "session_id": sid, "kind": kind, "text": text,
                        "meta": json.loads(meta or "{}"), "ts": ts, "score": best[mid]})
        out.sort(key=lambda r: -r["score"])
        return out[:top_k]

    async def search(self, query: str, *, top_k: Optional[int] = None, exclude_session: Optional[str] = None,
                     collections: tuple[str, ...] = ("text", "vl"), min_score: Optional[float] = None) -> list[dict]:
        """Semantic search across past dialogs. Runs both query embeddings concurrently."""
        top_k = top_k or settings.MEMORY_TOP_K
        min_score = settings.MEMORY_MIN_SCORE if min_score is None else min_score
        query = (query or "").strip()
        if not query:
            return []
        active = [c for c in collections if len(self._ids.get(c, [])) > 0]
        if not active:
            return []

        async def one(coll: str) -> list[dict]:
            try:
                vec = (await self.nim.embed(COLLECTION_MODEL[coll], [query[:6000]], "query"))[0]
            except Exception as e:  # noqa: BLE001
                log.warning("query embedding failed (%s): %s", coll, e)
                return []
            return self._search_vec(coll, self._norm(vec), top_k, exclude_session, min_score)

        results = await asyncio.gather(*(one(c) for c in active))
        merged = [r for rs in results for r in rs]
        merged.sort(key=lambda r: -r["score"])
        return merged[:top_k]

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute("SELECT id, collection, session_id, kind, text, ts FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "collection": r[1], "session_id": r[2], "kind": r[3], "text": r[4], "ts": r[5]} for r in rows]

    def format_for_prompt(self, items: list[dict], max_chars: int = 6000) -> str:
        if not items:
            return ""
        lines = []
        used = 0
        for it in items:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(it["ts"]))
            body = it["text"].strip()
            if len(body) > 1500:
                body = body[:1500] + " …"
            entry = f"[{when} · {it['kind']} · relevance {it['score']:.2f}]\n{body}"
            if used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry)
        return "\n\n".join(lines)
