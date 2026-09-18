"""Long-term memory: RAG over past dialogs, stored on the client (client/data/memory.sqlite3).

The server keeps nothing: it only turns texts (and data:image/… URIs) into vectors through its /embed
proxy. Two collections, because the two embedding models live in different vector spaces:
  * "text" — dialog turns and hand-written notes, embedded by nvidia/nemotron-3-embed-1b;
  * "vl"   — turns with attached images (question + answer + the images), embedded by
             nvidia/llama-nemotron-embed-vl-1b-v2 (text and images in one space).

Storage: SQLite (source of truth) + an in-memory normalized numpy matrix per collection for cosine
search; thousands of records search in well under a millisecond.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from .config import settings

log = logging.getLogger("memory")

Embedder = Callable[[str, list[str], str], Awaitable[list[list[float]]]]   # (kind, inputs, input_type) -> vectors
COLLECTIONS = ("text", "vl")

# closing formulas the model loves to append; they only reinforce themselves once stored
_FILLER_RE = re.compile(
    r"(?:^|(?<=[.!?…\n])\s*)(?:(?:чем|как) (?:ещё |еще )?(?:я )?(?:могу|смогу) (?:вам |тебе )?(?:помочь|быть полезен|быть полезна)"
    r"|(?:если|когда) (?:вам |тебе )?(?:понадобится|нужно|нужна|надо|захотите|хотите)(?: будет)? (?:ещё |еще )?(?:что-то|что-нибудь|помощь)[^.!?\n]{0,40}"
    r"|обращайтесь,? если[^.!?\n]{0,30}|(?:всегда )?(?:рад|рада|готов|готова) помочь(?: ещё| еще)?(?: чем-нибудь| чем-то)?"
    r"|how (?:else )?(?:can|may) i (?:help|assist)(?: you)?|is there anything else(?: i can (?:help|do)(?: you)?(?: with)?)?"
    r"|(?:just |please )?let me know if (?:you need|you have|there(?:'s| is)) [^.!?\n]{0,30})\s*[.!?…]*\s*$", re.I)


def strip_filler(text: str) -> str:
    out = (text or "").rstrip()
    for _ in range(3):
        new = _FILLER_RE.sub("", out).rstrip()
        if new == out:
            break
        out = new
    return out.rstrip(" \n\t,;:—-") or (text or "").strip()


class MemoryStore:
    def __init__(self, embed: Embedder, db_path: Path):
        self.embed = embed
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
        for coll in COLLECTIONS:
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
        log.info("memory: %s", self.count())

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
    async def remember_dialog(self, session_id: Optional[str], user_text: str, assistant_text: str, meta: Optional[dict] = None) -> Optional[int]:
        """Index one user/assistant exchange in the text collection."""
        user_text = (user_text or "").strip()
        assistant_text = strip_filler((assistant_text or "").strip())
        if not user_text and not assistant_text:
            return None
        # small talk and test chatter ("Проверка." / "Спасибо") only pollute recall with near-identical junk
        if len(user_text) < 12 and len(assistant_text) < 60:
            return None
        doc = f"User: {user_text[:4000]}\nAssistant: {assistant_text[:6000]}"
        with self._lock:
            dup = self._db.execute("SELECT id FROM memories WHERE collection='text' AND text=? LIMIT 1", (doc,)).fetchone()
        if dup:
            return int(dup[0])
        try:
            vec = (await self.embed("text", [doc], "passage"))[0]
        except Exception as e:  # noqa: BLE001
            log.warning("embedding failed (text): %s", e)
            return None
        return self._insert("text", session_id, "dialog", doc, meta or {}, [self._norm(vec)])

    async def remember_media(self, session_id: Optional[str], question: str, answer: str, files: list[dict],
                             image_data_uris: Optional[list[str]] = None) -> Optional[int]:
        """Index a turn with attached images in the VL collection: the text and every image each get their own
        vector, all pointing at the same record, so it is reachable by text or by visual similarity."""
        names = ", ".join(f.get("name", "?") for f in files) if files else ""
        doc = f"Attachments: {names}\nUser: {question[:2000]}\nAssistant: {strip_filler(answer)[:6000]}"
        inputs = [doc] + list(image_data_uris or [])[:8]
        try:
            vecs = await self.embed("vl", inputs, "passage")
        except Exception as e:  # noqa: BLE001
            log.warning("embedding failed (vl): %s", e)
            try:  # images may be rejected (too big etc.) — fall back to text only
                vecs = await self.embed("vl", [doc], "passage")
            except Exception as e2:  # noqa: BLE001
                log.warning("embedding failed (vl, text only): %s", e2)
                return None
        meta = {"files": [{"name": f.get("name"), "mime": f.get("mime")} for f in files]}
        return self._insert("vl", session_id, "media", doc, meta, [self._norm(v) for v in vecs])

    async def add_note(self, text: str) -> Optional[int]:
        """A memory written by hand: a fact, a preference, an agreement."""
        text = (text or "").strip()
        if not text:
            return None
        vec = (await self.embed("text", [text[:6000]], "passage"))[0]
        return self._insert("text", None, "note", text, {"manual": True}, [self._norm(vec)])

    async def update_text(self, mid: int, text: str) -> bool:
        """Replace a record's text and its text vector (a media record keeps its image vectors)."""
        text = (text or "").strip()
        row = self._db.execute("SELECT collection FROM memories WHERE id=?", (mid,)).fetchone()
        if not row or not text:
            return False
        coll = row[0]
        vec = self._norm((await self.embed(coll, [text[:6000]], "passage"))[0]).astype(np.float32)
        with self._lock:
            self._db.execute("UPDATE memories SET text=? WHERE id=?", (text, mid))
            first = self._db.execute("SELECT id FROM vectors WHERE memory_id=? ORDER BY id LIMIT 1", (mid,)).fetchone()
            if first:   # the first vector of a record is always the text one
                self._db.execute("UPDATE vectors SET dim=?, vec=? WHERE id=?", (int(vec.size), vec.tobytes(), first[0]))
            else:
                self._db.execute("INSERT INTO vectors(memory_id, collection, dim, vec) VALUES(?,?,?,?)", (mid, coll, int(vec.size), vec.tobytes()))
            self._db.commit()
        self._load()
        return True

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
                     collections: tuple[str, ...] = COLLECTIONS, min_score: Optional[float] = None) -> list[dict]:
        """Semantic search across past dialogs; both query embeddings run concurrently."""
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
                vec = (await self.embed(coll, [query[:6000]], "query"))[0]
            except Exception as e:  # noqa: BLE001
                log.warning("query embedding failed (%s): %s", coll, e)
                return []
            return self._search_vec(coll, self._norm(vec), top_k, exclude_session, min_score)

        results = await asyncio.gather(*(one(c) for c in active))
        merged = [r for rs in results for r in rs]
        merged.sort(key=lambda r: -r["score"])
        return merged[:top_k]

    # ------------------------------------------------------------- maintenance
    def list_items(self, limit: int = 500, offset: int = 0) -> list[dict]:
        rows = self._db.execute("SELECT id, collection, session_id, kind, text, ts FROM memories ORDER BY id DESC LIMIT ? OFFSET ?",
                                (max(1, min(limit, 2000)), max(0, offset))).fetchall()
        return [dict(zip(("id", "collection", "session_id", "kind", "text", "ts"), r)) for r in rows]

    def delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            ph = ",".join("?" * len(ids))
            self._db.execute(f"DELETE FROM vectors WHERE memory_id IN ({ph})", ids)
            cur = self._db.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
            self._db.commit()
        self._load()
        return cur.rowcount

    def find_turn(self, session_ids: list[str], user_text: str) -> list[int]:
        """Records of one exchange (dialog or media) from the given sessions, found by the question they store.
        Used for turns remembered before chat messages were linked to their memory records."""
        question = (user_text or "").strip()
        ids = [str(s) for s in session_ids if s]
        if not question or not ids:
            return []
        with self._lock:
            ph = ",".join("?" * len(ids))
            rows = self._db.execute(f"SELECT id, kind, text FROM memories WHERE session_id IN ({ph})", ids).fetchall()
        out = []
        for mid, kind, text in rows:
            if kind == "dialog" and text.startswith(f"User: {question[:4000]}\nAssistant:"):
                out.append(int(mid))
            elif kind == "media" and f"\nUser: {question[:2000]}\nAssistant:" in text:
                out.append(int(mid))
        return out

    def delete_sessions(self, session_ids: list[str]) -> int:
        """Forget everything remembered from the given server sessions (a chat deleted in the sidebar)."""
        ids = [str(s) for s in session_ids if s]
        if not ids:
            return 0
        with self._lock:
            ph = ",".join("?" * len(ids))
            rows = [r[0] for r in self._db.execute(f"SELECT id FROM memories WHERE session_id IN ({ph})", ids).fetchall()]
        return self.delete(rows)

    def prune(self, min_user_chars: int = 12, min_answer_chars: int = 60) -> int:
        """Remove trivial dialog records (short question and short answer) and exact duplicates; cut the
        "Чем могу помочь?" endings out of stored answers."""
        rows = self._db.execute("SELECT id, text FROM memories WHERE collection='text' AND kind='dialog' ORDER BY id").fetchall()
        seen: set[str] = set()
        victims: list[int] = []
        for mid, text in rows:
            user_part, _, answer = text.partition("\nAssistant: ")
            user_part = user_part.removeprefix("User: ")
            cleaned = strip_filler(answer.strip())
            if cleaned != answer.strip():
                text = f"User: {user_part}\nAssistant: {cleaned}"
                with self._lock:
                    self._db.execute("UPDATE memories SET text=? WHERE id=?", (text, mid))
                    self._db.commit()
                answer = cleaned
            if (len(user_part.strip()) < min_user_chars and len(answer.strip()) < min_answer_chars) or text in seen:
                victims.append(mid)
            seen.add(text)
        return self.delete(victims)

    def clear(self) -> int:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self._db.execute("DELETE FROM vectors")
            self._db.execute("DELETE FROM memories")
            self._db.commit()
        self._load()
        return int(n)
