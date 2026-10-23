"""Хранилище памяти Айзель: SQLite + векторы в BLOB.

Одна база, несколько таблиц, разная «скорость жизни» у данных:

    messages   — дословные реплики. Прирастает на каждую.
    episodes   — смысловые куски разговора (то, что вспоминается).
    facts      — устойчивые факты («любит астрономию»). Живут дольше эпизодов.
    core       — ядро памяти: короткие строки, всегда в контексте.
    meta       — служебное (схема, счётчики).

Векторы лежат рядом с текстом (BLOB float32): поиск не ходит в отдельные файлы,
база остаётся единым переносимым артефактом — скопировал aizel.db, перенёс память.

WAL включён: ядро и чат могут работать с базой одновременно.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    session     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    text         TEXT NOT NULL,
    vec          BLOB,
    dim          INTEGER NOT NULL DEFAULT 0,
    importance   REAL NOT NULL DEFAULT 0.5,
    mood         REAL NOT NULL DEFAULT 0.0,
    topic        TEXT NOT NULL DEFAULT '',
    recall_count INTEGER NOT NULL DEFAULT 0,
    last_recall  REAL NOT NULL DEFAULT 0.0,
    consolidated INTEGER NOT NULL DEFAULT 0,
    session      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    text         TEXT NOT NULL,
    vec          BLOB,
    dim          INTEGER NOT NULL DEFAULT 0,
    kind         TEXT NOT NULL DEFAULT 'user',
    importance   REAL NOT NULL DEFAULT 0.6,
    confidence   REAL NOT NULL DEFAULT 0.5,
    hits         INTEGER NOT NULL DEFAULT 0,
    superseded   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS core (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    slot      TEXT NOT NULL,
    text      TEXT NOT NULL,
    updated   REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_core_slot ON core(slot, text);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class MemoryHit:
    """Найденное воспоминание со всеми составляющими оценки."""

    id: int
    text: str
    ts: float
    score: float
    similarity: float
    importance: float
    age_days: float
    mood: float = 0.0
    source: str = "episode"


def _blob_to_vec(blob: bytes | None, dim: int):
    if not blob or dim <= 0:
        return None
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._check_version()

    def _check_version(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES('schema',?)", (str(SCHEMA_VERSION),)
            )
            self.db.commit()
            return
        if int(row["value"]) > SCHEMA_VERSION:
            raise RuntimeError(
                f"База {self.path.name} создана более новой версией Айзель "
                f"(схема {row['value']}, ожидается ≤{SCHEMA_VERSION})."
            )

    def close(self) -> None:
        self.db.close()

    def purge(self) -> None:
        """Полностью стирает все воспоминания, оставляя структуру таблиц."""
        for table in ("episodes", "facts", "messages", "core"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()

    # ---------------------------------------------------------------- записи

    def add_message(self, role: str, text: str, session: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO messages(ts, role, text, session) VALUES(?,?,?,?)",
            (time.time(), role, text, session),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_episode(
        self,
        text: str,
        vec,
        *,
        importance: float = 0.5,
        mood: float = 0.0,
        topic: str = "",
        session: str = "",
    ) -> int:
        blob = bytes(vec.astype("float32").tobytes()) if vec is not None else None
        dim = int(vec.shape[0]) if vec is not None else 0
        cur = self.db.execute(
            """INSERT INTO episodes(ts, text, vec, dim, importance, mood, topic, session)
               VALUES(?,?,?,?,?,?,?,?)""",
            (time.time(), text, blob, dim, importance, mood, topic, session),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_fact(self, text: str, vec, *, kind: str = "user", importance: float = 0.6) -> int:
        blob = bytes(vec.astype("float32").tobytes()) if vec is not None else None
        dim = int(vec.shape[0]) if vec is not None else 0
        cur = self.db.execute(
            """INSERT INTO facts(ts, text, vec, dim, kind, importance)
               VALUES(?,?,?,?,?,?)""",
            (time.time(), text, blob, dim, kind, importance),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def set_core(self, slot: str, items: list[str]) -> None:
        """Полностью заменяет содержимое слота ядра памяти."""
        now = time.time()
        self.db.execute("DELETE FROM core WHERE slot=?", (slot,))
        for text in items:
            if text:
                self.db.execute(
                    "INSERT INTO core(slot, text, updated) VALUES(?,?,?)", (slot, text, now)
                )
        self.db.commit()

    def get_core(self, slot: str) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM core WHERE slot=? ORDER BY updated", (slot,)
        ).fetchall()
        return [r["text"] for r in rows]

    def upsert_core(self, slot: str, text: str, *, singleton: bool = True) -> None:
        """Записывает сведения в ядро.

        singleton=True — «значение одно»: новый город отменяет старый. Иначе
        это просто ещё одна заметка, и дубликат по тексту не плодим.
        """
        text = text.strip()
        if not text:
            return
        now = time.time()
        if singleton:
            self.db.execute("DELETE FROM core WHERE slot=?", (slot,))
        else:
            exists = self.db.execute(
                "SELECT 1 FROM core WHERE slot=? AND text=?", (slot, text)
            ).fetchone()
            if exists:
                return
        self.db.execute("INSERT INTO core(slot, text, updated) VALUES(?,?,?)", (slot, text, now))
        self.db.commit()

    def all_core(self) -> list[str]:
        """Всё ядро одним списком «слот: значение» — то, что идёт в контекст всегда."""
        rows = self.db.execute(
            "SELECT slot, text FROM core ORDER BY slot, updated"
        ).fetchall()
        return [f"{r['slot']}: {r['text']}" for r in rows]

    # -------------------------------------------------------------- чтения

    def all_episodes(self, limit: int = 20000):
        """Все невспомненные эпизоды с векторами. Brute-force поиск по ним.

        Ограничение по количеству — страховка: пока выборка не упрётся в лимит,
        точный поиск честнее приближённого, а 20k векторов по 1024 f32
        считаются за десятки миллисекунд.
        """
        return self.db.execute(
            """SELECT id, ts, text, vec, dim, importance, mood, topic, recall_count
               FROM episodes WHERE vec IS NOT NULL ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def all_facts(self, limit: int = 5000):
        return self.db.execute(
            """SELECT id, ts, text, vec, dim, importance FROM facts
               WHERE superseded=0 AND vec IS NOT NULL ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def touch_recall(self, ids: list[int]) -> None:
        """Отмечаем, что воспоминание вспомнили: частое укрепляется (Фаза 4)."""
        if not ids:
            return
        now = time.time()
        self.db.executemany(
            "UPDATE episodes SET recall_count=recall_count+1, last_recall=? WHERE id=?",
            [(now, i) for i in ids],
        )
        self.db.commit()

    def recent_messages(self, limit: int = 12, session: str | None = None) -> list[dict]:
        if session:
            rows = self.db.execute(
                "SELECT role, text FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
                (session, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"role": r["role"], "content": r["text"]} for r in reversed(rows)]

    def stats(self) -> dict:
        q = lambda t: self.db.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]  # noqa: E731
        return {
            "messages": q("messages"),
            "episodes": q("episodes"),
            "facts": q("facts"),
            "core": q("core"),
            "size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0,
        }

    def dump_debug(self, limit: int = 5) -> str:
        rows = self.db.execute(
            "SELECT ts, text, importance, recall_count FROM episodes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            stamp = time.strftime("%d.%m %H:%M", time.localtime(r["ts"]))
            out.append(
                f"  [{stamp}] imp={r['importance']:.2f} recalled={r['recall_count']} "
                f"| {r['text'][:110]}"
            )
        return "\n".join(out)
