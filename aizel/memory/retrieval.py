"""Извлечение воспоминаний: не «ближайший вектор», а «что всплывает в голове».

Голый косинус похожести — это поиск в Википедии, а не память. Живое существо
вспоминает иначе, поэтому итоговый скоринг собирается из четырёх множителей:

    score = similarity × importance × recency × mood_resonance

  similarity      — косинус bge-m3, смысл, а не совпадение слов.
  importance      — значимость на записи (0.2 бытовуха … 1.0 важное).
  recency         — exp(-возраст/τ). Свежее вспоминается легче.
  mood_resonance  — резонанс с текущим настроением: грустной вспоминается
                    грустное. Работает только по эпизодам с записанным mood.

Дополнительно частое вспоминание укрепляет след (recall_count) — это мостик
к forgetting.py: то, что не вспоминается годами, в Фазе 4 сожмётся в смысл.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .store import MemoryHit, MemoryStore

# Постоянная свежести: 30 дней → воспоминание месячной давности имеет ×0.37.
RECENCY_TAU_DAYS = 30.0
# Насколько настроение влияет на отбор. 0 = выключено.
MOOD_WEIGHT = 0.35
# Усиление от того, сколько раз вспомнили. Ключевое: буст ПРОПОРЦИОНАЛЕН
# похожести, а не добавляется к итогу. Иначе за несколько запросов подряд
# одна и та же находка накапливает буст и становится «магнитом», перебивая
# настоящую похожесть — именно это мы и наблюдали на замерах.
RECALL_GAIN = 0.06
RECALL_CAP = 0.2


@dataclass
class RetrievalSettings:
    top_k: int = 6
    min_score: float = 0.18
    # Порог похожести, ниже которого совпадение считается шумом. Подобран по
    # матрице похожестей BGE-M3 на бытовых фразах: осмысленные совпадения там
    # 0.50–0.65, откровенный шум — 0.1–0.4. Выше 0.50 поднимать нельзя: туда
    # попадают и настоящие, но формулированно далёкие совпадения.
    min_similarity: float = 0.42
    # Не даём одному сеансу занять весь контекст похожими эпизодами.
    max_per_session: int = 2


class Retriever:
    def __init__(self, store: MemoryStore, settings: RetrievalSettings | None = None):
        self.store = store
        self.cfg = settings or RetrievalSettings()

    # ------------------------------------------------------------------ поиск

    def search(
        self,
        query_vec,
        *,
        now: float | None = None,
        mood: float = 0.0,
        top_k: int | None = None,
    ) -> list[MemoryHit]:
        """Возвращает воспоминания, отсортированные по «всплываемости»."""
        now = now or time.time()
        top_k = top_k or self.cfg.top_k
        if query_vec is None:
            return []

        q = np.asarray(query_vec, dtype="float32")
        hits: list[MemoryHit] = []

        for row in self.store.all_episodes():
            sim = float(np.dot(q, _row_vec(row)))
            if sim < self.cfg.min_similarity:
                continue
            age_days = max(0.0, (now - row["ts"]) / 86400.0)
            recency = math.exp(-age_days / RECENCY_TAU_DAYS)
            # 0.5 + 0.5·sim сглаживает: даже точное совпадение не должно
            # полностью перекрывать важность и свежесть.
            base = 0.45 + 0.55 * sim
            importance = 0.4 + 0.6 * float(row["importance"])
            # Резонанс настроения: |m| велик → вспоминается то, что было так же.
            resonance = 1.0
            if MOOD_WEIGHT and mood:
                same_sign = 1.0 if (row["mood"] >= 0) == (mood >= 0) else -1.0
                strength = min(abs(mood), abs(float(row["mood"])))
                resonance = 1.0 + MOOD_WEIGHT * same_sign * strength
            boost = min(RECALL_GAIN * int(row["recall_count"]), RECALL_CAP) * sim
            score = base * importance * (0.55 + 0.45 * recency) * resonance * (1.0 + boost)
            hits.append(
                MemoryHit(
                    id=int(row["id"]),
                    text=row["text"],
                    ts=float(row["ts"]),
                    score=score,
                    similarity=sim,
                    importance=float(row["importance"]),
                    age_days=age_days,
                    mood=float(row["mood"]),
                    source="episode",
                    session=str(row["session"] or ""),
                )
            )

        for row in self.store.all_facts():
            sim = float(np.dot(q, _row_vec(row)))
            if sim < self.cfg.min_similarity:
                continue
            age_days = max(0.0, (now - row["ts"]) / 86400.0)
            # Факты — про «навсегда»: свежесть давит на них гораздо слабее.
            recency = math.exp(-age_days / (RECENCY_TAU_DAYS * 6))
            base = 0.45 + 0.55 * sim
            importance = 0.5 + 0.5 * float(row["importance"])
            score = base * importance * (0.8 + 0.2 * recency)
            hits.append(
                MemoryHit(
                    id=int(row["id"]),
                    text=row["text"],
                    ts=float(row["ts"]),
                    score=score,
                    similarity=sim,
                    importance=float(row["importance"]),
                    age_days=age_days,
                    source="fact",
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        # Сначала фильтруем по качеству, затем разнообразим выдачу. Иначе
        # несколько почти одинаковых эпизодов из одной беседы вытесняют
        # остальные воспоминания из короткого system-контекста.
        selected: list[MemoryHit] = []
        seen_text: set[tuple[str, str]] = set()
        per_session: dict[str, int] = {}
        for hit in hits:
            if hit.score < self.cfg.min_score:
                continue
            key = _normalize_memory_text(hit.text)
            # Дубликаты эпизодов убираем внутри одного сеанса: одинаковая
            # формулировка в другой день может описывать новое событие.
            # Факты дедуплицируем глобально, но не ограничиваем их числом на сеанс.
            if hit.source == "fact":
                session_key = f"fact:{hit.id}"
                dedupe_key = ("facts", key)
            else:
                session_key = hit.session or f"episode:{hit.id}"
                dedupe_key = (session_key, key)
                if per_session.get(session_key, 0) >= max(1, self.cfg.max_per_session):
                    continue
            if key and dedupe_key in seen_text:
                continue
            if key:
                seen_text.add(dedupe_key)
            if hit.source == "episode":
                per_session[session_key] = per_session.get(session_key, 0) + 1
            selected.append(hit)
            if len(selected) >= top_k:
                break
        return selected

    def format_for_context(self, hits: list[MemoryHit]) -> list[str]:
        """Превращает находки в строки контекста с человеческими метками времени.

        «3 дня назад» важнее для тона, чем голое предложение: «он тогда расстроился»
        и «он когда-то давно расстроился» — разная информация.
        """
        out = []
        for h in hits:
            when = _human_age(h.age_days)
            tag = "факт" if h.source == "fact" else "было"
            out.append(f"{when} ({tag}) {h.text}")
        return out


def _normalize_memory_text(text: str) -> str:
    """Сводит различия регистра, пробелов и пунктуации к одному ключу."""
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _row_vec(row) -> np.ndarray:
    from .store import _blob_to_vec

    v = _blob_to_vec(row["vec"], int(row["dim"]))
    if v is None:
        return np.zeros(1, dtype="float32")
    return v


def _human_age(days: float) -> str:
    if days < 1 / 24:
        return "только что"
    if days < 1:
        hours = int(days * 24)
        return f"{hours} ч назад"
    if days < 2:
        return "вчера"
    if days < 7:
        return f"{int(days)} дн назад"
    if days < 31:
        return f"{int(days / 7)} нед назад"
    if days < 365:
        return f"{int(days / 30)} мес назад"
    return f"{days / 365:.1f} г назад"
