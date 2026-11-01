"""Фасад памяти: единая точка, через которую чат и сознание говорят с памятью.

Прячет три разнородных куска (SQLite, BGE-M3, скоринг) за двумя глаголами —
`remember` и `recall`. Если загрузить BGE-M3 не удалось (нет интернета, нет
свободной RAM), фасад не падает, а честно переходит на лексический поиск и
один раз сообщает об этом: молча деградировать хуже, чем сказать.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import unicodedata
import uuid
from pathlib import Path

from ..core.config import ROOT
from ..core.embeddings import DIM, Embedder, EmbedderUnavailable
from .retrieval import Retriever, RetrievalSettings
from .sanitize import prepare
from .semantic import ExtractedFact
from .store import MemoryHit, MemoryStore

log = logging.getLogger("aizel.memory")

# Кэш весов живёт отдельно от базы памяти и НЕ зависит от AIZEL_DATA_DIR:
# гигабайты весов переиспользуются всеми прогонами, перекачивать их ради
# изолированной отладки бессмысленно.
DEFAULT_MODELS_DIR = ROOT / "data" / "models"

# Хэш-векторы дают косинусы принципиально другой величины, чем BGE-M3
# (совпадение основ слов → редкие размерности, скалярное произведение мелкое).
# Порог 0.42 из RetrievalSettings под семантику их бы отбраковывал все.
LEXICAL_MIN_SIMILARITY = 0.12

# Только эти поля разрешено помещать в постоянное ядро из ответа модели.
# Одиночные значения заменяют прежнее; семейные сведения могут накапливаться.
CORE_SINGLETON_SLOTS = frozenset({"имя", "город", "возраст", "работа"})
CORE_MULTI_SLOTS = frozenset({"семья"})
CORE_SLOTS = CORE_SINGLETON_SLOTS | CORE_MULTI_SLOTS


def _memory_key(text: str) -> str:
    """Нормализованный ключ для дедупликации фактов и сведений ядра."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\\w+", normalized, flags=re.UNICODE))


class Memory:
    def __init__(
        self,
        data_dir: Path,
        *,
        force_lexical: bool = False,
        models_dir: Path | None = None,
    ):
        """models_dir — куда кладётся кэш BGE-M3. Отдельно от памяти: веса
        переиспользуются между прогонами и тестами, а не скачиваются в tmp."""
        self.store = MemoryStore(data_dir / "aizel.db")
        cache = models_dir if models_dir is not None else DEFAULT_MODELS_DIR
        self.embedder = Embedder(cache_dir=cache)
        self.retriever = Retriever(self.store, RetrievalSettings())
        # Лексический фолбэк и BGE-M3 дают несопоставимые косинусы: у хэш-векторов
        # они от природы в разы меньше. Один порог на оба режима невозможен —
        # поэтому шкалу подбираем под активный движок.
        if force_lexical:
            self.retriever.cfg.min_similarity = LEXICAL_MIN_SIMILARITY
        self.session = uuid.uuid4().hex[:8]
        self.degraded = force_lexical
        self._warned = False
        if force_lexical:
            log.warning("память в лексическом режиме (эмбеддинги выключены)")

    # ------------------------------------------------------------- кодирование

    def _encode_docs(self, texts: list[str]):
        if self.degraded:
            return [Embedder.lexical_fallback(t) for t in texts]
        try:
            return self.embedder.encode_docs(texts)
        except EmbedderUnavailable as exc:
            self._degrade(exc)
            return [Embedder.lexical_fallback(t) for t in texts]

    def _encode_query(self, text: str):
        if self.degraded:
            return Embedder.lexical_fallback(text)
        try:
            return self.embedder.encode_query(text)
        except EmbedderUnavailable as exc:
            self._degrade(exc)
            return Embedder.lexical_fallback(text)

    def _degrade(self, exc: Exception) -> None:
        self.degraded = True
        self.retriever.cfg.min_similarity = LEXICAL_MIN_SIMILARITY
        if not self._warned:
            self._warned = True
            log.warning(
                "BGE-M3 недоступна (%s) — память переключается на поиск по словам. "
                "Смысловые совпадения без общих слов найдены не будут.",
                exc,
            )

    # ---------------------------------------------------------------- запись

    def remember_message(self, role: str, text: str) -> None:
        self.store.add_message(role, text, session=self.session)

    def remember_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        mood: float = 0.0,
        importance: float | None = None,
        topic: str = "",
    ) -> int:
        """Один обмен репликами = один эпизод.

        Храним реплику пользователя как носитель смысла: именно она потом
        вспомнится как «он спрашивал про …». Ответ модели — контекст эпизода.

        Шум («ага», «ну», служебные строки консоли) в память не попадает
        вообще: короткие мусорные строки исчисляются сотнями и вытесняют
        настоящие воспоминания при отборе.
        """
        text = prepare(user_text)
        if not text:
            return 0
        # Не превращаем голые вопросы («как меня зовут?») в воспоминания.
        # Вопрос с личным утверждением («я люблю звёзды, знаешь?») сохраняем.
        if text.rstrip().endswith("?") and not re.search(
            r"^\s*(?:я\s+(?:люблю|обожаю|ненавижу|боюсь|мечтаю|хочу|работаю|учусь|живу|играю|занимаюсь|увлекаюсь|предпочитаю|создал|создала)|меня\s+зовут\s+[А-ЯЁA-Z]|у\s+меня\s+есть|мне\s+\d{1,2}\s*(?:лет|год|года)|мой\s+\w+|моя\s+\w+|моё\s+\w+|мои\s+\w+)\b",
            text, re.I
        ):
            return 0
        if importance is None:
            importance = _guess_importance(text)
        vec = self._encode_docs([text])[0]
        return self.store.add_episode(
            text,
            vec,
            importance=importance,
            mood=mood,
            topic=topic or text[:60],
            session=self.session,
        )

    def remember_fact(self, text: str, importance: float = 0.6) -> int:
        clean = prepare(text)
        key = _memory_key(clean)
        if not clean or not key:
            return 0
        known = {
            _memory_key(row["text"])
            for row in self.store.db.execute(
                "SELECT text FROM facts WHERE superseded=0"
            )
        }
        if key in known:
            return 0
        vec = self._encode_docs([clean])[0]
        return self.store.add_fact(clean, vec, importance=importance)

    def remember_facts(self, pairs: list[tuple[str, float]]) -> int:
        """Добавляет только новые факты; дубли внутри пачки тоже отбрасываются."""
        if not pairs:
            return 0
        known = {
            _memory_key(row["text"])
            for row in self.store.db.execute(
                "SELECT text FROM facts WHERE superseded=0"
            )
        }
        fresh: list[tuple[str, float]] = []
        for raw_text, importance in pairs:
            clean = prepare(raw_text)
            key = _memory_key(clean)
            if not clean or not key or key in known:
                continue
            known.add(key)
            fresh.append((clean, importance))
        if not fresh:
            return 0
        vecs = self._encode_docs([text for text, _ in fresh])
        for (text, importance), vec in zip(fresh, vecs):
            self.store.add_fact(text, vec, importance=importance)
        return len(fresh)

    def quick_core(self, user_text: str) -> list[str]:
        """Мгновенно вылавливает имя/город/возраст/работу — без обращения к модели.

        Фоновый прогон извлечения стоит 20–40 секунд и ждёт своей очереди к
        движку. За это время человек успевает задать «как меня зовут?» и
        услышать, что она не знает, — именно на это и жаловалось «почти не
        помнит». Крупные личные сведения имеют устойчивую форму и выхватываются
        регуляркой сразу, в той же реплике. Возвращает список слотов, которые
        этим текстом обновились, — по нему чат решает, надо ли догружать ядро.
        """
        from .semantic import quick_core as detect_core

        body = prepare(user_text)
        if not body:
            return []
        touched: list[str] = []
        for fact in detect_core(body):
            self.store.upsert_core(fact.slot, fact.text)
            touched.append(fact.slot)
        if touched:
            log.info("ядро обновлено сразу (без модели): %s", ", ".join(touched))
        return touched

    def apply_facts(self, facts: list[ExtractedFact]) -> dict[str, int]:
        """Раскладывает извлечённое по двум разным хранилищам.

        Разделение принципиальное, оно стоит за измерением: «где я живу» и
        «как меня зовут» по вектору ищутся плохо — правильные и случайные
        кандидаты различаются в третьем знаке. Поэтому одиночные устойчивые
        сведения (slot) кладутся в ядро и попадают в контекст ВСЕГДА, а всё
        прочее — в векторный поиск, где они сильны.
        """
        core_count = 0
        searchable: list[tuple[str, float]] = []
        seen_core: set[tuple[str, str]] = set()

        existing_multi = {
            (slot, _memory_key(text))
            for slot in CORE_MULTI_SLOTS
            for text in self.store.get_core(slot)
        }
        for fact in facts:
            clean = prepare(fact.text)
            if not clean:
                continue
            slot = (fact.slot or "").strip().casefold()
            if slot not in CORE_SLOTS:
                # Неизвестный slot не должен создавать произвольное поле ядра.
                # Сам текст при этом не теряем: он остаётся обычным фактом.
                searchable.append((clean, fact.importance))
                continue

            key = _memory_key(clean)
            pair = (slot, key)
            if not key or pair in seen_core:
                continue
            seen_core.add(pair)

            singleton = slot in CORE_SINGLETON_SLOTS
            if not singleton and pair in existing_multi:
                continue
            self.store.upsert_core(slot, clean, singleton=singleton)
            existing_multi.add(pair)
            core_count += 1

        added = self.remember_facts(searchable)
        return {"core": core_count, "facts": added}

    # --------------------------------------------------------------- извлечение

    def recall(
        self, query: str, *, mood: float = 0.0, top_k: int | None = None
    ) -> list[MemoryHit]:
        if not query.strip():
            return []
        qvec = self._encode_query(query)
        hits = self.retriever.search(qvec, mood=mood, top_k=top_k)
        if hits:
            self.store.touch_recall([h.id for h in hits if h.source == "episode"])
        return hits

    def context_lines(self, query: str, *, mood: float = 0.0, top_k: int = 6) -> list[str]:
        """Готовые строки для системного блока."""
        hits = self.recall(query, mood=mood, top_k=top_k)
        return self.retriever.format_for_context(hits)

    # ------------------------------------------------------------------ служебное

    def warm_up(self) -> str:
        """Прогревает модель эмбеддингов. Возвращает человекочитаемый статус."""
        if self.degraded:
            return "лексический режим (без BGE-M3)"
        try:
            self.embedder.load()
            return f"BGE-M3, dim={self.embedder.dim or DIM}, cpu"
        except EmbedderUnavailable as exc:
            self._degrade(exc)
            return "лексический режим (BGE-M3 не загрузилась)"

    def close(self) -> None:
        self.store.close()

    def probe_text(self, history: list[dict], current: str) -> str:
        """Собирает текст для запроса памяти из истории + текущего сообщения.

        Если спросить память только по «привет», она ответит «ничего не помню».
        Но если передать последние 3-4 реплики из истории, она вспомнит,
        о чём говорили 5 минут назад — и может использовать их как контекст.
        """
        if not history:
            return current
        # Берём последние 6 реплик (3 оборота) и добавляем текущее сообщение.
        tail = history[-6:]
        parts = [f"{r['role']}: {r['content']}" for r in tail if r['content']]
        parts.append(current)
        return " ".join(parts)

    def forget_all(self, journal_dir: Path | None = None) -> Path:
        """Стирает всю память, предварительно упаковав её в архив.

        Сперва копия, потом стирание — наоборот быть не может: «забыть» должно
        оставаться обратимым хотя бы до конца дня. Базу копируем через
        sqlite3 backup API, а не shutil.copy(): при WAL-режиме голый копир
        файла рискует отдать состояние без незакрытых транзакций.
        """
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = self.store.path.parent / "forget_backups" / stamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(backup_dir / "aizel.db")) as dst:
            self.store.db.backup(dst)
        if journal_dir and journal_dir.exists():
            shutil.copytree(journal_dir, backup_dir / "dialoglog")

        self.store.purge()
        log.info("память стёрта, копия: %s", backup_dir)
        return backup_dir


# Ключевые слова, по которым на глаз определяем значимость. Грубо, но до
# Фазы 3, когда важность начнёт считать сама модель по контексту, достаточно.
_HIGH_SIGNAL = (
    "люб", "ненавист", "боюсь", "боишься", "мечта", "друг", "подруг", "брат",
    "сестр", "мать", "матери", "отец", "родител", "женат", "замуж", "разошл",
    "разошлись", "умер", "сдох", "похорон", "работ", "уволь", "учусь", "универ",
    "болезн", "врач", "диагноз", "переех", "день рожден", "годах", "зовут",
    "имя", "поссор", "обид", "бросил", "бросила", "поехал", "поехала",
)
_LOW_SIGNAL = ("привет", "как дела", "пока", "спасибо", "ок", "ага", "ну")


def _guess_importance(text: str) -> float:
    """Оценивает значимость реплики без обращения к модели.

    Вопрос по определению не содержит факта: «как меня зовут?» содержит слово
    «зовут», но помнить тут нечего — помнить надо ответ. Раньше именно это
    давало «как меня зовут?» важность 0.85; накопив 22 вспоминания, такой
    вопрос начинал затыкать собой весь поиск.

    Поэтому признак вопроса (точка с вопросом в конце) проверяется ДО
    ключевых слов, а не после. Порядок здесь — не косметика.
    """
    low = text.strip().lower()
    if low.endswith("?"):
        return 0.25
    # Низкая важность только если реплика состоит из одного такого слова
    # или очень короткая (меньше 20 символов). «спасибо за помощь…» —
    # это уже благодарность + контекст, важность 0.5.
    low_words = re.split(r"\W+", low)
    if any(k in low_words for k in _LOW_SIGNAL):
        if len(low) >= 20:
            return 0.5  # длинная фраза — не просто «спасибо»
        return 0.25
    if any(k in low for k in _HIGH_SIGNAL):
        return 0.85
    return 0.5
