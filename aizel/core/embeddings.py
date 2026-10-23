"""Смысловое ядро памяти: BAAI/bge-m3 на CPU.

Почему на CPU, а не на GPU вместе с LLM: 12B уже заняла почти все 8 ГБ VRAM,
любой ещё один резидентный тензор вытеснит слои LLM обратно в RAM и генерация
встанет. BGE-M3 на CPU — это ~0.1–0.3 с на запрос, для фона это несущественно,
зато скорость разговора не страдает.

Модель грузится лениво: `doctor` и чат без памяти не обязаны поднимать 2 ГБ весов.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

log = logging.getLogger("aizel.embeddings")

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024

# bge-m3 документирован с instruction-префиксом для веб-поиска. Мы измерили
# это на НАШЕЙ задаче (короткие бытовые факты о человеке, RU↔RU) — и документация
# проиграла: с английским web-search-префиксом правильные и неправильные кандидаты
# давали 0.383 против 0.374, то есть порог между ними провести нельзя.
# Без префикса: 4/4 верных и скоры 0.59–0.65. Причина проста: префикс разворачивает
# запрос в «режим поиска по википедии», а наши документы — реплики от первого лица.
# Префикс оставили настраиваемым, но по умолчанию он пуст.
QUERY_PREFIX = ""
DOC_PREFIX = ""


class EmbedderUnavailable(RuntimeError):
    pass


class Embedder:
    """Однопоточная ленивая обёртка над SentenceTransformer.

    Токенизация не тред-безопасна в общем случае, а фоновое сознание и чат
    будут дёргать память из разных мест — поэтому всё через блокировку.
    """

    def __init__(self, cache_dir: Path | None = None, device: str = "cpu"):
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderUnavailable(
                    "sentence-transformers не установлен. "
                    "pip install -r requirements.txt (секция Фазы 2)"
                ) from exc
            log.info("гружу %s на %s (первый запуск качает ~2.3 ГБ)…", MODEL_NAME, self.device)
            kwargs = {"device": self.device, "trust_remote_code": False}
            if self.cache_dir:
                kwargs["cache_folder"] = str(self.cache_dir)
            try:
                # Если веса уже лежат локально — не дёргаем сеть вообще.
                # Иначе каждый запуск превращается в запрос к HF, а без
                # интернета Айзель вовсе не проснётся.
                if self._cached():
                    self._model = SentenceTransformer(MODEL_NAME, local_files_only=True, **kwargs)
                else:
                    self._model = SentenceTransformer(MODEL_NAME, **kwargs)
            except Exception:
                if not self._cached():
                    raise
                # Локальная копия есть, но не устроила (например, не хватает
                # файла конфигa) — пробуем докачать, если сеть доступна.
                try:
                    self._model = SentenceTransformer(MODEL_NAME, **kwargs)
                except Exception as exc:
                    raise EmbedderUnavailable(
                        f"не удалось загрузить {MODEL_NAME} ни локально, ни из сети: {exc}"
                    ) from exc
            log.info("%s готова, dim=%d", MODEL_NAME, self.dim)

    def _cached(self) -> bool:
        """Есть ли уже скачанные веса (тогда сеть не нужна).

        Незавершённые .incomplete не считаем: они от прерванной загрузки и
        модель из них не соберётся.
        """
        if not self.cache_dir:
            return False
        root = Path(self.cache_dir)
        return any(
            p.is_file() and p.stat().st_size > 100_000_000 and not p.name.endswith(".incomplete")
            for p in root.glob("models--*--bge-m3/blobs/*")
        )

    @property
    def dim(self) -> int:
        if self._model is None:
            return DIM
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except Exception:
            return DIM

    # ------------------------------------------------------------------ encode

    def encode_docs(self, texts: list[str]):
        """Кодирует документы (воспоминания, факты)."""
        self.load()
        with self._lock:
            vecs = self._model.encode(
                [DOC_PREFIX + t for t in texts],
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vecs.astype("float32")

    def encode_query(self, text: str):
        """Кодирует запрос поиска (со служебной инструкцией bge-m3)."""
        self.load()
        with self._lock:
            vec = self._model.encode(
                [QUERY_PREFIX + text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vec[0].astype("float32")

    # ------------------------------------------------- дешёвый резервный вариант

    @staticmethod
    def lexical_fallback(text: str, dim: int = DIM):
        """Хэш-эмбеддинги по основам слов, если модели нет.

        Это НЕ семантика: работает на совпадении основ. Нужна, чтобы чат не
        падал без BGE-M3, а не чтобы делать вид, что память работает.
        """
        import math
        import numpy as np

        v = np.zeros(dim, dtype="float32")
        for token in _tokens(text):
            h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")
            v[h % dim] += 1.0
        norm = math.sqrt(float((v * v).sum()))
        if norm > 0:
            v /= norm
        return v


# Хвосты, которые срезаем, чтобы «звёздами» и «звёзды» стали одним словом.
# Без этого фолбэк на русской морфологии бесполезен: совпадений почти нет.
# Порядок важен: сначала длинные окончания, потом короткие.
_SUFFIXES = (
    "иями", "ого", "его", "ому", "ему", "ыми", "ими", "ыми",
    "ами", "ями", "ах", "ях", "ов", "ев", "ий", "ый", "ая", "яя",
    "ое", "ее", "ые", "ие", "ой", "ей", "ом", "ем", "ам", "ям",
    "ую", "юю", "ла", "ло", "ли", "ти", "ть", "а", "о", "е", "ы",
    "и", "я", "ь", "у", "ю",
)


def _stem(word: str) -> str:
    """Эвристическая основа слова: ё→е + срезание частых окончаний."""
    w = word.replace("ё", "е")
    if len(w) <= 4:
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    import re

    words = re.findall(r"[а-яa-z0-9]+", text.lower())
    return [_stem(w) for w in words]
