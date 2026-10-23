"""Настройки Айзель: где живёт мозг, сколько памяти, что писать в лог.

Всё читается из config.yaml в корне проекта, недостающее берётся из дефолтов.
Пароли и ключи сюда не кладём — Айзель полностью локальная.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Корень проекта: .../Aizel  (файл лежит в Aizel/aizel/core/config.py)
ROOT = Path(__file__).resolve().parents[2]
# Память можно вынести переменной: отладочные прогоны обязаны идти в свою
# папку, иначе тестовые реплики въедаются в живую память (уже случалось).
DATA_DIR = Path(os.getenv("AIZEL_DATA_DIR", str(ROOT / "data")))
SOUL_DIR = ROOT / "aizel" / "soul"
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class BrainConfig:
    """Подключение к KoboldCpp (OpenAI-совместимый API)."""

    base_url: str = "http://localhost:5001/v1"
    model: str = "local"
    # Родной чат-шаблон модели. В логе KoboldCpp видно «Alpaca will be used» —
    # при use_jinja=true идёт встроенный Llama-3 шаблон, тон заметно точнее.
    use_jinja: bool = True
    max_tokens: int = 420
    temperature: float = 0.85
    top_p: float = 0.92
    top_k: int = 40
    min_p: float = 0.05
    rep_pen: float = 1.12
    freq_pen: float = 0.3
    # Контекст у нас ~8300 токенов (Autofit), поэтому потолок держим ниже.
    context_limit: int = 8192
    timeout_sec: float = 300.0
    retries: int = 2

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "data/aizel.log"

    @property
    def path(self) -> Path:
        return ROOT / self.file


@dataclass
class MemoryConfig:
    """Память. BGE-M3 живёт на CPU — в VRAM уже почти не осталось места под LLM."""

    force_lexical: bool = False   # true = работать без BGE-M3 (поиск по словам)
    top_k: int = 6                # сколько воспоминаний подмешивать в контекст
    extract_on_exit: bool = True  # извлекать устойчивые факты при /exit
    embeddings_device: str = "cpu"


@dataclass
class AizelConfig:
    brain: BrainConfig = field(default_factory=BrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    soul_file: str = "aizel.yaml"

    @property
    def soul_path(self) -> Path:
        return SOUL_DIR / self.soul_file

    def ensure_dirs(self) -> None:
        """Папки создаём лениво: data/ в gitignore и появляется только при запуске."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)


def _merge(obj, data: dict) -> None:
    """Поверхностно накладываем поля из YAML на датакласс, незнакомые игнорируем."""
    for key, value in (data or {}).items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> AizelConfig:
    """Читает config.yaml, если он есть. Отсутствие файла — не ошибка."""
    cfg = AizelConfig()
    target = path or CONFIG_PATH
    if target.exists():
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        _merge(cfg, {k: v for k, v in raw.items() if k not in ("brain", "logging", "memory")})
        _merge(cfg.brain, raw.get("brain", {}))
        _merge(cfg.logging, raw.get("logging", {}))
        _merge(cfg.memory, raw.get("memory", {}))
    # Переменная окружения полезнее файла: удобно для запуска и тестов.
    if os.getenv("AIZEL_BASE_URL"):
        cfg.brain.base_url = os.environ["AIZEL_BASE_URL"]
    cfg.ensure_dirs()
    return cfg
