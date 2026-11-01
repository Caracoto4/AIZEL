"""Детерминированная модель внутренних побуждений AIZEL.

Побуждения — вычисляемые поведенческие приоритеты, а не реальные чувства
или автономные потребности. Слой не вызывает LLM и не отправляет сообщения
самостоятельно: он лишь добавляет мягкий фокус в контекст следующего ответа.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_BASELINE = {
    "curiosity": 0.22,
    "engagement": 0.28,
    "continuity": 0.18,
    "reflection": 0.12,
    "initiative": 0.16,
}
_DECAY = 0.10


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class DriveState:
    """Состояния побуждений в диапазоне [0, 1], с инерцией и затуханием."""

    values: dict[str, float] = field(default_factory=lambda: dict(_BASELINE))
    last_intention: str = "respond"

    def __post_init__(self) -> None:
        for name, baseline in _BASELINE.items():
            self.values[name] = _clamp(self.values.get(name, baseline))

    def reset(self) -> None:
        self.values = dict(_BASELINE)
        self.last_intention = "respond"

    def observe(self, user_text: str, *, has_history: bool = False, personality=None) -> None:
        """Обновляет побуждения по явным признакам текущего хода."""
        for name, baseline in _BASELINE.items():
            current = self.values.get(name, baseline)
            self.values[name] = _clamp(current + (baseline - current) * _DECAY)

        text = " ".join((user_text or "").lower().replace("ё", "е").split())
        if not text:
            return

        if "?" in text or text.startswith(
            ("как ", "почему ", "зачем ", "что ", "когда ", "где ", "можно ли ")
        ):
            self._add("curiosity", 0.18)
        if any(term in text for term in ("расскажи", "объясни", "интересно", "подробнее", "хочу узнать")):
            self._add("curiosity", 0.12)

        if len(text) >= 55:
            self._add("engagement", 0.10)
        if any(term in text for term in ("продолжи", "еще", "давай дальше", "расскажи больше")):
            self._add("engagement", 0.20)

        if has_history and any(
            term in text
            for term in ("а если", "тогда", "а почему", "к этому", "насчет", "вернемся")
        ):
            self._add("continuity", 0.22)
        if any(term in text for term in ("ты не поняла", "неправильно", "я имел в виду", "я имела в виду", "исправь")):
            self._add("reflection", 0.24)

        openness = self._trait(personality, "O", 0.5)
        extraversion = self._trait(personality, "E", 0.5)
        if "?" in text or len(text) >= 55:
            self._add("initiative", 0.035 * (0.5 + openness))
        if any(term in text for term in ("продолжи", "подробнее", "интересно")):
            self._add("initiative", 0.045 * (0.5 + extraversion))

    def _add(self, name: str, amount: float) -> None:
        self.values[name] = _clamp(self.values.get(name, _BASELINE[name]) + amount)

    @staticmethod
    def _trait(personality, name: str, fallback: float) -> float:
        if personality is None:
            return fallback
        try:
            return _clamp(personality.ocean.get(name, fallback))
        except AttributeError:
            return fallback

    def choose_intention(self, personality=None) -> str:
        """Выбирает один мягкий фокус; при слабых сигналах — обычный ответ."""
        openness = self._trait(personality, "O", 0.5)
        extraversion = self._trait(personality, "E", 0.5)
        scores = {
            "clarify": self.values["curiosity"] * (0.65 + 0.35 * openness),
            "continue": self.values["engagement"] * (0.65 + 0.35 * extraversion),
            "connect": self.values["continuity"] * 0.95,
            "reflect": self.values["reflection"] * 0.95,
            "suggest": self.values["initiative"] * (0.45 + 0.30 * openness + 0.25 * extraversion),
        }
        intention, score = max(scores.items(), key=lambda item: item[1])
        self.last_intention = intention if score >= 0.34 else "respond"
        return self.last_intention

    def directive(self, personality=None) -> str:
        """Короткая инструкция для system-контекста, без принуждения к инициативе."""
        intention = self.choose_intention(personality)
        instructions = {
            "respond": "Сейчас достаточно просто ответить на текущую реплику; не ищи повод проявлять инициативу.",
            "clarify": "Есть повод проявить любопытство: если это действительно поможет, задай один уместный уточняющий вопрос.",
            "continue": "Тема явно развита собеседником: можно содержательно продолжить её, не повторяя уже сказанное.",
            "connect": "Свяжи текущую реплику с предыдущим контекстом, если связь действительно есть.",
            "reflect": "Проверь, не ошиблась ли ты ранее; признай конкретную ошибку и скорректируй ответ, если это нужно.",
            "suggest": "Можно предложить один конкретный следующий шаг, только если он прямо полезен в текущей теме.",
        }
        return (
            "Текущий поведенческий фокус (модель побуждений): "
            + instructions[intention]
            + " Это предпочтение, а не обязательная цель. Всегда сначала отвечай на запрос."
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"values": self.values, "last_intention": self.last_intention},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "DriveState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                values={**_BASELINE, **(data.get("values") or {})},
                last_intention=data.get("last_intention", "respond"),
            )
        except (OSError, ValueError, TypeError):
            return cls()
