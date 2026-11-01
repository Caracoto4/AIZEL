"""Оценка эмоционально значимых сигналов в реплике собеседника.

Это лёгкий локальный слой: не вызывает LLM и не пытается угадывать настроение
человека по любой негативной фразе. Реагирует только на явную благодарность,
недовольство ответом или прямую враждебность в адрес Айзель.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalReaction:
    pleasure: float
    arousal: float
    dominance: float
    cause: str
    sensitivity: float = 1.0


# Только достаточно явные формулировки. Нейтральная критика темы или проблемы
# не должна автоматически восприниматься как личная обида.
_APPRECIATION = (
    "спасибо",
    "благодарю",
    "ты помогла",
    "мне помогло",
    "это полезно",
    "классно",
    "круто",
    "отлично",
    "идеально",
    "мне нравится",
    "хорошо получилось",
    "вот это то",
    "супер",
)

_FRUSTRATION = (
    "не то",
    "это не то",
    "опять не то",
    "ты опять",
    "неправильно",
    "ты не понимаешь",
    "ты не поняла",
    "не нравится",
    "мимо",
)

_DIRECT_HOSTILITY = (
    "ты тупая",
    "ты бесполезная",
    "ты идиотка",
    "ненавижу тебя",
    "бесишь",
    "достала",
    "замолчи",
    "отстань",
)


def _has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def appraise_user_message(
    text: str,
    *,
    touchiness: float = 0.5,
) -> EmotionalReaction | None:
    """Возвращает небольшой PAD-импульс или None для нейтральной реплики.

    Нейротизм (touchiness) усиливает только негативные сигналы. Положительная
    реакция не должна зависеть от ранимости.
    """
    normalized = " ".join((text or "").lower().replace("ё", "е").split())
    if not normalized:
        return None

    sensitivity = max(0.0, min(1.0, float(touchiness)))

    if _has_phrase(normalized, _DIRECT_HOSTILITY):
        return EmotionalReaction(
            pleasure=-0.12,
            arousal=0.10,
            dominance=-0.04,
            cause="прямая враждебность",
            sensitivity=0.75 + sensitivity * 0.5,
        )

    if _has_phrase(normalized, _FRUSTRATION):
        return EmotionalReaction(
            pleasure=-0.055,
            arousal=0.035,
            dominance=-0.015,
            cause="недовольство ответом",
            sensitivity=0.85 + sensitivity * 0.3,
        )

    if _has_phrase(normalized, _APPRECIATION):
        return EmotionalReaction(
            pleasure=0.075,
            arousal=0.025,
            dominance=0.02,
            cause="благодарность или похвала",
            sensitivity=1.0,
        )

    return None
