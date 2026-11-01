"""Persistent, explicit user preferences for conversational style.

Only clear, directly stated feedback is stored. The module deliberately avoids
LLM-based inference so casual questions are not silently turned into preferences.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


_GREETING_WORDS = ("привет", "здравств", "поздоров", "приветств")
_REQUEST_WORDS = (
    "не ", "реже", "пореже", "без ", "хватит", "перестань",
    "не надо", "не нужно", "избегай", "поменьше",
)


class ConversationPreferences:
    def __init__(self, *, avoid_repeated_greetings: bool = False):
        self.avoid_repeated_greetings = bool(avoid_repeated_greetings)

    def observe_feedback(self, text: str) -> bool:
        """Persist an explicit request to reduce unnecessary greetings."""
        normalized = " ".join((text or "").lower().replace("ё", "е").split())
        if not normalized:
            return False
        mentions_greeting = any(word in normalized for word in _GREETING_WORDS)
        requests_change = any(word in normalized for word in _REQUEST_WORDS)
        if mentions_greeting and requests_change and not self.avoid_repeated_greetings:
            self.avoid_repeated_greetings = True
            return True
        return False

    def directive(self) -> str:
        if not self.avoid_repeated_greetings:
            return ""
        return (
            "УСТОЙЧИВОЕ ПРЕДПОЧТЕНИЕ СОБЕСЕДНИКА: не начинай обычные ответы "
            "с приветствия и не повторяй приветствие посреди продолжающегося "
            "разговора. Отвечай сразу по существу. Приветствие при настоящем "
            "начале новой сессии допустимо; это предпочтение не запрещает его "
            "полностью. Не нужно каждый раз проговаривать, что ты учла эту просьбу."
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"avoid_repeated_greetings": self.avoid_repeated_greetings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ConversationPreferences":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                avoid_repeated_greetings=data.get("avoid_repeated_greetings", False)
            )
        except (OSError, ValueError, TypeError):
            return cls()
